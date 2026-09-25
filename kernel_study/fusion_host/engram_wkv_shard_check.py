#!/usr/bin/env python3
"""Engram wkv as a column-parallel layer: is each rank's half bit-identical, and how fast?

The Engram wkv is a ReplicatedLinear [25600, 6144] (MXFP8 e4m3 weight, ue8m0
32x32 block scales) on both TP ranks: each rank reads the whole 157 MB weight
per Engram layer (2 layers a step). Column-parallel with the output gathered
(ColumnParallelLinear(gather_output=True)) has each rank compute 12800 output
columns and all-gather them. That is bit-exact only if the production GEMM
computes each output column the same way whatever N is (no N-dependent split-K
or tile choice).

Runs in the serving image on a GPU, real weights read-only from the pack
(--snapshot). The production path is FlashInferCutlassMxfp8LinearKernel
(backend "auto" -> b12x on sm_121): mxfp8_e4m3_quantize(x, swizzled) +
mm_mxfp8(q, W.t(), s, swizzle_mxfp8_scale(scale_2d)).

1. bitwise: layers 1 and 14, M in 1..8 and 16..2048 (decode and prefill
   chunk sizes), four activation distributions; y_full[:, :12800] must equal
   y_half0 and y_full[:, 12800:] y_half1, raw bf16 bits, every element.
2. timing: the layer call (activation quant + GEMM) captured in CUDA graphs at
   M 1/3/4/6/8, full vs one half, >= 300 replays alternating, a ~90 us GPU
   sleep before the start event hides the graph launch, cold L2 (a 128 MiB
   write before each replay; GB10 L2 is 24 MiB) and warm; achieved GB/s =
   weight + scale bytes / time.
Prints one JSON line; exit 1 on any bit difference.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch

N_FULL, K = 25600, 6144
LAYERS = (1, 14)


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def summarize(v):
    return dict(n=len(v), median=round(statistics.median(v), 2), p10=round(pct(v, 0.10), 2),
                p90=round(pct(v, 0.90), 2))


def load(snapshot, layer):
    """(weight e4m3 [N, K], per-row ue8m0 scale [N, K/32]) as the KMxfp8Static loader expands it."""
    from safetensors import safe_open

    idx = json.load(open(os.path.join(snapshot, "model.safetensors.index.json")))["weight_map"]
    name = f"layers.{layer}.engram.wkv"
    with safe_open(os.path.join(snapshot, idx[name + ".weight"]), framework="pt") as fh:
        w = fh.get_tensor(name + ".weight")
    with safe_open(os.path.join(snapshot, idx[name + ".scale"]), framework="pt") as fh:
        s = fh.get_tensor(name + ".scale").view(torch.uint8)
    assert tuple(w.shape) == (N_FULL, K) and tuple(s.shape) == (N_FULL // 32, K // 32), (w.shape, s.shape)
    return w.cuda().contiguous(), s.repeat_interleave(32, dim=0).cuda().contiguous()


def make_x(m, dist, g):
    x = torch.randn(m, K, generator=g, device="cuda")
    if dist == "lognormal":
        x = x * torch.exp(1.5 * torch.randn(m, K, generator=g, device="cuda"))
    elif dist == "outlier":
        x[:, torch.randint(0, K, (8,), generator=g, device="cuda")] *= 300.0
    elif dist == "edge":
        pick = torch.randint(0, 6, (m, K), generator=g, device="cuda")
        e = torch.randint(-20, 20, (m, K), generator=g, device="cuda").float()
        x = torch.where(pick == 0, torch.exp2(e), x)
        x = torch.where(pick == 1, 448.0 * torch.exp2(e), x)
        x = torch.where(pick == 2, torch.zeros_like(x), x)
        x = torch.where(pick == 3, x * 1e-30, x)
        x[:, :32] = 0.0
    return x.to(torch.bfloat16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
        swizzle_mxfp8_scale,
    )
    from vllm.utils import flashinfer as vfi

    def prep(w, s2d):
        n = w.shape[0]
        return w.contiguous(), swizzle_mxfp8_scale(s2d.contiguous(), M=n, K=K).contiguous()

    def call(x, wt):
        w, ssw = wt
        q, s = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
        return vfi.mm_mxfp8(q, w.t(), s, ssw, out_dtype=torch.bfloat16, backend="auto")

    g = torch.Generator(device="cuda").manual_seed(11)
    out = {"torch": torch.__version__, "device": torch.cuda.get_device_name(), "bitwise": {}, "timing_us": {}}
    bad = []
    half = N_FULL // 2
    tensors = {}
    for layer in LAYERS:
        w, s2d = load(args.snapshot, layer)
        tensors[layer] = {"full": prep(w, s2d), "h0": prep(w[:half], s2d[:half]), "h1": prep(w[half:], s2d[half:])}
        del w, s2d
    torch.cuda.synchronize()

    # 1. bitwise
    ms = list(range(1, 9)) + [16, 32, 64, 128, 256, 512, 1024, 2048]
    n_cases = n_diff_cases = 0
    for layer in LAYERS:
        t = tensors[layer]
        for m in ms:
            for dist in ("normal", "lognormal", "outlier", "edge"):
                x = make_x(m, dist, g)
                yf = call(x, t["full"])
                y0 = call(x, t["h0"])
                y1 = call(x, t["h1"])
                torch.cuda.synchronize()
                d0 = int((yf[:, :half].view(torch.int16) != y0.view(torch.int16)).sum())
                d1 = int((yf[:, half:].view(torch.int16) != y1.view(torch.int16)).sum())
                n_cases += 1
                if d0 or d1:
                    n_diff_cases += 1
                    bad.append(f"layer {layer} m {m} {dist}: {d0} + {d1} of {yf.numel()} elements differ")
    out["bitwise"] = {"cases": n_cases, "cases_with_differences": n_diff_cases, "m": ms}

    # 2. timing (layer 1)
    t = tensors[LAYERS[0]]
    flush = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
    wbytes = {"full": N_FULL * K + N_FULL * K // 32, "h0": half * K + half * K // 32}
    for m in (1, 3, 4, 6, 8):
        x = make_x(m, "normal", g)
        graphs = {}
        for name in ("full", "h0"):
            call(x, t[name])
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                call(x, t[name])
            graphs[name] = gr
        for l2 in ("cold", "warm"):
            times = {"full": [], "h0": []}
            for it in range(args.warmup + args.iters):
                for name in (("full", "h0") if it % 2 == 0 else ("h0", "full")):
                    if l2 == "cold":
                        flush.fill_(it & 0xFF)
                    torch.cuda._sleep(200_000)
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    graphs[name].replay()
                    e1.record()
                    torch.cuda.synchronize()
                    if it >= args.warmup:
                        times[name].append(e0.elapsed_time(e1) * 1e3)
            cell = {k: summarize(v) for k, v in times.items()}
            for k in cell:
                cell[k]["GBps"] = round(wbytes[k] / (cell[k]["median"] * 1e-6) / 1e9, 1)
                cell[k]["pct_of_250"] = round(100 * cell[k]["GBps"] / 250, 1)
            cell["delta_us"] = round(cell["full"]["median"] - cell["h0"]["median"], 2)
            out["timing_us"][f"m{m}_{l2}"] = cell
    out["mismatches"] = bad
    print(json.dumps(out))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
