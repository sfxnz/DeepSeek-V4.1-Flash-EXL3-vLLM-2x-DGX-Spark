#!/usr/bin/env python3
"""Indexer weights_proj (bf16 [32, 5120], M = decode rows): cuBLAS vs a Triton GEMV.

cuBLAS runs this shape as cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1
on 2 one-warp CTAs (serve trace: 70-114 us per call on the indexer's aux stream,
the last branch to finish at the qkv join in 8 layers a step). The Triton kernel
keeps each output tile's MMA chain in the same k order (tl.dot accumulating
over k), so its bf16 output must equal cuBLAS's bit for bit; it only schedules
the loads better.

Runs in the serving image on a GPU. Real weights: the 8 indexer weights_proj
tensors of the pack. For every config: bitwise check vs F.linear at M 2..8 on
all 8 layers (at M = 1 cuBLAS picks another kernel with another reduction
order: a first sweep over 9 configs matched it only by chance, 1-3 of 16 cases
off per config, so the lever leaves M = 1 to cuBLAS); the serve wrapper
(self-test, per-call verify, stock fallback at M 1/9/16, tuple return) and 20
CUDA-graph replays of it, bitwise; then timing in CUDA graphs, cold L2 (a
128 MiB write between replays; GB10 L2 is 24 MiB) and warm, alone and beside
a DRAM-saturating copy on another stream, >= 300 replays, arms alternating.
Prints one JSON line.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys

import torch

sys.path.insert(0, "/opt/dsv41-patch")

SNAP = "/hf/hub/models--sfxnz--DeepSeek-V4.1-Flash-EXL3/snapshots/2.0bpw-mcg-lmhead-mxfp8"
LAYERS = (2, 8, 14, 20, 24, 28, 32, 36)


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def summarize(v):
    return dict(n=len(v), median=statistics.median(v), p10=pct(v, 0.10), p90=pct(v, 0.90), mean=statistics.fmean(v))


def load_weights():
    from safetensors import safe_open

    idx = json.load(open(SNAP + "/model.safetensors.index.json"))["weight_map"]
    out = []
    for layer in LAYERS:
        name = f"layers.{layer}.attn.indexer.weights_proj.weight"
        with safe_open(SNAP + "/" + idx[name], framework="pt") as f:
            out.append(f.get_tensor(name).cuda().contiguous())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--configs", default="256:4:3,512:2:3,512:4:4")
    args = ap.parse_args()

    import indexer_wp_gemv as wpg
    from vllm.triton_utils import tl, triton

    kernel = wpg._build_kernel(tl, triton)
    ws = load_weights()
    dev = torch.device("cuda")
    g = torch.Generator(device="cuda").manual_seed(0)
    configs = [tuple(int(v) for v in c.split(":")) for c in args.configs.split(",")]
    out = {"torch": torch.__version__, "configs": {}, "bad": []}

    def run(cfg, x, w, o):
        bk, warps, stages = cfg
        wpg.launch(kernel, triton, x, w, o, bk, warps, stages)

    # bitwise: every config, every layer, M = MIN_M..MAX_M (cuBLAS takes another
    # kernel at M = 1, so the lever leaves M = 1 to it), x ~ N(0,1) and wide range
    for cfg in configs:
        ok = True
        for li, w in enumerate(ws):
            for m in range(wpg.MIN_M, wpg.MAX_M + 1):
                for scale in (1.0, 30.0):
                    x = (torch.randn(m, 5120, device=dev, generator=g) * scale).to(torch.bfloat16)
                    ref = torch.nn.functional.linear(x, w)
                    o = torch.empty(m, w.shape[0], device=dev, dtype=torch.bfloat16)
                    run(cfg, x, w, o)
                    torch.cuda.synchronize()
                    if not torch.equal(o.view(torch.int16), ref.view(torch.int16)):
                        ok = False
                        out["bad"].append(f"cfg={cfg} layer={LAYERS[li]} m={m} scale={scale}")
        out["configs"][":".join(map(str, cfg))] = {"bitwise": ok}

    # the serve wrapper (make_forward) on a stand-in layer: self-test, verify,
    # stock fallback for M outside MIN_M..MAX_M, tuple return like ReplicatedLinear
    import types as _types

    wpg._STATE.update(armed=True, verify_left=8, engaged=False)
    for li, w in enumerate(ws):
        layer = _types.SimpleNamespace(weight=w, bias=None, return_bias=True)
        stock = lambda x, w=w: (torch.nn.functional.linear(x, w), None)
        fwd = wpg.make_forward(torch, triton, kernel, stock)
        for m in (1, 2, 3, 4, 6, 8, 9, 16):
            x = torch.randn(m, 5120, device=dev, generator=g).to(torch.bfloat16)
            got, bias_out = fwd(layer, x)
            ref = torch.nn.functional.linear(x, w)
            torch.cuda.synchronize()
            if bias_out is not None or not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
                out["bad"].append(f"wrapper layer={LAYERS[li]} m={m}")
    out["wrapper_state"] = dict(wpg._STATE)
    if not wpg._STATE["engaged"] or not wpg._STATE["armed"]:
        out["bad"].append("wrapper not engaged")

    # graph capture of the wrapper, replayed with new rows
    x_s = torch.randn(4, 5120, device=dev, generator=g).to(torch.bfloat16)
    layer = _types.SimpleNamespace(weight=ws[0], bias=None, return_bias=True)
    fwd = wpg.make_forward(torch, triton, kernel, lambda x: (torch.nn.functional.linear(x, ws[0]), None))
    fwd(layer, x_s)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        g_out, _ = fwd(layer, x_s)
    for rep in range(20):
        x_s.copy_(torch.randn(4, 5120, device=dev, generator=g).to(torch.bfloat16))
        gr.replay()
        ref = torch.nn.functional.linear(x_s, ws[0])
        torch.cuda.synchronize()
        if not torch.equal(g_out.view(torch.int16), ref.view(torch.int16)):
            out["bad"].append(f"graph replay {rep}")
    out["graph_replays"] = 20

    flush = torch.empty(128 << 20, dtype=torch.uint8, device=dev)
    side = torch.cuda.Stream()
    big_src = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
    big_dst = torch.empty_like(big_src)
    w = ws[4]
    timing = {}
    for m in (1, 3, 4, 6, 8):
        x = torch.randn(m, 5120, device=dev, generator=g).to(torch.bfloat16)
        o = torch.empty(m, 32, device=dev, dtype=torch.bfloat16)
        arms = {"cublas": lambda: torch.nn.functional.linear(x, w, out=None)}
        for cfg in configs:
            arms[":".join(map(str, cfg))] = (lambda cfg=cfg: run(cfg, x, w, o))
        graphs = {}
        for name, fn in arms.items():
            fn()
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                fn()
            graphs[name] = gr
        names = list(graphs)
        for mode in ("cold", "warm", "beside_copy"):
            t = {n: [] for n in names}
            for it in range(args.warmup + args.iters):
                order = names[it % len(names):] + names[: it % len(names)]
                for name in order:
                    if mode == "cold":
                        flush.fill_(it & 0xFF)
                    if mode == "beside_copy":
                        side.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(side):
                            big_dst.copy_(big_src)
                        torch.cuda._sleep(20000)
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    graphs[name].replay()
                    e1.record()
                    torch.cuda.synchronize()
                    if it >= args.warmup:
                        t[name].append(e0.elapsed_time(e1) * 1e3)
            timing[f"m{m}_{mode}"] = {n: summarize(v) for n, v in t.items()}
    out["timing_us"] = timing
    clk = torch.cuda.clock_rate() if hasattr(torch.cuda, "clock_rate") else None
    out["sm_clock_mhz"] = clk
    print(json.dumps(out))
    return 1 if out["bad"] else 0


if __name__ == "__main__":
    sys.exit(main())
