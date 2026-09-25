#!/usr/bin/env python3
# Adopted from the k3 dense-gemv review (the reviewer's independent harness); only the output
# path and comments that described the since-fixed smem-attribute defect were changed.
"""Reviewer correctness hunt (k3 dense-gemv).

1. Edge activations through the real hooked vLLM path (apply_weights armed vs disarmed = stock
   mxfp8_e4m3_quantize + b12x), every M 1..8, bitwise (bf16 bits incl. NaN payloads):
   zero rows (CUDA-graph padding rows), one live row + zero rows, blocks whose absmax/448 is an
   exact power of two, bf16 subnormals, bf16 max (e4m3 saturation / ue8m0 clamp), -0.0,
   single +inf, single NaN, constant rows, a row-strided (non-contiguous) input.
2. Race hunt: GEMV launches on stream A while co-resident noise kernels run on stream B
   (perturbs warp scheduling / CTA start skew), N launches per (shape, M, config), each output
   compared bitwise on-GPU to the stock reference (mismatch counter, no per-iteration sync).
Each shape is armed right before it is exercised (serve-order arming: serve_hook_test.py).
"""
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, "/repo/kernel_study/dense_gemv")
import weights  # noqa: E402

assert os.environ.get("DSV41_DENSE_GEMV") == "1"
import dense_gemv  # noqa: E402
from torch.nn import Parameter  # noqa: E402
from vllm.model_executor.kernels.linear.mxfp8 import flashinfer as fi_mod  # noqa: E402
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize  # noqa: E402

Kern = fi_mod.FlashInferCutlassMxfp8LinearKernel
BF16_MAX = 3.3895313892515355e38


def arm(name, li):
    _, N, K, _ = weights.SHAPES[name]
    w, s2d = weights.load(name, li)
    layer = torch.nn.Module()
    layer.weight = Parameter(w, requires_grad=False)
    layer.weight_scale = Parameter(s2d, requires_grad=False)
    kern = object.__new__(Kern)
    kern.process_weights_after_loading(layer)
    assert layer._dsv41_gemv is not None, name
    return kern, layer, N, K


def edge_inputs(M, K, gen):
    dev = "cuda"
    base = torch.randn(M, K, generator=gen, device=dev)
    out = {}
    out["zeros_all"] = torch.zeros(M, K, device=dev)
    z = torch.zeros(M, K, device=dev)
    z[0] = base[0]
    out["one_live_rest_zero_rows"] = z  # graph padding: live row(s) + zeroed pad rows
    p = base.clone()
    blk = p.view(M, K // 32, 32)
    blk[:, :, 0] = 448.0 * torch.exp2(torch.randint(-20, 20, (M, K // 32), generator=gen, device=dev).float())
    blk[:, :, 1:] = blk[:, :, 1:].clamp(-1, 1) * blk[:, :, :1].abs() * 0.5
    out["pow2_absmax"] = p
    out["subnormal"] = base * 1e-39
    out["tiny_mixed"] = base * torch.exp2(torch.randint(-140, -100, (M, K), generator=gen, device=dev).float())
    hb = base.clone()
    hb[:, ::97] = BF16_MAX
    out["bf16_max"] = hb
    out["huge_mixed"] = base * 1e35
    nz = base.clone()
    nz[:, ::3] = -0.0
    out["neg_zero"] = nz
    out["const_rows"] = torch.full((M, K), 0.3333, device=dev)
    inf = base.clone()
    inf[0, 5] = float("inf")
    out["one_inf"] = inf
    nan = base.clone()
    nan[0, 7] = float("nan")
    out["one_nan"] = nan
    return {k: v.to(torch.bfloat16) for k, v in out.items()}


def edge_check(kern, layer, N, K, gen, res, name):
    armed = layer._dsv41_gemv
    rows = {}
    for M in range(1, 9):
        if armed.plan(M) is None:
            continue
        cases = edge_inputs(M, K, gen)
        big = torch.randn(M, K + 128, generator=gen, device="cuda").to(torch.bfloat16)
        cases["row_strided"] = big[:, :K]
        for cname, x in cases.items():
            layer._dsv41_gemv = armed
            try:
                y = kern.apply_weights(layer, x)
                used = True
            except Exception as exc:  # noqa: BLE001
                rows[f"{cname}@M{M}"] = f"EXC {exc!r}"[:160]
                continue
            layer._dsv41_gemv = None
            try:
                yref = kern.apply_weights(layer, x)
            except Exception as exc:  # noqa: BLE001
                rows[f"{cname}@M{M}"] = f"stock EXC {exc!r}"[:160]
                layer._dsv41_gemv = armed
                continue
            layer._dsv41_gemv = armed
            torch.cuda.synchronize()
            eq = torch.equal(y.view(torch.int16), yref.view(torch.int16))
            if not eq:
                d = (y.float() - yref.float()).abs()
                fin = torch.isfinite(y.float()) & torch.isfinite(yref.float())
                rows[f"{cname}@M{M}"] = (f"MISMATCH nbits_diff={int((y.view(torch.int16) != yref.view(torch.int16)).sum())} "
                                         f"maxabs_finite={float(d[fin].max()) if fin.any() else 'n/a'} "
                                         f"nan_y={int(torch.isnan(y.float()).sum())} nan_ref={int(torch.isnan(yref.float()).sum())}")
            else:
                rows[f"{cname}@M{M}"] = "ok"
    bad = {k: v for k, v in rows.items() if v != "ok"}
    res["edge"][name] = {"cases": len(rows), "bad": bad}
    print(f"edge {name:15s}: {len(rows) - len(bad)}/{len(rows)} bitwise" + (f" BAD: {json.dumps(bad)[:1500]}" if bad else ""),
          flush=True)


def race_hunt(kern, layer, N, K, gen, res, name, iters):
    armed = layer._dsv41_gemv
    ext = dense_gemv._ext()
    sB = torch.cuda.Stream()
    noise = torch.randn(8 << 20, device="cuda")
    out = {}
    for M in (1, 3, 4, 6, 8):
        pl = armed.plan(M)
        if pl is None:
            continue
        W, S, MR, grid = pl
        ext.plan_grid(W, S, armed.kc, MR, armed.smode, 0, N, K)
        x = torch.randn(M, K, generator=gen, device="cuda").to(torch.bfloat16)
        layer._dsv41_gemv = None
        ref = kern.apply_weights(layer, x).reshape(M, N)
        q, s = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
        layer._dsv41_gemv = armed
        torch.cuda.synchronize()
        bad = torch.zeros((), dtype=torch.int64, device="cuda")
        y = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        wu = layer.weight.view(torch.uint8)
        for i in range(iters):
            with torch.cuda.stream(sB):
                for _ in range(1 + (i % 3)):
                    noise.mul_(1.0000001)
            if i % 2 == 0:
                ext.gemv(x, None, None, wu, armed.scales, armed.smode, y, W, S, armed.kc, MR, grid, False)
            else:  # pre-quantized input path (same numerics by construction)
                ext.gemv(None, q.view(torch.uint8), s.view(torch.uint8), wu, armed.scales, armed.smode, y, W, S,
                         armed.kc, MR, grid, False)
            bad += (y.view(torch.int16) != ref.view(torch.int16)).any()
            if i % 500 == 499:
                torch.cuda.current_stream().wait_stream(sB)
        torch.cuda.synchronize()
        out[M] = {"iters": iters, "mismatch_launches": int(bad)}
    res["race"][name] = out
    print(f"race {name:15s}: " + ", ".join(f"M{m} {v['mismatch_launches']}/{v['iters']}" for m, v in out.items()), flush=True)


def main() -> int:
    gen = torch.Generator(device="cuda").manual_seed(31337)
    res = {"edge": {}, "race": {}}
    iters = int(os.environ.get("RACE_ITERS", "3000"))
    for name, li in (("qkv_a", 7), ("wq_b", 7), ("wo_b", 7), ("shared_gate_up", 7), ("shared_down", 7),
                     ("engram_wkv", 1), ("main_proj", 0)):
        kern, layer, N, K = arm(name, li)
        edge_check(kern, layer, N, K, gen, res, name)
        race_hunt(kern, layer, N, K, gen, res, name, iters if name not in ("engram_wkv", "main_proj") else 300)
        del kern, layer
        torch.cuda.empty_cache()
    json.dump(res, open("/repo/results/2026-09-25-kernels/dense-gemv/edge-stress.json", "w"), indent=1)
    nbad = sum(len(v["bad"]) for v in res["edge"].values()) + sum(
        r["mismatch_launches"] for v in res["race"].values() for r in v.values())
    print(f"TOTAL problems: {nbad}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
