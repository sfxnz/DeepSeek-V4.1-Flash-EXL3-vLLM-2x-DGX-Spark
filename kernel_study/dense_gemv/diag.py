#!/usr/bin/env python3
"""Where does the v2 GEMV lose to b12x? Cold timing of diagnostic variants.

DIAG1 = cp.async ring only (no ldmatrix/MMA), DIAG2 = no activation staging,
plus a pipeline-depth / warps-per-CTA / grid sweep. Same cold protocol as
timing.py (rotation + double hashed flush, ABBA, events)."""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weights  # noqa: E402
from benchutil import Flusher, Rotation, copies_for, load_ext, stats, time_arms  # noqa: E402
from timing import build_copies  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shapes", default="qkv_a,wo_b,wq_b")
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    ext = load_ext("dgemv_v1", ["gemv_ext.cu"])
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
    from vllm.utils import flashinfer as vfi

    fl = Flusher(ext)
    o = torch.zeros(4, dtype=torch.int32, device="cuda")
    res = {}
    M = args.M
    for name in args.shapes.split(","):
        _, N, K, _ = weights.SHAPES[name]
        R = copies_for(N * K)
        copies = build_copies(name, R)
        x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
        q, s = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
        y = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        T = N // 16
        fns = {
            "b12x_gemm": lambda c: vfi.mm_mxfp8(q, c[0].view(torch.float8_e4m3fn).t(), s, c[1],
                                                 out_dtype=torch.bfloat16, backend="auto"),
            "read": lambda c: ext.read_flat(c[0], o, 48, 2),
            "diag1_nomma_w4s6": lambda c: ext.gemv(x, None, None, c[0], c[2], 11, y, 4, 6, 128, (T + 3) // 4, False),
            "diag2_nostage_w4s6": lambda c: ext.gemv(x, None, None, c[0], c[2], 12, y, 4, 6, 128, (T + 3) // 4, False),
        }
        for W, S, KS in [(4, 4, 128), (4, 6, 128), (4, 8, 128), (2, 4, 128), (2, 6, 128), (2, 8, 128), (2, 12, 128),
                         (2, 4, 256), (2, 6, 256), (1, 8, 128), (1, 16, 128), (8, 3, 128)]:
            smem = (K // 32) * 264 + W * S * (16 * KS + 16)
            if K % KS or smem > 101376:
                continue
            grids = {(T + W - 1) // W}
            if name == "wq_b":
                grids |= {48 * g for g in (1, 2, 3) if 48 * g * W <= T}
            for g in sorted(grids):
                fns[f"gv_w{W}s{S}k{KS}g{g}"] = (lambda c, W=W, S=S, KS=KS, g=g:
                                                 ext.gemv(x, None, None, c[0], c[2], 0, y, W, S, KS, g, False))
        names = list(fns)
        rot = Rotation(copies, narms=len(names))
        arms = {n: (lambda n=n, k=k: fns[n](rot.get(k))) for k, n in enumerate(names)}
        t = time_arms(arms, iters=args.iters, pre=fl)
        res[name] = {n: stats(v) for n, v in t.items()}
        print(f"== {name} M={M} (cold median us, p10, p90)", flush=True)
        for n in names:
            st = res[name][n]
            print(f"   {n:28s} {st['median']:8.2f} {st['p10']:8.2f} {st['p90']:8.2f}", flush=True)
        del copies
        torch.cuda.empty_cache()
    json.dump(res, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
