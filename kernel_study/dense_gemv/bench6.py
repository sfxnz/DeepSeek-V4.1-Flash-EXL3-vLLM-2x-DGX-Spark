#!/usr/bin/env python3
"""v6 study: staging warps (SW = 1, 2) vs the v5 schedule (SW = 0) on the production
configs (dense_gemv.CONFIGS). Bitwise check vs b12x, then cold timing (protocol:
benchutil docstring; activation re-touched after the flush)."""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../docker/patch"))
import dense_gemv  # noqa: E402
import weights  # noqa: E402
from benchutil import PEAK_GBPS, Flusher, Rotation, copies_for, load_ext, stats, time_arms  # noqa: E402
from correctness import b12x_ref, make_x  # noqa: E402
from timing import build_copies  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shapes", default="qkv_a,wo_b,wq_b,shared_gate_up,shared_down")
    ap.add_argument("--ms", default="1,4,8")
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    ext = load_ext("dgemv_v1", ["gemv_ext.cu"])
    e6 = load_ext("dgemv_v6", ["gemv6_ext.cu"])
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
    from vllm.utils import flashinfer as vfi

    fl = Flusher(ext)
    o = torch.zeros(4, dtype=torch.int32, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(5)
    report = {}
    for name in args.shapes.split(","):
        _, N, K, _ = weights.SHAPES[name]
        key = (K, N)
        _, kc, smode, buckets = dense_gemv.CONFIGS[key]
        R = max(copies_for(N * K), 64 if N * K < (64 << 20) else 4)
        copies = build_copies(name, R)
        scs = {}
        for c in copies:
            w, wsw, sc = c
            from correctness import unswizzle
            s2d = unswizzle(wsw, N, K // 32)
            scs[id(w)] = dense_gemv.build_scales(s2d, N, K, kc, smode)
        rep = report[name] = {"correct": {}, "M": {}}
        for M in [int(m) for m in args.ms.split(",")]:
            W, S, MR = dense_gemv.pick(buckets, M)
            grids = {}
            for sw in (0, 1, 2):
                e6.set_sw(sw)
                grids[sw] = int(e6.plan_grid(W, S, kc, MR, smode, 0, N, K))
            # correctness (2 layers x 2 distributions, both input kinds)
            nbad = 0
            for c in copies[:2]:
                for dist in ("normal", "lognormal"):
                    x = make_x(M, K, dist, gen)
                    q, s, yref = b12x_ref(x, c[0].view(torch.float8_e4m3fn), c[1])
                    for sw in (0, 1, 2):
                        e6.set_sw(sw)
                        y1 = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
                        e6.gemv(x, None, None, c[0], scs[id(c[0])], smode, y1, W, S, kc, MR, grids[sw], False)
                        y2 = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
                        e6.gemv(None, q.view(torch.uint8), s.view(torch.uint8), c[0], scs[id(c[0])], smode, y2,
                                W, S, kc, MR, grids[sw], False)
                        torch.cuda.synchronize()
                        ok = torch.equal(y1.view(torch.int16), yref.view(torch.int16)) and torch.equal(
                            y2.view(torch.int16), yref.view(torch.int16))
                        nbad += not ok
            rep["correct"][M] = {"bad": nbad, "cases": 2 * 2 * 3}
            x = torch.randn(M, K, generator=gen, device="cuda").to(torch.bfloat16)
            q, s = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
            qu, su = q.view(torch.uint8), s.view(torch.uint8)
            y = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

            def b12x_e2e(c):
                qq, ss = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
                vfi.mm_mxfp8(qq, c[0].view(torch.float8_e4m3fn).t(), ss, c[1], out_dtype=torch.bfloat16,
                             backend="auto")

            fns = {"b12x_e2e": b12x_e2e,
                   "b12x_gemm": lambda c: vfi.mm_mxfp8(q, c[0].view(torch.float8_e4m3fn).t(), s, c[1],
                                                       out_dtype=torch.bfloat16, backend="auto")}
            for sw in (0, 1, 2):
                def f_fused(c, sw=sw):
                    e6.set_sw(sw)
                    e6.gemv(x, None, None, c[0], scs[id(c[0])], smode, y, W, S, kc, MR, grids[sw], False)

                def f_preq(c, sw=sw):
                    e6.set_sw(sw)
                    e6.gemv(None, qu, su, c[0], scs[id(c[0])], smode, y, W, S, kc, MR, grids[sw], False)
                fns[f"fused_sw{sw}"] = f_fused
                fns[f"preq_sw{sw}"] = f_preq
            names = list(fns)
            rot = Rotation(copies, narms=len(names))
            arms = {n: (lambda n=n, k=k: fns[n](rot.get(k))) for k, n in enumerate(names)}

            def pre():
                fl()
                for t in (x, qu, su):
                    ext.read_flat(t.view(torch.uint8), o, 8, 2)

            t = time_arms(arms, iters=args.iters, pre=pre)
            rep["M"][M] = {n: stats(v) for n, v in t.items()}
            med = {n: rep["M"][M][n]["median"] for n in names}
            print(f"{name:14s} M={M} bad={nbad} e2e {med['b12x_e2e']:7.2f} gemm {med['b12x_gemm']:7.2f} | "
                  + " ".join(f"{n} {med[n]:7.2f}" for n in names if "sw" in n), flush=True)
        del copies
        torch.cuda.empty_cache()
        json.dump(report, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
