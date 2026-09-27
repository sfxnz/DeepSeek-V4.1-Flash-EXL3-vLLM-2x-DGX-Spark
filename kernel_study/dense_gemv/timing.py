#!/usr/bin/env python3
"""Dense GEMV vs production b12x: cold/warm per-call timing on real weights.

Protocol: benchutil docstring (rotation over >= 16 distinct copies of real
layer weights, >= 512 MiB, double hashed 64 MiB flush before every timed call
for cold; 300 us memory-free spin for warm; CUDA events per call; ABBA
interleave of all arms in one process; >= 200 timed iterations after warmup).

Arms (all on the same copies):
  b12x_e2e   serve path from bf16: FlashInfer mxfp8_quantize + mm_mxfp8(auto=b12x)
  b12x_gemm  mm_mxfp8 on a pre-quantized activation (the wq_b path)
  gv_fused   GEMV from bf16 (fused quant)              -> replaces b12x_e2e
  gv_preq    GEMV from the pre-quantized activation    -> replaces b12x_gemm
  read       flat streaming read of the weight bytes only (in-process roofline)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weights  # noqa: E402
from benchutil import PEAK_GBPS, Flusher, Rotation, copies_for, load_ext, stats, time_arms  # noqa: E402


def build_copies(name: str, R: int):
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import swizzle_mxfp8_scale

    kind, N, K, layers = weights.SHAPES[name]
    base = []
    for layer in layers[: min(len(layers), R)]:
        w, s2d = weights.load(name, layer)
        wsw = swizzle_mxfp8_scale(s2d, M=N, K=K).contiguous()
        sc = weights.compact_scale(s2d)
        base.append((w.view(torch.uint8), wsw, sc))
        del s2d
    copies = list(base)
    i = 0
    while len(copies) < R:  # distinct allocations of real layers
        w, wsw, sc = base[i % len(base)]
        copies.append((w.clone(), wsw.clone(), None if sc is None else sc.clone()))
        i += 1
    return copies


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shapes", default="qkv_a,wq_b,wo_b,shared_gate_up,shared_down,indexer_wq_b")
    ap.add_argument("--ms", default="1,3,4,6,8")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--cfgs", default="4,4,128", help="';'-separated W,STAGES,KSPAN[,grid] GEMV configs")
    ap.add_argument("--no-warm", action="store_true")
    ap.add_argument("--min-total-mib", type=int, default=512)
    ap.add_argument("--impl", default="v1", choices=["v1", "v3"])
    ap.add_argument("--pdl", action="store_true", help="launch the GEMV arms with the PDL attribute")
    args = ap.parse_args()
    ext = load_ext("dgemv_v1", ["gemv_ext.cu"])
    ext3 = load_ext("dgemv_v3", ["gemv3_ext.cu"]) if args.impl == "v3" else None
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
    from vllm.utils import flashinfer as vfi

    flusher = Flusher(ext)
    o = torch.zeros(4, dtype=torch.int32, device="cuda")
    cfgs = [tuple(int(v) for v in c.split(",")) for c in args.cfgs.split(";") if c]
    gen = torch.Generator(device="cuda").manual_seed(7)
    res = {"protocol": "rotation+double-hashed-flush; warm = 300us spin, same copy", "shapes": {}}
    for name in [s for s in args.shapes.split(",") if s]:
        kind, N, K, _ = weights.SHAPES[name]
        wbytes = N * K
        R = copies_for(wbytes, min_total=args.min_total_mib << 20)
        copies = build_copies(name, R)
        sm = 0 if copies[0][2] is not None else 1
        res["shapes"][name] = {"N": N, "K": K, "copies": R, "scale_mode": sm, "M": {}}
        for M in [int(m) for m in args.ms.split(",")]:
            x = (torch.randn(M, K, generator=gen, device="cuda")).to(torch.bfloat16)
            q, s = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
            qu, su = q.view(torch.uint8), s.view(torch.uint8)
            y = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
            arms = {}

            def mk_b12x_e2e(c):
                w, wsw, _ = c
                qq, ss = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
                vfi.mm_mxfp8(qq, w.view(torch.float8_e4m3fn).t(), ss, wsw, out_dtype=torch.bfloat16, backend="auto")

            def mk_b12x_gemm(c):
                w, wsw, _ = c
                vfi.mm_mxfp8(q, w.view(torch.float8_e4m3fn).t(), s, wsw, out_dtype=torch.bfloat16, backend="auto")

            names = ["b12x_e2e", "b12x_gemm", "read"]
            fns = {"b12x_e2e": mk_b12x_e2e, "b12x_gemm": mk_b12x_gemm,
                   "read": lambda c: ext.read_flat(c[0], o, 48, 2)}
            for cfg in cfgs:
                W, STG, KSP = cfg[:3]
                if ext3 is not None:
                    smem = ext3.smem3(W, STG, KSP, K)
                else:
                    smem = (K // 32) * 264 + W * STG * (16 * KSP + (16 if sm == 0 else 16 * (KSP // 32)))
                if K % KSP or smem > 101376:
                    continue
                if ext3 is not None:
                    cps = max(1, 102400 // (smem + 1024))
                    grid = cfg[3] if len(cfg) > 3 else min((N // 16 + W - 1) // W, 48 * cps)
                else:
                    grid = cfg[3] if len(cfg) > 3 else (N // 16 + W - 1) // W
                tag = f"w{W}s{STG}k{KSP}g{grid}"
                if ext3 is not None:
                    v3s = {id(c[0]): weights.v3_scales_from(c, N, K, KSP) for c in copies}

                    def f_fused(c, W=W, STG=STG, KSP=KSP, grid=grid, v3s=v3s):
                        m3, t3 = v3s[id(c[0])]
                        ext3.gemv3(x, None, None, c[0], t3, m3, y, W, STG, KSP, grid, args.pdl)

                    def f_preq(c, W=W, STG=STG, KSP=KSP, grid=grid, v3s=v3s):
                        m3, t3 = v3s[id(c[0])]
                        ext3.gemv3(None, qu, su, c[0], t3, m3, y, W, STG, KSP, grid, args.pdl)
                else:
                    def f_fused(c, W=W, STG=STG, KSP=KSP, grid=grid):
                        w, wsw, sc = c
                        ext.gemv(x, None, None, w, sc if sm == 0 else wsw, sm, y, W, STG, KSP, grid, False)

                    def f_preq(c, W=W, STG=STG, KSP=KSP, grid=grid):
                        w, wsw, sc = c
                        ext.gemv(None, qu, su, w, sc if sm == 0 else wsw, sm, y, W, STG, KSP, grid, False)

                fns[f"gv_fused_{tag}"] = f_fused
                fns[f"gv_preq_{tag}"] = f_preq
                names += [f"gv_fused_{tag}", f"gv_preq_{tag}"]
            rot = Rotation(copies, narms=len(names))
            arms = {n: (lambda n=n, k=k: fns[n](rot.get(k))) for k, n in enumerate(names)}
            cold = time_arms(arms, iters=args.iters, pre=flusher)
            warm = None
            if not args.no_warm:
                warm_arms = {n: (lambda n=n: fns[n](copies[0])) for n in names}
                warm = time_arms(warm_arms, iters=args.iters, pre=lambda: ext.spin(657000))
            sc_bytes = (N // 32) * (K // 32) if sm == 0 else N * (K // 32)
            out = {}
            for n in names:
                if n == "read":
                    b = wbytes
                elif n.startswith("b12x"):
                    b = wbytes + N * (K // 32) + (M * K * 2 if n == "b12x_e2e" else M * K * 33 // 32) + M * N * 2
                elif n.startswith("gv_fused"):
                    b = wbytes + sc_bytes + M * K * 2 + M * N * 2
                else:
                    b = wbytes + sc_bytes + M * K * 33 // 32 + M * N * 2
                c_st = stats(cold[n])
                row = {"bytes": b, "cold": c_st,
                       "cold_GBps": b / c_st["median"] / 1e3,
                       "cold_pct_250": 100.0 * b / c_st["median"] / 1e3 / PEAK_GBPS,
                       "cold_weight_GBps": wbytes / c_st["median"] / 1e3}
                if warm is not None:
                    w_st = stats(warm[n])
                    row.update(warm=w_st, warm_GBps=b / w_st["median"] / 1e3)
                out[n] = row
            res["shapes"][name]["M"][M] = out
            base_e2e = out["b12x_e2e"]["cold"]["median"]
            base_gemm = out["b12x_gemm"]["cold"]["median"]
            line = (f"{name:14s} M={M} cold med us: b12x_e2e {base_e2e:7.2f} b12x_gemm {base_gemm:7.2f} "
                    f"read {out['read']['cold']['median']:7.2f}")
            for n in names:
                if n.startswith("gv_"):
                    v = out[n]["cold"]["median"]
                    ref = base_e2e if n.startswith("gv_fused") else base_gemm
                    line += f" | {n} {v:7.2f} ({100 * (ref / v - 1):+5.1f}%)"
            print(line, flush=True)
        del copies
        torch.cuda.empty_cache()
    json.dump(res, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
