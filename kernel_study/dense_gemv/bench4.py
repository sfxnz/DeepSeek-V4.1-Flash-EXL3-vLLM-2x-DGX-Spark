#!/usr/bin/env python3
"""v4 dense GEMV: bitwise check vs production b12x, then cold/warm timing.

Correctness gate (fixed up front): bitwise equality with b12x (bf16 as int16)
for M in 1..8, 4 activation distributions, 2 real layers, fused-quant (bf16 in)
and pre-quantized input; CUDA-graph replay == eager (also after rewriting the
captured input). Timing protocol: benchutil docstring.
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
from correctness import DISTS, b12x_ref, cmp, make_x, unswizzle  # noqa: E402
from timing import build_copies  # noqa: E402

LIMIT = 101376


def v4_scales(copy, N: int, K: int, KC: int):
    _, wsw, sc = copy
    nspan, kbs = K // KC, KC // 32
    if sc is not None:
        scb = (kbs + 15) // 16 * 16
        out = torch.zeros(N // 32, nspan, scb, dtype=torch.uint8, device=wsw.device)
        out[:, :, :kbs] = sc.view(N // 32, nspan, kbs)
        return 0, out
    s2d = unswizzle(wsw, N, K // 32)
    return 1, s2d.reshape(N // 16, 16, nspan, kbs).permute(0, 2, 1, 3).contiguous()


# (W, S, KC, MR) candidates compiled into gemv4_ext.cu
CFGS = [(4, 2, 512, 4), (2, 2, 512, 8), (3, 2, 512, 8), (2, 3, 512, 8), (2, 2, 1024, 4), (4, 2, 512, 8),
        (4, 2, 640, 8), (2, 2, 1280, 8), (4, 2, 384, 8), (2, 2, 1152, 8), (2, 2, 768, 8)]
CFGS_V5_EXTRA = [(4, 3, 512, 4), (2, 2, 640, 8), (4, 2, 640, 4), (2, 2, 384, 8), (2, 2, 512, 4), (1, 2, 512, 8)]


# configs compiled into docker/patch/dense_gemv_kernel.cu (kTable), per scale mode
PROD_CFGS = {0: [(4, 2, 512, 4), (3, 2, 512, 8), (4, 2, 512, 8), (2, 2, 640, 8), (4, 2, 384, 8), (2, 2, 512, 4),
                 (2, 2, 512, 8)],
             1: [(3, 2, 512, 8)]}


def valid_cfgs(ext4, K: int, M: int, sm: int):
    out = []
    if hasattr(ext4, "plan_grid"):
        cand = PROD_CFGS[sm]
    else:
        extra = CFGS_V5_EXTRA if getattr(ext4, "gemv4", None) is not None and not hasattr(ext4, "__file__") else []
        cand = CFGS + extra
    for W, S, KC, MR in cand:
        if K % KC or M > MR or ext4.smem4(W, S, KC, MR, sm, K) > LIMIT:
            continue
        out.append((W, S, KC, MR))
    return out


def grid_for(ext4, cfg, N: int, K: int, sm: int) -> int:
    W, S, KC, MR = cfg
    if hasattr(ext4, "plan_grid"):
        return ext4.plan_grid(W, S, KC, MR, sm, 0, N, K)
    smem = ext4.smem4(W, S, KC, MR, sm, K)
    cps = max(1, min(102400 // (smem + 1024), 1536 // (32 * W)))
    return min((N // 16 + W - 1) // W, 48 * cps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shapes", default="qkv_a,wq_b,wo_b,shared_gate_up,shared_down,indexer_wq_b")
    ap.add_argument("--ms", default="1,3,4,6,8")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--no-warm", action="store_true")
    ap.add_argument("--no-correct", action="store_true")
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--cfg", default="", help="restrict to W,S,KC,MR")
    ap.add_argument("--pdl", action="store_true")
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--impl", default="v4", choices=["v4", "v5", "prod"])
    args = ap.parse_args()
    ext = load_ext("dgemv_v1", ["gemv_ext.cu"])
    if args.impl == "prod":
        # the serve's kernel file itself (docker/patch/dense_gemv_kernel.cu)
        prod = load_ext("dsv41_dense_gemv_study", ["../../docker/patch/dense_gemv_kernel.cu"])

        def _stage_bytes(KC, sm):
            kbs = KC // 32
            sc = (kbs + 15) // 16 * 16 if sm == 0 else 16 * kbs
            return (16 * KC + sc + 127) // 128 * 128

        class ext4:  # bench4's call shape on the production entry points
            gemv4 = staticmethod(lambda *a: prod.gemv(*a[:13]))
            smem4 = staticmethod(lambda W, S, KC, MR, sm, K: W * S * _stage_bytes(KC, sm) + MR * K
                                 + (MR * (K // 32) + 15) // 16 * 16)
            plan_grid = staticmethod(prod.plan_grid)
    elif args.impl == "v5":
        e5 = load_ext("dgemv_v5", ["gemv5_ext.cu"])

        class ext4:  # same call signature, v5 kernel
            gemv4 = staticmethod(e5.gemv5)
            smem4 = staticmethod(e5.smem5)
    else:
        ext4 = load_ext("dgemv_v4", ["gemv4_ext.cu"])
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
    from vllm.utils import flashinfer as vfi

    only_cfg = tuple(int(v) for v in args.cfg.split(",")) if args.cfg else None
    fl = Flusher(ext)
    o = torch.zeros(4, dtype=torch.int32, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(20260925)
    report = {"shapes": {}}
    all_ok = True
    for name in [s for s in args.shapes.split(",") if s]:
        _, N, K, layers = weights.SHAPES[name]
        wbytes = N * K
        R = max(copies_for(wbytes), 64 if wbytes < (64 << 20) else 4)
        copies = build_copies(name, R)
        sm0 = 0 if copies[0][2] is not None else 1
        rep = report["shapes"][name] = {"N": N, "K": K, "copies": R, "scale_mode": sm0, "correct": [], "M": {}}
        scales = {}

        def sc_for(c, KC):
            key = (id(c[0]), KC)
            if key not in scales:
                scales[key] = v4_scales(c, N, K, KC)
            return scales[key]

        # ---------------- correctness (2 real layers)
        if not args.no_correct:
            nbad = ncase = 0
            for c in copies[:2]:
                w8 = c[0].view(torch.float8_e4m3fn)
                for M in range(1, 9):
                    cfgs = [cf for cf in valid_cfgs(ext4, K, M, sm0) if only_cfg is None or cf == only_cfg]
                    for dist in DISTS:
                        x = make_x(M, K, dist, gen)
                        q, s, yref = b12x_ref(x, w8, c[1])
                        for cf in cfgs:
                            W, S, KC, MR = cf
                            smd, sct = sc_for(c, KC)
                            g = grid_for(ext4, cf, N, K, smd)
                            y1 = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
                            ext4.gemv4(x, None, None, c[0], sct, smd, y1, W, S, KC, MR, g, False)
                            y2 = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
                            ext4.gemv4(None, q.view(torch.uint8), s.view(torch.uint8), c[0], sct, smd, y2,
                                       W, S, KC, MR, g, False)
                            torch.cuda.synchronize()
                            c1, c2 = cmp(y1, yref), cmp(y2, yref)
                            ok = c1["bitwise"] and c2["bitwise"]
                            ncase += 1
                            nbad += not ok
                            if not ok:
                                print(f"MISMATCH {name} M={M} {dist} cfg={cf} fused={c1} preq={c2}", flush=True)
                                rep["correct"].append({"M": M, "dist": dist, "cfg": cf, "fused": c1, "preq": c2})
                # graph capture + replay per config (M = 1, 4, 8)
                for M in (1, 4, 8):
                    for cf in [cf for cf in valid_cfgs(ext4, K, M, sm0) if only_cfg is None or cf == only_cfg]:
                        W, S, KC, MR = cf
                        smd, sct = sc_for(c, KC)
                        g = grid_for(ext4, cf, N, K, smd)
                        xg = make_x(M, K, "normal", gen)
                        y = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
                        ext4.gemv4(xg, None, None, c[0], sct, smd, y, W, S, KC, MR, g, True)
                        torch.cuda.synchronize()
                        eager = y.clone()
                        graph = torch.cuda.CUDAGraph()
                        st = torch.cuda.Stream()
                        with torch.cuda.stream(st):
                            with torch.cuda.graph(graph):
                                ext4.gemv4(xg, None, None, c[0], sct, smd, y, W, S, KC, MR, g, True)
                        y.zero_()
                        graph.replay()
                        torch.cuda.synchronize()
                        r1 = torch.equal(y.view(torch.int16), eager.view(torch.int16))
                        xg.copy_(make_x(M, K, "lognormal", gen))
                        graph.replay()
                        ye = torch.empty_like(y)
                        ext4.gemv4(xg, None, None, c[0], sct, smd, ye, W, S, KC, MR, g, False)
                        torch.cuda.synchronize()
                        r2 = torch.equal(y.view(torch.int16), ye.view(torch.int16))
                        ncase += 1
                        nbad += not (r1 and r2)
                        if not (r1 and r2):
                            print(f"GRAPH MISMATCH {name} M={M} cfg={cf} {r1} {r2}", flush=True)
                        del graph
            all_ok &= nbad == 0
            rep["correct_cases"] = ncase
            rep["correct_bad"] = nbad
            print(f"{name:15s} correctness: {ncase - nbad}/{ncase} bitwise (incl. graph replay)", flush=True)

        # ---------------- timing
        if not args.no_time:
            for M in [int(m) for m in args.ms.split(",")]:
                x = torch.randn(M, K, generator=gen, device="cuda").to(torch.bfloat16)
                q, s = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
                qu, su = q.view(torch.uint8), s.view(torch.uint8)
                y = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

                def b12x_e2e(c):
                    qq, ss = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
                    vfi.mm_mxfp8(qq, c[0].view(torch.float8_e4m3fn).t(), ss, c[1], out_dtype=torch.bfloat16,
                                 backend="auto")

                def b12x_gemm(c):
                    vfi.mm_mxfp8(q, c[0].view(torch.float8_e4m3fn).t(), s, c[1], out_dtype=torch.bfloat16,
                                 backend="auto")

                fns = {"b12x_e2e": b12x_e2e, "b12x_gemm": b12x_gemm,
                       "read": lambda c: ext.read_flat(c[0], o, 48, 2)}
                for cf in [cf for cf in valid_cfgs(ext4, K, M, sm0) if only_cfg is None or cf == only_cfg]:
                    W, S, KC, MR = cf
                    for c in copies:
                        sc_for(c, KC)
                    g = grid_for(ext4, cf, N, K, sm0)
                    tag = f"w{W}s{S}k{KC}m{MR}g{g}"
                    fns[f"gv_fused_{tag}"] = (lambda c, cf=cf, g=g: ext4.gemv4(
                        x, None, None, c[0], sc_for(c, cf[2])[1], sm0, y, *cf, g, args.pdl))
                    fns[f"gv_preq_{tag}"] = (lambda c, cf=cf, g=g: ext4.gemv4(
                        None, qu, su, c[0], sc_for(c, cf[2])[1], sm0, y, *cf, g, args.pdl))
                    if args.diag:
                        for d in (1, 2):
                            fns[f"gv_preq_diag{d}_{tag}"] = (lambda c, cf=cf, g=g, d=d: ext4.gemv4(
                                None, qu, su, c[0], sc_for(c, cf[2])[1], sm0, y, *cf, g, args.pdl, d))
                names = list(fns)
                rot = Rotation(copies, narms=len(names))
                arms = {n: (lambda n=n, k=k: fns[n](rot.get(k))) for k, n in enumerate(names)}

                def pre_cold():
                    # flush, then re-touch the activation: in the serve it was just written by the
                    # previous kernel, so it is L2-resident when the GEMM starts
                    fl()
                    for t in (x, qu, su):
                        ext.read_flat(t.view(torch.uint8), o, 8, 2)

                cold = time_arms(arms, iters=args.iters, pre=pre_cold)
                warm = None
                if not args.no_warm:
                    warm = time_arms({n: (lambda n=n: fns[n](copies[0])) for n in names}, iters=args.iters,
                                     pre=lambda: ext.spin(657000))
                out = {}
                for n in names:
                    if n == "read":
                        b = wbytes
                    elif n.startswith("b12x"):
                        b = wbytes + N * (K // 32) + (M * K * 2 if n == "b12x_e2e" else M * K * 33 // 32) + M * N * 2
                    else:
                        scb = (N // 32) * (K // 32) if sm0 == 0 else N * (K // 32)
                        b = wbytes + scb + (M * K * 2 if "fused" in n else M * K * 33 // 32) + M * N * 2
                    cs = stats(cold[n])
                    row = {"bytes": b, "cold": cs, "cold_GBps": b / cs["median"] / 1e3,
                           "cold_pct_250": 100 * b / cs["median"] / 1e3 / PEAK_GBPS}
                    if warm is not None:
                        ws = stats(warm[n])
                        row.update(warm=ws, warm_GBps=b / ws["median"] / 1e3)
                    out[n] = row
                rep["M"][M] = out
                e2e, gm = out["b12x_e2e"]["cold"]["median"], out["b12x_gemm"]["cold"]["median"]
                line = f"{name:14s} M={M} cold us: b12x_e2e {e2e:7.2f} b12x_gemm {gm:7.2f} read {out['read']['cold']['median']:7.2f}"
                for kind, ref in (("fused", e2e), ("preq", gm)):
                    ks = [n for n in names if n.startswith(f"gv_{kind}_")]
                    if ks:
                        b = min(ks, key=lambda n: out[n]["cold"]["median"])
                        v = out[b]["cold"]["median"]
                        line += f" | best {kind} {b[len(kind) + 4:]} {v:7.2f} ({100 * (ref / v - 1):+5.1f}%)"
                print(line, flush=True)
        del copies, scales
        torch.cuda.empty_cache()
        json.dump(report, open(args.out, "w"), indent=1)  # keep partial results if a later shape fails
    report["all_ok"] = all_ok
    json.dump(report, open(args.out, "w"), indent=1)
    print("ALL_OK" if all_ok else "FAILURES", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
