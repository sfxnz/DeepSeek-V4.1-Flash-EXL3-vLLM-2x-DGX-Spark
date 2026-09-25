#!/usr/bin/env python3
# Adopted from the k3 dense-gemv review (the reviewer's independent harness); only the output
# path and comments that described the since-fixed smem-attribute defect were changed.
"""Reviewer re-bench (k3 dense-gemv), independent of the builder's harness.

Per (shape, M) and arm, one CUDA graph runs the call back to back over R distinct real-layer
weight copies (as the serve replays graphs; each weight is read once per replay and R copies
>> 24 MiB L2, so it is cold by construction). A 2 x 64 MiB read flush also runs before every
timed replay (outside the events). 'warm' = the same graph shape with all R calls on copy 0.
Arms (same process, ABBA-alternated, 20 warmup + ITERS timed replays each):
  stock: the serve's call - mxfp8_e4m3_quantize + flashinfer mm_mxfp8(backend=auto) for a bf16
         input; mm_mxfp8 alone for wq_b (its producer hands a QuantizedActivation); DeepGEMM
         fp8_einsum with the prepacked scale for wo_a (DSV41_WOA_PREPACK=1).
  gemv:  dense_gemv production ext, (W,S,MR) from dense_gemv.CONFIGS, grid from plan_grid.
Per-call time = replay time / R. Reported: median p10 p90 mean min over replays, GB/s from each
arm's bytes (weights + weight scales + activation + output) and weight-only GB/s, % of 250.
Bitwise: every call's output compared between arms after a replay.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, "/repo/kernel_study/dense_gemv")
import weights  # noqa: E402

import dense_gemv  # noqa: E402  (/opt/dsv41-patch: the serve's module + kernel)

PEAK = 250.0


def pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(math.ceil(q * len(s))) - 1))]


def st(xs):
    return {"median": statistics.median(xs), "p10": pct(xs, 0.1), "p90": pct(xs, 0.9),
            "mean": statistics.fmean(xs), "min": min(xs), "n": len(xs)}


class Flush:
    def __init__(self):
        self.bufs = [torch.randint(0, 255, (64 << 20,), dtype=torch.uint8, device="cuda") for _ in range(2)]

    def __call__(self):
        for b in self.bufs:
            b.sum(dtype=torch.int32)


def time_graphs(graphs: dict, R: int, iters: int, flush) -> dict:
    names = list(graphs)
    for _ in range(20):
        for n in names:
            flush()
            graphs[n].replay()
    torch.cuda.synchronize()
    ev = {n: [] for n in names}
    for i in range(iters):
        for n in (names if i % 2 == 0 else names[::-1]):
            flush()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            graphs[n].replay()
            b.record()
            ev[n].append((a, b))
    torch.cuda.synchronize()
    return {n: [a.elapsed_time(b) * 1000.0 / R for a, b in ev[n]] for n in names}


def capture(fn, R):
    fn(0)  # eager warm-up (JIT, attributes)
    torch.cuda.synchronize()
    outs = []
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(R):
            outs.append(fn(i))
    return g, outs


def dense_shape(name, layers, Ms, iters, flush, report, clones=0):
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize, swizzle_mxfp8_scale
    from vllm.utils import flashinfer as vfi

    _, N, K, _ = weights.SHAPES[name]
    key = (K, N)
    nm, kc, smode, buckets = dense_gemv.CONFIGS[key]
    ext = dense_gemv._ext()
    copies = []
    t0 = time.time()
    for li in layers:
        w, s2d = weights.load(name, li)
        wsw = swizzle_mxfp8_scale(s2d, M=N, K=K).contiguous()
        sc = dense_gemv.build_scales(s2d, N, K, kc, smode)
        copies.append((w, wsw, sc))
    base = list(copies)
    for i in range(clones):  # distinct allocations of the same real layer(s)
        w, wsw, sc = base[i % len(base)]
        copies.append((w.clone(), wsw.clone(), sc.clone()))
    R = len(copies)
    wbytes = N * K
    print(f"{name}: {R} copies ({R * wbytes / 2**20:.0f} MiB) loaded in {time.time() - t0:.1f} s", flush=True)
    preq = name == "wq_b"
    gen = torch.Generator(device="cuda").manual_seed(1234 + N + K)
    for M in Ms:
        pl = dense_gemv.pick(buckets, M)
        if pl is None:
            continue
        W, S, MR = pl
        grid = int(ext.plan_grid(W, S, kc, MR, smode, 0, N, K))  # the persistent grid dense_gemv plans
        x = torch.randn(M, K, generator=gen, device="cuda").to(torch.bfloat16)
        q, s = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
        qu, su = q.view(torch.uint8), s.view(torch.uint8)
        row = {"cfg": [W, S, kc, MR, grid], "R": R}
        for mode in ("cold", "warm"):
            pick_copy = (lambda i: copies[i]) if mode == "cold" else (lambda i: copies[0])

            def stock(i, pick_copy=pick_copy):
                c = pick_copy(i)
                if preq:
                    return vfi.mm_mxfp8(q, c[0].t(), s, c[1], out_dtype=torch.bfloat16, backend="auto")
                qq, ss = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
                return vfi.mm_mxfp8(qq, c[0].t(), ss, c[1], out_dtype=torch.bfloat16, backend="auto")

            def gemv(i, pick_copy=pick_copy):
                c = pick_copy(i)
                y = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
                if preq:
                    ext.gemv(None, qu, su, c[0].view(torch.uint8), c[2], smode, y, W, S, kc, MR, grid, False)
                else:
                    ext.gemv(x, None, None, c[0].view(torch.uint8), c[2], smode, y, W, S, kc, MR, grid, False)
                return y

            gs, os_ = capture(stock, R)
            gg, og = capture(gemv, R)
            gs.replay()
            gg.replay()
            torch.cuda.synchronize()
            eq = all(torch.equal(a.view(torch.int16), b.reshape(M, N).view(torch.int16)) for a, b in zip(og, os_))
            t = time_graphs({"stock": gs, "gemv": gg}, R, iters, flush if mode == "cold" else (lambda: None))
            act = M * K * 33 // 32 if preq else M * K * 2
            b_stock = wbytes + N * (K // 32) + act + M * N * 2
            b_gemv = wbytes + (N // 32) * (K // 32) * (1 if smode == 0 else 32) + act + M * N * 2
            r = {}
            for arm, b in (("stock", b_stock), ("gemv", b_gemv)):
                s_ = st(t[arm])
                s_["GBps"] = b / s_["median"] / 1e3
                s_["pct250"] = 100 * s_["GBps"] / PEAK
                s_["w_GBps"] = wbytes / s_["median"] / 1e3
                r[arm] = s_
            r["gain_pct_median"] = 100 * (r["stock"]["median"] / r["gemv"]["median"] - 1)
            r["gain_pct_mean"] = 100 * (r["stock"]["mean"] / r["gemv"]["mean"] - 1)
            r["bitwise_all_calls"] = bool(eq)
            row[mode] = r
            print(f"{name:14s} M={M} {mode}: stock {r['stock']['median']:8.2f} [{r['stock']['p10']:.2f},{r['stock']['p90']:.2f}] "
                  f"gemv {r['gemv']['median']:8.2f} [{r['gemv']['p10']:.2f},{r['gemv']['p90']:.2f}] "
                  f"-> {r['gain_pct_median']:+5.1f}% (mean {r['gain_pct_mean']:+5.1f}%), gemv {r['gemv']['GBps']:.0f} GB/s "
                  f"= {r['gemv']['pct250']:.0f}% | bitwise {eq}", flush=True)
            del gs, gg, os_, og
        report.setdefault(name, {})[M] = row
    del copies, base
    torch.cuda.empty_cache()


def woa(layers, Ms, iters, flush, report):
    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import fused_inv_rope_fp8_quant
    from vllm.utils.deep_gemm import fp8_einsum, transform_sf_into_required_layout

    G, D, K, recipe = dense_gemv.WOA_GROUPS, dense_gemv.WOA_D, dense_gemv.WOA_K, (1, 1, 32)
    nm, kc, smode, buckets = dense_gemv.WOA_CONFIG
    ext = dense_gemv._ext()
    copies = []
    for li in layers:
        w, s2d = weights.load("wo_a", li)
        s_f32 = torch.exp2(s2d.view(G, D, K // 32).float() - 127.0)
        sp = transform_sf_into_required_layout(s_f32, D, K, recipe, G, False)
        sc = dense_gemv.build_scales(s2d, G * D, K, kc, smode)
        copies.append((w, sp, sc))
    R = len(copies)
    print(f"wo_a: {R} copies loaded", flush=True)
    cos_sin = torch.randn(8192, 64, device="cuda", dtype=torch.float32)
    gen = torch.Generator(device="cuda").manual_seed(77)
    for M in Ms:
        W, S, MR = dense_gemv.pick(buckets, M)
        grid = int(ext.plan_grid(W, S, kc, MR, smode, 1, G * D, K, G))
        oa = torch.randn(M, 32, 512, generator=gen, device="cuda").to(torch.bfloat16)
        pos = torch.randint(0, 8000, (M,), generator=gen, device="cuda")
        x8, xs = fused_inv_rope_fp8_quant(oa, pos, cos_sin, n_groups=G, heads_per_group=8, nope_dim=448,
                                          rope_dim=64, quant_group_size=32, tma_aligned_scales=True)
        row = {"cfg": [W, S, kc, MR, grid], "R": R}
        for mode in ("cold", "warm"):
            pick_copy = (lambda i: copies[i]) if mode == "cold" else (lambda i: copies[0])

            def stock(i, pick_copy=pick_copy):
                c = pick_copy(i)
                z = torch.empty(M, G, D, device="cuda", dtype=torch.bfloat16)
                fp8_einsum("bhr,hdr->bhd", (x8, xs), (c[0].view(G, D, K), c[1]), z, recipe=recipe)
                return z

            def gemv(i, pick_copy=pick_copy):
                c = pick_copy(i)
                z = torch.empty(M, G, D, device="cuda", dtype=torch.bfloat16)
                ext.gemv_grouped(x8.view(torch.uint8), xs, c[0].view(torch.uint8), c[2], smode, z.view(M, G * D),
                                 W, S, kc, MR, grid, False, G)
                return z

            gs, os_ = capture(stock, R)
            gg, og = capture(gemv, R)
            gs.replay()
            gg.replay()
            torch.cuda.synchronize()
            eq = all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(og, os_))
            t = time_graphs({"stock": gs, "gemv": gg}, R, iters, flush if mode == "cold" else (lambda: None))
            wbytes = G * D * K
            r = {}
            for arm, b in (("stock", wbytes + G * D * K // 32 + M * G * K * 33 // 32 + M * G * D * 2),
                           ("gemv", wbytes + (G * D // 32) * (K // 32) + M * G * K * 33 // 32 + M * G * D * 2)):
                s_ = st(t[arm])
                s_["GBps"] = b / s_["median"] / 1e3
                s_["pct250"] = 100 * s_["GBps"] / PEAK
                r[arm] = s_
            r["gain_pct_median"] = 100 * (r["stock"]["median"] / r["gemv"]["median"] - 1)
            r["gain_pct_mean"] = 100 * (r["stock"]["mean"] / r["gemv"]["mean"] - 1)
            r["bitwise_all_calls"] = bool(eq)
            row[mode] = r
            print(f"{'wo_a':14s} M={M} {mode}: stock {r['stock']['median']:8.2f} [{r['stock']['p10']:.2f},{r['stock']['p90']:.2f}] "
                  f"gemv {r['gemv']['median']:8.2f} [{r['gemv']['p10']:.2f},{r['gemv']['p90']:.2f}] "
                  f"-> {r['gain_pct_median']:+5.1f}% (mean {r['gain_pct_mean']:+5.1f}%), gemv {r['gemv']['GBps']:.0f} GB/s "
                  f"= {r['gemv']['pct250']:.0f}% | bitwise {eq}", flush=True)
            del gs, gg, os_, og
        report.setdefault("wo_a", {})[M] = row
    del copies
    torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["a", "b"])
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    flush = Flush()
    rep = {"method": __doc__, "iters": args.iters}
    if args.part == "a":
        ms = [1, 3, 4, 6, 8]
        dense_shape("qkv_a", list(range(32)), ms, args.iters, flush, rep)
        dense_shape("wq_b", list(range(24)), ms, args.iters, flush, rep)
        dense_shape("wo_b", list(range(24)), ms, args.iters, flush, rep)
        dense_shape("shared_gate_up", list(range(32)), ms, args.iters, flush, rep)
        dense_shape("shared_down", list(range(40)), ms, args.iters, flush, rep)
    else:
        dense_shape("engram_wkv", [1, 14], [1, 4, 8], args.iters, flush, rep, clones=2)
        dense_shape("main_proj", [0], [1, 3, 4], args.iters, flush, rep, clones=3)
        dense_shape("lm_head", [0], [1, 4, 8], args.iters, flush, rep, clones=1)
        woa(list(range(24)), [1, 3, 4, 6, 8], args.iters, flush, rep)
    json.dump(rep, open(f"/repo/results/2026-09-25-kernels/dense-gemv/chain-{args.part}.json", "w"), indent=1)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
