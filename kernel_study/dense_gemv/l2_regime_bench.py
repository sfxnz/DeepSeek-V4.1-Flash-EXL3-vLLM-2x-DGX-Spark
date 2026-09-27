#!/usr/bin/env python3
"""GEMV vs stock when part or all of the weight is L2-resident (k3 review: 'L2-regime').

Question (fixed before any result): the serve reads every dense weight cold, except that
k3 comm's DSV41_AR_L2_PREFETCH=1 prefetches the first 5.5 MiB of the next layer's qkv_a
(fused_wqa_wkv weight + weight_scale, the same fraction of each: ar_l2_prefetch.py
split_budget) into L2 during the MoE all-reduce. Does the GEMV still beat the stock path
on qkv_a with that prefetch stacked? Shared gate_up / down with the same budget are
reported as information for a possible second window (no such lever exists), and 'warm'
(everything the arm reads prefetched) is the ceiling.

Per timed call (benchutil protocol otherwise: real rank-0 weights rotating over R distinct
layer copies, double hashed 64 MiB read flush, CUDA events, >= 200 samples, all arms of a
(shape, M) interleaved ABBA in one process):
  flush -> [prefetch: k3 comm's shipped TMA kernel (l2pf_k3comm.py) on the call's copy]
  -> 60 us memory-free spin (the AR + mHC window; 5.5 MiB lands in ~24 us)
  -> re-touch the activation (the previous kernel just wrote it) -> timed call.
Regimes: cold (no prefetch), pf (weight + stock weight_scale prefixes, budget split as
the lever does; the GEMV's own compact scale is not prefetched, as in the serve), warm
(whole weight + the arm's own scale tensor).
Arms: stock = FlashInfer mxfp8_e4m3_quantize + mm_mxfp8(auto) (the serve's call from a
bf16 activation); gemv = the serve module's kernel with its production config for M.
Correctness: both arms' outputs compared bitwise on the same copy and input.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import l2pf_k3comm  # noqa: E402
import weights  # noqa: E402
from benchutil import PEAK_GBPS, Flusher, load_ext, stats  # noqa: E402
from correctness import unswizzle  # noqa: E402
from timing import build_copies  # noqa: E402

BUDGET = int(5.5 * 2**20)  # DSV41_AR_L2_PREFETCH_MIB default
WINDOW_CYCLES = 131400  # 60 us at 2190 MHz


def split_budget(sizes, budget):
    """ar_l2_prefetch.split_budget: the same fraction of each tensor, 16-B aligned."""
    frac = min(1.0, budget / sum(sizes))
    return [int(n * frac) & ~15 for n in sizes]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shapes", default="qkv_a,shared_gate_up,shared_down")
    ap.add_argument("--ms", default="1,3,4,6,8")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--copies", type=int, default=32)
    args = ap.parse_args()
    import dense_gemv  # /opt/dsv41-patch (SERVE_PATCH=1): the serve's module and kernel
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
    from vllm.utils import flashinfer as vfi

    ext = load_ext("dgemv_v1", ["gemv_ext.cu"])  # flush / spin / read helpers
    prod = dense_gemv._ext()
    pf = l2pf_k3comm.launcher(torch)
    fl = Flusher(ext)
    o = torch.zeros(4, dtype=torch.int32, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(20260926)
    report = {"budget_bytes": BUDGET, "window_cycles": WINDOW_CYCLES, "method": __doc__, "shapes": {}}
    all_ok = True
    for name in args.shapes.split(","):
        _, N, K, _ = weights.SHAPES[name]
        key = (K, N)
        _, kc, smode, buckets = dense_gemv.CONFIGS[key]
        copies = [(w, wsw) for w, wsw, _ in build_copies(name, args.copies)]
        # the GEMV's compact scale from each copy's stock per-row scale (same layer bytes)
        gscales = [dense_gemv.build_scales(unswizzle(c[1], N, K // 32), N, K, kc, smode) for c in copies]
        wbytes, sbytes = N * K, copies[0][1].numel() * copies[0][1].element_size()
        pw, ps = split_budget([wbytes, sbytes], BUDGET)
        rep = report["shapes"][name] = {"N": N, "K": K, "copies": len(copies), "weight_bytes": wbytes,
                                        "stock_scale_bytes": sbytes, "pf_weight_bytes": pw, "pf_scale_bytes": ps,
                                        "pf_weight_frac": pw / wbytes, "M": {}}
        for M in [int(m) for m in args.ms.split(",")]:
            pl = dense_gemv.pick(buckets, M)
            if pl is None:
                continue
            W, S, MR = pl
            grid = int(prod.plan_grid(W, S, kc, MR, smode, 0, N, K))
            x = torch.randn(M, K, generator=gen, device="cuda").to(torch.bfloat16)
            y = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

            def stock(i):
                q, s = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
                return vfi.mm_mxfp8(q, copies[i][0].view(torch.float8_e4m3fn).t(), s, copies[i][1],
                                    out_dtype=torch.bfloat16, backend="auto")

            def gemv(i):
                prod.gemv(x, None, None, copies[i][0], gscales[i], smode, y, W, S, kc, MR, grid, False)
                return y

            # correctness first: same copy, same input, bitwise
            eq = True
            for i in range(4):
                a = stock(i).clone()
                b = gemv(i).clone()
                torch.cuda.synchronize()
                eq &= bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))
            all_ok &= eq

            def pre(regime, arm, i):
                fl()
                if regime == "pf":
                    pf(copies[i][0], pw)
                    pf(copies[i][1], ps)
                elif regime == "warm":
                    pf(copies[i][0], wbytes)
                    if arm == "stock":
                        pf(copies[i][1], sbytes)
                    else:
                        pf(gscales[i], gscales[i].numel())
                ext.spin(WINDOW_CYCLES)
                ext.read_flat(x.view(torch.uint8), o, 8, 2)

            arms = [(r, a) for r in ("cold", "pf", "warm") for a in ("stock", "gemv")]
            fns = {"stock": stock, "gemv": gemv}
            R = len(copies)
            turn = {arm: k * R // len(arms) for k, arm in enumerate(arms)}  # arms read different copies
            ev = {arm: [] for arm in arms}
            for it in range(args.iters + 20):
                for arm in (arms if it % 2 == 0 else arms[::-1]):
                    i = turn[arm] % R
                    turn[arm] += 1
                    pre(arm[0], arm[1], i)
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    fns[arm[1]](i)
                    e.record()
                    if it >= 20:
                        ev[arm].append((s, e))
            torch.cuda.synchronize()
            row = {"cfg": [W, S, kc, MR, grid], "bitwise": eq}
            for arm in arms:
                st = stats([s.elapsed_time(e) * 1000.0 for s, e in ev[arm]])
                b = wbytes + (sbytes if arm[1] == "stock" else gscales[0].numel()) + M * K * 2 + M * N * 2
                st["GBps"] = b / st["median"] / 1e3
                st["pct_250"] = 100 * st["GBps"] / PEAK_GBPS
                row[f"{arm[0]}_{arm[1]}"] = st
            for r in ("cold", "pf", "warm"):
                row[f"{r}_gain_pct"] = 100 * (row[f"{r}_stock"]["median"] / row[f"{r}_gemv"]["median"] - 1)
            rep["M"][M] = row
            print(f"{name:14s} M={M} " + " | ".join(
                f"{r}: stock {row[f'{r}_stock']['median']:6.2f} [{row[f'{r}_stock']['p10']:.2f},"
                f"{row[f'{r}_stock']['p90']:.2f}] gemv {row[f'{r}_gemv']['median']:6.2f} "
                f"[{row[f'{r}_gemv']['p10']:.2f},{row[f'{r}_gemv']['p90']:.2f}] {row[f'{r}_gain_pct']:+5.1f}%"
                for r in ("cold", "pf", "warm")) + f" | bitwise {eq}", flush=True)
        del copies, gscales
        torch.cuda.empty_cache()
        json.dump(report, open(args.out, "w"), indent=1)
    report["all_ok"] = all_ok
    json.dump(report, open(args.out, "w"), indent=1)
    print("ALL_OK" if all_ok else "FAILURES", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
