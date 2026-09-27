#!/usr/bin/env python3
"""Pure streaming-read roofline for the dense decode weight sizes on GB10.

For every per-rank dense shape, time a read of exactly the fp8 weight bytes
(N*K) with three access patterns and a sweep of grid / bytes in flight:
  flat   grid-stride 16 B loads (the ideal streaming pattern)
  tiles  one warp per 16-row tile of the row-major [N, K] matrix (the GEMV
         access pattern on the stock layout)
  chunks one warp per contiguous chunk (the pattern a fragment-native
         repack of the weight would give)
Cold (64 MiB read-flush before each call) and warm (memory-free spacer).
Output: JSON with per-config stats; the best config per shape and pattern is
the calibrated per-shape roofline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchutil import Flusher, load_ext, summarize, time_arms  # noqa: E402

SHAPES = [  # name, N, K (per rank)
    ("shared_down", 5120, 1152),
    ("indexer_wq_b", 4096, 1280),
    ("qkv_a", 1792, 5120),
    ("shared_gate_up", 2304, 5120),
    ("wq_b", 16384, 1280),
    ("wo_b", 5120, 4096),
    ("main_proj", 5120, 15360),
    ("engram_wkv", 25600, 6144),
    ("lm_head", 64640, 5120),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--only", default="")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    ext = load_ext("dgemv_calib", ["calib.cu"])
    flusher = Flusher(ext)
    out_scalar = torch.zeros(4, dtype=torch.int32, device="cuda")
    only = {s for s in args.only.split(",") if s}
    res = {"device": torch.cuda.get_device_name(), "shapes": {}}
    for name, N, K in SHAPES:
        if only and name not in only:
            continue
        nbytes = N * K
        w = torch.randint(0, 255, (nbytes,), dtype=torch.uint8, device="cuda")
        arms = {}
        # flat grid-stride
        for grid in (48, 96, 192, 384):
            for u in (2, 4, 8):
                for pf in (False, True):
                    if args.quick and (pf or grid in (96,)):
                        continue
                    arms[f"flat_g{grid}_u{u}_pf{int(pf)}"] = (
                        lambda grid=grid, u=u, pf=pf: ext.flat(w, out_scalar, grid, 256, u, pf))
        ntiles = N // 16
        # tiles: one tile per warp (grid covers all tiles) and persistent grids
        for block in (128, 256):
            wpb = block // 32
            grids = {"all": (ntiles + wpb - 1) // wpb, "p48": 48, "p96": 96, "p144": 144}
            for gname, grid in grids.items():
                if grid * wpb > ntiles and gname != "all":
                    continue
                for d in (2, 4, 8):
                    for pf in (False, True):
                        if args.quick and pf:
                            continue
                        arms[f"tiles_b{block}_{gname}_d{d}_pf{int(pf)}"] = (
                            lambda grid=grid, block=block, d=d, pf=pf:
                            ext.tiles(w, out_scalar, N, K, grid, block, d, pf))
        # chunks: contiguous chunk per warp; nwarps = ntiles (tile-sized chunks)
        for block in (128, 256):
            wpb = block // 32
            for nw in (ntiles, 48 * 16, 48 * 32):
                if nw > ntiles:
                    continue
                grid = (nw + wpb - 1) // wpb
                chunk = ((nbytes + nw - 1) // nw + 511) // 512 * 512
                for d in (2, 4, 8):
                    if args.quick and d == 2:
                        continue
                    arms[f"chunks_b{block}_w{nw}_d{d}"] = (
                        lambda grid=grid, block=block, chunk=chunk, d=d:
                        ext.chunks(w, out_scalar, chunk, grid, block, d, False))
        cold = summarize(time_arms(arms, iters=args.iters, pre=flusher), nbytes)
        best = min(cold, key=lambda k: cold[k]["median"])
        by_pattern = {}
        for pat in ("flat", "tiles", "chunks"):
            ks = [k for k in cold if k.startswith(pat)]
            if ks:
                b = min(ks, key=lambda k: cold[k]["median"])
                by_pattern[pat] = {"arm": b, **cold[b]}
        # warm: the best arm of each pattern, spacer instead of flush
        warm_arms = {v["arm"]: arms[v["arm"]] for v in by_pattern.values()}
        warm = summarize(time_arms(warm_arms, iters=args.iters,
                                   pre=lambda: ext.spin(20000)), nbytes)
        res["shapes"][name] = {"N": N, "K": K, "bytes": nbytes, "cold": cold, "best_cold": best,
                               "best_by_pattern_cold": by_pattern, "warm_best_arms": warm}
        print(f"{name:14s} {nbytes/1e6:8.2f} MB  best cold {best}: "
              f"{cold[best]['median']:.2f} us ({cold[best]['GBps_median']:.1f} GB/s, "
              f"{cold[best]['pct_of_250']:.1f}%)", flush=True)
        for pat, v in by_pattern.items():
            wv = warm[v["arm"]]
            print(f"    {pat:7s} {v['arm']:28s} cold med {v['median']:8.2f} p10 {v['p10']:8.2f} "
                  f"p90 {v['p90']:8.2f}  {v['GBps_median']:6.1f} GB/s | warm med {wv['median']:8.2f} "
                  f"({wv['GBps_median']:.1f} GB/s)", flush=True)
        del w
        torch.cuda.empty_cache()
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
