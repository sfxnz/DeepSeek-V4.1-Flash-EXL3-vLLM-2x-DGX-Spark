#!/usr/bin/env python3
"""Pure streaming-read roofline, rotation protocol (see benchutil docstring).

Same patterns as calib.py (flat / tiles / chunks), each arm reading a
different copy per iteration from a pool of >= 16 copies (>= 512 MiB), double
hashed flush before every timed call. Output: per-config stats; best per shape.
"""
from __future__ import annotations
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchutil import Flusher, Rotation, copies_for, load_ext, summarize, time_arms  # noqa: E402
from calib import SHAPES  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    ext = load_ext("dgemv_calib", ["calib.cu"])
    flusher = Flusher(ext)
    o = torch.zeros(4, dtype=torch.int32, device="cuda")
    only = {s for s in args.only.split(",") if s}
    res = {"device": torch.cuda.get_device_name(), "protocol": "rotation+double-hashed-flush", "shapes": {}}
    for name, N, K in SHAPES:
        if only and name not in only:
            continue
        nbytes = N * K
        R = copies_for(nbytes)
        bufs = [torch.randint(0, 255, (nbytes,), dtype=torch.uint8, device="cuda") for _ in range(R)]
        ntiles = N // 16
        cfgs = {}
        for grid in (48, 96, 192, 384):
            for u in (2, 4, 8):
                cfgs[f"flat_g{grid}_u{u}"] = lambda b, grid=grid, u=u: ext.flat(b, o, grid, 256, u, False)
        for block in (128, 256):
            wpb = block // 32
            for gname, grid in {"all": (ntiles + wpb - 1) // wpb, "p48": 48, "p96": 96, "p144": 144}.items():
                if gname != "all" and grid * wpb > ntiles:
                    continue
                for d in (2, 4, 8):
                    cfgs[f"tiles_b{block}_{gname}_d{d}"] = (
                        lambda b, grid=grid, block=block, d=d: ext.tiles(b, o, N, K, grid, block, d, False))
        for block in (128, 256):
            wpb = block // 32
            for nw in sorted({ntiles, 48 * 8, 48 * 16, 48 * 32}):
                if nw > ntiles:
                    continue
                grid = (nw + wpb - 1) // wpb
                chunk = ((nbytes + nw - 1) // nw + 511) // 512 * 512
                for d in (2, 4, 8):
                    cfgs[f"chunks_b{block}_w{nw}_d{d}"] = (
                        lambda b, grid=grid, block=block, chunk=chunk, d=d: ext.chunks(b, o, chunk, grid, block, d, False))
        names = list(cfgs)
        rot = Rotation(bufs, narms=len(names))
        arms = {n: (lambda n=n, k=k: cfgs[n](rot.get(k))) for k, n in enumerate(names)}
        cold = summarize(time_arms(arms, iters=args.iters, pre=flusher), nbytes)
        by_pattern = {}
        for pat in ("flat", "tiles", "chunks"):
            ks = [k for k in cold if k.startswith(pat)]
            b = min(ks, key=lambda k: cold[k]["median"])
            by_pattern[pat] = {"arm": b, **cold[b]}
        best = min(cold, key=lambda k: cold[k]["median"])
        res["shapes"][name] = {"N": N, "K": K, "bytes": nbytes, "copies": R, "cold": cold,
                               "best_cold": best, "best_by_pattern_cold": by_pattern}
        print(f"{name:14s} {nbytes/1e6:8.2f} MB x{R:3d} copies  best {best}: {cold[best]['median']:.2f} us "
              f"({cold[best]['GBps_median']:.1f} GB/s, {cold[best]['pct_of_250']:.1f}%)", flush=True)
        for pat, v in by_pattern.items():
            print(f"    {pat:7s} {v['arm']:26s} med {v['median']:8.2f} p10 {v['p10']:8.2f} p90 {v['p90']:8.2f} "
                  f"mean {v['mean']:8.2f}  {v['GBps_median']:6.1f} GB/s", flush=True)
        del bufs, rot, arms
        torch.cuda.empty_cache()
    json.dump(res, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
