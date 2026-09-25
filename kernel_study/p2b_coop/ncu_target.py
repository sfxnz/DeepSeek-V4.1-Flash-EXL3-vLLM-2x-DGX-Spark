#!/usr/bin/env python3
"""Target process for Nsight Compute: a few flushed calls per variant on the real layer.

  PROFILE=1 kernel_study/p2b_coop/spark2.sh <tag> --target-processes all \
      --kernel-name regex:p2b_moe --clock-control none --set full -o /repo/<file> \
      python3 kernel_study/p2b_coop/ncu_target.py --variants 0,1 --m 4 --calls 2

Each call uses a fresh census routing of the loaded layer; the L2 is flushed before every call.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import bench_r3 as br  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="0,1")
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--calls", type=int, default=2)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--source", default="census")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    ext = br.build_ext(False)
    layer = br.Layer(args.layer, 0, "cuda")
    routes = br.Routings(args.layer, random.Random(args.seed))
    b = br.Bench(ext, layer, "cuda", args.seed)
    flush = br.Flusher("cuda")
    x, rw = b.x(args.m), b.rw(args.m)
    out = torch.empty_like(x)
    rows = [routes.draw(args.source, args.m) for _ in range(args.calls)]
    for v in (int(s) for s in args.variants.split(",")):
        for r in rows:
            flush()
            b.call(v, x, b.ids(r), rw, out)
            torch.cuda.synchronize()
            print(f"variant {v} unique {br.n_unique(r)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
