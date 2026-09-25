#!/usr/bin/env python3
"""Race/intermittency stress of the production kernel (dense_gemv_kernel.cu with the
dense_gemv.CONFIGS table): for every tuned shape and decode M, REPS fresh random
activations (4 distributions in turn, a different real layer each time), fused and
pre-quantized inputs, each compared bit for bit with b12x; then a captured CUDA graph
replayed REPS times with new inputs copied into its static buffer, each replay
compared with b12x. Any mismatch is a failure (no tolerance)."""
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
from correctness import DISTS, b12x_ref, make_x, unswizzle  # noqa: E402
from timing import build_copies  # noqa: E402

NAMES = {v[0]: k for k, v in dense_gemv.CONFIGS.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shapes", default=",".join(NAMES))
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--copies", type=int, default=8)
    args = ap.parse_args()
    from benchutil import load_ext
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize

    ext = load_ext("dsv41_dense_gemv_study", ["../../docker/patch/dense_gemv_kernel.cu"])
    gen = torch.Generator(device="cuda").manual_seed(424242)
    report = {"reps": args.reps, "shapes": {}}
    all_ok = True
    for name in args.shapes.split(","):
        K, N = NAMES[name]
        _, kc, smode, buckets = dense_gemv.CONFIGS[(K, N)]
        copies = build_copies(name, min(args.copies, len(weights.SHAPES[name][3])))
        scs = [dense_gemv.build_scales(unswizzle(c[1], N, K // 32), N, K, kc, smode) for c in copies]
        rep = report["shapes"][name] = {}
        for M in (1, 3, 4, 6, 8):
            pl = dense_gemv.pick(buckets, M)
            if pl is None:
                continue
            W, S, MR = pl
            grid = int(ext.plan_grid(W, S, kc, MR, smode, 0, N, K))
            bad = 0
            for r in range(args.reps):
                ci = r % len(copies)
                c, sc = copies[ci], scs[ci]
                x = make_x(M, K, DISTS[r % len(DISTS)], gen)
                q, s, yref = b12x_ref(x, c[0].view(torch.float8_e4m3fn), c[1])
                y1 = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
                ext.gemv(x, None, None, c[0], sc, smode, y1, W, S, kc, MR, grid, False)
                y2 = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
                ext.gemv(None, q.view(torch.uint8), s.view(torch.uint8), c[0], sc, smode, y2, W, S, kc, MR, grid,
                         False)
                bad += not torch.equal(y1.view(torch.int16), yref.view(torch.int16))
                bad += not torch.equal(y2.view(torch.int16), yref.view(torch.int16))
            # graph replay stress on copy 0
            c, sc = copies[0], scs[0]
            xs = torch.zeros(M, K, dtype=torch.bfloat16, device="cuda")
            ys = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
            ext.gemv(xs, None, None, c[0], sc, smode, ys, W, S, kc, MR, grid, False)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                ext.gemv(xs, None, None, c[0], sc, smode, ys, W, S, kc, MR, grid, False)
            gbad = 0
            for r in range(args.reps):
                x = make_x(M, K, DISTS[r % len(DISTS)], gen)
                xs.copy_(x)
                g.replay()
                _, _, yref = b12x_ref(x, c[0].view(torch.float8_e4m3fn), c[1])
                gbad += not torch.equal(ys.view(torch.int16), yref.view(torch.int16))
            del g
            torch.cuda.synchronize()
            rep[M] = {"cfg": [W, S, kc, MR, grid], "eager_checks": 2 * args.reps, "eager_bad": bad,
                      "graph_replays": args.reps, "graph_bad": gbad}
            all_ok &= bad == 0 and gbad == 0
            print(f"{name:14s} M={M} cfg W{W} S{S} KC{kc} MR{MR} g{grid}: eager bad {bad}/{2 * args.reps}, "
                  f"graph bad {gbad}/{args.reps}", flush=True)
        del copies, scs
        torch.cuda.empty_cache()
        json.dump(report, open(args.out, "w"), indent=1)
    report["all_ok"] = all_ok
    json.dump(report, open(args.out, "w"), indent=1)
    print("ALL_OK" if all_ok else "FAILURES", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
