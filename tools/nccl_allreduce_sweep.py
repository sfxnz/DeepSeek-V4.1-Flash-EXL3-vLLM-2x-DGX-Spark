#!/usr/bin/env python3
"""Two-node bf16 all_reduce sweep over torch.distributed (NCCL), serve down.

Stand-in for nccl-tests all_reduce_perf: the recipe image has no nccl-tests
binaries and no MPI, and this uses the same libnccl the serve loads. Sizes
double from --min to --max bytes. busbw = algbw * 2(n-1)/n, as in nccl-tests.

  run:     python3 -S nccl_allreduce_sweep.py run --rank R --world 2 \
             --master 10.100.8.1:29531 --json /out/arm.json
  compare: python3 nccl_allreduce_sweep.py compare base.json cand.json

tools/nccl_dualrail.sh drives both ranks per arm.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Gate (dual-rail-nccl): bandwidth up where prefill all-reduces live
# (8k-chunk AR is ~80 MiB), latency not worse where decode ARs live (~51 KB).
BW_SIZES = (32 << 20, 64 << 20, 128 << 20)
LAT_SIZES = tuple(1 << i for i in range(3, 17))  # 8 B .. 64 KiB
BW_MIN_GAIN = 0.10
LAT_MAX_LOSS = 0.05


def sizes(lo: int, hi: int) -> list[int]:
    out, s = [], lo
    while s <= hi:
        out.append(s)
        s *= 2
    return out


def busbw(nbytes: int, seconds: float, world: int) -> float:
    """GB/s, nccl-tests convention for all_reduce."""
    return nbytes / seconds / 1e9 * 2 * (world - 1) / world


def iters_for(nbytes: int) -> int:
    return 200 if nbytes <= (1 << 20) else 50 if nbytes <= (32 << 20) else 20


def run(args) -> int:
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method=f"tcp://{args.master}", rank=args.rank,
                            world_size=args.world)
    rows = []
    for n in sizes(args.min, args.max):
        x = torch.ones(max(1, n // 2), dtype=torch.bfloat16, device="cuda")
        it = iters_for(n)
        for _ in range(5):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dist.barrier()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(it):
            dist.all_reduce(x)
        end.record()
        torch.cuda.synchronize()
        sec = start.elapsed_time(end) / 1e3 / it
        rows.append({"bytes": n, "us": sec * 1e6, "busbw_GBps": busbw(n, sec, args.world), "iters": it})
        if args.rank == 0:
            print(f"{n:>12d} {sec * 1e6:10.1f} us {rows[-1]['busbw_GBps']:8.2f} GB/s", flush=True)
    dist.destroy_process_group()
    if args.rank == 0 and args.json:
        Path(args.json).write_text(json.dumps({"world": args.world, "rows": rows}, indent=2) + "\n")
    return 0


def compare(base: dict, cand: dict) -> dict:
    """Gate verdict for cand vs base (both `run` JSON)."""
    b = {r["bytes"]: r for r in base["rows"]}
    c = {r["bytes"]: r for r in cand["rows"]}
    bw = {n: c[n]["busbw_GBps"] / b[n]["busbw_GBps"] - 1 for n in BW_SIZES if n in b and n in c}
    lat = {n: c[n]["us"] / b[n]["us"] - 1 for n in LAT_SIZES if n in b and n in c}
    ok_bw = bool(bw) and min(bw.values()) >= BW_MIN_GAIN
    ok_lat = bool(lat) and max(lat.values()) <= LAT_MAX_LOSS
    return {"bw_gain": bw, "lat_loss": lat, "pass": ok_bw and ok_lat}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--rank", type=int, required=True)
    r.add_argument("--world", type=int, default=2)
    r.add_argument("--master", required=True, help="host:port reachable from both ranks")
    r.add_argument("--min", type=int, default=8)
    r.add_argument("--max", type=int, default=256 << 20)
    r.add_argument("--json")
    c = sub.add_parser("compare")
    c.add_argument("base")
    c.add_argument("cand")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return run(args)
    res = compare(json.loads(Path(args.base).read_text()), json.loads(Path(args.cand).read_text()))
    for n, g in res["bw_gain"].items():
        print(f"busbw {n >> 20:4d} MiB {100 * g:+6.1f}%  (need >= {100 * BW_MIN_GAIN:+.0f}%)")
    worst = max(res["lat_loss"].items(), key=lambda kv: kv[1], default=(0, 0.0))
    print(f"worst small-message latency change {100 * worst[1]:+.1f}% at {worst[0]} B "
          f"(allow <= {100 * LAT_MAX_LOSS:+.0f}%)")
    print("PASS" if res["pass"] else "FAIL")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
