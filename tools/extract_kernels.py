#!/usr/bin/env python3
"""Stream GPU kernel events out of a gzipped torch-profiler trace.

UMA-safe: ijson streaming under an RLIMIT_AS cap, never json.load (a 5M-event
trace json.load'ed with the serve resident hung spark1, RESULTS.md Round 11).
Run on an idle host, one rank's trace at a time.

Output is a pickle of (names, rows):
  names: {name_id: kernel name[:160]}
  rows:  [(ts_us, dur_us, name_id, grid, block, tid, regs, smem), ...]
tid is the CUDA stream for kernel events. For two-rank AR alignment, compare
per-layer AllReduce durations across ranks and anchor on kernel END (ts+dur).

  python3 tools/extract_kernels.py rank0.pt.trace.json.gz rank0.pkl
"""
from __future__ import annotations

import argparse
import gzip
import pickle
import resource


def extract(path: str) -> tuple[dict[int, str], list[tuple]]:
    import ijson

    names: dict[str, int] = {}
    rows: list[tuple] = []
    with gzip.open(path, "rb") as fh:
        for e in ijson.items(fh, "traceEvents.item", use_float=True):
            if e.get("cat") != "kernel":
                continue
            nid = names.setdefault(e["name"][:160], len(names))
            a = e.get("args") or {}
            rows.append((
                float(e["ts"]),
                float(e.get("dur", 0)),
                nid,
                tuple(a.get("grid") or ()),
                tuple(a.get("block") or ()),
                e.get("tid"),
                a.get("registers per thread"),
                a.get("shared memory"),
            ))
    return {v: k for k, v in names.items()}, rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("trace", help="*.pt.trace.json.gz")
    ap.add_argument("out", help="output pickle")
    ap.add_argument("--rlimit-gib", type=int, default=3, help="RLIMIT_AS cap (default 3)")
    args = ap.parse_args()
    cap = args.rlimit_gib << 30
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    names, rows = extract(args.trace)
    with open(args.out, "wb") as fh:
        pickle.dump((names, rows), fh)
    print(len(rows), len(names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
