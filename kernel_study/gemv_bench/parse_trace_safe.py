#!/usr/bin/env python3
"""Crash-safe chrome-trace analyzer for DGX Spark UMA hosts.

Streams the gzipped torch-profiler trace with ijson instead of json.load:
a full load of a 5M-event trace materializes ~15-20 GB of Python objects and
OOMs the host while the serve is resident (kernel oom-killer cascade,
2026-09-19). This variant caps its own address space first and keeps only
small per-event tuples.
"""
from __future__ import annotations

import gzip
import resource
import sys
from collections import Counter, defaultdict

# Die with MemoryError instead of taking the host down.
resource.setrlimit(resource.RLIMIT_AS, (6 << 30, 6 << 30))

import ijson  # noqa: E402


def main(path: str) -> int:
    kernels: dict[str, list] = defaultdict(lambda: [0, 0.0])
    # correlation id -> (op name, dur) for cpu ops, only for ops that launch
    # kernels we care about (we cannot know in advance, so keep all cpu_ops
    # but only id/name/dur as compact tuples; a few hundred thousand max).
    cpu_ops: dict[int, tuple[str, float]] = {}
    target_kernels: dict[int, str] = {}
    annotations = Counter()
    ann_dur: dict[str, float] = defaultdict(float)

    with gzip.open(path, "rb") as fh:
        events = ijson.items(fh, "traceEvents.item")
        for e in events:
            cat = e.get("cat")
            if cat == "kernel":
                name = e["name"]
                d = float(e.get("dur", 0))
                k = kernels[name[:100]]
                k[0] += 1
                k[1] += d
                corr = (e.get("args") or {}).get("correlation")
                if corr is not None:
                    target_kernels[corr] = name[:60]
            elif cat == "cpu_op":
                corr = (e.get("args") or {}).get("correlation")
                if corr is not None:
                    cpu_ops[corr] = (e.get("name", "?")[:80], float(e.get("dur", 0)))
            elif cat == "gpu_user_annotation":
                name = e.get("name", "?")[:80]
                annotations[name] += 1
                ann_dur[name] += float(e.get("dur", 0))

    tot = sum(v[1] for v in kernels.values())
    print(f"total kernel time: {tot/1000:.1f} ms")
    print("\n== top kernels ==")
    for n, (c, d) in sorted(kernels.items(), key=lambda kv: -kv[1][1])[:20]:
        print(f"{d/1000:10.2f} ms {c:7d} {d/max(c,1):9.1f} us  {n}")

    print("\n== gpu annotations ==")
    for n, c in annotations.most_common(10):
        print(f"{ann_dur[n]/1000:10.2f} ms {c:7d}  {n}")

    # Correlate every kernel to its launching cpu op, aggregated.
    print("\n== kernel time by launching cpu op ==")
    by_op: dict[tuple[str, str], list] = defaultdict(lambda: [0, 0.0])
    for corr, kname in target_kernels.items():
        op = cpu_ops.get(corr)
        if op is None:
            continue
        key = (op[0], kname)
        by_op[key][0] += 1
        by_op[key][1] += 0.0  # kernel dur unknown here; count only
    for (op, kname), (c, _) in sorted(
        by_op.items(), key=lambda kv: -kv[1][0]
    )[:25]:
        print(f"{c:7d} calls  {op}  ->  {kname}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
