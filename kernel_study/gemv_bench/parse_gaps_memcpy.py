#!/usr/bin/env python3
"""Third pass: include gpu_memcpy/gpu_memset in device-busy; re-bucket gaps."""
from __future__ import annotations
import bisect, gzip, resource, sys
from collections import Counter

resource.setrlimit(resource.RLIMIT_AS, (6 << 30, 6 << 30))
import ijson

trace = sys.argv[1]
MIN_GAP = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

dev_events = []  # (ts, te, kind)
cpu_by_tid = {}
memcpy_by_name = Counter()
with gzip.open(trace, "rb") as fh:
    for e in ijson.items(fh, "traceEvents.item"):
        cat = e.get("cat")
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            ts = float(e.get("ts", 0)); dur = float(e.get("dur", 0))
            dev_events.append((ts, ts + dur, cat))
            if cat == "gpu_memcpy":
                memcpy_by_name[(e.get("name") or "?")[:80]] += dur
        elif cat in ("cpu_op", "python_function", "cuda_runtime", "cuda_driver"):
            tid = e.get("tid"); ts = float(e.get("ts", 0)); dur = float(e.get("dur", 0))
            if dur <= 300_000.0:
                cpu_by_tid.setdefault(tid, []).append((ts, ts + dur, (e.get("name") or "?")[:120]))

dev_events.sort()
merged = []
for s, e, kind in dev_events:
    if merged and s <= merged[-1][1]:
        merged[-1][1] = max(merged[-1][1], e)
    else:
        merged.append([s, e, kind])
span = merged[-1][1] - merged[0][0]
busy = sum(e - s for s, e, _ in merged)
print(f"window {span/1e3:.1f} ms, device-busy(kernels+memcpy) {busy/1e3:.1f} ms, "
      f"idle {(span-busy)/1e3:.1f} ms")

# pure-idle gaps
gaps = [(e1, s2, s2 - e1) for (s1, e1, _), (s2, e2, _) in zip(merged, merged[1:]) if s2 - e1 >= MIN_GAP * 1000]
print(f"pure-idle gaps >= {MIN_GAP}ms: {len(gaps)}, total {sum(g for _,_,g in gaps)/1e3:.1f} ms")

# how much memcpy time total, and per-step
print("\n== memcpy summary ==")
tot = sum(v for v in memcpy_by_name.values())
print(f"total memcpy: {tot/1e3:.1f} ms")
for n, d in memcpy_by_name.most_common(8):
    print(f"{d/1e3:9.1f} ms  {n}")

# what launches during former kernel-only gaps: check cpu frames at each gap
for tid in cpu_by_tid:
    cpu_by_tid[tid].sort(key=lambda t: t[0])
starts = {tid: [t[0] for t in evs] for tid, evs in cpu_by_tid.items()}
owner = Counter()
for gs, ge, g in gaps:
    best = None
    for tid, evs in cpu_by_tid.items():
        i = bisect.bisect_right(starts[tid], gs) - 1
        cov = []
        steps = 0
        while i >= 0 and steps < 300:
            ts, te, name = evs[i]
            if te > gs and ts <= gs:
                cov.append((te - ts, name))
            elif ts >= ge:
                break
            i -= 1; steps += 1
        if cov and (best is None or len(cov) > best[0]):
            best = (len(cov), sorted(cov))
    if best is None:
        owner["UNMATCHED"] += g
        continue
    cov = best[1]
    names = [n for _, n in cov]
    key = next((n for n in names if "engram" in n or "prepare_inputs" in n or "synchronize" in n), names[0])
    owner[key[:100]] += g
print("\n== pure-idle gap ownership ==")
for k, ms in owner.most_common(10):
    print(f"{ms/1e3:9.1f} ms  {k}")
