#!/usr/bin/env python3
"""Second-pass: full python stacks above streams.py synchronize for big gaps."""
from __future__ import annotations
import bisect, gzip, resource, sys
from collections import Counter

resource.setrlimit(resource.RLIMIT_AS, (6 << 30, 6 << 30))
import ijson

trace = sys.argv[1]
MIN_GAP = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0

kernels = []
cpu_by_tid = {}
with gzip.open(trace, "rb") as fh:
    for e in ijson.items(fh, "traceEvents.item"):
        cat = e.get("cat")
        if cat == "kernel":
            ts = float(e.get("ts", 0)); kernels.append((ts, ts + float(e.get("dur", 0))))
        elif cat in ("cpu_op", "python_function", "cuda_runtime", "cuda_driver"):
            tid = e.get("tid"); ts = float(e.get("ts", 0)); dur = float(e.get("dur", 0))
            if dur <= 300_000.0:
                cpu_by_tid.setdefault(tid, []).append((ts, ts + dur, (e.get("name") or "?")[:160]))

kernels.sort()
merged = []
for s, e in kernels:
    if merged and s <= merged[-1][1]:
        merged[-1][1] = max(merged[-1][1], e)
    else:
        merged.append([s, e])
gaps = [(e1, s2, s2 - e1) for (s1, e1), (s2, e2) in zip(merged, merged[1:]) if s2 - e1 >= MIN_GAP * 1000]
print(f"{len(gaps)} gaps >= {MIN_GAP}ms, total {sum(g for _,_,g in gaps)/1e3:.1f} ms")

for tid in cpu_by_tid:
    cpu_by_tid[tid].sort(key=lambda t: t[0])
starts = {tid: [t[0] for t in evs] for tid, evs in cpu_by_tid.items()}

caller_counter = Counter()
shown = 0
for gs, ge, g in gaps:
    best = None
    for tid, evs in cpu_by_tid.items():
        i = bisect.bisect_right(starts[tid], gs) - 1
        cov = []
        steps = 0
        while i >= 0 and steps < 400:
            ts, te, name = evs[i]
            if te > gs and ts <= gs:
                cov.append((te - ts, name))
            elif te <= gs and ts < ge and te > gs:
                cov.append((te - ts, name + "  [ends inside gap]"))
            elif ts >= ge:
                break
            i -= 1; steps += 1
        if cov and (best is None or len(cov) > best[0]):
            best = (len(cov), tid, sorted(cov))
    if best is None:
        continue
    _, tid, cov = best
    stack = [n for _, n in cov]  # innermost first
    # caller = first vllm/torch frame below a runtime frame
    key = next((n for n in stack if "streams.py" in n or "graphs.py" in n or "synchronize" in n.lower()), stack[0])
    # find what's ABOVE streams.py (outer frames = larger dur = later in cov)
    try:
        idx = next(i for i, n in enumerate(stack) if "streams.py(245)" in n)
        outer = stack[idx+1:idx+8]
    except StopIteration:
        outer = stack[1:8]
    caller_counter[" <- ".join(outer[:5])] += g
    if shown < 8 and g >= 4_000:
        print(f"\n-- gap {g/1000:.1f} ms @+{(gs-merged[0][0])/1e3:.1f} ms (tid {tid})")
        for n in stack[:22]:
            print(f"     {n}")
        shown += 1

print("\n== gap ms by outer-caller (frames above streams.py synchronize) ==")
for caller, ms in caller_counter.most_common(12):
    print(f"{ms/1e3:9.1f} ms  {caller}")
