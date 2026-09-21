#!/usr/bin/env python3
"""Round-22 pass: for each pure-idle gap >=2ms, split gap time into
(a) time the CPU thread was BLOCKED inside the covering cudaEventSynchronize
    runtime call (event fires late), vs
(b) time AFTER that call returned (post-sync CPU work before next launch).
Same safety pattern as parse_gaps_*: ijson stream + 6GB RLIMIT_AS.
"""
import bisect, gzip, resource, sys
from collections import Counter

resource.setrlimit(resource.RLIMIT_AS, (6 << 30, 6 << 30))
import ijson

trace = sys.argv[1]
MIN_GAP = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0

dev_events, cpu_by_tid = [], {}
with gzip.open(trace, "rb") as fh:
    for e in ijson.items(fh, "traceEvents.item"):
        cat = e.get("cat")
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            ts = float(e.get("ts", 0))
            dev_events.append((ts, ts + float(e.get("dur", 0))))
        elif cat in ("cuda_runtime", "cuda_driver", "python_function", "cpu_op"):
            tid = e.get("tid"); ts = float(e.get("ts", 0)); dur = float(e.get("dur", 0))
            if dur <= 300_000.0:
                cpu_by_tid.setdefault(tid, []).append((ts, ts + dur, (e.get("name") or "?")[:120]))

dev_events.sort()
merged = []
for s, e in dev_events:
    if merged and s <= merged[-1][1]:
        merged[-1][1] = max(merged[-1][1], e)
    else:
        merged.append([s, e])
gaps = [(e1, s2, s2 - e1) for (_, e1), (s2, _) in zip(merged, merged[1:]) if s2 - e1 >= MIN_GAP * 1000]
print(f"{len(gaps)} gaps >= {MIN_GAP}ms, total {sum(g for _,_,g in gaps)/1e3:.1f} ms")

for tid in cpu_by_tid:
    cpu_by_tid[tid].sort(key=lambda t: t[0])
starts = {tid: [t[0] for t in evs] for tid, evs in cpu_by_tid.items()}

blocked_ms, post_ms, post_names = 0.0, 0.0, Counter()
for gs, ge, g in gaps:
    # find the covering cudaEventSynchronize (or any cuda_runtime) on the
    # tid with the deepest covering stack — reuse simple heuristic: the tid
    # whose covering set is largest
    best = None
    for tid, evs in cpu_by_tid.items():
        i = bisect.bisect_right(starts[tid], gs) - 1
        cov, steps = [], 0
        while i >= 0 and steps < 300:
            ts, te, name = evs[i]
            if te > gs and ts <= gs:
                cov.append((te - ts, te, name))
            elif ts >= ge:
                break
            i -= 1; steps += 1
        if cov and (best is None or len(cov) > len(best)):
            best = cov
    if not best:
        continue
    # the sync call = shortest-duration covering cuda_runtime-ish frame
    sync = next(((te, n) for _, te, n in sorted(best) if "synchronize" in n.lower() or "Synchronize" in n), None)
    if sync is None:
        post_ms += g
        post_names["NO_SYNC_FRAME"] += g
        continue
    sync_end, sync_name = sync
    in_sync = max(0.0, min(sync_end, ge) - gs)
    after = max(0.0, ge - sync_end)
    blocked_ms += in_sync
    post_ms += after
    if after > 500:
        post_names["post-sync tail"] += after

n = len(gaps) or 1
print(f"blocked INSIDE cudaEventSynchronize across gap starts: {blocked_ms/1e3:.1f} ms ({blocked_ms/1e3/n:.2f} ms/gap)")
print(f"AFTER sync returned (post-sync CPU work): {post_ms/1e3:.1f} ms ({post_ms/1e3/n:.2f} ms/gap)")
for k, v in post_names.most_common(6):
    print(f"  {v/1e3:9.1f} ms  {k}")
