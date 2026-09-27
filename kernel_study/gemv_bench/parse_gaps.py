#!/usr/bin/env python3
"""Round-18 gap attribution parser (crash-safe).

Streams a gzipped torch-profiler trace with ijson (never json.load — see the
2026-09-19 host-OOM incident, RESULTS.md round 11). Caps RLIMIT_AS at 6 GB.

Extends parse_trace_safe.py with GPU-idle bucketing:
  - collects kernel events (ts, dur, stream) and CPU-track events
    (cpu_op / python_function / cuda_runtime) as compact tuples,
  - merges kernel intervals per device into busy spans,
  - for every idle gap > 0.5 ms on the main device, finds the CPU thread
    frames active at gap start (innermost + outermost named frames) and the
    cuda_runtime call if any, and buckets gap time by owner class.

Usage: parse_gaps.py TRACE.json.gz [--device 0] [--min-gap-ms 0.5]
"""
from __future__ import annotations

import argparse
import gzip
import resource
import sys
from collections import Counter, defaultdict

resource.setrlimit(resource.RLIMIT_AS, (6 << 30, 6 << 30))

import ijson  # noqa: E402

# Owner-class rules (name substrings, first match wins per frame, evaluated
# innermost-first so the deepest python frame names the owner).
RULES = [
    # class #2 — eager region between verify graph and draft graph
    ("EAGER_SAMPLER_VERIFY_TO_DRAFT", (
        "rejection", "sample_tokens", "postprocess_sampled", "post_update",
        "gumbel_sample", "softmax", "argmax", "multinomial",
        "prepare_dflash_inputs", "propose", "generate_draft",
        "_flatten_sampled", "dflash", "speculator",
    )),
    # class #1 — host critical path / scheduler / input prep
    ("HOST_SCHED_INPUT_PREP", (
        "prepare_inputs", "build_normal_batch", "schedule", "scheduler",
        "apply_staged_writes", "engram_disk", "stage", "_read_rows",
        "Future.result", "copy_to_cpu", "get_output", "process_outputs",
        "detokenize", "AsyncLLm", "output_handler", "add_request",
        "gather_mm_embeddings", "update_from_output",
    )),
    # class #3 — cross-stream sync / events
    ("STREAM_SYNC_EVENT", (
        "cudaStreamWaitEvent", "cudaEventSynchronize", "StreamSynchronize",
        "cudaMemcpyAsync", "cudaEventRecord", "synchronize", "query",
    )),
]
FALLBACK = "OTHER"


def classify(frames):
    """frames: list of event names, innermost first."""
    for name in frames:
        low = name.lower()
        for cls, keys in RULES:
            if any(k in low for k in keys):
                return cls
        # python module path frames
        if "/" in name or name.startswith("vllm") or name.startswith("torch"):
            return "HOST_SCHED_INPUT_PREP"
    return FALLBACK


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--min-gap-ms", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    kernels_by_dev = defaultdict(list)
    # cpu track: tid -> list of (ts, dur, name) ; kept bounded by clipping to
    # the kernel time window later. python_function can be huge, so keep only
    # events shorter than 200 ms (long-lived thread frames are useless for
    # attribution) — bounded memory.
    cpu_by_tid = defaultdict(list)
    tids = {}  # tid -> is_main_thread_track heuristic unused, keep all
    thread_names = {}
    total_events = 0

    opener = gzip.open if args.trace.endswith(".gz") else open
    with opener(args.trace, "rb") as fh:
        events = ijson.items(fh, "traceEvents.item")
        for e in events:
            total_events += 1
            cat = e.get("cat")
            if cat == "kernel":
                pid = e.get("pid")
                # torch traces: pid == device id for kernel events (mostly);
                # keep device-0-ish unless empty
                kernels_by_dev[pid].append(
                    (float(e.get("ts", 0)), float(e.get("ts", 0)) + float(e.get("dur", 0)))
                )
            elif cat in ("cpu_op", "python_function", "cuda_runtime", "cuda_driver"):
                tid = e.get("tid")
                ts = float(e.get("ts", 0)); dur = float(e.get("dur", 0))
                if dur <= 200_000.0:  # µs cap
                    name = (e.get("name") or "?")[:120]
                    cpu_by_tid[tid].append((ts, ts + dur, name))
            elif cat == "thread_name" or e.get("name") == "thread_name":
                pass
            if total_events % 1_000_000 == 0:
                print(f".. streamed {total_events/1e6:.0f}M events", file=sys.stderr)

    print(f"events streamed: {total_events}")
    for dev, ks in sorted(kernels_by_dev.items()):
        print(f"device {dev}: {len(ks)} kernels, busy computing…")
    dev = args.device if args.device in kernels_by_dev else max(
        kernels_by_dev, key=lambda d: len(kernels_by_dev[d]))
    ks = sorted(kernels_by_dev[dev])
    # merge busy spans
    merged = []
    for s, e in ks:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    busy = sum(e - s for s, e in merged)
    span = merged[-1][1] - merged[0][0]
    print(f"\ndevice {dev}: window {span/1e3:.1f} ms, busy {busy/1e3:.1f} ms, "
          f"idle {(span-busy)/1e3:.1f} ms, {len(merged)} busy spans")

    # idle gaps
    gaps = []
    for (s1, e1), (s2, e2) in zip(merged, merged[1:]):
        g = s2 - e1
        if g >= args.min_gap_ms * 1000:
            gaps.append((e1, s2, g))

    gap_total = sum(g for _, _, g in gaps)
    print(f"gaps >= {args.min_gap_ms} ms: {len(gaps)}, total {gap_total/1e3:.1f} ms")

    # Bucket each gap by co-temporal CPU frames: sort each tid's events by
    # start, then for every gap find the events covering the gap-start
    # timestamp (the moment the GPU went idle waiting on the host).
    import bisect
    bucket_ms = Counter()
    bucket_examples = defaultdict(list)
    unmatched = 0
    for tid in cpu_by_tid:
        cpu_by_tid[tid].sort(key=lambda t: t[0])
    tid_starts = {tid: [t[0] for t in evs] for tid, evs in cpu_by_tid.items()}

    for gi, (gs, ge, g) in enumerate(gaps):
        # candidate frames: events active at gs (covering gs) — innermost =
        # shortest duration among covering events
        best_tid, best = None, None
        for tid, evs in cpu_by_tid.items():
            starts = tid_starts[tid]
            i = bisect.bisect_right(starts, gs) - 1
            # walk back to find events covering gs (bounded walk, 200 max)
            covering = []
            steps = 0
            while i >= 0 and steps < 200:
                ts, te, name = evs[i]
                if te <= gs:
                    # ended before gap start: if it ended within the gap, it is
                    # co-temporal work inside the gap
                    if te > gs - 0 and ts < ge and te > gs:
                        covering.append((te - ts, name, "inside"))
                elif ts <= gs < te:
                    covering.append((te - ts, name, "covering"))
                elif ts >= ge:
                    break
                i -= 1
                steps += 1
            if covering:
                # prefer covering frames (host was inside them when GPU idled)
                cov = [c for c in covering if c[2] == "covering"]
                use = cov or covering
                # innermost = min duration
                use_sorted = sorted(use, key=lambda c: c[0])
                score = len(use) + (1 if cov else 0)
                if best is None or score > best[0]:
                    best = (score, tid, use_sorted)
        if best is None:
            unmatched += g
            bucket_ms[FALLBACK] += g
            continue
        _, tid, use_sorted = best
        names = [n for _, n, _ in use_sorted[:12]]
        cls = classify(names)
        bucket_ms[cls] += g
        if len(bucket_examples[cls]) < 6:
            bucket_examples[cls].append(
                (g / 1000, gs - merged[0][0], names[:6]))

    print(f"\n== idle-gap ownership (device {dev}, gaps >= {args.min_gap_ms} ms) ==")
    for cls, ms in bucket_ms.most_common():
        print(f"{ms/1e3:9.2f} ms  {cls}")
    if unmatched:
        print(f"(unmatched {unmatched/1e3:.2f} ms -> {FALLBACK})")
    print("\n== examples per class ==")
    for cls, exs in bucket_examples.items():
        print(f"-- {cls}")
        for g, rel, names in exs:
            print(f"   gap {g:6.2f} ms @+{rel/1e3:8.1f} ms : {' <- '.join(names)}")

    # kernel summary (same as parse_trace_safe)
    print("\n(detailed kernel table: run parse_trace_safe.py on the same trace)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
