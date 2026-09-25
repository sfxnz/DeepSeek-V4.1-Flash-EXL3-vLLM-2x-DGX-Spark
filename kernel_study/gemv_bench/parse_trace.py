#!/usr/bin/env python3
"""Parse torch-profiler chrome traces: aggregate CUDA kernel time by name.

Usage: parse_trace.py TRACE.json [TRACE2.json ...]
Prints: top kernels by total device time, plus phase-group rollup.
"""
import json
import sys
from collections import defaultdict


def rollup(name: str) -> str:
    n = name.lower()
    if "p2b" in n or "moe_batched" in n:
        return "MoE p2b kernel"
    if "nccl" in n:
        return "NCCL comms"
    if "had" in n and ("r_128" in n or "hf" in n):
        return "Hadamard (non-p2b)"
    if "gemm" in n or "cutlass" in n or "nvjet" in n or "matmul" in n or "s16816" in n or "hmma" in n:
        return "GEMM/attention"
    if "flash" in n or "mla" in n or "attn" in n or "paged" in n or "fmha" in n:
        return "Attention"
    if "engram" in n or "page" in n or "memcpy" in n or "memset" in n:
        return "Paging/memcpy"
    if "sampl" in n or "softmax" in n or "topk" in n or "routing" in n or "argmax" in n:
        return "Sampling/routing"
    if "norm" in n or "rms" in n:
        return "Norms"
    if "markov" in n or "draft" in n or "dspark" in n or "ngram" in n:
        return "Draft (DSpark)"
    if "elementwise" in n or "vectorized" in n or "reduce" in n or "cat" in n or "copy" in n:
        return "Elementwise/copy"
    return "other"

def main():
    files = sys.argv[1:]
    per_name = defaultdict(lambda: [0, 0])  # name -> [total_us, count]
    span = None
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        for ev in data.get("traceEvents", []):
            if ev.get("ph") != "X":
                continue
            cat = ev.get("cat", "")
            if cat not in ("kernel", "gpu_memcpy", "gpu_memset"):
                continue
            name = ev.get("name", "?")
            dur_us = ev.get("dur", 0.0)
            per_name[name][0] += dur_us
            per_name[name][1] += 1
            ts = ev.get("ts", 0)
            te = ts + dur_us
            if span is None:
                span = [ts, te]
            else:
                span[0] = min(span[0], ts)
                span[1] = max(span[1], te)

    total = sum(v[0] for v in per_name.values())
    print(f"traces: {files}")
    print(f"device-busy total: {total/1e6:.3f} s   timeline span: {(span[1]-span[0])/1e6:.3f} s   busy%: {100*total/(span[1]-span[0]):.1f}\n")
    print(f"{'device us':>12} {'count':>7}  name")
    for name, (us, cnt) in sorted(per_name.items(), key=lambda kv: -kv[1][0])[:30]:
        print(f"{us:12.0f} {cnt:7}  {name[:130]}")
    print("\n--- rollup ---")
    groups = defaultdict(float)
    for name, (us, cnt) in per_name.items():
        groups[rollup(name)] += us
    for g, us in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"{us/1e6:9.3f} s  {100*us/total:5.1f}%  {g}")

if __name__ == "__main__":
    main()
