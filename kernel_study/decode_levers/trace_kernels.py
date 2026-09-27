#!/usr/bin/env python3
"""Per-step kernel sums from torch-profiler traces (stdlib only, CPU).

Primary gate for the decode levers. Each saves ~1% of a ~65 ms step, which is
below L.A.I.L noise, so the decision uses the per-step time of these kernels
from a short profiled window.

  python3 kernel_study/decode_levers/trace_kernels.py TRACE.json[.gz] [...] \
      [--json out.json]

Steps = calls of --step-kernel / --per-step (default: p2b_moe_batched_kernel,
40 MoE layers per verify step, as in the 2026-09-21 trace parses). Rows are
grouped by (pattern, grid), so the wo_a pack at grid (22,4,1) is reported
apart from the draft-MoE pack at (12,1,1).
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys

PATTERNS = {
    "woa_pack": r"transpose_and_pack_fp32_into_ue8m0",
    "woa_einsum": r"sm120_fp8_fp4_gemm_1d1d_impl<0u, [34]u, 4096u",
    "mhc_prenorm_gemm": r"sm120_tf32_hc_prenorm_gemm",
    "mhc_pre_norm_fused": r"mhc_pre_big_fuse_with_norm_tilelang",
    "markov_gemm": r"cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x1",
    "topk": r"(?i)topk|radix|sort",
    "argmax": r"(?i)argmax",
}


_WS = re.compile(r"[\s,]*")


def iter_events(path: str, chunk: int = 1 << 22):
    """Stream traceEvents objects; a serve trace is ~2 GB of JSON (no json.load)."""
    opener = gzip.open if path.endswith(".gz") else open
    dec = json.JSONDecoder()
    with opener(path, "rt") as fh:
        buf = ""
        while '"traceEvents"' not in buf:
            more = fh.read(chunk)
            if not more:
                return
            buf += more
        pos = buf.index("[", buf.index('"traceEvents"')) + 1
        while True:
            pos = _WS.match(buf, pos).end()
            if buf.startswith("]", pos):
                return
            try:
                obj, pos = dec.raw_decode(buf, pos)
            except json.JSONDecodeError:
                more = fh.read(chunk)
                if not more:
                    return
                buf, pos = buf[pos:] + more, 0
                continue
            yield obj


def load_events(path: str) -> list[dict]:
    """Kernel events only, trimmed to name/dur/grid."""
    return [
        {"name": e.get("name", ""), "dur": e.get("dur", 0.0), "args": {"grid": (e.get("args") or {}).get("grid")}}
        for e in iter_events(path)
        if e.get("ph") == "X" and e.get("cat") == "kernel"
    ]


def summarize(events: list[dict], step_kernel: str, per_step: int, patterns=PATTERNS) -> dict:
    step_calls = sum(1 for e in events if step_kernel in e.get("name", ""))
    steps = step_calls / per_step if per_step else 0.0
    total_us = sum(float(e.get("dur", 0.0)) for e in events)
    rows: dict[tuple, dict] = {}
    for e in events:
        name = e.get("name", "")
        for key, pat in patterns.items():
            if re.search(pat, name):
                grid = tuple((e.get("args") or {}).get("grid") or ())
                r = rows.setdefault((key, grid), {"pattern": key, "grid": list(grid), "calls": 0,
                                                  "total_us": 0.0, "name": name[:100]})
                r["calls"] += 1
                r["total_us"] += float(e.get("dur", 0.0))
    out = []
    for r in sorted(rows.values(), key=lambda r: (r["pattern"], -r["total_us"])):
        r["us_per_call"] = round(r["total_us"] / r["calls"], 2)
        r["calls_per_step"] = round(r["calls"] / steps, 2) if steps else None
        r["ms_per_step"] = round(r["total_us"] / 1000.0 / steps, 3) if steps else None
        r["total_us"] = round(r["total_us"], 1)
        out.append(r)
    return {
        "steps": round(steps, 1),
        "kernel_ms_per_step": round(total_us / 1000.0 / steps, 3) if steps else None,
        "rows": out,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--step-kernel", default="p2b_moe_batched_kernel")
    ap.add_argument("--per-step", type=int, default=40)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    events = []
    for path in args.traces:
        events += load_events(path)
    res = summarize(events, args.step_kernel, args.per_step)
    print(f"steps={res['steps']} kernel_ms_per_step={res['kernel_ms_per_step']}")
    for r in res["rows"]:
        print(f"{r['pattern']:<20} grid={r['grid']!s:<14} calls/step={r['calls_per_step']!s:<7} "
              f"us/call={r['us_per_call']:<8} ms/step={r['ms_per_step']}  {r['name'][:60]}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
