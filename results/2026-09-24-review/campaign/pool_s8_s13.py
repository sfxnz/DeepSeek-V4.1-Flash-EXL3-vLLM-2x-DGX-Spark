#!/usr/bin/env python3
"""Pool the per-run values of the two identical-config round-34 boots (s8, s13).

s8-S-sparse-markov and s13-promote-final ran the same env and engine argv.
Reads the per-run lines of 04-bench-decode.log (values as logged, 2 decimals)
and 03-/11-lail-prose-*.log, and writes pooled-s8-s13.json next to this file.
bench_decode per-stream medians pool every stream (c=2: 2 streams per run),
aggregate medians pool every run, as bench_decode.py itself does per boot.
"""
import json
import re
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOOTS = ["s8-S-sparse-markov", "s13-promote-final"]
BENCH = re.compile(
    r"phase=(\w+) c=(\d+) run=\d+ .*agg=([\d.]+) tok/s per_stream=\[([^\]]*)\] "
    r"ttft=\[([^\]]*)\] acc=([\d.]+) ms_step=\[([^\]]*)\]"
)


def floats(s):
    return [float(x) for x in s.split(",")]


def bench():
    cells = {}
    for b in BOOTS:
        for line in (HERE / b / "04-bench-decode.log").read_text().splitlines():
            m = BENCH.match(line)
            if not m:
                continue
            c = cells.setdefault(f"{m[1]} c={m[2]}", {"runs": 0, "agg": [], "decode": [], "ttft": [], "ms_step": [], "acc": []})
            c["runs"] += 1
            c["agg"].append(float(m[3]))
            c["decode"] += floats(m[4])
            c["ttft"] += floats(m[5])
            c["acc"].append(float(m[6]))
            c["ms_step"] += floats(m[7])
    out = {}
    for k, c in cells.items():
        out[k] = {
            "n_runs": c["runs"],
            "n_streams": len(c["decode"]),
            "median_decode_tok_s": round(st.median(c["decode"]), 2),
            "median_agg_tok_s": round(st.median(c["agg"]), 2),
            "median_ttft_s": round(st.median(c["ttft"]), 3),
            "median_ms_per_step": round(st.median(c["ms_step"]), 2),
            "median_run_acceptance_len": round(st.median(c["acc"]), 3),
            "per_run_agg_tok_s": c["agg"],
            "per_stream_decode_tok_s": c["decode"],
        }
    return out


def lail(name):
    rows = []
    for b in BOOTS:
        for line in (HERE / b / name).read_text().splitlines():
            if line.startswith("run="):
                rows.append(json.loads(line.split(" ", 1)[1]))
    return {
        "n": len(rows),
        "median_lail_tok_s": round(st.median(r["lail_tok_s"] for r in rows), 2),
        "median_ttft_s": round(st.median(r["ttft_s"] for r in rows), 3),
        "median_ms_per_step": round(st.median(1000 * r["decode_s"] / r["chunks"] for r in rows), 2),
        "median_acceptance_len": round(st.median(r["acceptance_len"] for r in rows), 3),
        "per_run_lail_tok_s": [r["lail_tok_s"] for r in rows],
    }


def main():
    out = {
        "boots": BOOTS,
        "note": "pooled s8+s13, 2 boots, identical config (round-34 defaults on canonical-e13)",
        "bench_decode": bench(),
        "lail_prose_fresh": lail("03-lail-prose-fresh.log"),
        "lail_prose_after_c2_stress": lail("11-lail-prose-after-c2.log"),
    }
    (HERE / "pooled-s8-s13.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, v in out["bench_decode"].items():
        print(k, v["n_runs"], v["n_streams"], v["median_decode_tok_s"], v["median_agg_tok_s"], v["median_ttft_s"], v["median_ms_per_step"])
    for k in ("lail_prose_fresh", "lail_prose_after_c2_stress"):
        v = out[k]
        print(k, v["n"], v["median_lail_tok_s"], v["median_ttft_s"], v["median_ms_per_step"], v["median_acceptance_len"])


if __name__ == "__main__":
    main()
