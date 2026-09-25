#!/usr/bin/env python3
"""Per-step saving projection from the fixed-config chain bench (host-side, stdlib only).

saving/step = sum over shapes of (stock - gemv) cold median per call at M x calls per step
(k3 profile at c=1: 43 each for qkv_a, wq_b, wo_b, wo_a, gate_up, down = 40 target + 3 draft
layers; 2 for Engram wkv (layers 1, 14) and lm_head; 1 for main_proj). The corrected figure
replaces the isolated gate_up + down deltas with concur.json's per-layer shared-expert saving
measured with the serve's side-stream router GEMM running concurrently.
Usage: projection.py RESULTS_DIR [M]
"""
from __future__ import annotations

import json
import os
import sys

CALLS = {"qkv_a": 43, "wq_b": 43, "wo_b": 43, "wo_a": 43, "shared_gate_up": 43, "shared_down": 43,
         "engram_wkv": 2, "lm_head": 2, "main_proj": 1}
STEP_MS = 64.45  # k3 profile decode step at c=1 (rank 0)


def main(argv: list[str]) -> int:
    d = argv[1]
    m = argv[2] if len(argv) > 2 else "4"
    rows = {}
    for part in ("a", "b"):
        rows.update(json.load(open(os.path.join(d, f"chain-{part}.json"))))
    concur = json.load(open(os.path.join(d, "concur.json")))
    out = {"M": int(m), "per_shape": {}, "calls_per_step": CALLS}
    total = 0.0
    for name, calls in CALLS.items():
        if m not in rows[name]:  # no GEMV bucket at this M (main_proj M > 4): the stock path stays
            out["per_shape"][name] = {"delta_us": 0.0, "saving_us_per_step": 0.0, "note": "stock path at this M"}
            continue
        r = rows[name][m]["cold"]
        delta = r["stock"]["median"] - r["gemv"]["median"]
        out["per_shape"][name] = {"stock_us": r["stock"]["median"], "gemv_us": r["gemv"]["median"],
                                  "delta_us": delta, "gain_pct": r["gain_pct_median"],
                                  "gemv_pct_of_250": r["gemv"]["pct250"], "bitwise": r["bitwise_all_calls"],
                                  "saving_us_per_step": delta * calls}
        total += delta * calls
    shared_iso = sum(out["per_shape"][n]["delta_us"] for n in ("shared_gate_up", "shared_down"))
    shared_conc = concur[m]["saving_us_with_router"]
    corrected = total - 43 * shared_iso + 43 * shared_conc
    out.update(isolated_ms_per_step=total / 1e3, shared_expert_isolated_us=shared_iso,
               shared_expert_with_router_us=shared_conc, corrected_ms_per_step=corrected / 1e3,
               corrected_pct_of_step=100 * corrected / 1e3 / STEP_MS, step_ms=STEP_MS)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
