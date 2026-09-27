#!/usr/bin/env python3
"""Tag each NCCL sweep run (arm, rep) with the spark1 GPU utilization seen
before its containers started (first 2 s of the run's nvidia-smi log). With the
serve down, any utilization there belongs to another process: from 07:27 UTC a
Playwright chromium GPU process (type G) shared spark1's GB10.

  nccl_contamination.py SWEEP_DIR > contamination.json
"""
import glob
import json
import os
import statistics
import sys

root = sys.argv[1]
out = {}
for f in sorted(glob.glob(os.path.join(root, "*", "rep*.smi.spark1.csv"))):
    arm = os.path.basename(os.path.dirname(f))
    rep = os.path.basename(f).split(".")[0]
    rows = [ln.strip().split(", ") for ln in open(f) if ln.strip()]
    head = [int(r[5].rstrip(" %")) for r in rows[:4] if len(r) > 5 and r[5].rstrip(" %").isdigit()]
    out.setdefault(arm, {})[rep] = {
        "start_local": rows[0][0][11:19] if rows else None,
        "pre_run_gpu_util_pct": head,
        "clean": bool(head) and max(head) == 0,
        "has_result": os.path.exists(os.path.join(root, arm, rep + ".rank0.json")),
    }
json.dump(out, sys.stdout, indent=1)
print()
clean = sum(v["clean"] for a in out.values() for v in a.values())
total = sum(len(a) for a in out.values())
print(f"clean runs {clean}/{total}", file=sys.stderr)
