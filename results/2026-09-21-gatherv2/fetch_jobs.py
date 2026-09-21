#!/usr/bin/env python3
"""Fetch L.A.I.L job results -> ndjson rows (Round 23)."""
import json, sys, urllib.request

TOKEN = None
for ln in open("/home/sfxnz/projects/ai-lab/local-ai-lab/.env"):
    if ln.startswith("LAIL_TOKEN="):
        TOKEN = ln.strip().split("=", 1)[1]
        break

tag = sys.argv[1]
out_path = sys.argv[2]
rows = []
for jid in sys.argv[3:]:
    req = urllib.request.Request(
        f"http://localhost:8765/api/jobs/{jid}",
        headers={"X-Lail-Token": TOKEN},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        d = json.load(r)
    res = d.get("result") or {}
    head = ((res.get("metrics") or {}).get("headline")) or {}
    arm = (((res.get("metrics") or {}).get("arms") or [{}])[0])
    rows.append({
        "tag": tag,
        "job_id": jid,
        "run_id": res.get("run_id"),
        "status": d.get("status"),
        "decode_c1": head.get("decode_tok_per_s_median_c1"),
        "prefill_c1": head.get("prefill_tok_per_s_median_c1"),
        "aggregate": arm.get("aggregate_tok_per_s"),
    })
with open(out_path, "a") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
for r in rows:
    print(r["tag"], r["job_id"], r["decode_c1"], r["prefill_c1"])
