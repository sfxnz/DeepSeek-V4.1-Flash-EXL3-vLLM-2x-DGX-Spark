#!/usr/bin/env python3
"""Fetch a L.A.I.L perf job result and print run_id + decode_c1 tok/s."""
import json
import re
import sys
import urllib.request

job = sys.argv[1]
env = open("/home/sfxnz/projects/ai-lab/local-ai-lab/.env").read()
token = re.search(r"LAIL_TOKEN=(\S+)", env).group(1)
req = urllib.request.Request(
    f"http://localhost:8765/api/jobs/{job}", headers={"X-Lail-Token": token}
)
d = json.load(urllib.request.urlopen(req, timeout=10))
res = d.get("result", {})


def walk(o):
    out = []
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, (int, float)) and (
                "tok" in k.lower() or k == "decode_c1"
            ):
                out.append((k, v))
            out.extend(walk(v))
    elif isinstance(o, list):
        for v in o:
            out.extend(walk(v))
    return out


print(json.dumps(walk(res)))
runs = res.get("runs") or []
for r in runs:
    print(r.get("run_id"), r.get("decode_c1"), r.get("aggregate"))
