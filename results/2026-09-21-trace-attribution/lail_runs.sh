#!/usr/bin/env bash
# L.A.I.L decode-prose bench driver: warmup (discard) + n real runs, job ids.
# Usage: lail_runs.sh N LABEL
set -euo pipefail
N="${1:-3}"
LABEL="${2:-run}"
TOKEN=$(grep -oP '(?<=LAIL_TOKEN=).*' /home/sfxnz/projects/ai-lab/local-ai-lab/.env)
for i in $(seq 1 "$N"); do
  ID=$(curl -s --max-time 10 "http://localhost:8765/api/bench/perf" -H "X-Lail-Token: $TOKEN" -H 'Content-Type: application/json' \
    -d '{"runner":"decode","workload":"prose","concurrencies":[1]}' | python3 -c "import json,sys; print(json.load(sys.stdin)['job_id'])")
  echo "$LABEL job[$i]=$ID"
  # poll until done (max 240s)
  for t in $(seq 1 48); do
    sleep 5
    ST=$(curl -s --max-time 10 "http://localhost:8765/api/jobs/$ID" -H "X-Lail-Token: $TOKEN")
    echo "$ST" | python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: print('poll err'); raise SystemExit
st=d.get('status') or d.get('state')
if st in ('done','completed','error','failed','cancelled'):
    print('STATUS', st); print(json.dumps(d)[:2000]); raise SystemExit(0)
raise SystemExit(1)
" && break || true
  done
done
