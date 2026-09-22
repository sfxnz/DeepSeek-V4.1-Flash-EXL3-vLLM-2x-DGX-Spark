#!/usr/bin/env bash
# Submit N L.A.I.L decode/prose jobs sequentially, collect medians.
# Usage: bash lail_bench.sh <n> <tag>   -> writes lail_<tag>.json
set -euo pipefail
N="${1:-5}"; TAG="${2:-run}"
TOKEN=$(grep '^LAIL_TOKEN=' /home/sfxnz/projects/ai-lab/local-ai-lab/.env | cut -d= -f2-)
OUT="$(dirname "$0")/lail_${TAG}.json"
echo "[" > "$OUT.tmp"
for i in $(seq 1 "$N"); do
  JID=$(curl -s -X POST http://127.0.0.1:8765/api/bench/perf \
    -H "X-Lail-Token: $TOKEN" -H 'Content-Type: application/json' \
    -d '{"runner":"decode","workload":"prose"}' \
    | python3 -c "import json,sys;print(json.load(sys.stdin)['job_id'])")
  for t in $(seq 1 60); do
    R=$(curl -s "http://127.0.0.1:8765/api/jobs/$JID" -H "X-Lail-Token: $TOKEN")
    ST=$(echo "$R" | python3 -c "import json,sys;print(json.load(sys.stdin).get('status'))")
    [ "$ST" = "completed" ] || [ "$ST" = "failed" ] && break
    sleep 10
  done
  echo "$R" | python3 -c "
import json,sys
d=json.load(sys.stdin)
h=d.get('result',{}).get('metrics',{}).get('headline',{})
row={'job_id':d.get('job_id'),'status':d.get('status'),
     'decode':h.get('decode_tok_per_s_median_c1'),
     'aggregate':(d.get('result',{}).get('metrics',{}).get('arms') or [{}])[0].get('aggregate_tok_per_s')}
print(json.dumps(row))"
  echo "$R" | python3 -c "
import json,sys
d=json.load(sys.stdin)
h=d.get('result',{}).get('metrics',{}).get('headline',{})
print(json.dumps({'job_id':d.get('job_id'),'decode':h.get('decode_tok_per_s_median_c1')}))" >> "$OUT.tmp"
  [ "$i" -lt "$N" ] && echo "," >> "$OUT.tmp"
done
echo "]" >> "$OUT.tmp"
mv "$OUT.tmp" "$OUT"
python3 - "$OUT" <<'EOF'
import json,statistics,sys
rows=[r for r in json.load(open(sys.argv[1])) if r.get('decode')]
vals=[r['decode'] for r in rows]
print(f"{sys.argv[1]}: n={len(vals)} decode_medians={vals} pooled_median={statistics.median(vals):.2f}")
EOF
