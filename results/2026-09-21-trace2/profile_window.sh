#!/usr/bin/env bash
# Profile ONE L.A.I.L prose request on the live serve (round-18 boot 1).
# Sends the frozen L.A.I.L prose prompt as a streaming request, starts the
# profiler just before, stops after ~10s of decode (~130 steps @ ~78ms).
set -euo pipefail
API=http://127.0.0.1:8000
TOKEN_HDR=('Content-Type: application/json')

echo "== warmup (profiler off) =="
curl -s --max-time 120 "$API/v1/chat/completions" -H "${TOKEN_HDR[0]}" -d '{
  "messages":[{"role":"user","content":"Continue this essay in the same voice. Do not stop.\n\nDecode throughput and time-to-first-token feel different when a coding agent shares a long system prompt across tabs on a DGX Spark with unified memory. The KV cache is the product, not a leftover after util. "}],
  "max_tokens": 512, "min_tokens": 512, "ignore_eos": true, "temperature": 0.2,
  "stream": false, "chat_template_kwargs": {"thinking": false}
}' > /tmp/lail-warmup.json
python3 -c "import json; r=json.load(open('/tmp/lail-warmup.json')); print('warmup completion tokens:', r['usage']['completion_tokens'])"

echo "== start profile =="
curl -s --max-time 30 -X POST "$API/start_profile"
echo
echo "== profiled request (stream, 512 tok) =="
S=$(date +%s.%N)
curl -sN --max-time 180 "$API/v1/chat/completions" -H "${TOKEN_HDR[0]}" -d '{
  "messages":[{"role":"user","content":"Continue this essay in the same voice. Do not stop.\n\nDecode throughput and time-to-first-token feel different when a coding agent shares a long system prompt across tabs on a DGX Spark with unified memory. The KV cache is the product, not a leftover after util. "}],
  "max_tokens": 512, "min_tokens": 512, "ignore_eos": true, "temperature": 0.2,
  "stream": true, "stream_options": {"include_usage": true},
  "chat_template_kwargs": {"thinking": false}
}' > /tmp/lail-profiled.sse
E=$(date +%s.%N)
python3 -c "
import json
usage=None
for line in open('/tmp/lail-profiled.sse'):
    if line.startswith('data: ') and '[DONE]' not in line:
        try:
            d=json.loads(line[6:])
            u=d.get('usage')
            if u: usage=u
        except Exception: pass
print('profiled req usage:', usage)
"
echo "== stop profile =="
curl -s --max-time 120 -X POST "$API/stop_profile"
echo
date; sleep 15
echo "== trace files in container =="
docker exec dsv41-flash-exl3 sh -c 'ls -la /tmp/dsv41-traces/ 2>/dev/null'
