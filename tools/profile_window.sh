#!/usr/bin/env bash
# Profile ONE L.A.I.L prose request on a serve booted with
#   EXTRA_ARGS='--profiler-config {"profiler":"torch","torch_profiler_dir":"/tmp/dsv41-traces"}'
# (run.sh forwards EXTRA_ARGS to the worker, so both ranks write traces).
#
# Protocol (R18/R22/R25 + two-rank fix): warmup (profiler off) -> start_profile
# -> one streaming 512-token prose request -> stop_profile -> flush wait ->
# list traces on BOTH ranks -> docker cp on each host BEFORE ./stop.sh
# (docker rm destroys them). Traces stay on the host that wrote them; parse
# there on an idle host with tools/extract_kernels.py (RLIMIT_AS, never json.load).
# Watch MemAvail: it dipped to 2 GiB during cp in R22/R25.
#
# This script never stops the serve. Run ./stop.sh yourself afterwards.
set -euo pipefail
PORT="${PORT:-8000}"
API="${API:-http://127.0.0.1:$PORT}"
CONTAINER_NAME="${CONTAINER_NAME:-dsv41-flash-exl3}"
WORKER_HOST="${WORKER_HOST:-spark2}"
TRACE_DIR="${TRACE_DIR:-/tmp/dsv41-traces}"
OUT_DIR="${OUT_DIR:-$HOME/projects/data/dsv41-traces/$(date +%Y%m%d-%H%M%S)}"
FLUSH_S="${FLUSH_S:-15}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

PROMPT='Continue this essay in the same voice. Do not stop.\n\nDecode throughput and time-to-first-token feel different when a coding agent shares a long system prompt across tabs on a DGX Spark with unified memory. The KV cache is the product, not a leftover after util. '
body() {
  printf '{"messages":[{"role":"user","content":"%s"}],"max_tokens":512,"min_tokens":512,"ignore_eos":true,"temperature":0.2,%s,"chat_template_kwargs":{"thinking":false}}' \
    "$PROMPT" "$1"
}

echo "== warmup (profiler off) =="
curl -s --max-time 120 "$API/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "$(body '"stream":false')" >"$TMP/warmup.json"
python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print('warmup completion tokens:', r['usage']['completion_tokens'])" "$TMP/warmup.json"

echo "== start profile =="
curl -s --max-time 30 -X POST "$API/start_profile"
echo
echo "== profiled request (stream, 512 tok) =="
S=$(date +%s.%N)
curl -sN --max-time 180 "$API/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "$(body '"stream":true,"stream_options":{"include_usage":true}')" >"$TMP/profiled.sse"
E=$(date +%s.%N)
python3 - "$TMP/profiled.sse" "$S" "$E" <<'PY'
import json, sys
usage, chunks = None, 0
for line in open(sys.argv[1]):
    if line.startswith("data: ") and "[DONE]" not in line:
        try:
            d = json.loads(line[6:])
        except ValueError:
            continue
        if d.get("usage"):
            usage = d["usage"]
        if d.get("choices"):
            chunks += 1
print("profiled req usage:", usage, "content chunks (~steps):", chunks)
print("wall s:", round(float(sys.argv[3]) - float(sys.argv[2]), 2))
PY
echo "== stop profile =="
curl -s --max-time 120 -X POST "$API/stop_profile"
echo
date
sleep "$FLUSH_S"

echo "== trace files: rank 0 (head) =="
docker exec "$CONTAINER_NAME" sh -c "ls -la $TRACE_DIR/"
echo "== trace files: rank 1 ($WORKER_HOST) =="
ssh "$WORKER_HOST" docker exec "$CONTAINER_NAME" sh -c "'ls -la $TRACE_DIR/'"

echo "== MemAvail before cp =="
free -h | sed -n 1,2p
ssh "$WORKER_HOST" free -h | sed -n 1,2p

echo "== docker cp BEFORE stop: rank 0 -> $(hostname):$OUT_DIR/rank0 =="
mkdir -p "$OUT_DIR"
docker cp "$CONTAINER_NAME:$TRACE_DIR" "$OUT_DIR/rank0"
ls -la "$OUT_DIR/rank0"
echo "== docker cp BEFORE stop: rank 1 -> $WORKER_HOST:$OUT_DIR/rank1 =="
ssh "$WORKER_HOST" "mkdir -p '$OUT_DIR' && docker cp '$CONTAINER_NAME:$TRACE_DIR' '$OUT_DIR/rank1' && ls -la '$OUT_DIR/rank1'"

echo "== MemAvail after cp =="
free -h | sed -n 1,2p
ssh "$WORKER_HOST" free -h | sed -n 1,2p
echo "Traces saved. Now ./stop.sh, then parse each rank on its own idle host."
