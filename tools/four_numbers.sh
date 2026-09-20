#!/usr/bin/env bash
# Four-numbers capture for ONE campaign arm — HTTP-only, serialized, zero GPU
# state change. Run from anywhere; cds to the repo root.
#
#   1. prose decode c=1 tok/s   — 9-run median   (bench_decode.py, unmodified)
#   2. cold prefill tok/s @8k/@32k — 3 runs each (benches/micro.py; fresh doc
#      per run, prefix-cache bust built in)
#   3. MoE/attention ms per layer — NO live probe exists; recorded as a
#      documented fallback + TODO in the JSON notes. Do NOT build a profiler
#      here; profiling is a separate gated step.
#   4. MemAvailable GiB on spark1 AND spark2, read immediately after the 32k
#      prefill (free -h / free -b — never nvidia-smi)
# Plus DSpark acceptance (/metrics deltas) and L.A.I.L prose median
# (tools/measure_lail_prose.py — CLI twin of L.A.I.L's own bench math).
#
# Usage: tools/four_numbers.sh --arm NAME [--out DIR]
#   DIR default: results/$(date +%Y-%m-%d)-NAME
# Output: DIR/four_numbers.json + numbered logs. Logs are tee'd live; on any
# early exit the EXIT trap writes a PARTIAL marker and best-effort partial
# JSON via tools/four_numbers_parse.py — partial data is never lost.
#
# Capture order is serialized on purpose (MAX_NUM_SEQS=2; concurrent
# prefill chunks contaminate each other): provenance -> prose 9x -> micro
# 8k+32k -> free both nodes (right after the 32k prefill) -> L.A.I.L 3x.
# Runtime budget ~10-15 min.
set -euo pipefail
cd "$(dirname "$0")/.."

arm="" out=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --arm) arm="${2:?--arm needs a value}"; shift 2 ;;
    --out) out="${2:?--out needs a value}"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$arm" ]] || { echo "usage: tools/four_numbers.sh --arm NAME [--out DIR]" >&2; exit 2; }
out="${out:-results/$(date +%Y-%m-%d)-${arm}}"
mkdir -p "$out"
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "arm=$arm ts=$ts host=$(hostname -s) out=$out" | tee "$out/00-header.log"

json_done=0
finish_partial() {
  local rc=$?
  trap - EXIT
  if [[ "$json_done" != 1 ]]; then
    printf 'PARTIAL capture: rc=%s at=%s\n' "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      | tee "$out/PARTIAL" >&2
    python3 tools/four_numbers_parse.py "$arm" "$ts" "$out" partial \
      >"$out/05-parse-partial.log" 2>&1 || true
  fi
  exit "$rc"
}
trap finish_partial EXIT

# --- provenance (read-only docker inspect) --------------------------------
docker inspect dsv41-flash-exl3 --format '{{.Config.Image}}' \
  >"$out/serve_image.txt" 2>/dev/null \
  || echo "(docker inspect image failed)" >"$out/serve_image.txt"
docker inspect dsv41-flash-exl3 --format '{{json .Config.Cmd}}' \
  >"$out/serve_cmd.json" 2>/dev/null \
  || echo "[]" >"$out/serve_cmd.json"

# --- 1) prose decode, c=1, 9 runs (median) --------------------------------
python3 bench_decode.py --phase prose --concurrency 1 --max-tokens 200 \
  --runs 9 2>&1 | tee "$out/01-prose.log"

# --- 2) cold prefill 8k + 32k (fresh doc per run; cache-bust built in) ----
python3 benches/micro.py --contexts 8192 32768 --runs 3 2>&1 \
  | tee "$out/02-micro.log"

# --- 4) MemAvailable BOTH nodes, immediately after the 32k prefill --------
{
  echo "== spark1 $(hostname -s) $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
  free -h
  free -b
  echo "== spark2 =="
  ssh -o ConnectTimeout=10 -o BatchMode=yes spark2 free -h
  ssh -o ConnectTimeout=10 -o BatchMode=yes spark2 free -b
} 2>&1 | tee "$out/03-mem.log"

# --- L.A.I.L prose 3x (the user-visible number) ---------------------------
python3 tools/measure_lail_prose.py --runs 3 2>&1 | tee "$out/04-lail.log"

# --- parse everything into one JSON ----------------------------------------
python3 tools/four_numbers_parse.py "$arm" "$ts" "$out" complete \
  2>&1 | tee "$out/05-parse.log"

json_done=1
trap - EXIT
echo "four-numbers capture complete: $out/four_numbers.json"
