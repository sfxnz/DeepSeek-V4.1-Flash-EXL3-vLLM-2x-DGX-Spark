#!/usr/bin/env bash
# Merge two-Spark EXL3 shards, rebuild the index, rsync the pack to the worker.
set -euo pipefail
DST="${DST:-$HOME/.cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-EXL3/snapshots/2.0bpw-mcg}"
WORKER="${WORKER:-10.100.8.2}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
log() { printf '==> %s\n' "$*"; }

need=( )
for i in $(seq 1 48); do
  need+=("$(printf 'model-%05d-of-00048.safetensors' "$i")")
done

log "pull expert shards 23-42 from $WORKER"
mkdir -p "$DST"
list="$(mktemp)"
for i in $(seq 23 42); do
  printf 'model-%05d-of-00048.safetensors\n' "$i"
done >"$list"
rsync -a --partial --info=progress2 -e 'ssh -o StrictHostKeyChecking=accept-new' \
  --files-from="$list" "$WORKER:$DST/" "$DST/"
rm -f "$list"

missing=()
for f in "${need[@]}"; do
  if [[ ! -f "$DST/$f" ]]; then
    missing+=("$f")
  fi
done
if ((${#missing[@]})); then
  echo "pack incomplete, missing ${#missing[@]} shards: ${missing[*]}" >&2
  exit 1
fi

log "rebuild index"
python3 "$ROOT/tools/rebuild_index.py" --dst "$DST"
if ! python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); q=c.get("quantization_config") or {}; sys.exit(0 if q.get("quant_method")=="exl3" else 1)' \
  "$DST/config.json"; then
  echo "config.json is not an EXL3 pack" >&2
  exit 1
fi

log "push pack to $WORKER"
ssh -o StrictHostKeyChecking=accept-new "$WORKER" "mkdir -p '$DST'"
rsync -a --partial --info=progress2 -e 'ssh -o StrictHostKeyChecking=accept-new' \
  "$DST/" "$WORKER:$DST/"
log "assembled $DST"
