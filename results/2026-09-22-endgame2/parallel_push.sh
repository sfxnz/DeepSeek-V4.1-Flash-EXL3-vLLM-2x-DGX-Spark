#!/usr/bin/env bash
# Parallel pack push (Round 29): 4 concurrent rsync streams over the 10G link
# (single-stream TCP is ~11ms-RTT bound at ~22MB/s; 4 streams -> ~4x).
set -uo pipefail
DST="$HOME/.cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-EXL3/snapshots/2.0bpw-mcg-g8"
SSH='ssh -o StrictHostKeyChecking=accept-new'
ssh spark2 "mkdir -p '$DST'"
cd "$DST"
JOBS=4
running=0
for f in model-*-of-00048.safetensors; do
  # skip if size already matches on the remote (resumable)
  rsync -e "$SSH" -rltD --partial --inplace --info=name0 "$f" "spark2:$DST/" &
  running=$((running+1))
  if (( running >= JOBS )); then wait -n; running=$((running-1)); fi
done
wait
# push index + configs too
rsync -e "$SSH" -rltD --info=name0 model.safetensors.index.json config.json "spark2:$DST/"
echo "parallel push done $(date -Is)"
