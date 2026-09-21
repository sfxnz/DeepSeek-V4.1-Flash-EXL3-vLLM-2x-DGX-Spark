#!/usr/bin/env bash
# Launch PF-G8 pack rebuild (REBUILD-PLAN.md step 2), spark2, expert shards 23-42.
# Detached: docker run -d -> container owned by the dockerd daemon, survives
# any session exit. Log written by the container into this results dir.
# NOTE: run ON spark2. The repo tools on spark2 were synced from spark1 @3242152
# (quantize_experts_exl3.py + pack_meta.py + mxfp4.py) because spark2's checkout
# is on an older commit without the pfg8 fold.
set -euo pipefail

R=/home/sfxnz/projects/ai-lab/recipes/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark
HUB=/cache/huggingface/hub
IMAGE=dsv41-flash-exl3-sm121:canonical-e12
NAME=dsv41-quant-g8-r2
SRC=$HUB/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277
DST=$HUB/models--sfxnz--DeepSeek-V4.1-Flash-EXL3/snapshots/2.0bpw-mcg-g8

mkdir -p "$R/results/2026-09-21-pfg8"
docker rm -f "$NAME" >/dev/null 2>&1 || true
: > "$R/results/2026-09-21-pfg8/rebuild-spark2.log"
docker run -d \
  --name "$NAME" \
  --restart no \
  --gpus all \
  --ipc host \
  --shm-size 32g \
  --ulimit memlock=-1:-1 \
  -v "$HOME/.cache/huggingface:/cache/huggingface" \
  -v "$R:/repo" \
  -v "$R/results/2026-09-21-pfg8:/logs" \
  -e DSV41_PACK_PF_G8=1 \
  --entrypoint bash \
  "$IMAGE" \
  -c "exec python3 /repo/tools/quantize_experts_exl3.py \
  --codebook mcg \
  --allow-partial --batch 8 --greedy --beam 16 \
  --src $SRC --dst $DST \
  --only-files $(seq -f 'model-%05g-of-00048.safetensors' 23 42 | tr '\n' ' ') \
  >> /logs/rebuild-spark2.log 2>&1"

date -Is | tee "$R/results/2026-09-21-pfg8/rebuild-spark2.started"
docker ps --filter "name=$NAME" --format '{{.ID}} {{.Image}} {{.Status}} {{.Names}}'
