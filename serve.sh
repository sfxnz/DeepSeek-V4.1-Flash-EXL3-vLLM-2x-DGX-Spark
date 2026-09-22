#!/usr/bin/env bash
# Single reproducible launch for the DeepSeek-V4.1-Flash EXL3 recipe.
#
# This delegates to run.sh (generated from recipe.yaml), which orchestrates
# both DGX Spark nodes: worker on WORKER_HOST first, then the head rank, and
# waits for the OpenAI API. All knobs are env vars — see flags.md.
#
#   ./serve.sh                    # defaults from recipe.yaml
#   MAX_NUM_BATCHED_TOKENS=16384 ./serve.sh
#
# Exclusive GPUs: stop any other --gpus all container on both nodes first.
set -euo pipefail
cd "$(dirname "$0")"

exec ./run.sh "$@"
