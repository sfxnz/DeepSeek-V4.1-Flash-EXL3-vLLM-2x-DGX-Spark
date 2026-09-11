#!/usr/bin/env bash
# H1: SPEC=dspark NUM_SPECULATIVE_TOKENS=5, otherwise recipe defaults (eager).
set -euo pipefail
cd "$(dirname "$0")/../.."
./stop.sh || true
SPEC=dspark ./run.sh
python3 smoke_chat.py
python3 bench_decode.py --phase prose --concurrency 1 --runs 3 --max-tokens 200
