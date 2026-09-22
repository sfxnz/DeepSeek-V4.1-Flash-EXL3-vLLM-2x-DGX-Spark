#!/usr/bin/env bash
# Isolated prefill (pp) and decode (tg32) at fixed contexts, fresh docs per
# run so prefix caching stays out of the measurement.
# Usage: benches/micro.sh [--contexts 512 4096 16384 65536] [--runs 2] ...
set -euo pipefail
cd "$(dirname "$0")/.."

exec python3 benches/micro.py "$@"
