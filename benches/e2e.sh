#!/usr/bin/env bash
# Real-workload e2e: coding-agent turn, 64k doc recall+summary, tool/JSON.
# Plus the frozen L.A.I.L prose harness that published numbers use.
# Usage: benches/e2e.sh [--runs 1]
set -euo pipefail
cd "$(dirname "$0")/.."

python3 benches/e2e.py "$@"
echo
python3 tools/measure_lail_prose.py
