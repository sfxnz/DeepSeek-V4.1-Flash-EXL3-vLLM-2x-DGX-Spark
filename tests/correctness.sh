#!/usr/bin/env bash
# Fixed correctness eval for the live serve. Deterministic settings.
# Usage: tests/correctness.sh [--full]
#   --full adds the 64k-context recall check (about one minute of prefill).
set -euo pipefail
cd "$(dirname "$0")/.."

python3 tests/correctness.py "$@"
