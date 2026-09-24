#!/usr/bin/env bash
# Capture a full baseline/experiment record against the live serve.
# Usage: tools/capture_cells.sh <label>
# Serializes micro, correctness, and e2e — never run them concurrently
# (MAX_NUM_SEQS=2 and shared prefill chunks contaminate each other).
# benches/e2e.py exits 1 when a phase fails its quality check; with pipefail
# that aborts the capture after 02-e2e on purpose (fast garbage is no cell).
set -euo pipefail
cd "$(dirname "$0")/.."

label="${1:?usage: capture_cells.sh <label>}"
out="results/${label}"
mkdir -p "$out"

echo "== micro ==" | tee "$out/00-micro.log"
benches/micro.sh 2>&1 | tee -a "$out/00-micro.log"

echo "== correctness (full) ==" | tee "$out/01-correctness.log"
tests/correctness.sh --full 2>&1 | tee -a "$out/01-correctness.log"

echo "== e2e ==" | tee "$out/02-e2e.log"
python3 benches/e2e.py 2>&1 | tee -a "$out/02-e2e.log"
python3 tools/measure_lail_prose.py 2>&1 | tee -a "$out/02-e2e.log"

echo "== decode reference ==" | tee "$out/03-decode.log"
python3 bench_decode.py --phase prose --concurrency 1 2>&1 | tee -a "$out/03-decode.log"

echo "captured $out"
