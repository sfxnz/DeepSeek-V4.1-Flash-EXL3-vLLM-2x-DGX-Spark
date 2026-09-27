#!/usr/bin/env bash
# Disarm scan: grep both ranks' `docker logs` for every LOG_DISARMED marker
# (the constants next to each docker/patch file, listed by
# tools/engagement_audit.py). The post-ready audit reads the boot log once;
# a lever can also turn itself off later (prefetch v3 after 3 worker errors
# did, on both ranks, on 2026-09-24). Any hit makes a capture invalid for an
# A/B. Read-only: only the matching lines leave the container.
#
#   tools/disarm_scan.sh     # "== spark1 ==" / "== spark2 ==" sections
#
# Exit 0 when no rank has a disarm line, 1 when one does.
set -uo pipefail
cd "$(dirname "$0")/.."
CONTAINER_NAME="${CONTAINER_NAME:-dsv41-flash-exl3}"
WORKER_HOST="${WORKER_HOST:-spark2}"

mapfile -t markers < <(python3 -c 'import sys; sys.path.insert(0, "tools"); import engagement_audit as a; print("\n".join(sorted(set(a.expectations({})[1]))))')
pat=""
for m in "${markers[@]}"; do
  pat+=" -e $(printf '%q' "$m")"
done
cmd="docker logs $(printf '%q' "$CONTAINER_NAME") 2>&1 | grep -F$pat"

# grep: 0 = disarm lines, 1 = none, else the scan itself failed (counted as dirty).
found=0
scan() {
  echo "== $1 =="
  shift
  "$@"
  local rc=$?
  (( rc > 1 )) && echo "disarm scan failed (rc=$rc)"
  (( rc != 1 )) && found=1
  return 0
}
scan spark1 bash -c "$cmd"
scan spark2 ssh -o ConnectTimeout=10 -o BatchMode=yes "$WORKER_HOST" "$cmd"
exit "$found"
