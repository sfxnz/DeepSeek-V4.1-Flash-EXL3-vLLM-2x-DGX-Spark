#!/usr/bin/env bash
# GPU contamination guard for serve arms (spark1's GB10 is shared with a foreign headless Chromium).
#   gpu_guard.sh HOST OUT [SECONDS]   sample HOST's GPU processes every SECONDS (default 10) until
#                                     killed; HOST = local or an ssh host (spark2). One line per
#                                     process: "<utc> ok|FOREIGN <pmon row>".
#   gpu_guard.sh --check OUT...       exit 1 (and print them) if any sample was FOREIGN.
# A process is ours when its host PID belongs to the serve container (docker top CONTAINER_NAME)
# or its command is Xorg / gnome-shell (display, no compute). Anything else (chrome --type=gpu,
# a microbench, the requant) is FOREIGN: the numbers taken while it ran are void.
set -uo pipefail
if [ "${1:-}" = "--check" ]; then
  shift
  bad=$(grep -h ' FOREIGN ' "$@" || true)
  n=$(printf '%s' "$bad" | grep -c . || true)
  echo "gpu_guard: $n FOREIGN samples in $*"
  [ "$n" -eq 0 ] || { printf '%s\n' "$bad" | head -20; exit 1; }
  exit 0
fi
host=$1 out=$2 every=${3:-10}
name=${CONTAINER_NAME:-dsv41-flash-exl3}
run() { if [ "$host" = local ]; then bash -c "$1"; else ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" "$1"; fi; }
while true; do
  ts=$(date -u +%FT%TZ)
  pids=$(run "docker top $name -eo pid 2>/dev/null | tail -n +2 | tr '\n' ' '" || true)
  rows=$(run "nvidia-smi pmon -c 1" 2>&1) || rows="pmon-failed $rows"
  printf '%s\n' "$rows" | awk -v ts="$ts" -v pids=" $pids " '
    /^#/ { next }
    /pmon-failed/ { print ts, "FOREIGN", $0; next }
    $2 == "-" { next }
    { tag = (index(pids, " " $2 " ") || $10 == "Xorg" || $10 == "gnome-shell") ? "ok" : "FOREIGN"; print ts, tag, $0 }
  ' >> "$out"
  sleep "$every"
done
