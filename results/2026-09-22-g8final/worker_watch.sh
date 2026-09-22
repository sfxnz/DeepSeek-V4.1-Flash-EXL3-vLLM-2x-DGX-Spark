#!/usr/bin/env bash
# G8 boot watcher on spark2: poll worker anon-rss + swap every 15s, capture
# docker logs, abort (docker stop) if anon+swap > ABORT_GIB (default 60).
set -u
ABORT_GIB="${ABORT_GIB:-60}"
LOG="${WATCH_LOG:-/tmp/g8watch.log}"
DOCKER_LOG="${WATCH_DLOG:-/tmp/g8worker-$(date +%H%M%S).log}"
echo "watcher start $(date -Is) abort> ${ABORT_GIB}GiB" >> "$LOG"
while true; do
  pid=$(pgrep -f "VLLM::Worker_TP" | head -1 || true)
  if [[ -n "${pid:-}" ]]; then
    read -r rss swap <<< "$(awk '/^VmRSS:/{r=$2}/^VmSwap:/{s=$2}END{print r+0, s+0}' /proc/$pid/status 2>/dev/null)"
    gib=$(( (rss + swap) / 1024 / 1024 ))
    echo "$(date -Is) pid=$pid rss=${rss}kB swap=${swap}kB total=${gib}GiB" >> "$LOG"
    docker logs dsv41-flash-exl3 > "$DOCKER_LOG" 2>&1 || true
    if [[ "${ABORT_GIB}" != "never" && "${gib}" -gt "${ABORT_GIB}" ]]; then
      echo "$(date -Is) ABORT: worker ${gib}GiB > ${ABORT_GIB}GiB — stopping container" >> "$LOG"
      # capture smaps summary + python stack before death
      grep -E "^(Rss|Pss|Anonymous|Swap):" /proc/$pid/smaps_rollup >> "$LOG" 2>/dev/null || true
      cat /proc/$pid/smaps 2>/dev/null | awk '/^[0-9a-f]+-/{a=$1} /^Rss:/{r=$2} /^Anonymous:/{an=$2} an+0>262144 && r+0>0 {print a, r"kB", an"kB anon"}' | sort -k3 -rn | head -20 >> "$LOG" || true
      docker stop dsv41-flash-exl3 >/dev/null 2>&1 || true
      echo "$(date -Is) container stopped" >> "$LOG"
      exit 42
    fi
  fi
  # exit when container is gone and no pid for 3 consecutive rounds
  if ! docker ps --format '{{.Names}}' | grep -q dsv41-flash-exl3; then
    c=$(( ${c:-0} + 1 )); [[ $c -ge 3 ]] && { echo "$(date -Is) container gone, watcher exit" >> "$LOG"; exit 0; }
  else c=0; fi
  sleep 15
done
