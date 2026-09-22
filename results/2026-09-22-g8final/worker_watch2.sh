#!/usr/bin/env bash
# G8 boot watcher v2: poll worker rss; at >DUMP_GiB dump smaps (top anon maps)
# + py-spy/faulthandler stack every poll; abort at >ABORT_GIB.
set -u
ABORT_GIB="${ABORT_GIB:-40}"
DUMP_GIB="${DUMP_GIB:-15}"
LOG="${WATCH_LOG:-/tmp/g8watch.log}"
D="${DUMP_DIR:-/tmp/g8dumps}"; mkdir -p "$D"
echo "watcher2 start $(date -Is) abort>${ABORT_GIB} dump>${DUMP_GIB}" >> "$LOG"
while true; do
  pid=$(pgrep -f "VLLM::Worker_TP" | head -1 || true)
  if [[ -n "${pid:-}" && -d /proc/$pid ]]; then
    read -r rss swap <<< "$(awk '/^VmRSS:/{r=$2}/^VmSwap:/{s=$2}END{print r+0, s+0}' /proc/$pid/status 2>/dev/null)"
    gib=$(( (rss + swap) / 1024 / 1024 ))
    echo "$(date -Is) pid=$pid total=${gib}GiB" >> "$LOG"
    if [[ "$gib" -gt "$DUMP_GIB" ]]; then
      ts=$(date +%H%M%S)
      # top anonymous mappings by size
      awk '/^[0-9a-f]+-/{a=$1;an=0} /^Anonymous:/{an=$2} /^Rss:/{r=$2} an>65536 {print an, r, a}' /proc/$pid/smaps 2>/dev/null | sort -rn | head -15 > "$D/smaps_$ts.txt"
      grep -E '^(Rss|Anonymous|Swap|Private)' /proc/$pid/smaps_rollup >> "$D/smaps_$ts.txt" 2>/dev/null
      # python stack via gdb-free faulthandler not possible; capture /proc stack + wchan + fds count
      cat /proc/$pid/wchan 2>/dev/null >> "$D/smaps_$ts.txt"
      echo "fds=$(ls /proc/$pid/fd 2>/dev/null | wc -l)" >> "$D/smaps_$ts.txt"
      ls -la /proc/$pid/fd 2>/dev/null | awk '{print $NF}' | sort | uniq -c | sort -rn | head -8 >> "$D/smaps_$ts.txt"
    fi
    if [[ "$gib" -gt "$ABORT_GIB" ]]; then
      echo "$(date -Is) ABORT at ${gib}GiB" >> "$LOG"
      docker stop dsv41-flash-exl3 >/dev/null 2>&1 || true
      exit 42
    fi
  fi
  if ! docker ps --format '{{.Names}}' | grep -q dsv41-flash-exl3; then
    c=$(( ${c:-0} + 1 )); [[ $c -ge 30 ]] && { echo "$(date -Is) container gone, watcher exit" >> "$LOG"; exit 0; }
  else c=0; fi
  sleep 15
done
