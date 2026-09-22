#!/usr/bin/env bash
# v3: abort on ANON rss only (smaps_rollup Anonymous + Swap) > 60 GiB.
# v2 was wrong: it aborted on rss+swap TOTAL, which counts the ~79 GiB/rank
# of reclaimable file-backed pack mmap pages that flow through RSS during a
# normal (stock) load too. The OOM floor from the task is anon-rss > 60 GiB.
set -u
LOG=/tmp/g8watch3.log
echo "watcher3 start $(date -Is) abort-anon>60GiB" >> "$LOG"
while true; do
  pid=$(pgrep -f "VLLM::Worker_TP" | head -1 || true)
  if [[ -n "${pid:-}" && -d /proc/$pid ]]; then
    anon=$(awk '/^Anonymous:/{print $2}' /proc/$pid/smaps_rollup 2>/dev/null)
    swap=$(awk '/^Swap:/{print $2}' /proc/$pid/smaps_rollup 2>/dev/null)
    gib=$(( (${anon:-0} + ${swap:-0}) / 1024 / 1024 ))
    echo "$(date -Is) anon=${anon:-0}kB swap=${swap:-0}kB total=${gib}GiB" >> "$LOG"
    if [[ "$gib" -gt 60 ]]; then
      echo "$(date -Is) ABORT anon-total ${gib}GiB > 60" >> "$LOG"
      exit 1
    fi
  fi
  sleep 15
done
