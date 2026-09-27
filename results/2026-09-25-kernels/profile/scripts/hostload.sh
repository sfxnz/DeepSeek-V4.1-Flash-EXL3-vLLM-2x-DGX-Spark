#!/usr/bin/env bash
while true; do
  echo "$(date -u +%T) chrome_procs=$(pgrep -c -f ms-playwright) chrome_cpu=$(ps -eo pcpu,args | grep '[m]s-playwright' | awk '{s+=$1} END {print int(s+0)}') load1=$(cut -d' ' -f1 /proc/loadavg) memavail_MB=$(free -m | awk '/Mem:/{print $7}')"
  sleep 5
done
