#!/usr/bin/env bash
# exit when spark1's GPU shows no other users for 24 consecutive 5 s samples (2 min)
ok=0
while true; do
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  g=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  gp=$(nvidia-smi | grep -c " G  ")
  if [ "$u" -le 1 ] && [ "$gp" -eq 0 ]; then ok=$((ok+1)); else ok=0; fi
  echo "$(date -u +%T) util=$u graphics_procs=$gp compute_procs=$g ok=$ok"
  [ $ok -ge 24 ] && break
  sleep 5
done
echo GPUFREE
