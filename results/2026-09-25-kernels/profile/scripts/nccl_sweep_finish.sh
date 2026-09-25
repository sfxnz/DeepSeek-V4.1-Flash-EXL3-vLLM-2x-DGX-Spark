#!/usr/bin/env bash
# Finish the NCCL sweep once spark1's GPU is exclusive: rep 3 of the 21 phase-1 arms, then phase 2.
# Each block (<= 7 arms, < 10 min) waits until nvidia-smi shows no other GPU process, then runs under the lock.
set -uo pipefail
W=/home/sfxnz/projects/ai-lab/recipes/.worktrees/kernels-r3
L=/tmp/claude-1000/-home-sfxnz-projects-ai-lab-recipes-DeepSeek-V4-1-Flash-EXL3-vLLM-2x-DGX-Spark/00916df6-3e19-4085-96e2-41192ed78f01/scratchpad/gpu-spark1.lock
OUT=/home/sfxnz/projects/data/dsv41-nccl-decode/phase1
gpu_clear() { [ "$(nvidia-smi | grep -cE ' (G|C) ')" -eq 0 ] && [ "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')" -le 1 ]; }
wait_clear() { local n=0; until gpu_clear; do n=$((n+1)); [ $((n % 12)) -eq 0 ] && echo "  waiting for exclusive GPU $(date -u +%T)"; sleep 5; done; }
run_block() {  # $1 arms, $2 rep, $3 port
  wait_clear
  echo "=== block rep $2: $1 ($(date -u +%T))"
  ARMS="$1" REPS=1 REP0=$2 OUT=$OUT NCCL_BENCH_PORT=$3 flock -w 7200 $L $W/tools/nccl_decode_sweep.sh 2>&1 | grep -E "^== rep|FAILED"
}
P1="keep bare proto_ll proto_simple proto_ll128 algo_ring algo_tree ch1 ch2 ch4 minch8 nt64 nt128 nt256 nt512 ib_inline proxy_big proxy_little ignore_cpu_aff graph_mixing0 gdr_c2c"
read -r -a A <<<"$P1"
port=29800
for c in 0 7 14; do port=$((port + 10)); run_block "${A[*]:$c:7}" 3 $port; done
run_block "algo_tree" 1 29850
port=29860
for rep in 4 5 6; do
  port=$((port + 10))
  arms="mix0_proxy_big mix0_bare mix0_so0 graph_mixing0 keep"
  [[ $rep == 5 ]] && arms="keep graph_mixing0 mix0_so0 mix0_bare mix0_proxy_big"
  run_block "$arms" $rep $port
done
echo "=== done $(date -u +%T)"
