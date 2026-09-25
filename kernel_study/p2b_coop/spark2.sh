#!/usr/bin/env bash
# From spark1: sync this worktree to spark2, run gpu_run.sh under spark2's GPU lock, sync the
# coop-moe results back.   kernel_study/p2b_coop/spark2.sh <tag> <python3 args...>
# PROFILE=1 is passed through (ncu runs as root with SYS_ADMIN).
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
root=/home/sfxnz/projects/ai-lab/recipes/.worktrees-spark2
dst=$root/$(basename "$here")
res=results/2026-09-25-kernels/coop-moe
ssh spark2 "mkdir -p $dst"
rsync -a --delete --exclude .git --exclude "$res/runs/" "$here/" "spark2:$dst/"
set +e
ssh spark2 "PROFILE=${PROFILE:-0} flock -w 7200 $root/.gpu.lock bash $dst/kernel_study/p2b_coop/gpu_run.sh $dst $(printf '%q ' "$@")"
rc=$?
set -e
mkdir -p "$here/$res/runs"
rsync -a "spark2:$dst/$res/" "$here/$res/"
exit $rc
