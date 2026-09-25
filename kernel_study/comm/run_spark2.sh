#!/usr/bin/env bash
# Run one single-GPU microbench from this worktree on spark2 under its GPU lock.
#   kernel_study/comm/run_spark2.sh OUTDIR python3 kernel_study/comm/X.py --json OUTDIR/x.json
# OUTDIR is relative to the repo root. Rsyncs the worktree to spark2, runs
# gpu_run.sh there under the lock, copies OUTDIR back.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTDIR="$1"
shift
R=/home/sfxnz/projects/ai-lab/recipes/.worktrees-spark2
NAME="${NAME:-k3-comm}"
ssh -o BatchMode=yes spark2 "mkdir -p $R/$NAME"
rsync -a --delete --exclude .git "$ROOT/" "spark2:$R/$NAME/"
ssh -o BatchMode=yes spark2 "cd $R/$NAME && IMAGE=${IMAGE:-} AS_ROOT=${AS_ROOT:-0} DOCKER_EXTRA=$(printf '%q' "${DOCKER_EXTRA:-}") flock -w 7200 $R/.gpu.lock kernel_study/comm/gpu_run.sh $R/$NAME $OUTDIR $(printf '%q ' "$@")"
mkdir -p "$ROOT/$OUTDIR"
rsync -a "spark2:$R/$NAME/$OUTDIR/" "$ROOT/$OUTDIR/"
tail -3 "$ROOT/$OUTDIR/stdout.txt"
