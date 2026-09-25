#!/usr/bin/env bash
# Run one command in the serve image on spark2's GPU, under spark2's shared GPU lock.
#
#   kernel_study/mhc_det/spark2.sh <tag> <python3 args...>
#   SANITIZER=1 kernel_study/mhc_det/spark2.sh <tag> kernel_study/mhc_det/san.sh
#       (bash entrypoint; the host's compute-sanitizer mounted read-only)
#
# 1. rsync this worktree to spark2 (.worktrees-spark2/k3-mhc-det), no .git.
# 2. Under flock on .worktrees-spark2/.gpu.lock: wait until no foreign GPU process
#    is present (Xorg / gnome-shell are the idle desktop and are allowed), record
#    `nvidia-smi pmon -c 1` before and after, log clocks/power every 200 ms, run
#    `python3 <args>` in the image (network none, 16 GB, uid 1000, timeout 590 s). --init so
#    PID 1 forwards SIGTERM, -k 15 escalates to SIGKILL, and the named container is removed
#    before the lock is released: a hung job cannot hold the shared GPU.
# 3. rsync results/2026-09-25-kernels/mhc-det back (-u: never over a file edited here meanwhile).
# Logs land in results/2026-09-25-kernels/mhc-det/runs/<tag>/.
set -euo pipefail

TAG="$1"; shift
WT="$(cd "$(dirname "$0")/../.." && pwd)"
R=/home/sfxnz/projects/ai-lab/recipes/.worktrees-spark2
NAME=k3-mhc-det
IMAGE="${IMAGE:-dsv41-flash-exl3-sm121:canonical-e13}"
SNAPROOT=/home/sfxnz/.cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-EXL3
RUNDIR="results/2026-09-25-kernels/mhc-det/runs/$TAG"
CNAME="mhcdet-$(printf %s "$TAG" | tr -c 'A-Za-z0-9_.-' _)-$$"
ENTRY=python3
SANMOUNT=""
if [ "${SANITIZER:-0}" = 1 ]; then
  ENTRY=bash
  SANMOUNT="-v /usr/local/cuda-13.0/compute-sanitizer:/usr/local/cuda-13.0/compute-sanitizer:ro"
fi
mkdir -p "$WT/$RUNDIR"

ssh -o BatchMode=yes spark2 "mkdir -p $R/$NAME"
rsync -a --delete --exclude .git --exclude .run-state --exclude '__pycache__' "$WT/" "spark2:$R/$NAME/"

ARGS=$(printf '%q ' "$@")
ssh -o BatchMode=yes spark2 bash -s <<EOF
set -euo pipefail
cd $R/$NAME
mkdir -p $RUNDIR
exec 9>$R/.gpu.lock
flock -w 7200 9
foreign() { nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v '^\$' || true; nvidia-smi pmon -c 1 | awk '!/^#/ && \$NF!="Xorg" && \$NF!="gnome-shell"'; }
for i in \$(seq 1 60); do
  f=\$(foreign)
  [ -z "\$f" ] && break
  echo "foreign GPU process present, waiting (\$i): \$f" | tee -a $RUNDIR/gate.log
  sleep 10
done
f=\$(foreign)
if [ -n "\$f" ]; then echo "ABORT: foreign GPU process still present: \$f" | tee -a $RUNDIR/gate.log; exit 3; fi
{ date -u +%FT%TZ; nvidia-smi pmon -c 1; } > $RUNDIR/pmon_before.txt
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv -lms 200 > $RUNDIR/smi.csv &
SMI=\$!
set +e
timeout -k 15 590 docker run --init --rm --name $CNAME --gpus all --network none --memory 16g --user 1000:1000 \
  -e HOME=/tmp/h -e PYTHONDONTWRITEBYTECODE=1 $SANMOUNT \
  -v $R/$NAME:/repo -v $SNAPROOT:/snaproot:ro -w /repo \
  --entrypoint $ENTRY $IMAGE $ARGS > $RUNDIR/stdout.txt 2> $RUNDIR/stderr.txt
RC=\$?
docker rm -f $CNAME >/dev/null 2>&1 || true
set -e
kill \$SMI 2>/dev/null || true
{ date -u +%FT%TZ; nvidia-smi pmon -c 1; } > $RUNDIR/pmon_after.txt
echo "rc=\$RC" > $RUNDIR/rc.txt
EOF
# -u: a file edited here while the job ran is newer than its uploaded copy and is kept.
rsync -a -u "spark2:$R/$NAME/results/2026-09-25-kernels/mhc-det/" "$WT/results/2026-09-25-kernels/mhc-det/"
cat "$WT/$RUNDIR/rc.txt"
tail -n 60 "$WT/$RUNDIR/stdout.txt"
if [ -s "$WT/$RUNDIR/stderr.txt" ]; then echo "--- stderr (tail)"; tail -n 25 "$WT/$RUNDIR/stderr.txt"; fi
