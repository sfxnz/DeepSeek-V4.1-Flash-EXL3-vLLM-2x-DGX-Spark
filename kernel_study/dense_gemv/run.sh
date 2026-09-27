#!/usr/bin/env bash
# Dense-gemv study runner.
#   run.sh build  <ext-python-snippet>   compile-only container on spark1 (no GPU)
#   run.sh gpu    <tag> <cmd...>         rsync to spark2, run <cmd> in the image on
#                                        spark2's GPU under the shared lock, rsync back
#   DOCKER_EXTRA='...' adds docker run args (e.g. the serve's patch-dir mounts).
#   SERVE_PATCH=1 mounts docker/patch like run.sh does (sitecustomize + /opt/dsv41-patch).
# GPU runs record `nvidia-smi pmon -c 1` before and after, refuse to start while a
# foreign compute process is on spark2's GPU (Xorg / gnome-shell with no SM use
# are the desktop and are allowed), and sample SM/mem clocks + power every 250 ms.
set -euo pipefail

WT="$(cd "$(dirname "$0")/../.." && pwd)"
NAME="k3-dense-gemv"
IMAGE="dsv41-flash-exl3-sm121:canonical-e13"
REMOTE_ROOT="/home/sfxnz/projects/ai-lab/recipes/.worktrees-spark2"
REMOTE="$REMOTE_ROOT/$NAME"
LOCK="$REMOTE_ROOT/.gpu.lock"
PACK="/home/sfxnz/.cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-EXL3/snapshots"
LOGDIR="$WT/results/2026-09-25-kernels/dense-gemv/runs"

mode="${1:?build|gpu}"
shift
if [[ "${SERVE_PATCH:-0}" == 1 ]]; then
  DOCKER_EXTRA="${DOCKER_EXTRA:-} -v $REMOTE/docker/patch:/opt/dsv41-patch:ro -v $REMOTE/docker/patch/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro"
fi

case "$mode" in
build)
  snippet="${1:?python snippet}"
  free -h | sed -n 2p
  docker run --rm --network none --memory 12g --cpus 6 \
    -v "$WT:/repo" -w /repo/kernel_study/dense_gemv \
    -e TORCH_CUDA_ARCH_LIST=12.1a -e MAX_JOBS=6 \
    --entrypoint bash "$IMAGE" -c \
    "python3 -c \"$snippet\" 2>&1 | grep -v -E 'dsv41-patch|vllm._C|interface.py|importing.py' ; rc=\${PIPESTATUS[0]}; chown -R $(id -u):$(id -g) /repo/kernel_study/dense_gemv/build; exit \$rc"
  ;;
gpu)
  tag="${1:?tag}"
  shift
  mkdir -p "$LOGDIR"
  log="$LOGDIR/$tag.log"
  ssh spark2 "mkdir -p $REMOTE"
  rsync -a --delete --exclude .git --exclude 'results/2026-09-25-kernels/dense-gemv/runs' \
    "$WT/" "spark2:$REMOTE/"
  cmd="$*"
  # shellcheck disable=SC2029
  ssh spark2 "bash -s" <<EOF 2>&1 | tee "$log"
set -uo pipefail
cd $REMOTE
exec 9>$LOCK
echo "waiting for $LOCK at \$(date -u +%FT%TZ)"
flock -w 7200 9 || { echo 'lock timeout'; exit 3; }
echo "lock acquired \$(date -u +%FT%TZ)"
for try in \$(seq 1 60); do
  pm=\$(nvidia-smi pmon -c 1)
  foreign=\$(echo "\$pm" | awk 'NR>2 && \$2 ~ /^[0-9]+$/ && \$10 != "Xorg" && \$10 != "gnome-shell" {print}')
  sm_busy=\$(echo "\$pm" | awk 'NR>2 && \$4 ~ /^[0-9]+$/ && \$4 > 0 {print}')
  if [ -z "\$foreign" ] && [ -z "\$sm_busy" ]; then break; fi
  echo "foreign GPU process present (try \$try), waiting 30 s:"; echo "\$pm"
  sleep 30
done
if [ -n "\$foreign\$sm_busy" ]; then echo 'GPU never became free; aborting'; exit 4; fi
echo "== pmon before"; echo "\$pm"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,utilization.gpu,temperature.gpu --format=csv -lms 250 > /tmp/$NAME-clocks.csv &
smi=\$!
timeout 600 docker run --rm --gpus all --network none --memory 16g ${DOCKER_EXTRA:-} \
  -v $REMOTE:/repo -v $PACK:/pack:ro -w /repo/kernel_study/dense_gemv \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  --entrypoint bash $IMAGE -c "$cmd 2>&1 | grep -v -E 'dsv41-patch|vllm._C|interface.py|importing.py'; rc=\\\${PIPESTATUS[0]}; chown -R $(id -u):$(id -g) /repo/kernel_study/dense_gemv /repo/results/2026-09-25-kernels/dense-gemv 2>/dev/null; exit \\\$rc"
rc=\$?
kill \$smi 2>/dev/null
echo "== pmon after"; nvidia-smi pmon -c 1
echo "== clocks (sm MHz, mem, W): samples, median sm, min sm, median W"
python3 - <<'PY'
import csv, statistics
rows = list(csv.reader(open('/tmp/$NAME-clocks.csv')))[1:]
sm = [float(r[1].split()[0]) for r in rows if r[1].strip().split()[0].replace('.','',1).isdigit()]
pw = [float(r[3].split()[0]) for r in rows if r[3].strip().split()[0].replace('.','',1).isdigit()]
busy = [s for s, r in zip(sm, rows) if int(r[4].split()[0]) > 0] if sm else []
mem = sorted({r[2].strip() for r in rows})
print(f"samples {len(rows)} busy {len(busy)} sm_median {statistics.median(busy or sm or [0]):.0f} "
      f"sm_min_busy {min(busy or [0]):.0f} mem {mem} power_median {statistics.median(pw or [0]):.1f} W")
PY
cp /tmp/$NAME-clocks.csv $REMOTE/kernel_study/dense_gemv/last-clocks.csv
echo "rc=\$rc"
exit \$rc
EOF
  rc=${PIPESTATUS[0]}
  rsync -a "spark2:$REMOTE/results/2026-09-25-kernels/dense-gemv/" "$WT/results/2026-09-25-kernels/dense-gemv/" --exclude runs
  rsync -a "spark2:$REMOTE/kernel_study/dense_gemv/last-clocks.csv" "$LOGDIR/$tag.clocks.csv" || true
  exit "$rc"
  ;;
*)
  echo "usage: run.sh build|gpu" >&2
  exit 2
  ;;
esac
