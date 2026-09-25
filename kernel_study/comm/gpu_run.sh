#!/usr/bin/env bash
# Executed on the GPU host under its lock (see run_spark2.sh):
#   gpu_run.sh REPO_DIR OUTDIR CMD...
# Waits for a GPU with no process besides Xorg/gnome-shell, saves pmon before/after and
# nvidia-smi clocks/power during the run, runs CMD in the image with the patch dir
# mounted the way run.sh mounts it. AS_ROOT=1 runs the container as root (default: the
# caller's uid); DOCKER_EXTRA adds docker run arguments (e.g. --device ...).
# Hard wall-clock limit GPU_RUN_TIMEOUT_S (default 590: locked runs stay under 10 min): the
# docker client runs under `timeout -k 10`, the container is --rm --init and named, and is
# removed if it outlives its client, so a hung run (e.g. one rank of a 2-rank NCCL test
# spinning in an all-reduce after its peer died) cannot keep the GPU lock.
set -uo pipefail
REPO="$1"
OUTDIR="$2"
shift 2
IMAGE="${IMAGE:-dsv41-flash-exl3-sm121:canonical-e13}"
cd "$REPO"
mkdir -p "$OUTDIR"
foreign() { nvidia-smi pmon -c 1 | awk 'NR>2 && $2 != "-" && $NF != "Xorg" && $NF != "gnome-shell" {print $NF}'; }
until [[ -z "$(foreign)" ]]; do
  echo "foreign GPU process: $(foreign); waiting $(date -u +%T)"
  sleep 30
done
nvidia-smi pmon -c 1 >"$OUTDIR/pmon_before.txt"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu \
  --format=csv,noheader -lms 500 >"$OUTDIR/smi.csv" 2>/dev/null &
smi=$!
rc=0
user_args=(--user "$(id -u):$(id -g)")
[[ "${AS_ROOT:-0}" == "1" ]] && user_args=()
read -ra extra_args <<<"${DOCKER_EXTRA:-}"
cname="gpurun-$(date +%s)-$$"
timeout -k 10 "${GPU_RUN_TIMEOUT_S:-590}" \
docker run --rm --init --name "$cname" --gpus all --network none --memory 16g --ipc host \
  "${user_args[@]}" "${extra_args[@]}" -e HOME=/tmp -e TORCH_CUDA_ARCH_LIST=12.1a \
  -v "$REPO:/repo" -w /repo \
  -v "$REPO/docker/patch:/opt/dsv41-patch:ro" \
  -v "$REPO/docker/patch/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro" \
  --entrypoint "" "$IMAGE" "$@" >"$OUTDIR/stdout.txt" 2>&1 || rc=$?
if docker ps -aq --filter "name=^${cname}\$" | grep -q .; then  # outlived its client (timeout)
  echo "gpu_run: $cname still up after the client exited (rc=$rc); removing it" >>"$OUTDIR/stdout.txt"
  docker rm -f "$cname" >/dev/null 2>&1 || true
fi
kill "$smi" 2>/dev/null || true
if [[ "${AS_ROOT:-0}" == "1" ]]; then  # hand root-written outputs back to the caller
  docker run --rm --network none -v "$REPO/$OUTDIR:/o" --entrypoint chown "$IMAGE" -R "$(id -u):$(id -g)" /o
fi
echo "rc=$rc" >>"$OUTDIR/stdout.txt"
nvidia-smi pmon -c 1 >"$OUTDIR/pmon_after.txt"
exit "$rc"
