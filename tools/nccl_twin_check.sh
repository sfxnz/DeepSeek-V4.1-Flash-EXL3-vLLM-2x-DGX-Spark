#!/usr/bin/env bash
# Two-node check of DSV41_NCCL_EAGER_TWIN on the serve's decode collective pattern.
# SERVE DOWN and both GPUs exclusive (no dsv41 serve, no requant, no foreign GPU process).
# Per arm and rep: rank 0 here, rank 1 on WORKER_HOST, each running tools/nccl_twin_check.py.
# Arms alternate inside each rep (ABAB); nvidia-smi pmon is saved before each run and
# clocks/power are sampled during it on both nodes.
#
# Usage: tools/nccl_twin_check.sh                        (keep twin, REPS=3, c=1 sizes)
#        ARMS="keep twin twin_so0" REPS=2 ROWS="--m 8 --md 6" tools/nccl_twin_check.sh   (c=2)
#        ARMS="keep keep_qos twin twin_qos" tools/nccl_twin_check.sh   (with the PM QoS arms)
#        CHECK=--device-check REPS=1 ARMS="keep twin" tools/nccl_twin_check.sh
#            (every pipelined step verified on the device; the phase-A1 correctness gate)
# Then:  python3 tools/nccl_twin_check.py compare $OUT
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${IMAGE:-dsv41-flash-exl3-sm121:canonical-e13}"
WORKER_HOST="${WORKER_HOST:-spark2}"
HEAD_IP="${HEAD_IP:-10.100.8.1}"
IFACE="${IFACE:-enp1s0f1np1}"
PORT0="${NCCL_TWIN_PORT:-29661}"
REPS="${REPS:-3}"
ARMS="${ARMS:-$(python3 "$ROOT/tools/nccl_twin_check.py" arms)}"
ROWS="${ROWS:---m 4 --md 3}"
STEPS="${STEPS:-300}"
CHECK="${CHECK:-}"  # extra run options for both ranks, e.g. --device-check
OUT="${OUT:-$HOME/projects/data/dsv41-nccl-twin/$(date +%Y%m%d-%H%M)}"
TIMEOUT_S="${TIMEOUT_S:-420}"
RDIR=/tmp/dsv41-nccl-twin

busy() { docker ps --format '{{.Names}}' | grep -E '^(dsv41-flash-exl3|dsv41-quant|dsv41-requant)' || true; }
wbusy() { ssh -o BatchMode=yes "$WORKER_HOST" "docker ps --format '{{.Names}}'" | grep -E '^(dsv41-flash-exl3|dsv41-quant|dsv41-requant)' || true; }
# Compute or graphics processes other than the desktop's Xorg/gnome-shell.
foreign() { nvidia-smi pmon -c 1 | awk 'NR>2 && $2 != "-" && $NF != "Xorg" && $NF != "gnome-shell" {print $NF}'; }
wforeign() { ssh -o BatchMode=yes "$WORKER_HOST" "nvidia-smi pmon -c 1" | awk 'NR>2 && $2 != "-" && $NF != "Xorg" && $NF != "gnome-shell" {print $NF}'; }
if [[ -n "$(busy)$(wbusy)" ]]; then
  echo "refusing: GPU holders up ($(busy) $(wbusy)). ./stop.sh / stop the requant first." >&2
  exit 1
fi

mkdir -p "$OUT"
ssh -o BatchMode=yes "$WORKER_HOST" "mkdir -p $RDIR/tools $RDIR/docker/patch $RDIR/out"
scp -q "$ROOT/tools/nccl_twin_check.py" "$WORKER_HOST:$RDIR/tools/"
scp -q "$ROOT/docker/patch/nccl_eager_twin.py" "$ROOT/docker/patch/pm_qos.py" "$WORKER_HOST:$RDIR/docker/patch/"

docker_args() {  # $1 = arm env (KEY=VAL lines), $2 = repo-like dir, $3 = out dir, $4 = name
  local who=(--user "$(id -u):$(id -g)")
  # PM QoS arms: root in the container opens /dev/cpu_dma_latency (host-wide while the run lasts)
  if grep -q '^DSV41_PM_QOS_US=' <<<"$1"; then who=(--device /dev/cpu_dma_latency); fi
  local a=(--rm --init --name "$4" --gpus all --network host --ipc host --device /dev/infiniband
    --cap-add IPC_LOCK --ulimit memlock=-1:-1 --entrypoint python3
    "${who[@]}" -e HOME=/tmp
    -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages
    -e NCCL_SOCKET_IFNAME="$IFACE" -e GLOO_SOCKET_IFNAME="$IFACE" -e NCCL_DEBUG=WARN
    -v "$2:/bench:ro" -v "$3:/out")
  local kv
  while IFS= read -r kv; do [[ -n "$kv" ]] && a+=(-e "$kv"); done <<<"$1"
  printf '%q ' "${a[@]}"
}

smi() { echo "nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader -lms 500"; }

n=0
for ((rep = 1; rep <= REPS; rep++)); do
  for arm in $ARMS; do
    n=$((n + 1))
    port=$((PORT0 + n % 50))
    env_kv="$(python3 "$ROOT/tools/nccl_twin_check.py" arm-env "$arm")"
    d="$OUT/$arm"
    mkdir -p "$d"
    until [[ -z "$(foreign)$(wforeign)" ]]; do
      echo "  foreign GPU process ($(foreign) $(wforeign)); waiting $(date -u +%T)"
      sleep 30
    done
    nvidia-smi pmon -c 1 >"$d/rep$rep.pmon.spark1.txt"
    ssh -o BatchMode=yes "$WORKER_HOST" "nvidia-smi pmon -c 1" >"$d/rep$rep.pmon.spark2.txt"
    echo "== rep $rep arm $arm (port $port)"
    $(smi) >"$d/rep$rep.smi.spark1.csv" 2>/dev/null &
    smi0=$!
    ssh -o BatchMode=yes "$WORKER_HOST" "$(smi) > $RDIR/out/smi.csv 2>/dev/null & echo \$! > $RDIR/smi.pid" || true
    wcmd="docker run $(docker_args "$env_kv" "$RDIR" "$RDIR/out" nccl-twin-r1) $IMAGE -S /bench/tools/nccl_twin_check.py run --rank 1 --master $HEAD_IP:$port --arm $arm --steps $STEPS $ROWS $CHECK --json /out/rep$rep.rank1.json"
    timeout -k 10 "$TIMEOUT_S" ssh -o BatchMode=yes "$WORKER_HOST" "$wcmd" >"$d/rep$rep.rank1.txt" 2>&1 &
    wpid=$!
    rc0=0
    # shellcheck disable=SC2046,SC2086
    eval timeout -k 10 "$TIMEOUT_S" docker run $(docker_args "$env_kv" "$ROOT" "$d" nccl-twin-r0) "$IMAGE" \
      -S /bench/tools/nccl_twin_check.py run --rank 0 --master "$HEAD_IP:$port" --arm "$arm" \
      --steps "$STEPS" $ROWS $CHECK --json "/out/rep$rep.rank0.json" >"$d/rep$rep.rank0.txt" 2>&1 || rc0=$?
    rc1=0
    wait "$wpid" || rc1=$?
    kill "$smi0" 2>/dev/null || true
    ssh -o BatchMode=yes "$WORKER_HOST" "kill \$(cat $RDIR/smi.pid) 2>/dev/null; true"
    scp -q "$WORKER_HOST:$RDIR/out/smi.csv" "$d/rep$rep.smi.spark2.csv" || true
    scp -q "$WORKER_HOST:$RDIR/out/rep$rep.rank1.json" "$d/rep$rep.rank1.json" 2>/dev/null || true
    ssh -o BatchMode=yes "$WORKER_HOST" "rm -f $RDIR/out/rep$rep.rank1.json"
    if [[ $rc0 != 0 || $rc1 != 0 ]]; then
      echo "  FAILED rc0=$rc0 rc1=$rc1 (see $d/rep$rep.rank*.txt; rc 3 = watchdog hang, 2 = mismatch)"
      docker rm -f nccl-twin-r0 >/dev/null 2>&1 || true
      ssh -o BatchMode=yes "$WORKER_HOST" "docker rm -f nccl-twin-r1 >/dev/null 2>&1; true"
    fi
    grep -h '"step_us"' "$d/rep$rep.rank0.txt" | sed 's/^/  /' || true
  done
done
echo "results in $OUT"
