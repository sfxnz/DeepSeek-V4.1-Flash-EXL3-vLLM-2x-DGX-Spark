#!/usr/bin/env bash
# NCCL latency sweep at the exact decode collective sizes, SERVE DOWN, both nodes.
# Per arm and rep: one rank-0 container here and one rank-1 container on
# WORKER_HOST, each running tools/nccl_decode_sweep.py (vLLM PyNcclCommunicator,
# the serve's libnccl). nvidia-smi clocks/power are sampled on both nodes.
#
# Usage: tools/nccl_decode_sweep.sh                     (all arms, REPS=3)
#        ARMS="keep proto_ll" REPS=1 OUT=dir tools/nccl_decode_sweep.sh
# Reps loop outside arms, so drift spreads over every arm.
# Then: python3 tools/nccl_decode_sweep.py summarize $OUT --out $OUT/nccl-sweep.json
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${IMAGE:-dsv41-flash-exl3-sm121:canonical-e13}"
WORKER_HOST="${WORKER_HOST:-spark2}"
HEAD_IP="${HEAD_IP:-10.100.8.1}"
IFACE="${IFACE:-enp1s0f1np1}"
PORT0="${NCCL_BENCH_PORT:-29541}"
REPS="${REPS:-3}"
REP0="${REP0:-1}"
ARMS="${ARMS:-$(python3 "$ROOT/tools/nccl_decode_sweep.py" arms)}"
OUT="${OUT:-$HOME/projects/data/dsv41-nccl-decode/$(date +%Y%m%d-%H%M)}"  # raw logs are large; commit the summary
TIMEOUT_S="${TIMEOUT_S:-300}"
RDIR=/tmp/dsv41-nccl-decode

busy() { docker ps --format '{{.Names}}' | grep -E '^(dsv41-flash-exl3|dsv41-quant)' || true; }
wbusy() { ssh -o BatchMode=yes "$WORKER_HOST" "docker ps --format '{{.Names}}'" | grep -E '^(dsv41-flash-exl3|dsv41-quant)' || true; }
if [[ -n "$(busy)$(wbusy)" ]]; then
  echo "refusing: GPU holders up ($(busy) $(wbusy)). ./stop.sh first." >&2
  exit 1
fi

mkdir -p "$OUT"
ssh -o BatchMode=yes "$WORKER_HOST" "mkdir -p $RDIR/out"
scp -q "$ROOT/tools/nccl_decode_sweep.py" "$WORKER_HOST:$RDIR/"

docker_args() {  # $1 = arm env (KEY=VAL lines), $2 = tools dir, $3 = out dir, $4 = name
  local a=(--rm --name "$4" --gpus all --network host --ipc host --device /dev/infiniband
    --cap-add IPC_LOCK --ulimit memlock=-1:-1 --entrypoint python3
    --user "$(id -u):$(id -g)" -e HOME=/tmp
    -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages
    -e NCCL_SOCKET_IFNAME="$IFACE" -e GLOO_SOCKET_IFNAME="$IFACE"
    -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,TUNING
    -v "$2:/bench:ro" -v "$3:/out")
  local kv
  while IFS= read -r kv; do [[ -n "$kv" ]] && a+=(-e "$kv"); done <<<"$1"
  printf '%q ' "${a[@]}"
}

smi() { echo "nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader -lms 500"; }

n=0
for ((rep = REP0; rep < REP0 + REPS; rep++)); do
  for arm in $ARMS; do
    n=$((n + 1))
    port=$((PORT0 + n % 50))
    env_kv="$(python3 "$ROOT/tools/nccl_decode_sweep.py" arm-env "$arm")"
    d="$OUT/$arm"
    mkdir -p "$d"
    echo "== rep $rep arm $arm (port $port): $(tr '\n' ' ' <<<"$env_kv")"
    $(smi) >"$d/rep$rep.smi.spark1.csv" 2>/dev/null &
    smi0=$!
    ssh -o BatchMode=yes "$WORKER_HOST" "$(smi) > $RDIR/out/smi.csv 2>/dev/null & echo \$! > $RDIR/smi.pid" || true
    wcmd="docker run $(docker_args "$env_kv" "$RDIR" "$RDIR/out" nccl-decode-r1) $IMAGE -S /bench/nccl_decode_sweep.py run --rank 1 --master $HEAD_IP:$port --arm $arm --json /out/rep$rep.rank1.json"
    timeout "$TIMEOUT_S" ssh -o BatchMode=yes "$WORKER_HOST" "$wcmd" >"$d/rep$rep.rank1.log" 2>&1 &
    wpid=$!
    rc0=0
    # shellcheck disable=SC2046
    eval timeout "$TIMEOUT_S" docker run $(docker_args "$env_kv" "$ROOT/tools" "$d" nccl-decode-r0) "$IMAGE" \
      -S /bench/nccl_decode_sweep.py run --rank 0 --master "$HEAD_IP:$port" --arm "$arm" --json "/out/rep$rep.rank0.json" \
      >"$d/rep$rep.rank0.log" 2>&1 || rc0=$?
    rc1=0
    wait "$wpid" || rc1=$?
    kill "$smi0" 2>/dev/null || true
    ssh -o BatchMode=yes "$WORKER_HOST" "kill \$(cat $RDIR/smi.pid) 2>/dev/null; true"
    scp -q "$WORKER_HOST:$RDIR/out/smi.csv" "$d/rep$rep.smi.spark2.csv" || true
    scp -q "$WORKER_HOST:$RDIR/out/rep$rep.rank1.json" "$d/rep$rep.rank1.json" 2>/dev/null || true
    ssh -o BatchMode=yes "$WORKER_HOST" "rm -f $RDIR/out/rep$rep.rank1.json"
    if [[ $rc0 != 0 || $rc1 != 0 ]]; then
      echo "  FAILED rc0=$rc0 rc1=$rc1 (see $d/rep$rep.rank*.log)"
      docker rm -f nccl-decode-r0 >/dev/null 2>&1 || true
      ssh -o BatchMode=yes "$WORKER_HOST" "docker rm -f nccl-decode-r1 >/dev/null 2>&1; true"
    fi
    grep -m3 -E "Algo|NCCL version|GPU Direct RDMA" "$d/rep$rep.rank0.log" | sed 's/^/  /' || true
  done
done
echo "results in $OUT"
