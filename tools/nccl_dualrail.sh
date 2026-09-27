#!/usr/bin/env bash
# dual-rail-nccl step 1: two-node all_reduce sweep per NCCL arm, SERVE DOWN.
#
# Arms (every arm pins only ACTIVE f1 HCAs; the DOWN f0 ports hang init):
#   single             live config: rocep1s0f1, CROSS_NIC=1, KEEP AR-tail set
#   single_bare        rocep1s0f1, CROSS_NIC=1, no KEEP set
#   dual               rocep1s0f1,roceP2p1s0f1, CROSS_NIC=0, no KEEP set
#   dual_keep          dual + KEEP set
#   dual_keep_nomerge  dual_keep + NCCL_IB_MERGE_NICS=0
# CROSS_NIC=0 keeps rails matched: 10.100.8.0/24 and 10.100.9.0/24 are
# separate point-to-point links, so rail0<->rail1 cannot route.
# MERGE_NICS may be a no-op: the two HCAs sit in different PCI domains
# (0000:01:00.1 vs 0002:01:00.1). The INFO log shows what NCCL chose.
#
# Usage: tools/nccl_dualrail.sh            (all arms)
#        ARMS="single dual_keep" tools/nccl_dualrail.sh
# Then:  python3 tools/nccl_allreduce_sweep.py compare $OUT/single.json $OUT/dual_keep.json
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${IMAGE:-dsv41-flash-exl3-sm121:canonical-e12}"
WORKER_HOST="${WORKER_HOST:-spark2}"
HEAD_IP="${HEAD_IP:-10.100.8.1}"
IFACE="${IFACE:-enp1s0f1np1}"
PORT="${NCCL_BENCH_PORT:-29531}"
ARMS="${ARMS:-single single_bare dual dual_keep dual_keep_nomerge}"
OUT="${OUT:-$ROOT/results/2026-09-24-review/research/nccl-dualrail-$(date +%Y%m%d-%H%M)}"
RAIL0=rocep1s0f1
RAIL1=roceP2p1s0f1
KEEP="NCCL_BUFFSIZE=1048576 NCCL_LL128_BUFFSIZE=262144 NCCL_PROTO=^LL128 NCCL_MAX_NCHANNELS=8"

arm_env() {
  case "$1" in
    single) echo "NCCL_IB_HCA=$RAIL0 NCCL_CROSS_NIC=1 $KEEP" ;;
    single_bare) echo "NCCL_IB_HCA=$RAIL0 NCCL_CROSS_NIC=1" ;;
    dual) echo "NCCL_IB_HCA=$RAIL0,$RAIL1 NCCL_CROSS_NIC=0" ;;
    dual_keep) echo "NCCL_IB_HCA=$RAIL0,$RAIL1 NCCL_CROSS_NIC=0 $KEEP" ;;
    dual_keep_nomerge) echo "NCCL_IB_HCA=$RAIL0,$RAIL1 NCCL_CROSS_NIC=0 NCCL_IB_MERGE_NICS=0 $KEEP" ;;
    *) echo "unknown arm $1" >&2; return 1 ;;
  esac
}

# Exclusive GPUs: refuse while the serve or a pack conversion holds them.
busy() { docker ps --format '{{.Names}}' | grep -E '^(dsv41-flash-exl3|dsv41-quant)' || true; }
wbusy() { ssh -o BatchMode=yes "$WORKER_HOST" "docker ps --format '{{.Names}}'" | grep -E '^(dsv41-flash-exl3|dsv41-quant)' || true; }
if [[ -n "$(busy)$(wbusy)" ]]; then
  echo "refusing: GPU holders up ($(busy) $(wbusy)). ./stop.sh first." >&2
  exit 1
fi

# Both f1 rails must be ACTIVE on both nodes (read-only sysfs).
rails='for h in '"$RAIL0 $RAIL1"'; do printf "%s %s\n" $h "$(cat /sys/class/infiniband/$h/ports/1/state)"; done'
for where in local "$WORKER_HOST"; do
  if [[ $where == local ]]; then st="$(bash -c "$rails")"; else st="$(ssh -o BatchMode=yes "$WORKER_HOST" "$rails")"; fi
  if grep -qv ACTIVE <<<"$st"; then
    echo "refusing: rail not ACTIVE on $where: $st" >&2
    exit 1
  fi
done

mkdir -p "$OUT"
ssh -o BatchMode=yes "$WORKER_HOST" "mkdir -p /tmp/dsv41-nccl"
scp -q "$ROOT/tools/nccl_allreduce_sweep.py" "$WORKER_HOST:/tmp/dsv41-nccl/"

docker_args() {  # $1 = arm env, $2 = tools dir, $3 = out dir
  local a=(--rm --gpus all --network host --ipc host --device /dev/infiniband
    --cap-add IPC_LOCK --ulimit memlock=-1:-1 --entrypoint python3
    --user "$(id -u):$(id -g)" -e HOME=/tmp
    -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages
    -e NCCL_SOCKET_IFNAME="$IFACE" -e NCCL_NET=IB -e NCCL_IB_DISABLE=0
    -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_ENABLE=0
    -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
    -v "$2:/bench:ro" -v "$3:/out")
  local kv
  for kv in $1; do a+=(-e "$kv"); done
  printf '%q ' "${a[@]}"
}

for arm in $ARMS; do
  env_kv="$(arm_env "$arm")"
  echo "== $arm: $env_kv"
  wcmd="docker run $(docker_args "$env_kv" /tmp/dsv41-nccl /tmp/dsv41-nccl) $IMAGE -S /bench/nccl_allreduce_sweep.py run --rank 1 --master $HEAD_IP:$PORT"
  timeout 900 ssh -o BatchMode=yes "$WORKER_HOST" "$wcmd" >"$OUT/$arm.rank1.log" 2>&1 &
  wpid=$!
  # shellcheck disable=SC2046
  eval timeout 900 docker run $(docker_args "$env_kv" "$ROOT/tools" "$OUT") "$IMAGE" \
    -S /bench/nccl_allreduce_sweep.py run --rank 0 --master "$HEAD_IP:$PORT" --json "/out/$arm.json" \
    >"$OUT/$arm.rank0.log" 2>&1 || echo "  rank0 failed (see $OUT/$arm.rank0.log)"
  wait "$wpid" || echo "  rank1 failed (see $OUT/$arm.rank1.log)"
  grep -m2 -E "NET/IB : Using|NCCL_IB_MERGE_NICS" "$OUT/$arm.rank0.log" | sed 's/^/  /' || true
  grep -cE "via NET/IB/[0-9]" "$OUT/$arm.rank0.log" | sed 's/^/  channels via NET\/IB: /' || true
done

echo "results in $OUT"
for arm in $ARMS; do
  [[ $arm == single || ! -f "$OUT/$arm.json" || ! -f "$OUT/single.json" ]] && continue
  echo "== gate: $arm vs single"
  python3 "$ROOT/tools/nccl_allreduce_sweep.py" compare "$OUT/single.json" "$OUT/$arm.json" || true
done
