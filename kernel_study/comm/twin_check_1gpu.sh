#!/usr/bin/env bash
# tools/nccl_twin_check.py with its two ranks on this one GPU (two_rank_mps.sh: MPS, NCCL_HOSTID
# per rank, NET/IB RoCE loopback, serve NCCL env): the real network transport (proxies,
# host-memory LL buffers, NIC DMA, graph host nodes) without the second node. Runs inside the
# image as root (PM QoS arms open /dev/cpu_dma_latency), e.g. from the repo root:
#   AS_ROOT=1 DOCKER_EXTRA="--device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
#     --device /dev/cpu_dma_latency" kernel_study/comm/run_spark2.sh OUT \
#     env OUT=OUT REPS=3 ARMS="keep_qos twin_qos" bash kernel_study/comm/twin_check_1gpu.sh
# Arms alternate inside each rep; one process pair per arm and rep (NCCL reads the mixing mode
# at communicator init); then `nccl_twin_check.py compare OUT`. RUN_ARGS adds run options to
# both ranks (e.g. "--device-check" to verify every pipelined step, "--inject-fault 123").
set -u
OUT="${OUT:?}"
REPS="${REPS:-3}"
ARMS="${ARMS:-keep twin keep_qos twin_qos}"
STEPS="${STEPS:-300}"
port="${PORT0:-29840}"
for ((rep = 1; rep <= REPS; rep++)); do
  for arm in $ARMS; do
    port=$((port + 1))
    env_kv=""
    [[ $arm == twin* ]] && env_kv="NCCL_GRAPH_MIXING_SUPPORT=0"
    [[ $arm == *_qos ]] && env_kv="$env_kv DSV41_PM_QOS_US=20"
    mkdir -p "$OUT/$arm"
    echo "== rep $rep arm $arm env [$env_kv] $(date -u +%T)"
    PYFLAGS=-S EXTRA_ENV="PYTHONPATH=/usr/local/lib/python3.12/dist-packages $env_kv" \
      bash kernel_study/comm/two_rank_mps.sh "$OUT/$arm/log$rep" tools/nccl_twin_check.py run \
      --master "127.0.0.1:$port" --arm "$arm" --steps "$STEPS" ${ROWS:-} ${RUN_ARGS:-} --json "$OUT/$arm/rep$rep.rank{R}.json"
    echo "   rc=$? $(grep -h '"step_us"' -A1 "$OUT/$arm/rep$rep.rank0.json" 2>/dev/null | tr -d '\n ' | cut -c1-80)"
  done
done
python3 -S tools/nccl_twin_check.py compare "$OUT"
