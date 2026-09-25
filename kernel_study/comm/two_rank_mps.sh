#!/usr/bin/env bash
# Two NCCL ranks on the one GB10 of this host, inside one container (the review's recipe):
# an MPS daemon with a private pipe dir so both ranks' kernels run concurrently, and a
# different NCCL_HOSTID per rank so NCCL treats them as two hosts and connects them through
# the net transport (NET/IB, RoCE loopback on the serve's HCA; GDR off, host-memory LL
# buffers, proxy threads), with the serve's NCCL env (run.sh). Needs the container to have
# --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1 (DOCKER_EXTRA of
# gpu_run.sh / run_spark2.sh).
#   two_rank_mps.sh OUTDIR SCRIPT ARGS...     ({R} in ARGS becomes the rank; --rank R is appended)
# EXTRA_ENV="K=V ..." adds env to both ranks (e.g. DSV41_PM_QOS_US=20, which needs root and
# --device /dev/cpu_dma_latency); NCCL_DEBUG / NCCL_DEBUG_SUBSYS pass through; PYFLAGS="-S"
# skips site (then PYTHONPATH must name dist-packages, see tools/nccl_twin_check.py).
set -u
OUT="$1"
shift
mkdir -p "$OUT"
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d && echo "MPS daemon started"
export NCCL_IB_HCA=rocep1s0f1 NCCL_CROSS_NIC=1 NCCL_NET=IB NCCL_IB_DISABLE=0 NCCL_NVLS_ENABLE=0 \
  NCCL_CUMEM_ENABLE=0 NCCL_BUFFSIZE=1048576 NCCL_LL128_BUFFSIZE=262144 NCCL_PROTO=^LL128 \
  NCCL_MAX_NCHANNELS=8 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
for kv in ${EXTRA_ENV:-}; do export "${kv?}"; done
a0=("${@//\{R\}/0}")
a1=("${@//\{R\}/1}")
read -ra pyf <<<"${PYFLAGS:-}"
# Hard stop per rank (RANK_TIMEOUT_S, default 540): a rank whose peer died spins in an NCCL
# kernel forever and would hold the GPU lock; the scripts' own watchdogs normally fire first.
to=(timeout -k 10 "${RANK_TIMEOUT_S:-540}")
NCCL_HOSTID=k3host0 "${to[@]}" python3 "${pyf[@]}" "${a0[@]}" --rank 0 >"$OUT/rank0.txt" 2>&1 &
p0=$!
NCCL_HOSTID=k3host1 "${to[@]}" python3 "${pyf[@]}" "${a1[@]}" --rank 1 >"$OUT/rank1.txt" 2>&1 &
p1=$!
wait $p0
rc0=$?
wait $p1
rc1=$?
echo quit | nvidia-cuda-mps-control
echo "rc0=$rc0 rc1=$rc1"
grep -h "NET/IB\|GPU Direct\|via NET\|Connected all" "$OUT"/rank*.txt | sort | uniq -c | head -20
[[ $rc0 == 0 && $rc1 == 0 ]]
