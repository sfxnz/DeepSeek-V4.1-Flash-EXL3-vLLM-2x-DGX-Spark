#!/usr/bin/env bash
# usage: run_diag.sh TAG PORT "EXTRA_ENV KV ..." [DIAG_SPIN]
set -uo pipefail
D=/tmp/claude-1000/-home-sfxnz-projects-ai-lab-recipes-DeepSeek-V4-1-Flash-EXL3-vLLM-2x-DGX-Spark/00916df6-3e19-4085-96e2-41192ed78f01/scratchpad/k3prof/diag
TAG=$1; PORT=$2; EXTRA=${3:-}; SPIN=${4:-0}
IMAGE=dsv41-flash-exl3-sm121:canonical-e13
ENVS="-e NCCL_IB_HCA=rocep1s0f1 -e NCCL_CROSS_NIC=1 -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_BUFFSIZE=1048576 -e NCCL_LL128_BUFFSIZE=262144 -e NCCL_PROTO=^LL128 -e NCCL_MAX_NCHANNELS=8 -e NCCL_SOCKET_IFNAME=enp1s0f1np1 -e GLOO_SOCKET_IFNAME=enp1s0f1np1 -e DIAG_SPIN=$SPIN"
for kv in $EXTRA; do ENVS="$ENVS -e $kv"; done
COMMON="--rm --gpus all --network host --ipc host --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1 --entrypoint python3 --user 1000:1000 -e HOME=/tmp -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages"
ssh spark2 "mkdir -p /tmp/dsv41-diag" && scp -q $D/graph_diag.py spark2:/tmp/dsv41-diag/
timeout 200 ssh spark2 "docker run $COMMON $ENVS --name diag-r1 -v /tmp/dsv41-diag:/d:ro $IMAGE -S /d/graph_diag.py 1 10.100.8.1:$PORT $TAG" > $D/$TAG.r1.log 2>&1 &
timeout 200 docker run $COMMON $ENVS --name diag-r0 -v $D:/d:ro $IMAGE -S /d/graph_diag.py 0 10.100.8.1:$PORT $TAG 2>&1 | grep -E "^$TAG|Error|error" 
wait
grep -E "^$TAG" $D/$TAG.r1.log
