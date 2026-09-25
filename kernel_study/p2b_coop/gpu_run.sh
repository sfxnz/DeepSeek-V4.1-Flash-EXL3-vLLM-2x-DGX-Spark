#!/usr/bin/env bash
# Run one GPU microbench step on spark2 (clean GPU) under the shared lock holder's shell.
# Invoked as:  ssh spark2 "flock -w 7200 <root>/.gpu.lock bash <dst>/kernel_study/p2b_coop/gpu_run.sh <dst> <tag> <cmd...>"
#   <dst>  the rsynced worktree on spark2 (mounted at /repo)
#   <tag>  log prefix under <dst>/results/2026-09-25-kernels/coop-moe/runs/
#   <cmd>  python3 arguments run inside dsv41-flash-exl3-sm121:canonical-e13 (cwd /repo)
# Before the run: nvidia-smi pmon -c 1 must show only the idle desktop (Xorg, gnome-shell, SM "-");
# a foreign process means wait 60 s and retry (20 tries), then give up (exit 3). During the run a
# 250 ms sampler records SM/mem clocks, power and temperature. PROFILE=1 runs ncu (the host's
# Nsight Compute mounted at /ncu) as root with SYS_ADMIN, <cmd> being ncu's arguments; otherwise
# python3 as uid 1000.
set -euo pipefail
dst=$1; tag=$2; shift 2
logdir=$dst/results/2026-09-25-kernels/coop-moe/runs
mkdir -p "$logdir"
log=$logdir/$tag.log
clk=$logdir/$tag.clocks.csv

foreign() {
    nvidia-smi pmon -c 1 | awk '!/^#/ && NF >= 9 {
        name = $NF; sm = $4
        if ((name == "Xorg" || name == "gnome-shell") && (sm == "-" || sm == "0")) next
        print
    }'
}

for try in $(seq 1 20); do
    f=$(foreign)
    [ -z "$f" ] && break
    echo "$(date -u +%FT%TZ) foreign GPU process, waiting (try $try): $f" | tee -a "$log"
    sleep 60
done
if [ -n "$(foreign)" ]; then
    echo "$(date -u +%FT%TZ) giving up: foreign GPU process still present" | tee -a "$log"
    exit 3
fi

{
    echo "# $(date -u +%FT%TZ) host $(hostname) tag $tag"
    echo "# pmon before:"; nvidia-smi pmon -c 1
    echo "# free -h:"; free -h
    echo "# cmd: $*"
} >> "$log"

nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu \
    --format=csv -lms 250 > "$clk" 2>&1 &
smi=$!
trap 'kill $smi 2>/dev/null || true' EXIT

hf=/home/sfxnz/.cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-EXL3
common=(--rm --gpus all --network none --memory 16g
        -v "$dst:/repo" -v "$hf:/hf:ro" -w /repo
        -e HOME=/tmp -e TORCH_CUDA_ARCH_LIST=12.1a
        -e TORCH_EXTENSIONS_DIR=/repo/kernel_study/p2b_coop/build/torch_ext
        )
if [ "${PROFILE:-0}" = 1 ]; then
    common+=(--cap-add SYS_ADMIN -v /opt/nvidia/nsight-compute/2025.3.1:/ncu:ro --entrypoint /ncu/ncu)
else
    common+=(--user 1000:1000 --entrypoint python3)
fi
set +e
# timeout: every locked GPU window stays under 10 minutes
timeout 590 docker run "${common[@]}" dsv41-flash-exl3-sm121:canonical-e13 "$@" 2>&1 \
    | grep -v -e '^dsv41-patch' -e '^dsv41-indexer' -e ' \[importing.py' -e ' \[interface.py' -e 'cpp_extension.py:' \
    | tee -a "$log"
rc=${PIPESTATUS[0]}
set -e
{
    echo "# $(date -u +%FT%TZ) rc $rc"
    echo "# pmon after:"; nvidia-smi pmon -c 1
} >> "$log"
exit "$rc"
