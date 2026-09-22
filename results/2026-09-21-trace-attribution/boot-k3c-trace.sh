#!/usr/bin/env bash
# BOOT 1 — trace attribution: boot-k3c.sh + ONLY addition = torch profiler
# (mounts /start_profile /stop_profile; round-5/7/11 precedent).
# EXTRA_ARGS JSON contains no spaces so run.sh's unquoted $EXTRA_ARGS is safe
# (same pattern as COMPILATION_CONFIG).
set -euo pipefail
cd /home/sfxnz/projects/ai-lab/recipes/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark
export IMAGE=dsv41-flash-exl3-sm121:canonical-e12
export MAX_NUM_BATCHED_TOKENS=8192
export DSV41_DROP_PAGE_CACHE=1
export DSV41_INDEXER_PREFILL_FACTOR=1
export DSV41_PREFILL_EMPTY_CACHE_TOKENS=8192
export DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB=2.5
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256
export NCCL_BUFFSIZE=1048576
export NCCL_LL128_BUFFSIZE=262144
export NCCL_PROTO='^LL128'
export NCCL_MAX_NCHANNELS=8
export NUM_SPECULATIVE_TOKENS=3
export FORCE_UNSAFE_CTX=1
export COMPILATION_CONFIG='{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,3,4,6,8],"custom_ops":["all"]}'
export EXTRA_ARGS='--profiler-config {"profiler":"torch","torch_profiler_dir":"/tmp/dsv41-traces"}'
exec ./serve.sh
