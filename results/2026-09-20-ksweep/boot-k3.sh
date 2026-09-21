#!/usr/bin/env bash
# K-sweep arm k=3: NCCL baseline envs (identical to results/2026-09-20-nccl/boot-arm.sh)
# + single lever NUM_SPECULATIVE_TOKENS 5->3 (FORCE_UNSAFE_CTX=1 to pass the %5 guard)
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
exec ./serve.sh
