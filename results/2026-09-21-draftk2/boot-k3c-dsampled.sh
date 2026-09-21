#!/usr/bin/env bash
# ARM A (draftk2 session): boot-k3c.sh + ONLY lever = SPEC_CONFIG override with
# draft_sample_method=probabilistic (sampled drafter at t=0.2). NOTE: the plan
# said "remove draft_sample_method", but vLLM's default is greedy — removing it
# is a provable no-op. The arm's intent (sampled drafter) = "probabilistic".
# Captures stay [1,3,4,6,8]; k=3.
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
export SPEC_CONFIG='{"method":"dspark","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}'
exec ./serve.sh
