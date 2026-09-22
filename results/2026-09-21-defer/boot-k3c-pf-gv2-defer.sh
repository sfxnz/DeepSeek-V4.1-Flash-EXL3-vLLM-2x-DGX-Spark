#!/usr/bin/env bash
# Round 24: single lever added to the live-best 31.37 config
# (boot-k3c-pf-gv2.sh = k3c + DSV41_ENGRAM_PREFETCH=1 + CENSUS=1 +
#  DSV41_ENGRAM_GATHER_V2=1):
#   DSV41_ENGRAM_DEFER=1  (off-thread next-step gather: persistent worker
#   predicts the next chunk (v3 rule), CPU-hashes (cpu-hash mirror),
#   gathers via v2 preadv into double-buffered pinned slots + side-stream
#   H2D; stage() consumes with a per-table DtoD at replay; ANY anomaly
#   self-disarms to the sync v2 path with one warning line).
# Everything else identical.
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
export DSV41_ENGRAM_PREFETCH=1
export DSV41_ENGRAM_CENSUS=1
export DSV41_ENGRAM_GATHER_V2=1
export DSV41_ENGRAM_DEFER=1
exec ./serve.sh
