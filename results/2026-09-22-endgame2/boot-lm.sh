#!/usr/bin/env bash
# PHASE F FALLBACK BOOT (Round 29): stock pack + lm_head MXFP8.
# Copy of results/2026-09-21-gatherv2/boot-k3c-pf-gv2.sh (round-23 live-best
# k3c + PREFETCH + CENSUS + GATHER_V2, baseline 31.37 tok/s) with:
#   SNAPSHOT_SHA=2.0bpw-mcg-lmhead-mxfp8  (stock pack copy; ONLY model-00043
#                                          rewritten: head.weight ->
#                                          lm_head.weight mxfp8 + e8m0 scale)
#   DSV41_LMHEAD_MXFP8=1                  (sitecustomize swaps the vocab head
#                                          onto the b12x dense mxfp8 path)
# lm_head is G8-independent. No DSV41_LOAD_PF_G8 (loader stock).
set -euo pipefail
cd /home/sfxnz/projects/ai-lab/recipes/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark
export IMAGE=dsv41-flash-exl3-sm121:canonical-e12
export SNAPSHOT_SHA=2.0bpw-mcg-lmhead-mxfp8
export DSV41_LMHEAD_MXFP8=1
LMPACK="$HOME/.cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-EXL3/snapshots/$SNAPSHOT_SHA"
[[ -f "$LMPACK/model-00043-of-00048.safetensors" ]] || { echo "FATAL: lm pack missing" >&2; exit 1; }
python3 -c "import json;c=json.load(open('$LMPACK/model.safetensors.index.json'));assert 'lm_head.weight' in c['weight_map'], 'no lm_head in index'" \
  || { echo "FATAL: index lacks lm_head.weight" >&2; exit 1; }

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
exec ./serve.sh
