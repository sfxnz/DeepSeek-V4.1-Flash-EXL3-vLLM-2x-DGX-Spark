#!/usr/bin/env bash
# PF-G8 BOOT (STAGED — DO NOT RUN until BOOT-CHAIN-AUDIT.md pre-boot
# checklist is green). Copy of results/2026-09-21-gatherv2/boot-k3c-pf-gv2.sh
# (round-23 live-best k3c + PREFETCH + CENSUS + GATHER_V2) with the G8 levers:
#   SNAPSHOT_SHA=2.0bpw-mcg-g8   (G8-folded pack, results/2026-09-21-pfg8/)
#   DSV41_LOAD_PF_G8=1           (loader TP narrow-dim swap + wiring)
#
# ============================================================================
# PRE-BOOT CHECKLIST — verify EVERY item before running this script.
# ============================================================================
# (0) PACK COMPLETE: both rebuild ranks done (WATCH.md verify block):
#       spark1:  ls .../snapshots/2.0bpw-mcg-g8/model-000{03..22}-of-00048.safetensors | wc -l  == 20
#       spark2:  ssh spark2 'ls .../model-000{23..42}-of-00048.safetensors | wc -l'            == 20
#     then assemble + verify:
#       CODEBOOK=mcg REV=2.0bpw-mcg-g8 DST=<pack> bash tools/assemble_pack.sh
#       python3 tools/permute_pack_group_major.py <pack>          # dry-run verifier
#     (shard-3 spot verification already PASS: results/2026-09-21-pfg8/shard3-verify.md,
#      folded shapes w1/w3 (18,320,256), w2 (40,144,256), reconstruct parity.)
#
# (1) IMAGE KERNELS G8-AWARE — THE GATING ITEM (MISSING as of 2026-09-21,
#     audit: BOOT-CHAIN-AUDIT.md risk R1). canonical-e12 contains NO G8-aware
#     kernels: prefill exl3_gemm B-load remap + p2b decode G8 reader exist
#     only as harness twins (kernel_study/gemv_bench/build_prefill/
#     exl3_gemm_inner_pf.cuh, bench5.cu DEC5). An image rebuild dispatch must
#     port them into the exllamav3_ext + vllm_exl3_c builds with a runtime
#     pf_g8 flag. Until then this script's guard below ABORTS — do not
#     bypass: a G8 pack on stock kernels silently misreads trellis (garbage).
#     Post-rebuild check (guard runs it automatically):
#       docker run --rm --network none --entrypoint bash <IMAGE> -c \
#         'strings /usr/local/lib/python3.12/dist-packages/exllamav3_ext.*.so \
#                 /usr/local/lib/python3.12/dist-packages/vllm_exl3_c.*.so | grep -c pf_g8'
#       must be >= 1 (the __constant__ int pf_g8 symbol survives compilation).
#
# (2) LOADER PATCH EXPECT LINES — at container start (both ranks), the boot
#     log MUST show one of (sitecustomize wiring, audit §wiring):
#         dsv41: pfg8 loader re-index installed
#         dsv41: pfg8 loader re-index already present
#     If instead you see "FATAL pfg8 loader re-index failed: ..." the image's
#     vllm_exl3/exl3.py no longer matches the patch anchors — the boot
#     self-aborts. A SILENT log (neither line) means sitecustomize did not
#     run: DO NOT let it serve.
#
# (3) ENV CHECK (after containers are up, both ranks):
#       docker inspect dsv41-flash-exl3 --format '{{.Config.Env}}' | tr ' ' '\n' \
#         | grep -E 'DSV41_LOAD_PF_G8|SNAPSHOT'   # not present by name — check:
#       docker inspect dsv41-flash-exl3 | grep -o 'DSV41_LOAD_PF_G8=[0-9]'   == 1
#     and the model path must resolve to snapshots/2.0bpw-mcg-g8.
#
# (4) FIRST-SHARD WEIGHT-LOAD SANITY — during load, docker logs -f must show
#     the EXL3 weight load completing with NO "EXL3 TP shard: dim ... not
#     divisible" errors and no "EXL3 weight load shape mismatch" (the
#     narrow-dim swap keeps 18 groups -> 9/rank and 144 -> 72/rank whole).
#     An 'dim 0/1 size ... not divisible by tp=2' ValueError = patch not
#     engaged (stock dims on a G8 pack): stop, do not smoke.
#
# (5) SMOKE GATE (only after (2)+(4) clean, API /health green):
#       python3 smoke_chat.py  (or curl /v1/chat/completions)
#     Prompt: '17 * 19 = ? Step by step, then answer.'  -> must contain 323.
#     G8 misreads produce fluent-but-wrong output or garbage: ANY deviation
#     from 323 = kill the boot, restore stock (rollback below).
#
# (6) ROLLBACK (stock pack untouched — verified 48/48 shards, mtime
#     2026-09-10, snapshots/2.0bpw-mcg):
#       ./stop.sh
#       bash results/2026-09-21-gatherv2/boot-k3c-pf-gv2.sh   # SNAPSHOT_SHA
#       # defaults to 2.0bpw-mcg, DSV41_LOAD_PF_G8 unset -> loader dormant
# ============================================================================
set -euo pipefail
cd /home/sfxnz/projects/ai-lab/recipes/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark

export IMAGE=dsv41-flash-exl3-sm121:canonical-e12
export SNAPSHOT_SHA=2.0bpw-mcg-g8
export DSV41_LOAD_PF_G8=1

# --- GUARD: refuse to serve a G8 pack on non-G8 kernels (checklist item 1) ---
kernel_g8_marks=$(docker run --rm --network none --entrypoint bash "$IMAGE" -c \
  'strings /usr/local/lib/python3.12/dist-packages/exllamav3_ext.*.so \
          /usr/local/lib/python3.12/dist-packages/vllm_exl3_c.*.so 2>/dev/null \
   | grep -c pf_g8 || true' 2>/dev/null || echo 0)
if [[ "$kernel_g8_marks" -lt 1 ]]; then
  echo "FATAL: image $IMAGE has no pf_g8 symbols in exllamav3_ext/vllm_exl3_c —" >&2
  echo "G8 kernels not built (BOOT-CHAIN-AUDIT.md R1). A G8 pack on stock kernels" >&2
  echo "misreads trellis (garbage). Aborting before serve." >&2
  exit 1
fi

# --- GUARD: pack must exist and have all 48 shards (checklist item 0) ---
G8PACK="$HOME/.cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-EXL3/snapshots/$SNAPSHOT_SHA"
if [[ ! -d "$G8PACK" ]]; then
  echo "FATAL: G8 pack missing: $G8PACK (assemble first — WATCH.md)" >&2
  exit 1
fi
n_shards=$(ls "$G8PACK"/model-*-of-00048.safetensors 2>/dev/null | wc -l)
if [[ "$n_shards" -ne 48 ]]; then
  echo "FATAL: $G8PACK has $n_shards/48 shards — run tools/assemble_pack.sh first" >&2
  exit 1
fi

# --- round-23 live-best config, verbatim from boot-k3c-pf-gv2.sh ---
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
