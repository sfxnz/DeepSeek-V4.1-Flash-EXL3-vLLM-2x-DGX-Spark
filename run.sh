#!/usr/bin/env bash
# DeepSeek-V4.1-Flash EXL3 on 2x DGX Spark (GB10) — vLLM TP=2
set -euo pipefail

# BEGIN generated from recipe.yaml — edit recipe.yaml and run kit/render.py
MODEL="${MODEL:-sfxnz/DeepSeek-V4.1-Flash-EXL3}"
SERVED_NAME="${SERVED_NAME:-deepseek-ai/DeepSeek-V4.1-Flash}"
IMAGE="${IMAGE:-dsv41-flash-exl3-sm121}"
CONTAINER_NAME="${CONTAINER_NAME:-dsv41-flash-exl3}"
PORT="${PORT:-8000}"
MASTER_PORT="${MASTER_PORT:-29524}"
HEAD_IP="${HEAD_IP:-10.100.8.1}"
WORKER_HOST="${WORKER_HOST:-spark2}"
IFACE="${IFACE:-enp1s0f1np1}"
HCA="${HCA:-rocep1s0f1}"
TP="${TP:-2}"
NNODES="${NNODES:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
UTIL="${UTIL:-0.75}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
KV_CACHE_MEMORY="${KV_CACHE_MEMORY:-4294967296}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-5}"
SPEC="${SPEC:-dspark}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"
QUANTIZATION="${QUANTIZATION:-exl3}"
DSV41_ENGRAM_DISK="${DSV41_ENGRAM_DISK:-1}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
FORCE_UNSAFE_CTX="${FORCE_UNSAFE_CTX:-0}"
FORCE_UNSAFE_ENGRAM="${FORCE_UNSAFE_ENGRAM:-0}"
FORCE_UNSAFE_QUANT="${FORCE_UNSAFE_QUANT:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
SNAPSHOT_SHA="${SNAPSHOT_SHA:-2.0bpw-mcg}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
ORCHESTRATE="${ORCHESTRATE:-auto}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
# END generated
HF_HOME_IN_CONTAINER="/cache/huggingface"

# Hub download writes a commit hash in refs/<rev>. Files live in snapshots/<commit>/.
# Assembled packs live at snapshots/<rev>/ with no refs file. Prefer refs when both exist.
resolve_snapshot() {
  local hub refs commit named
  hub="${HF_CACHE}/hub/models--${MODEL//\//--}"
  refs="${hub}/refs/${SNAPSHOT_SHA}"
  named="${hub}/snapshots/${SNAPSHOT_SHA}"
  if [[ -f "$refs" ]]; then
    commit="$(tr -d '[:space:]' <"$refs")"
    if [[ -n "$commit" && -d "${hub}/snapshots/${commit}" ]]; then
      printf '%s\n' "${hub}/snapshots/${commit}"
      return 0
    fi
  fi
  if [[ -d "$named" ]]; then
    printf '%s\n' "$named"
    return 0
  fi
  return 1
}

snapshot_in_container() {
  local host="$1"
  if [[ "$host" != "$HF_CACHE"/* ]]; then
    echo "snapshot $host is not under HF_CACHE=$HF_CACHE" >&2
    return 1
  fi
  printf '%s%s\n' "$HF_HOME_IN_CONTAINER" "${host#"$HF_CACHE"}"
}

if [[ "${RESOLVE_SNAPSHOT_ONLY:-0}" == "1" ]]; then
  if host="$(resolve_snapshot)"; then
    printf '%s\n' "$host"
    exit 0
  fi
  echo "snapshot missing for $MODEL revision $SNAPSHOT_SHA under $HF_CACHE" >&2
  exit 1
fi

if [[ -z "${SPEC_CONFIG:-}" ]]; then
  case "$SPEC" in
    dspark)
      SPEC_CONFIG='{"method":"dspark","num_speculative_tokens":'"$NUM_SPECULATIVE_TOKENS"',"draft_sample_method":"probabilistic"}'
      ;;
    none)
      SPEC_CONFIG=""
      ;;
    *)
      echo "Unknown SPEC=$SPEC (want dspark or none)" >&2
      exit 1
      ;;
  esac
fi

if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
  COMPILATION_CONFIG='{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["all"]}'
fi

if [[ "$QUANTIZATION" != exl3 && "$FORCE_UNSAFE_QUANT" != 1 ]]; then
  echo "QUANTIZATION=$QUANTIZATION. Native MXFP4 experts plus Engram do not fit 2x Spark UMA. This recipe serves an EXL3 pack. FORCE_UNSAFE_QUANT=1 overrides." >&2
  exit 1
fi
if [[ "$DSV41_ENGRAM_DISK" != 1 && "$FORCE_UNSAFE_ENGRAM" != 1 ]]; then
  echo "DSV41_ENGRAM_DISK=$DSV41_ENGRAM_DISK. Pinning Engram (~189 GiB) in UMA OOMs a Spark. FORCE_UNSAFE_ENGRAM=1 overrides." >&2
  exit 1
fi
if [[ "$MAX_MODEL_LEN" -gt 1048576 && "$FORCE_UNSAFE_CTX" != 1 ]]; then
  echo "fp8 KV pin cannot hold --max-model-len $MAX_MODEL_LEN. Native window is 1048576. FORCE_UNSAFE_CTX=1 overrides." >&2
  exit 1
fi
if [[ "$MAX_NUM_SEQS" -gt 2 && "$FORCE_UNSAFE_CTX" != 1 ]]; then
  echo "MAX_NUM_SEQS=$MAX_NUM_SEQS exceeds 2 on this pin. FORCE_UNSAFE_CTX=1 overrides." >&2
  exit 1
fi
if [[ "$KV_CACHE_MEMORY" -gt 8589934592 && "$FORCE_UNSAFE_CTX" != 1 ]]; then
  echo "KV_CACHE_MEMORY=$KV_CACHE_MEMORY exceeds 8 GiB pin 8589934592. CSA2 is 890 B/token; 4 GiB holds 1M x 2. FORCE_UNSAFE_CTX=1 overrides." >&2
  exit 1
fi
if [[ "$SPEC" == dspark && $((NUM_SPECULATIVE_TOKENS % 5)) -ne 0 && "$FORCE_UNSAFE_CTX" != 1 ]]; then
  echo "NUM_SPECULATIVE_TOKENS=$NUM_SPECULATIVE_TOKENS is not divisible by 5 (DSpark block size). FORCE_UNSAFE_CTX=1 overrides." >&2
  exit 1
fi

if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
  printf '==> validate-only spec=%s seqs=%s spec_tokens=%s quant=%s engram_disk=%s eager=%s image=%s\n' \
    "$SPEC" "$MAX_NUM_SEQS" "$NUM_SPECULATIVE_TOKENS" "$QUANTIZATION" "$DSV41_ENGRAM_DISK" "$ENFORCE_EAGER" "$IMAGE"
  exit 0
fi
