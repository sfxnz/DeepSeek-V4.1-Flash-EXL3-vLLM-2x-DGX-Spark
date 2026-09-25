#!/usr/bin/env bash
# DeepSeek-V4.1-Flash EXL3 on 2x DGX Spark (GB10) — vLLM TP=2
set -euo pipefail

# BEGIN generated from recipe.yaml — edit recipe.yaml and run kit/render.py
MODEL="${MODEL:-sfxnz/DeepSeek-V4.1-Flash-EXL3}"
SERVED_NAME="${SERVED_NAME:-deepseek-ai/DeepSeek-V4.1-Flash}"
IMAGE="${IMAGE:-dsv41-flash-exl3-sm121:canonical-e13}"
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
KV_CACHE_MEMORY="${KV_CACHE_MEMORY:-8589934592}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
SPEC="${SPEC:-dspark}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"
QUANTIZATION="${QUANTIZATION:-exl3}"
DSV41_ENGRAM_DISK="${DSV41_ENGRAM_DISK:-1}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-0}"
MM_ENCODER_TP_MODE="${MM_ENCODER_TP_MODE:-data}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
FORCE_UNSAFE_CTX="${FORCE_UNSAFE_CTX:-0}"
FORCE_UNSAFE_ENGRAM="${FORCE_UNSAFE_ENGRAM:-0}"
FORCE_UNSAFE_QUANT="${FORCE_UNSAFE_QUANT:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
DSV41_ALLOW_CUDA_GRAPHS="${DSV41_ALLOW_CUDA_GRAPHS:-1}"
VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-1}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
SNAPSHOT_SHA="${SNAPSHOT_SHA:-2.0bpw-mcg-lmhead-mxfp8}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
ORCHESTRATE="${ORCHESTRATE:-auto}"
AUDIT="${AUDIT:-warn}"
WARMUP="${WARMUP:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DSV41_ENGRAM_PREFETCH="${DSV41_ENGRAM_PREFETCH:-1}"
DSV41_ENGRAM_CENSUS="${DSV41_ENGRAM_CENSUS:-1}"
DSV41_ENGRAM_GATHER_V2="${DSV41_ENGRAM_GATHER_V2:-1}"
DSV41_ENGRAM_WILLNEED="${DSV41_ENGRAM_WILLNEED:-1}"
DSV41_LMHEAD_MXFP8="${DSV41_LMHEAD_MXFP8:-1}"
DSV41_STREAM_FEED="${DSV41_STREAM_FEED:-1}"
DSV41_WOA_PREPACK="${DSV41_WOA_PREPACK:-1}"
DSV41_DSPARK_SPARSE_MARKOV="${DSV41_DSPARK_SPARSE_MARKOV:-1}"
NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-1048576}"
NCCL_LL128_BUFFSIZE="${NCCL_LL128_BUFFSIZE:-262144}"
NCCL_PROTO="${NCCL_PROTO:-^LL128}"
NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-8}"
DSV41_DROP_PAGE_CACHE="${DSV41_DROP_PAGE_CACHE:-1}"
DSV41_INDEXER_PREFILL_FACTOR="${DSV41_INDEXER_PREFILL_FACTOR:-1}"
DSV41_PREFILL_EMPTY_CACHE_TOKENS="${DSV41_PREFILL_EMPTY_CACHE_TOKENS:-8192}"
DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB="${DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB:-2.5}"
VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-256}"
# END generated
HF_HOME_IN_CONTAINER="/cache/huggingface"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH_DIR="${DSV41_PATCH_DIR:-$SCRIPT_DIR/docker/patch}"
RUN_STATE="$SCRIPT_DIR/.run-state"
# Worker copy of PATCH_DIR, relative to the worker's $HOME: per-user, not a shared /tmp name.
WORKER_PATCH_DIR=".cache/dsv41-patch"

# Container env for BOTH ranks: `docker run -e` below and the worker ssh line.
# NAME=default; an empty value is not passed (unset in the container). Names
# with an empty default here take their default from the generated block.
# Every env a docker/patch/*.py file reads must be listed (tests check it).
FORWARD_ENVS=(
  VLLM_EXL3_MOE_KERNEL=native
  DSV41_PATCH_STRICT=1
  DSV41_ENGRAM_DISK=
  LANGUAGE_MODEL_ONLY=
  MM_ENCODER_TP_MODE=
  DSV41_DSPARK_MARKOV_SCALE=1
  DSV41_STEP_CENSUS=0
  DSV41_INDEX_TOPK=0
  DSV41_MHC_DECODE_SPLITS=0
  DSV41_ENGRAM_CACHE=0
  DSV41_ENGRAM_CACHE_ROWS=
  DSV41_ENGRAM_CENSUS=
  DSV41_ENGRAM_CENSUS_EVERY=
  DSV41_ENGRAM_DISK_CHUNK=
  DSV41_ENGRAM_DISK_THREADS=
  DSV41_ENGRAM_FAST_STAGE=1
  DSV41_ENGRAM_STAGE_THREADS=16
  DSV41_ENGRAM_PREFETCH=
  DSV41_ENGRAM_PF_DUMP=0
  DSV41_ENGRAM_PREFETCH_DEBUG=0
  DSV41_ENGRAM_CPU_HASH=0
  DSV41_ENGRAM_GATHER_V2=
  DSV41_ENGRAM_GATHER_V2_MAX_ROWS=
  DSV41_ENGRAM_WILLNEED=
  DSV41_ENGRAM_WILLNEED_MIN_ROWS=512
  DSV41_ENGRAM_DEFER=0
  DSV41_LOAD_PF_G8=0
  DSV41_STREAM_FEED=
  DSV41_MHC_NO_DEEPGEMM=0
  DSV41_DSPARK_DRAFT_TOPK=0
  DSV41_DSPARK_SPARSE_MARKOV=
  DSV41_DSPARK_SPARSE_MARKOV_TOPK=
  DSV41_WOA_PREPACK=
  DSV41_DSPARK_TAIL_NGRAM=0
  DSV41_DSPARK_TAIL_NGRAM_POS=3
  DSV41_DSPARK_SOFTMAX_VERIFY=0
  DSV41_DSPARK_REFINE_PASS=0
  DSV41_DSPARK_CONF_GATE=0
  DSV41_MLA_IO_WARPS=0
  DSV41_MLA_CHUNKS_PER_BLOCK=0
  DSV41_DENSE_DG_SMALLM=0
  DSV41_DENSE_DG_SHAPES=
  DSV41_PROBE_WO_A_EVERY=
  DSV41_LMHEAD_MXFP8=
  DSV41_DROP_PAGE_CACHE=
  DSV41_INDEXER_PREFILL_FACTOR=
  DSV41_PREFILL_EMPTY_CACHE_TOKENS=
  DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB=
  VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=
  VLLM_EXL3_FAT_THRESHOLD=
  DSV41_P2B_SRC_SORT=
  DSV41_ALLOW_CUDA_GRAPHS=
  VLLM_USE_BREAKABLE_CUDAGRAPH=
  VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=256
  NCCL_MIN_NCHANNELS=
  NCCL_MAX_NCHANNELS=
  NCCL_NTHREADS=
  NCCL_BUFFSIZE=
  NCCL_LL128_BUFFSIZE=
  NCCL_PROTO=
  NCCL_LAUNCH_CACHE=
  NCCL_CROSS_NIC=1
  NCCL_IB_MERGE_NICS=
)
for fwd in "${FORWARD_ENVS[@]}"; do
  fwd_name="${fwd%%=*}"
  if [[ -z "${!fwd_name:-}" ]]; then
    printf -v "$fwd_name" '%s' "${fwd#*=}"
  fi
done

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
      SPEC_CONFIG='{"method":"dspark","num_speculative_tokens":'"$NUM_SPECULATIVE_TOKENS"',"draft_sample_method":"greedy"}'
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
if [[ "$SPEC" == dspark && ! "$NUM_SPECULATIVE_TOKENS" =~ ^[1-5]$ && "$FORCE_UNSAFE_CTX" != 1 ]]; then
  echo "NUM_SPECULATIVE_TOKENS=$NUM_SPECULATIVE_TOKENS must be an integer 1..5 (DSpark block size 5). E6: k=10 collapsed L.A.I.L to 12.1 vs 22.6. FORCE_UNSAFE_CTX=1 overrides." >&2
  exit 1
fi
case "$AUDIT" in
  warn | strict | off) ;;
  *)
    echo "AUDIT=$AUDIT (want warn, strict or off)" >&2
    exit 1
    ;;
esac

# Cudagraph capture sizes match the verify batch: {1} + {s*k, s*(k+1)} for s in 1..MAX_NUM_SEQS.
# k=3, 2 seqs gives [1,3,4,6,8] (R16: +5.5% vs padded k5-era sizes).
if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
  spec_k=0
  if [[ "$SPEC" == dspark ]]; then
    spec_k="$NUM_SPECULATIVE_TOKENS"
  fi
  capture_sizes=(1)
  for ((s = 1; s <= MAX_NUM_SEQS; s++)); do
    capture_sizes+=($((s * spec_k)) $((s * (spec_k + 1))))
  done
  capture_list="$(printf '%s\n' "${capture_sizes[@]}" | awk '$1 > 0' | sort -nu | paste -sd, -)"
  COMPILATION_CONFIG='{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":['"$capture_list"'],"custom_ops":["all"]}'
fi

if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
  printf '==> validate-only spec=%s seqs=%s spec_tokens=%s quant=%s engram_disk=%s eager=%s lm_only=%s image=%s compilation_config=%s\n' \
    "$SPEC" "$MAX_NUM_SEQS" "$NUM_SPECULATIVE_TOKENS" "$QUANTIZATION" "$DSV41_ENGRAM_DISK" "$ENFORCE_EAGER" "$LANGUAGE_MODEL_ONLY" "$IMAGE" "$COMPILATION_CONFIG"
  exit 0
fi

log() { printf '==> %s\n' "$*"; }

host_short() { hostname -s | tr '[:upper:]' '[:lower:]'; }

detect_role() {
  if [[ -n "${ROLE:-}" ]]; then
    printf '%s\n' "$ROLE"
    return
  fi
  case "$(host_short)" in
    spark2*) printf 'worker\n' ;;
    *) printf 'head\n' ;;
  esac
}

hf_bin() {
  if command -v hf >/dev/null 2>&1; then
    echo hf
  elif command -v huggingface-cli >/dev/null 2>&1; then
    echo huggingface-cli
  else
    return 1
  fi
}

resolve_model() {
  printf '%s\n' "$SNAPSHOT_IN_CONTAINER"
}

maybe_drop_caches() {
  if sudo -n true >/dev/null 2>&1; then
    sync
    echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
  fi
}

ensure_image() {
  log "Ensuring image $IMAGE"
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Image $IMAGE not found. Build docker/Dockerfile from this repo first:" >&2
    echo "  docker build -f docker/Dockerfile -t $IMAGE docker" >&2
    exit 1
  fi
  if ! docker image inspect -f '{{json .Config.Labels}}' "$IMAGE" 2>/dev/null | grep -q '"dsv41.recipe.patches"'; then
    echo "WARNING: $IMAGE has no dsv41.recipe.patches label: not built from this repo's docker/Dockerfile, or built before the label existed. Rebuild: docker build -f docker/Dockerfile -t $IMAGE docker" >&2
  fi
}

ensure_weights() {
  local snap HF=""
  snap="$(resolve_snapshot || true)"
  if [[ -z "$snap" && "$SKIP_DOWNLOAD" != "1" ]]; then
    HF="$(hf_bin || true)"
    if [[ -n "$HF" ]]; then
      # Engram shards are ~95 GiB. xet must stay on.
      unset HF_HUB_DISABLE_XET
      export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
      export HF_HOME="$HF_CACHE"
      export HF_HUB_CACHE="${HF_CACHE}/hub"
      log "Downloading $MODEL revision $SNAPSHOT_SHA (resumes under $HF_CACHE)"
      "$HF" download "$MODEL" --revision "$SNAPSHOT_SHA"
      snap="$(resolve_snapshot || true)"
    else
      echo "No hf CLI on PATH and snapshot missing for $MODEL revision $SNAPSHOT_SHA under $HF_CACHE" >&2
      exit 1
    fi
  fi
  if [[ -z "$snap" ]]; then
    echo "Pinned snapshot missing for $MODEL revision $SNAPSHOT_SHA under $HF_CACHE" >&2
    exit 1
  fi
  SNAPSHOT="$snap"
  SNAPSHOT_IN_CONTAINER="$(snapshot_in_container "$snap")"
  log "Using pinned snapshot $SNAPSHOT"
  if [[ "$QUANTIZATION" == exl3 && "$FORCE_UNSAFE_QUANT" != 1 ]]; then
    if ! python3 -c 'import json,sys; p=sys.argv[1]; c=json.load(open(p)); q=c.get("quantization_config") or {}; sys.exit(0 if q.get("quant_method")=="exl3" else 1)' \
      "$SNAPSHOT/config.json"; then
      echo "Snapshot $SNAPSHOT is not an EXL3 pack (quant_method!=exl3)." >&2
      exit 1
    fi
  fi
}

refuse_foreign_serve() {
  local name devices
  while IFS= read -r name; do
    [[ -z "$name" || "$name" == "$CONTAINER_NAME" ]] && continue
    devices="$(docker inspect -f '{{json .HostConfig.DeviceRequests}} {{json .HostConfig.Devices}} {{json .Config.Env}}' "$name" 2>/dev/null || true)"
    if printf '%s' "$devices" | grep -Eqi 'gpu|nvidia|infiniband'; then
      echo "$name is using GPUs or InfiniBand. This recipe needs exclusive GPUs on both Sparks. Do not start. Do not docker rm that container from this script." >&2
      exit 1
    fi
  done < <(docker ps --format '{{.Names}}')
}

refuse_busy_port() {
  if (echo >/dev/tcp/127.0.0.1/"$PORT") >/dev/null 2>&1; then
    echo "Port $PORT is already in use. If it is this recipe's serve, stop it first: ./stop.sh" >&2
    exit 1
  fi
}

stop_local() {
  if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    log "Removing existing container $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME" >/dev/null
  fi
}

start_local() {
  local rank="$1"
  mkdir -p "$HF_CACHE"
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker not found" >&2
    exit 1
  fi
  refuse_foreign_serve
  maybe_drop_caches
  stop_local
  ensure_image
  ensure_weights

  local serve_model
  serve_model="$(resolve_model)"

  # The model is always a local snapshot path (resolve_model), so the
  # container gets no Hub token and no Hub access.
  local env_args=(
    -e "HF_HOME=$HF_HOME_IN_CONTAINER"
    -e "HF_HUB_OFFLINE=1"
    -e "TORCH_CUDA_ARCH_LIST=12.1a"
    -e "FLASHINFER_CUDA_ARCH_LIST=12.1a"
    -e "FLASHINFER_DISABLE_VERSION_CHECK=1"
    -e "VLLM_ENGINE_READY_TIMEOUT_S=3600"
    -e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    -e "VLLM_PLUGINS=vllm_exl3"
    -e "NCCL_SOCKET_IFNAME=$IFACE"
    -e "GLOO_SOCKET_IFNAME=$IFACE"
    -e "TP_SOCKET_IFNAME=$IFACE"
    -e "NCCL_IB_HCA=$HCA"
    -e "NCCL_NET=IB"
    -e "NCCL_IB_DISABLE=0"
    -e "NCCL_NVLS_ENABLE=0"
    -e "NCCL_CUMEM_ENABLE=0"
    -e "NCCL_DEBUG=WARN"
  )
  local fwd_name
  for fwd_name in "${FORWARD_ENVS[@]%%=*}"; do
    if [[ -n "${!fwd_name:-}" ]]; then
      env_args+=(-e "$fwd_name=${!fwd_name}")
    fi
  done
  local host_ip="$HEAD_IP"
  if [[ "$rank" != "0" ]]; then
    host_ip="$(ip -4 -o addr show "$IFACE" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)"
    host_ip="${host_ip:-10.100.8.2}"
  fi
  env_args+=(-e "VLLM_HOST_IP=$host_ip")

  local rank_args=()
  if [[ "$rank" == "0" ]]; then
    rank_args+=(--host 0.0.0.0 --port "$PORT")
  else
    rank_args+=(--headless)
  fi

  local eager_args=()
  if [[ "$ENFORCE_EAGER" == "1" ]]; then
    eager_args+=(--enforce-eager)
  else
    eager_args+=(--compilation-config "$COMPILATION_CONFIG")
  fi
  # FlashInfer SM120 sparse-MLA autotune feeds DeepGEMM paged-MQA which
  # asserts block_kv in {32, 64}; skip that warmup on this pack.
  local kernel_args=(
    --kernel-config '{"enable_flashinfer_autotune":false,"enable_jit_warmup":false}'
  )

  local vol_args=(-v "${HF_CACHE}:${HF_HOME_IN_CONTAINER}")
  if [[ -d "$PATCH_DIR" ]]; then
    vol_args+=(
      -v "${PATCH_DIR}:/opt/dsv41-patch:ro"
      -v "${PATCH_DIR}/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro"
    )
  fi
  local batched_args=()
  if [[ -n "$MAX_NUM_BATCHED_TOKENS" ]]; then
    batched_args+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
  fi
  local load_args=()
  if [[ "$LOAD_FORMAT" != "auto" ]]; then
    load_args+=(--load-format "$LOAD_FORMAT")
  fi
  local spec_args=()
  if [[ -n "$SPEC_CONFIG" ]]; then
    spec_args+=(--speculative-config "$SPEC_CONFIG")
  fi
  local lm_args=()
  if [[ "$LANGUAGE_MODEL_ONLY" == "1" ]]; then
    lm_args+=(--language-model-only)
  elif [[ -n "${MM_ENCODER_TP_MODE:-}" ]]; then
    lm_args+=(--mm-encoder-tp-mode "$MM_ENCODER_TP_MODE")
  fi
  local extra_args=()
  read -ra extra_args <<<"$EXTRA_ARGS"
  local kv_dtype_arg="$KV_CACHE_DTYPE"
  if [[ "$kv_dtype_arg" == fp8_e4m3 ]]; then
    kv_dtype_arg=fp8
  fi

  log "Starting $CONTAINER_NAME rank=$rank model=$serve_model ctx=$MAX_MODEL_LEN quant=$QUANTIZATION engram_disk=$DSV41_ENGRAM_DISK"
  docker run -d \
    --name "$CONTAINER_NAME" \
    --restart no \
    --gpus all \
    --network host \
    --ipc host \
    --shm-size 32g \
    --device /dev/infiniband \
    --cap-add IPC_LOCK \
    --ulimit memlock=-1:-1 \
    "${vol_args[@]}" \
    "${env_args[@]}" \
    --entrypoint vllm \
    "$IMAGE" \
    serve \
    "$serve_model" \
    --tensor-parallel-size "$TP" \
    --nnodes "$NNODES" \
    --node-rank "$rank" \
    --distributed-executor-backend mp \
    --master-addr "$HEAD_IP" \
    --master-port "$MASTER_PORT" \
    "${rank_args[@]}" \
    --max-model-len "$MAX_MODEL_LEN" \
    --kv-cache-dtype "$kv_dtype_arg" \
    --kv-cache-memory "$KV_CACHE_MEMORY" \
    --gpu-memory-utilization "$UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    "${batched_args[@]}" \
    "${eager_args[@]}" \
    "${kernel_args[@]}" \
    --block-size "$BLOCK_SIZE" \
    --quantization "$QUANTIZATION" \
    "${spec_args[@]}" \
    "${lm_args[@]}" \
    --tokenizer-mode deepseek_v41 \
    --tool-call-parser deepseek_v41 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v41 \
    --default-chat-template-kwargs '{"thinking":false,"reasoning_effort":"low"}' \
    "${load_args[@]}" \
    --served-model-name "$SERVED_NAME" \
    --trust-remote-code \
    "${extra_args[@]}"
}

# Worker helpers (head side). An ssh failure is "unknown", never "exited".
WORKER_STARTED=0
WORKER_LOG=""
# vLLM serve.py logs this when the headless rank starts connecting to HEAD_IP:MASTER_PORT.
WORKER_LAUNCH_MARK="headless multiproc executor"

worker_ssh() { ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" "$@"; }

# 0 = running, 1 = not running, 2 = unknown (ssh or docker failed).
worker_state() {
  local names
  names="$(worker_ssh "docker ps --format '{{.Names}}'" 2>/dev/null)" || return 2
  grep -qx -- "$CONTAINER_NAME" <<<"$names"
}

save_worker_logs() {
  [[ -n "$WORKER_LOG" ]] && return 0
  mkdir -p "$RUN_STATE"
  WORKER_LOG="$RUN_STATE/worker-$(date +%Y%m%d-%H%M%S).log"
  if worker_ssh "docker logs $(printf '%q' "$CONTAINER_NAME")" >"$WORKER_LOG" 2>&1; then
    echo "Worker logs saved to $WORKER_LOG" >&2
  else
    echo "Could not read worker logs from $WORKER_HOST (partial output in $WORKER_LOG)" >&2
  fi
}

fail_worker_exited() {
  echo "Worker $CONTAINER_NAME on $WORKER_HOST is not running. Last worker logs:" >&2
  save_worker_logs
  tail -n 120 "$WORKER_LOG" >&2
  exit 1
}

# EXIT trap while the head is starting: keep both ranks' logs, then remove
# both containers so a failed boot does not leave a ~75 GiB rank loaded on
# either node (a started head keeps waiting on the rendezvous otherwise).
HEAD_STARTED=0
cleanup_worker_on_failure() {
  local rc=$?
  trap - EXIT
  if [[ "$rc" != 0 && "$WORKER_STARTED" == 1 ]]; then
    echo "Head start failed (exit $rc). Removing $CONTAINER_NAME on $WORKER_HOST." >&2
    save_worker_logs
    worker_ssh "docker rm -f $(printf '%q' "$CONTAINER_NAME")" >/dev/null 2>&1 \
      || echo "Could not remove $CONTAINER_NAME on $WORKER_HOST. Run ./stop.sh." >&2
    if [[ "$HEAD_STARTED" == 1 ]] && docker ps -a --format '{{.Names}}' | grep -qx -- "$CONTAINER_NAME"; then
      local head_log
      head_log="$RUN_STATE/head-$(date +%Y%m%d-%H%M%S).log"
      mkdir -p "$RUN_STATE"
      docker logs "$CONTAINER_NAME" >"$head_log" 2>&1 || true
      echo "Head logs saved to $head_log. Removing $CONTAINER_NAME here." >&2
      docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 \
        || echo "Could not remove $CONTAINER_NAME here. Run ./stop.sh." >&2
    fi
  fi
  exit "$rc"
}

# Start the head once the worker reached its rendezvous (replaces a fixed 25 s sleep).
wait_worker_launch() {
  local i st logs
  for i in $(seq 1 60); do
    st=0
    worker_state || st=$?
    if [[ "$st" == 1 ]]; then
      fail_worker_exited
    fi
    logs="$(worker_ssh "docker logs $(printf '%q' "$CONTAINER_NAME") 2>&1" 2>/dev/null || true)"
    if grep -qF -- "$WORKER_LAUNCH_MARK" <<<"$logs"; then
      log "Worker is connecting to $HEAD_IP:$MASTER_PORT. Starting head."
      return 0
    fi
    sleep 2
  done
  log "No '$WORKER_LAUNCH_MARK' in the worker log after 120s. Starting head anyway."
}

wait_ready() {
  log "Waiting for http://127.0.0.1:${PORT}/health and /v1/models"
  local i body st
  for i in $(seq 1 720); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      body="$(curl -sf "http://127.0.0.1:${PORT}/v1/models" || true)"
      if [[ -n "$body" && "$body" == *"$SERVED_NAME"* ]]; then
        log "Ready → http://127.0.0.1:${PORT}/v1  (context=$MAX_MODEL_LEN)"
        printf '%s\n' "$body"
        echo
        return 0
      fi
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
      echo "Container exited early. Logs:" >&2
      docker logs "$CONTAINER_NAME" 2>&1 | tail -120 >&2
      exit 1
    fi
    if [[ "$WORKER_STARTED" == 1 ]] && (( i % 12 == 0 )); then
      st=0
      worker_state || st=$?
      if [[ "$st" == 1 ]]; then
        fail_worker_exited
      fi
    fi
    sleep 5
    if (( i % 12 == 0 )); then
      log "still loading… (${i}×5s) — docker logs -f $CONTAINER_NAME"
    fi
  done
  echo "Timed out waiting for API. Recent logs:" >&2
  docker logs "$CONTAINER_NAME" 2>&1 | tail -120 >&2
  exit 1
}

# Grep both ranks' logs for the engagement markers kept next to each patch.
audit_engagement() {
  [[ "$AUDIT" == off ]] && return 0
  local ts name logs=() envs=(ENFORCE_EAGER="$ENFORCE_EAGER")
  ts="$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$RUN_STATE"
  docker logs "$CONTAINER_NAME" >"$RUN_STATE/audit-head-$ts.log" 2>&1 || true
  logs+=("head=$RUN_STATE/audit-head-$ts.log")
  if [[ "$WORKER_STARTED" == 1 ]]; then
    worker_ssh "docker logs $(printf '%q' "$CONTAINER_NAME")" >"$RUN_STATE/audit-worker-$ts.log" 2>&1 || true
    logs+=("worker=$RUN_STATE/audit-worker-$ts.log")
  fi
  for name in "${FORWARD_ENVS[@]%%=*}"; do
    envs+=("$name=${!name:-}")
  done
  if ! env "${envs[@]}" python3 "$SCRIPT_DIR/tools/engagement_audit.py" --mode "$AUDIT" "${logs[@]}"; then
    echo "AUDIT=strict: a patch did not engage (see above). The serve is up; ./stop.sh tears it down." >&2
    exit 1
  fi
}

# First-request JIT (Triton, CuTeDSL) off the user's TTFT. Nonce prompts stay out of the prefix cache.
warmup() {
  [[ "$WARMUP" == 1 ]] || return 0
  log "Warmup: greedy, t=0.7 and a ~3k-token nonce prefill"
  python3 "$SCRIPT_DIR/tools/warmup.py" --url "http://127.0.0.1:${PORT}/v1/chat/completions" --model "$SERVED_NAME" \
    || echo "WARNING: warmup failed. The serve is up; the first requests pay the JIT." >&2
}

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  ensure_image
  ensure_weights
  exit 0
fi

ROLE="$(detect_role)"
log "role=$ROLE host=$(host_short)"

if [[ "$ORCHESTRATE" == "auto" && "$ROLE" == "head" ]]; then
  refuse_foreign_serve
  refuse_busy_port
  if [[ "$NNODES" -gt 1 ]]; then
    if ! command -v ssh >/dev/null 2>&1 || ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" true >/dev/null 2>&1; then
      echo "Cannot SSH to $WORKER_HOST. Refusing to start a TP=$TP head rank alone (NNODES=$NNODES)." >&2
      exit 1
    fi
    # run.sh knobs the worker needs, plus FORWARD_ENVS. %q keeps quotes and JSON intact.
    worker_config=(
      IMAGE CONTAINER_NAME PORT MASTER_PORT HEAD_IP IFACE HCA
      MAX_MODEL_LEN MAX_NUM_SEQS UTIL KV_CACHE_MEMORY KV_CACHE_DTYPE BLOCK_SIZE TP NNODES
      SERVED_NAME SKIP_DOWNLOAD SPEC SPEC_CONFIG NUM_SPECULATIVE_TOKENS ENFORCE_EAGER
      COMPILATION_CONFIG MAX_NUM_BATCHED_TOKENS FORCE_UNSAFE_CTX FORCE_UNSAFE_ENGRAM
      FORCE_UNSAFE_QUANT LOAD_FORMAT QUANTIZATION SNAPSHOT_SHA HF_CACHE MODEL EXTRA_ARGS
    )
    worker_env="ROLE=worker ORCHESTRATE=0 DSV41_PATCH_DIR=\$HOME/$WORKER_PATCH_DIR"
    for name in "${worker_config[@]}" "${FORWARD_ENVS[@]%%=*}"; do
      worker_env+=" $name=$(printf '%q' "${!name:-}")"
    done
    # Preflight both nodes (image, weights, EXL3 config) before any container starts.
    log "Preflight $(host_short)"
    ensure_image
    ensure_weights
    scp -q "$0" "${WORKER_HOST}:/tmp/dsv41-exl3-run.sh"
    log "Preflight $WORKER_HOST"
    if ! ssh "$WORKER_HOST" "$worker_env PREFLIGHT_ONLY=1 bash /tmp/dsv41-exl3-run.sh"; then
      echo "Preflight failed on $WORKER_HOST. No container was started." >&2
      exit 1
    fi
    mkdir -p "$RUN_STATE"
    printf '%s\n' "$WORKER_HOST" >"$RUN_STATE/worker_host"
    ssh "$WORKER_HOST" "rm -rf ~/$WORKER_PATCH_DIR && mkdir -p ~/.cache"
    scp -q -r "$PATCH_DIR" "${WORKER_HOST}:$WORKER_PATCH_DIR"
    trap cleanup_worker_on_failure EXIT
    trap 'trap - EXIT; echo "Interrupted. Containers keep loading; ./stop.sh stops both ranks." >&2; exit 130' INT
    WORKER_STARTED=1
    log "Starting worker on $WORKER_HOST first"
    ssh "$WORKER_HOST" "$worker_env bash /tmp/dsv41-exl3-run.sh"
    wait_worker_launch
  fi
  HEAD_STARTED=1
  start_local 0
  wait_ready
  trap - EXIT INT
  audit_engagement
  warmup
  log "Stop with: ./stop.sh"
elif [[ "$ROLE" == "worker" ]]; then
  start_local 1
  log "Worker rank 1 is up. Head should start next."
else
  start_local 0
  wait_ready
  audit_engagement
  warmup
  log "Stop with: ./stop.sh"
fi
