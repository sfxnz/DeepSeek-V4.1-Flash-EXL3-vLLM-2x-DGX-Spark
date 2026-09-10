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
BLOCK_SIZE="${BLOCK_SIZE:-128}"
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
SNAPSHOT="${HF_CACHE}/hub/models--${MODEL//\//--}/snapshots/${SNAPSHOT_SHA}"
SNAPSHOT_IN_CONTAINER="${HF_HOME_IN_CONTAINER}/hub/models--${MODEL//\//--}/snapshots/${SNAPSHOT_SHA}"

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

token_env() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    printf '%s' "$HF_TOKEN"
    return
  fi
  if [[ -f "$HOME/.cache/huggingface/token" ]]; then
    tr -d '[:space:]' <"$HOME/.cache/huggingface/token"
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
}

ensure_weights() {
  if [[ "$SKIP_DOWNLOAD" != "1" ]]; then
    local HF=""
    HF="$(hf_bin || true)"
    if [[ -d "$SNAPSHOT" ]]; then
      log "Using pinned snapshot $SNAPSHOT"
    elif [[ -n "$HF" ]]; then
      export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
      log "Downloading $MODEL revision $SNAPSHOT_SHA (resumes under $HF_CACHE)"
      "$HF" download "$MODEL" --revision "$SNAPSHOT_SHA"
    else
      echo "No hf CLI on PATH and snapshot $SNAPSHOT is missing" >&2
      exit 1
    fi
  fi
  if [[ ! -d "$SNAPSHOT" ]]; then
    echo "Pinned snapshot missing: $SNAPSHOT" >&2
    exit 1
  fi
  if [[ "$QUANTIZATION" == exl3 && "$FORCE_UNSAFE_QUANT" != 1 ]]; then
    if ! python3 -c 'import json,sys; p=sys.argv[1]; c=json.load(open(p)); q=c.get("quantization_config") or {}; sys.exit(0 if q.get("quant_method")=="exl3" else 1)' \
      "$SNAPSHOT/config.json"; then
      echo "Snapshot $SNAPSHOT is not an EXL3 pack (quant_method!=exl3). Run tools/quantize_experts_exl3.py first." >&2
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
    echo "Port $PORT is already in use" >&2
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

  local tok
  tok="$(token_env || true)"
  local env_args=(
    -e "HF_HOME=$HF_HOME_IN_CONTAINER"
    -e "TORCH_CUDA_ARCH_LIST=12.1a"
    -e "FLASHINFER_CUDA_ARCH_LIST=12.1a"
    -e "FLASHINFER_DISABLE_VERSION_CHECK=1"
    -e "VLLM_ENGINE_READY_TIMEOUT_S=3600"
    -e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    -e "VLLM_PLUGINS=vllm_exl3"
    -e "DSV41_ENGRAM_DISK=$DSV41_ENGRAM_DISK"
    -e "NCCL_SOCKET_IFNAME=$IFACE"
    -e "GLOO_SOCKET_IFNAME=$IFACE"
    -e "TP_SOCKET_IFNAME=$IFACE"
    -e "NCCL_IB_HCA=$HCA"
    -e "NCCL_NET=IB"
    -e "NCCL_IB_DISABLE=0"
    -e "NCCL_CROSS_NIC=1"
    -e "NCCL_NVLS_ENABLE=0"
    -e "NCCL_CUMEM_ENABLE=0"
    -e "NCCL_DEBUG=WARN"
  )
  local host_ip="$HEAD_IP"
  if [[ "$rank" != "0" ]]; then
    host_ip="$(ip -4 -o addr show "$IFACE" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)"
    host_ip="${host_ip:-10.100.8.2}"
  fi
  env_args+=(-e "VLLM_HOST_IP=$host_ip")
  if [[ -n "$tok" ]]; then
    env_args+=(-e "HF_TOKEN=$tok" -e "HUGGING_FACE_HUB_TOKEN=$tok")
  fi

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

  local vol_args=(-v "${HF_CACHE}:${HF_HOME_IN_CONTAINER}")
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
  fi
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
    $EXTRA_ARGS
}

wait_ready() {
  log "Waiting for http://127.0.0.1:${PORT}/health and /v1/models"
  local i body
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
    sleep 5
    if (( i % 12 == 0 )); then
      log "still loading… (${i}×5s) — docker logs -f $CONTAINER_NAME"
    fi
  done
  echo "Timed out waiting for API. Recent logs:" >&2
  docker logs "$CONTAINER_NAME" 2>&1 | tail -120 >&2
  exit 1
}

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
    log "Starting worker on $WORKER_HOST first"
    mkdir -p "${PWD}/.run-state"
    printf '%s\n' "$WORKER_HOST" >"${PWD}/.run-state/worker_host"
    scp -q "$0" "${WORKER_HOST}:/tmp/dsv41-exl3-run.sh"
    ssh "$WORKER_HOST" \
      "ROLE=worker ORCHESTRATE=0 IMAGE='$IMAGE' CONTAINER_NAME='$CONTAINER_NAME' PORT='$PORT' MASTER_PORT='$MASTER_PORT' HEAD_IP='$HEAD_IP' IFACE='$IFACE' HCA='$HCA' MAX_MODEL_LEN='$MAX_MODEL_LEN' MAX_NUM_SEQS='$MAX_NUM_SEQS' UTIL='$UTIL' KV_CACHE_MEMORY='$KV_CACHE_MEMORY' KV_CACHE_DTYPE='$KV_CACHE_DTYPE' BLOCK_SIZE='$BLOCK_SIZE' TP='$TP' NNODES='$NNODES' SERVED_NAME='$SERVED_NAME' SKIP_DOWNLOAD='$SKIP_DOWNLOAD' SPEC='$SPEC' SPEC_CONFIG='$SPEC_CONFIG' NUM_SPECULATIVE_TOKENS='$NUM_SPECULATIVE_TOKENS' ENFORCE_EAGER='$ENFORCE_EAGER' COMPILATION_CONFIG='$COMPILATION_CONFIG' MAX_NUM_BATCHED_TOKENS='$MAX_NUM_BATCHED_TOKENS' FORCE_UNSAFE_CTX='$FORCE_UNSAFE_CTX' FORCE_UNSAFE_ENGRAM='$FORCE_UNSAFE_ENGRAM' FORCE_UNSAFE_QUANT='$FORCE_UNSAFE_QUANT' LOAD_FORMAT='$LOAD_FORMAT' QUANTIZATION='$QUANTIZATION' DSV41_ENGRAM_DISK='$DSV41_ENGRAM_DISK' LANGUAGE_MODEL_ONLY='$LANGUAGE_MODEL_ONLY' SNAPSHOT_SHA='$SNAPSHOT_SHA' HF_CACHE='$HF_CACHE' MODEL='$MODEL' EXTRA_ARGS='$EXTRA_ARGS' bash /tmp/dsv41-exl3-run.sh"
    log "Worker container started. Waiting 25s for NCCL listen, then starting head"
    sleep 25
  fi
  start_local 0
  wait_ready
  log "Stop with: ./stop.sh"
elif [[ "$ROLE" == "worker" ]]; then
  start_local 1
  log "Worker rank 1 is up. Head should start next."
else
  start_local 0
  wait_ready
  log "Stop with: ./stop.sh"
fi
