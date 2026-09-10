#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-dsv41-flash-exl3}"
ORCHESTRATE="${ORCHESTRATE:-auto}"

if [[ -z "${WORKER_HOST:-}" && -f "${PWD}/.run-state/worker_host" ]]; then
  WORKER_HOST="$(tr -d '[:space:]' <"${PWD}/.run-state/worker_host")"
fi
WORKER_HOST="${WORKER_HOST:-spark2}"

stop_local() {
  if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "Stopping $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME" >/dev/null
    echo "Stopped local $CONTAINER_NAME"
  else
    echo "No local container named $CONTAINER_NAME"
  fi
}

host_short() { hostname -s | tr '[:upper:]' '[:lower:]'; }

stop_local

if [[ "$ORCHESTRATE" == "0" ]]; then
  exit 0
fi

if [[ "$ORCHESTRATE" == "auto" ]]; then
  case "$(host_short)" in
    spark2*) ;;
    *)
      if command -v ssh >/dev/null 2>&1 && ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" true >/dev/null 2>&1; then
        echo "Stopping $CONTAINER_NAME on $WORKER_HOST"
        ssh "$WORKER_HOST" "docker rm -f '$CONTAINER_NAME' >/dev/null 2>&1 && echo Stopped remote $CONTAINER_NAME || echo No remote container named $CONTAINER_NAME"
      else
        echo "Cannot SSH to $WORKER_HOST. Remote $CONTAINER_NAME may still be running. Set WORKER_HOST to the rank this head started." >&2
        exit 1
      fi
      ;;
  esac
fi
