#!/bin/bash
# scenario.sh NAME [ENV=VAL ...]: one CPU-only container, head env + overrides.
set -u
IDR=/tmp/claude-1000/-home-sfxnz-projects-ai-lab-recipes-DeepSeek-V4-1-Flash-EXL3-vLLM-2x-DGX-Spark/00916df6-3e19-4085-96e2-41192ed78f01/scratchpad/idr
PATCH=/home/sfxnz/projects/ai-lab/recipes/.worktrees/perf-review-0924/docker/patch
IMAGE=dsv41-flash-exl3-sm121:canonical-e12
name=$1; shift
out=$IDR/$name
rm -rf "$out"; mkdir -p "$out"
avail=$(free -g | awk '/^Mem:/{print $7}')
free -h | tee "$out/free-before.txt"
if [ "$avail" -lt 14 ]; then echo "ABORT: MemAvailable ${avail} GiB < 14" | tee "$out/ABORTED"; exit 2; fi
if [ "$(docker ps -q --filter name=dsv41-dryrun | wc -l)" != 0 ]; then echo "ABORT: another dry-run container is up"; exit 2; fi
envs=(--env-file "$IDR/head.env")
for kv in "$@"; do envs+=(-e "$kv"); done
printf '%s\n' "$@" > "$out/overrides.txt"
start=$(date +%s)
docker run --rm --name "dsv41-dryrun-$name" --network none --memory 10g --cpus 4 \
  -v "$PATCH:/opt/dsv41-patch:ro" \
  -v "$PATCH/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro" \
  -v "$IDR/driver.sh:/driver.sh:ro" \
  -v "$out:/out" \
  "${envs[@]}" -e PRE_WOA="${PRE_WOA:-0}" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  --entrypoint bash "$IMAGE" /driver.sh > "$out/docker.log" 2>&1
echo "docker rc=$? wall=$(( $(date +%s) - start ))s" | tee "$out/docker.rc"
free -h | tee "$out/free-after.txt"
