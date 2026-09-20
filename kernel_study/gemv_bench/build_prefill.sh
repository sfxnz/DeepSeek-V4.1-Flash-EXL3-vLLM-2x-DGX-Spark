#!/usr/bin/env bash
# Build the prefill harness extension inside the serve image WITHOUT GPUs.
#
# Host-side torch cpp_extension JIT is FORBIDDEN while the MCG serve is
# resident (host RAM is the GPU-memory pool on GB10 UMA; a JIT build has
# previously destabilized the serve). The serve container has no GPU device
# attached, its Python 3.12 + torch 2.13 + cu13 nvcc compile the extension
# CPU-side only. The built module is cached in build_prefill/ and imported
# by driver_prefill.py in a later GPU maintenance window.
#
# Usage (host, serve up or down — either is safe, no GPU touched):
#   kernel_study/gemv_bench/build_prefill.sh [container]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CT="${1:-dsv41-exl3-head}"

# 1. Generate the patched sources (pure CPU, host-side, no torch needed).
python3 "$HERE/make_bench_prefill.py"

# 2. Compile inside a THROWAWAY no-GPU container. The serve image's pip
#    nvidia/cu13 crt headers conflict with its own nvcc stub macros
#    (__cudaLaunch arity), so we build in the local bench image
#    dsv41-flash-exl3-sm121:latest which has the full CUDA toolkit
#    (cusparse et al.), the same torch 2.13+cu13 stack and the same
#    exllamav3_ext headers. The live serve container has GPUs attached —
#    never JIT there. Host torch JIT is likewise forbidden while the serve
#    is resident (UMA RAM).
BENCH_CT="dsv41-prefill-build"
docker rm -f "$BENCH_CT" >/dev/null 2>&1 || true
docker run -d --name "$BENCH_CT" --entrypoint sleep \
  dsv41-flash-exl3-sm121:latest 3600 >/dev/null

trap 'docker rm -f "$BENCH_CT" >/dev/null 2>&1' EXIT

docker exec "$BENCH_CT" mkdir -p /workspace/prefill
docker cp "$HERE/build_prefill/bench_prefill.cu" "$BENCH_CT:/workspace/prefill/"
docker cp "$HERE/build_prefill/exl3_gemm_kernel_pf.cuh" "$BENCH_CT:/workspace/prefill/"
docker cp "$HERE/build_prefill/exl3_gemm_inner_pf.cuh" "$BENCH_CT:/workspace/prefill/"

# 3. Compile inside the image, CPU-only (no GPU visible to the container).
docker exec "$BENCH_CT" bash -lc '
  set -e
  cd /workspace/prefill
  python3 - <<EOF
import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.1a"
from torch.utils.cpp_extension import load
ext = load(
    name="bench_prefill",
    sources=["bench_prefill.cu"],
    extra_include_paths=[".",
        "/usr/local/lib/python3.12/dist-packages/exllamav3/exllamav3_ext",
        "/usr/local/lib/python3.12/dist-packages/exllamav3/exllamav3_ext/quant"],
    extra_cuda_cflags=["-O3", "-std=c++17"],
    verbose=True,
)
print("BUILD OK:", ext)
EOF
'
