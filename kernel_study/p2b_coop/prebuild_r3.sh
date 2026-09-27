#!/usr/bin/env bash
# CPU only: generate build_r3/*.cu and JIT-build the bench modules (bench_r3, bench_r3_ts) into
# build/torch_ext with the serve image's toolchain, so the GPU window does not compile.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
IMG="${1:-dsv41-flash-exl3-sm121:canonical-e13}"
docker run --rm --network none --memory 12g --cpus 6 --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -e TORCH_CUDA_ARCH_LIST=12.1a -e TORCH_EXTENSIONS_DIR=/repo/kernel_study/p2b_coop/build/torch_ext \
    -v "$REPO:/repo" -w /repo --entrypoint python3 "$IMG" -c '
import sys; sys.path.insert(0, "kernel_study/p2b_coop")
import bench_r3
for stamps in (False, True):
    print("built", bench_r3.build_ext(stamps).__file__)
' 2>&1 | grep -v -e '^dsv41-patch' -e '^dsv41-indexer' -e 'importing.py' -e 'interface.py' -e 'cpp_extension.py:' -e '^unchanged'
