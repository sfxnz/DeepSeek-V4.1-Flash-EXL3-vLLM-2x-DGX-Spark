#!/usr/bin/env bash
# CPU-only: compile the p2b chain with and without widen_p2b_srcsort.py for
# sm_121a in a throwaway NO-GPU container, then require byte-identical SORT=0
# kernel machine code. Also compile-checks bench_srcsort.cu.
# No GPU is touched; the container is memory-capped (UMA host RAM is the GPU
# pool on GB10, so keep this small while a serve is resident).
#   kernel_study/p2b_srcsort/sass_identity.sh [image]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${1:-dsv41-flash-exl3-sm121:canonical-e12}"
python3 "$HERE/make_bench.py"
docker run --rm --network none --memory 8g --cpus 4 -v "$HERE/build:/w" --entrypoint bash "$IMG" -c '
set -e
cd /w
EXL=/usr/local/lib/python3.12/dist-packages/exllamav3/exllamav3_ext
T=/usr/local/lib/python3.12/dist-packages/torch/include
for f in chain_base chain_srcsort bench_srcsort; do
  nvcc -c $f.cu -o $f.o -std=c++17 -O3 -gencode arch=compute_121a,code=sm_121a \
    -I$EXL -I$T -I$T/torch/csrc/api/include -I/usr/include/python3.12 \
    -DTORCH_EXTENSION_NAME=p2b_srcsort_bench -DTORCH_API_INCLUDE_EXTENSION_H -D_GLIBCXX_USE_CXX11_ABI=1 \
    --expt-relaxed-constexpr -Xptxas -v 2> $f.ptxas.txt || { tail -30 $f.ptxas.txt; exit 1; }
  echo "compiled $f"
done
rm -rf cubin && mkdir -p cubin/base cubin/srcsort
(cd cubin/base && cuobjdump -xelf all ../../chain_base.o >/dev/null)
(cd cubin/srcsort && cuobjdump -xelf all ../../chain_srcsort.o >/dev/null)
chmod -R a+rwX /w
' 2>&1 | grep -v 'vllm._C\|cpp_extension.py'
python3 "$HERE/text_identity.py" "$HERE"/build/cubin/base/*.cubin "$HERE"/build/cubin/srcsort/*.cubin
