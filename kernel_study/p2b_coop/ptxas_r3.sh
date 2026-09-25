#!/usr/bin/env bash
# CPU only: compile build_r3/chain_r3.cu (the serve image's p2b TU incl. the round-3 patches) for
# sm_121a in a no-GPU container, print ptxas resource usage of the K=2 MCG kernels, and dump
# their SASS (host cuobjdump) to build_r3/sass/.   kernel_study/p2b_coop/ptxas_r3.sh [image]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${1:-dsv41-flash-exl3-sm121:canonical-e13}"
python3 "$HERE/make_bench_r3.py" >/dev/null
set +e
docker run --rm --network none --memory 12g --cpus 6 --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$HERE/build_r3:/w" --entrypoint bash "$IMG" -c '
set -e
cd /w
EXL=/usr/local/lib/python3.12/dist-packages/exllamav3/exllamav3_ext
T=/usr/local/lib/python3.12/dist-packages/torch/include
nvcc -c chain_r3.cu -o chain_r3.o -std=c++17 -O3 -gencode arch=compute_121a,code=sm_121a \
    -I$EXL -I$T -I$T/torch/csrc/api/include -I/usr/include/python3.12 \
    -DTORCH_EXTENSION_NAME=p2b_r3 -DTORCH_API_INCLUDE_EXTENSION_H -D_GLIBCXX_USE_CXX11_ABI=1 \
    --expt-relaxed-constexpr -Xptxas -v 2> chain_r3.ptxas.txt || { grep -B2 -A2 " error" chain_r3.ptxas.txt | head -40; exit 1; }
' 2>&1 | grep -v 'vllm._C\|cpp_extension.py\|^$'
rc=${PIPESTATUS[0]}
set -e
[ "$rc" = 0 ] || { echo "ptxas_r3: compile failed" >&2; exit 1; }
python3 - "$HERE/build_r3/chain_r3.ptxas.txt" <<'EOF'
import re, sys
lines = open(sys.argv[1]).read().splitlines()
name = None
for i, l in enumerate(lines):
    m = re.search(r"Compiling entry function '(\S+)'", l)
    if m:
        name = m.group(1)
    if name and re.search(r"p2b_\w*kernelILi2ELi1E", name) and not re.search(r"ILi2ELi1ELi1E", name):
        if "Used" in l or "spill" in l:
            short = re.sub(r"EvPK6__half.*", "", name)
            print(f"{short}: {l.strip()}")
EOF
CUOBJDUMP="$(command -v cuobjdump || echo /usr/local/cuda/bin/cuobjdump)"
rm -rf "$HERE/build_r3/sass" && mkdir -p "$HERE/build_r3/sass"
for fn in $("$CUOBJDUMP" -symbols "$HERE/build_r3/chain_r3.o" 2>/dev/null | grep -o '_Z[0-9]*p2b_[a-z0-9_]*kernelILi2ELi1E[A-Za-z0-9_]*' | grep -v 'ELi1ELi1E' | sort -u); do
    short=$(echo "$fn" | sed 's/EvPK6__half.*//')
    "$CUOBJDUMP" -sass -fun "$fn" "$HERE/build_r3/chain_r3.o" > "$HERE/build_r3/sass/$short.sass"
    python3 "$HERE/hot_loops.py" "$HERE/build_r3/sass/$short.sass" | sed "s#$HERE/build_r3/sass/##"
done
