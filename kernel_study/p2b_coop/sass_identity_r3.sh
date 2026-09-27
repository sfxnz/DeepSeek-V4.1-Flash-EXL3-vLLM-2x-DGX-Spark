#!/usr/bin/env bash
# CPU only: compile the round-2 image TU (chain + coop, docker/Dockerfile.e14) and the round-3
# TU (chain + coop + dataflow, docker/Dockerfile.e15, now docker/Dockerfile.e14) for sm_121a in one no-GPU container and
# require every p2b_moe_batched_kernel<BITS, CB, SORT> (SORT 0/1 for all six BITS/CB, SORT 2 for
# <2, 1>) to have byte-identical machine code, with exactly one new kernel,
# p2b_coop_df_kernel<2, 1>. So DSV41_P2B_COOP unset, 0 or 1 runs the same code as review-e14.
#   kernel_study/p2b_coop/sass_identity_r3.sh [image]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${1:-dsv41-flash-exl3-sm121:canonical-e13}"
python3 "$HERE/make_bench.py" >/dev/null
python3 "$HERE/make_bench_r3.py" >/dev/null
W="$HERE/build_r3/identity"
rm -rf "$W" && mkdir -p "$W"
cp "$HERE/build/chain_coop.cu" "$HERE/build_r3/chain_r3.cu" "$W/"
set +e
docker run --rm --network none --memory 12g --cpus 6 --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$W:/w" --entrypoint bash "$IMG" -c '
set -e
cd /w
EXL=/usr/local/lib/python3.12/dist-packages/exllamav3/exllamav3_ext
T=/usr/local/lib/python3.12/dist-packages/torch/include
for f in chain_coop chain_r3; do
  nvcc -cubin -o $f.cubin $f.cu -std=c++17 -O3 -gencode arch=compute_121a,code=sm_121a \
    -I$EXL -I$T -I$T/torch/csrc/api/include -I/usr/include/python3.12 \
    -DTORCH_EXTENSION_NAME=p2b_ident -DTORCH_API_INCLUDE_EXTENSION_H -D_GLIBCXX_USE_CXX11_ABI=1 \
    --expt-relaxed-constexpr 2> $f.nvcc.txt || { grep -B2 -A2 " error" $f.nvcc.txt | head -30; exit 1; }
done
nvcc --version | tail -1
' 2>&1 | grep -v 'vllm._C\|cpp_extension.py\|^$'
rc=${PIPESTATUS[0]}
set -e
[ "$rc" = 0 ] || { echo "sass_identity_r3: compile failed" >&2; exit 1; }
python3 - "$W/chain_coop.cubin" "$W/chain_r3.cubin" "$HERE/../p2b_srcsort" <<'PY'
import re, sys
sys.path.insert(0, sys.argv[3])
from text_identity import kernels, sections
a, b = kernels(sys.argv[1]), kernels(sys.argv[2])
bad = 0
for key in sorted(a):
    same = a[key] == b.get(key)
    bad += not same
    print(f"BITS={key[0]} CB={key[1]} SORT={key[2]}: {len(a[key])} vs {len(b.get(key, b''))} bytes "
          f"{'IDENTICAL' if same else 'DIFFERENT'}")
extra = sorted(k for k in b if k not in a)
new = sorted(s for s in sections(sys.argv[2]) if s.startswith(".text.") and "p2b_coop_df_kernel" in s)
old = sorted(s for s in sections(sys.argv[1]) if s.startswith(".text.") and "p2b_coop_df_kernel" in s)
print(f"new p2b_moe_batched_kernel instantiations: {extra}")
print(f"new dataflow kernels: {[re.sub(r'EvPK6__half.*', '', s[6:]) for s in new]}")
ok = not bad and not extra and not old and len(new) == 1 and "ILi2ELi1E" in new[0]
print("sass_identity_r3:", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
