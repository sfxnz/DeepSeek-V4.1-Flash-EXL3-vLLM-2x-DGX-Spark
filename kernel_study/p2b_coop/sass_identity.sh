#!/usr/bin/env bash
# CPU-only: compile the live p2b chain (srcsort) and the coop build for sm_121a
# in ONE throwaway NO-GPU container, require byte-identical SORT=0/1 kernel
# machine code, compile-check the bench TU and report the inner MMA loops
# (spills) of p2b <2,1,0> vs coop <2,1,2>. Informational: compare the
# chain's <2,1,0> kernel with the one in the image's vllm_exl3_c (the pin
# fixture vs what the image was built from; torch build flags may differ).
# Memory-capped: UMA host RAM is the GPU pool on GB10, keep it small while a
# serve is resident. Refuses to start below 14 GiB MemAvailable.
#   kernel_study/p2b_coop/sass_identity.sh [image]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${1:-dsv41-flash-exl3-sm121:canonical-e13}"
avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
if (( avail_kb < 14 * 1024 * 1024 )); then
  echo "MemAvailable $((avail_kb / 1024)) MiB < 14 GiB; not starting the compile container" >&2
  exit 2
fi
python3 "$HERE/make_bench.py"
docker run --rm --network none --memory 8g --cpus 4 -v "$HERE/build:/w" --entrypoint bash "$IMG" -c '
set -e
cd /w
nvcc --version | tail -2
EXL=/usr/local/lib/python3.12/dist-packages/exllamav3/exllamav3_ext
T=/usr/local/lib/python3.12/dist-packages/torch/include
for f in chain_srcsort chain_coop bench_coop; do
  python3 -S -c "import resource, subprocess, sys, time; t = time.time(); rc = subprocess.run(sys.argv[2:]).returncode; print(f\"{sys.argv[1]}: nvcc {time.time() - t:.0f} s, max child RSS {resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024} MiB\", file=sys.stderr); sys.exit(rc)" $f nvcc -c $f.cu -o $f.o -std=c++17 -O3 -gencode arch=compute_121a,code=sm_121a \
    -I$EXL -I$T -I$T/torch/csrc/api/include -I/usr/include/python3.12 \
    -DTORCH_EXTENSION_NAME=p2b_coop_bench -DTORCH_API_INCLUDE_EXTENSION_H -D_GLIBCXX_USE_CXX11_ABI=1 \
    --expt-relaxed-constexpr -Xptxas -v 2> $f.ptxas.txt || { tail -30 $f.ptxas.txt; exit 1; }
  tail -1 $f.ptxas.txt
  grep -A3 "_Z22p2b_moe_batched_kernelILi2ELi1ELi[02]E" $f.ptxas.txt | grep -o "Used.*\|[0-9]* bytes stack frame.*" || true
done
rm -rf cubin && mkdir -p cubin/srcsort cubin/coop cubin/live
(cd cubin/srcsort && cuobjdump -xelf all ../../chain_srcsort.o >/dev/null)
(cd cubin/coop && cuobjdump -xelf all ../../chain_coop.o >/dev/null)
SO=$(ls /usr/local/lib/python3.12/dist-packages/vllm_exl3_c*.so)
(cd cubin/live && cuobjdump -xelf all "$SO" >/dev/null) || echo "live .so cubin extract failed (informational)"
chmod -R a+rwX /w
' 2>&1 | grep -v 'vllm._C\|cpp_extension.py\|^$'
python3 "$HERE/text_identity.py" "$HERE"/build/cubin/srcsort/*.cubin "$HERE"/build/cubin/coop/*.cubin
# SASS needs nvdisasm, which the serve image lacks: use the host CUDA toolkit when present.
CUOBJDUMP="$(command -v cuobjdump || echo /usr/local/cuda/bin/cuobjdump)"
if [[ -x "$CUOBJDUMP" && -x "$(dirname "$CUOBJDUMP")/nvdisasm" ]]; then
  SIG=EvPK6__halfPKlS4_S4_S4_S4_S4_S4_S4_S4_PKiS2_PS0_S7_S7_S7_S7_S7_S7_Pfiiiif
  for k in 0 2; do
    "$CUOBJDUMP" -sass -fun "_Z22p2b_moe_batched_kernelILi2ELi1ELi${k}E$SIG" "$HERE"/build/cubin/coop/*.cubin > "$HERE/build/sort$k.sass"
  done
  python3 "$HERE/hot_loops.py" "$HERE/build/sort0.sass" "$HERE/build/sort2.sass" | sed "s#$HERE/build/##"
else
  echo "no host cuobjdump/nvdisasm: inner-loop spill report skipped"
fi
python3 - "$HERE/build" <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).parents[1] / "p2b_srcsort"))
from text_identity import kernels
b = Path(sys.argv[1]) / "cubin"
ours = kernels(str(next((b / "srcsort").glob("*.cubin"))))
live = {}
for c in sorted((b / "live").glob("*.cubin")):
    live.update(kernels(str(c)))
for key in sorted(ours):
    got = live.get(key)
    state = "absent" if got is None else ("IDENTICAL" if got == ours[key] else f"differs ({len(got)} vs {len(ours[key])} bytes)")
    print(f"live vllm_exl3_c <{', '.join(key)}> vs pin chain: {state} (informational)")
EOF
