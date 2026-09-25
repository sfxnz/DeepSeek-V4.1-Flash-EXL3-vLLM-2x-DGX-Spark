#!/bin/bash
# Apply the docker/Dockerfile p2b chain (shapes, mrow, cfg1, codebook, fshift, srcsort) to
# the pinned vllm-exl3 p2b_moe.cu and compile it to an object, CPU only (no GPU needed for nvcc -c).
set -u
OUT=/out
cd /tmp
# python3 -S: the patchers are stdlib only; -S keeps sitecustomize out of it.
python3 -S - <<'EOF' > "$OUT/chain.log" 2>&1
import importlib.util
from pathlib import Path
R = Path("/repo")
src = (R / "tests/fixtures/p2b_moe.pin.cu").read_text()
def load(n):
    s = importlib.util.spec_from_file_location(n, R / "docker/patch" / f"{n}.py")
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
for n in ("shapes", "mrow", "cfg1", "codebook", "fshift"):
    src = load(f"widen_p2b_{n}").patch_cu(src)
Path("/tmp/chain_base.cu").write_text(src)
ss = load("widen_p2b_srcsort")
out = ss.patch_cu(src)
assert ss.patch_cu(out) == out, "srcsort not idempotent"
Path("/tmp/chain_srcsort.cu").write_text(out)
print("base lines", src.count("\n"), "srcsort lines", out.count("\n"),
      "grid.sync", src.count("grid.sync()"), out.count("grid.sync()"))
EOF
echo "chain rc=$?" >> "$OUT/chain.log"
cp /tmp/chain_base.cu /tmp/chain_srcsort.cu "$OUT/" 2>/dev/null
EXL=/usr/local/lib/python3.12/dist-packages/exllamav3/exllamav3_ext
T=/usr/local/lib/python3.12/dist-packages/torch/include
PY=/usr/include/python3.12
NVCC=$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)
"$NVCC" --version > "$OUT/nvcc-version.txt" 2>&1
for f in chain_base chain_srcsort; do
  s=$(date +%s)
  timeout 1100 "$NVCC" -c /tmp/$f.cu -o /tmp/$f.o -std=c++17 -O3 -gencode arch=compute_121a,code=sm_121a \
    -I$EXL -I$T -I$T/torch/csrc/api/include -I$PY -DTORCH_EXTENSION_NAME=vllm_exl3_c -DTORCH_API_INCLUDE_EXTENSION_H \
    -D_GLIBCXX_USE_CXX11_ABI=1 --expt-relaxed-constexpr -Xptxas -v > "$OUT/$f.nvcc.log" 2>&1
  rc=$?
  echo "$f rc=$rc wall=$(( $(date +%s) - s ))s size=$(stat -c %s /tmp/$f.o 2>/dev/null)" >> "$OUT/compile.txt"
  if [ $rc = 0 ]; then
    cuobjdump -symbols /tmp/$f.o 2>/dev/null | grep -c "p2b" | sed "s/^/$f p2b symbols: /" >> "$OUT/compile.txt"
    cuobjdump -symbols /tmp/$f.o 2>/dev/null | grep -o "_Z[A-Za-z0-9_]*p2b[A-Za-z0-9_]*" | sort -u > "$OUT/$f.symbols.txt"
  fi
done
grep -E "error|warning" "$OUT"/chain_*.nvcc.log | grep -v "ptxas info" | sort | uniq -c | sort -rn | head -20 > "$OUT/nvcc-diagnostics.txt"
chown -R "${HOST_UID:-1000}:${HOST_GID:-1000}" "$OUT"
