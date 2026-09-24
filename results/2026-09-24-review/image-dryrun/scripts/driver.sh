#!/bin/bash
# In-container CPU-only dry-run of the sitecustomize patch chain.
# Mounts: /opt/dsv41-patch (ro), /usr/lib/python3.12/sitecustomize.py (ro), /out (rw).
# PRE_WOA=1: first apply fix_o_proj_woa_fp8.py stage 2 like docker/Dockerfile.woa-prepack.
set -u
OUT=/out
SP=/usr/local/lib/python3.12/dist-packages
mkdir -p "$OUT/diffs"
echo "nvidia-smi: $(command -v nvidia-smi || echo absent)  /dev/nvidia*: $(ls /dev/nvidia* 2>/dev/null | wc -l)" > "$OUT/env-check.txt"
env | sort | grep -E '^(DSV41_|VLLM_|LANGUAGE_MODEL_ONLY|MM_ENCODER|NCCL_|HF_HUB)' >> "$OUT/env-check.txt"

# Originals of every source the chain could touch (no python: python would run sitecustomize).
(cd "$SP" && find vllm vllm_exl3 flashinfer -type f \( -name '*.py' -o -name '*.cu' -o -name '*.cuh' \) -print0 \
  | tar --null -T - -cf /tmp/orig.tar)
touch /tmp/marker

if [ "${PRE_WOA:-0}" = 1 ]; then
  python3 -S /opt/dsv41-patch/fix_o_proj_woa_fp8.py "$SP/vllm/models/deepseek_v4/nvidia/ops/o_proj.py" > "$OUT/pre_woa.log" 2>&1
  echo "rc=$?" >> "$OUT/pre_woa.log"
  grep -c "_woa_prepacked_scale" "$SP/vllm/models/deepseek_v4/nvidia/ops/o_proj.py" >> "$OUT/pre_woa.log"
fi

# Run 1: sitecustomize is imported at interpreter start (that is how every serve process gets it).
( time python3 -c 'import sitecustomize, sys; print("imported", "sitecustomize" in sys.modules)' ) \
  > "$OUT/run1.stdout" 2> "$OUT/run1.stderr"
echo $? > "$OUT/run1.rc"
# Run 2: idempotency (a second serve process sees the already-rewritten tree).
python3 -c 'print("imported")' > "$OUT/run2.stdout" 2> "$OUT/run2.stderr"
echo $? > "$OUT/run2.rc"

find "$SP" /usr/lib/python3.12 -type f -newer /tmp/marker ! -name '*.pyc' ! -path '*/__pycache__/*' | sort > "$OUT/rewritten.txt"

mkdir -p /tmp/orig && tar -xf /tmp/orig.tar -C /tmp/orig
: > "$OUT/markers.txt"
while read -r f; do
  rel="${f#$SP/}"
  if [ -f "/tmp/orig/$rel" ]; then
    diff -u "/tmp/orig/$rel" "$f" > "$OUT/diffs/$(echo "$rel" | tr / _).diff"
    echo "== $rel (+$(grep -c '^+[^+]' "$OUT/diffs/$(echo "$rel" | tr / _).diff") lines)" >> "$OUT/markers.txt"
    grep -E '^\+.*(dsv41|DSV41|widen_|MARK|# ---|// ---)' "$OUT/diffs/$(echo "$rel" | tr / _).diff" | head -40 >> "$OUT/markers.txt"
  else
    echo "== $rel (new file)" >> "$OUT/markers.txt"
  fi
done < "$OUT/rewritten.txt"

# py_compile every rewritten .py (no site: pure syntax/bytecode check).
python3 -S - "$OUT/rewritten.txt" > "$OUT/py_compile.txt" 2>&1 <<'EOF'
import py_compile, sys
bad = 0
for f in open(sys.argv[1]).read().split():
    if not f.endswith(".py"):
        continue
    try:
        py_compile.compile(f, doraise=True)
        print("ok  ", f)
    except py_compile.PyCompileError as e:
        bad += 1
        print("FAIL", f, e.msg)
print("py_compile failures:", bad)
EOF

# Import every rewritten module (sitecustomize runs again first: already applied).
python3 - "$OUT/rewritten.txt" > "$OUT/imports.txt" 2> "$OUT/imports.stderr" <<'EOF'
import importlib, sys, traceback
SP = "/usr/local/lib/python3.12/dist-packages/"
mods = []
for f in open(sys.argv[1]).read().split():
    if f.endswith(".py") and f.startswith(SP):
        m = f[len(SP):-3].replace("/", ".")
        mods.append(m[:-9] if m.endswith(".__init__") else m)
extra = [m for m in sys.argv[2:]]
bad = 0
for m in mods:
    try:
        importlib.import_module(m)
        print("ok  ", m)
    except BaseException as e:
        bad += 1
        print("FAIL", m, repr(e)[:400])
        traceback.print_exc(limit=3, file=sys.stderr)
print("import failures:", bad)
EOF
echo "driver done" > "$OUT/done"
chown -R "${HOST_UID:-1000}:${HOST_GID:-1000}" "$OUT"
