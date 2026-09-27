#!/bin/bash
# In-container CPU-only strict dry-run of the sitecustomize patch chain + decode-lever installs.
# Adapted from results/2026-09-24-review/image-dryrun/scripts/driver.sh; adds the lever probe
# (levers_probe.py) and a scan for every LOG_DISARMED marker tools/engagement_audit.py lists.
# Mounts (dryrun.py builds them from run.sh's own docker run argv): /opt/dsv41-patch (ro) and
# /usr/lib/python3.12/sitecustomize.py (ro) exactly as run.sh, the worktree at /repo (ro), /out (rw).
set -u
OUT=/out
SCRIPTS=/repo/results/2026-09-25-kernels/integration/scripts
SP=/usr/local/lib/python3.12/dist-packages
mkdir -p "$OUT/diffs"
{
  echo "nvidia-smi: $(command -v nvidia-smi || echo absent)  /dev/nvidia*: $(ls /dev/nvidia* 2>/dev/null | wc -l)"
  echo "cpu_dma_latency: $(ls -l /dev/cpu_dma_latency 2>/dev/null || echo absent)"
  echo "sitecustomize sha256: $(sha256sum /usr/lib/python3.12/sitecustomize.py | cut -c1-16)"
  env | sort | grep -E '^(DSV41_|VLLM_|LANGUAGE_MODEL_ONLY|MM_ENCODER|NCCL_|HF_HUB)'
} > "$OUT/env-check.txt"

# Originals of every source the chain could touch (no python here: python would run sitecustomize).
(cd "$SP" && find vllm vllm_exl3 flashinfer -type f \( -name '*.py' -o -name '*.cu' -o -name '*.cuh' \) -print0 \
  | tar --null -T - -cf /tmp/orig.tar)
touch /tmp/marker

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
    d="$OUT/diffs/$(echo "$rel" | tr / _).diff"
    diff -u "/tmp/orig/$rel" "$f" > "$d"
    echo "== $rel (+$(grep -c '^+[^+]' "$d") lines)" >> "$OUT/markers.txt"
    grep -E '^\+.*(dsv41|DSV41|widen_|MARK|# ---|// ---)' "$d" | head -40 >> "$OUT/markers.txt"
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

# Lever probe: a fresh interpreter (sitecustomize ran at its start) reports what each round-3
# lever wrapped, imports every round-3 patch module, and reads the vllm_exl3_c markers.
python3 "$SCRIPTS/levers_probe.py" "$OUT/probe.json" > "$OUT/probe.stdout" 2> "$OUT/probe.stderr"
echo $? > "$OUT/probe.rc"

# Every LOG_DISARMED marker (engagement_audit's list, as tools/disarm_scan.sh) in all outputs.
python3 -S - "$OUT" > "$OUT/disarm_scan.txt" 2>&1 <<'EOF'
import sys
sys.path.insert(0, "/repo/tools")
import engagement_audit as a
out = sys.argv[1]
markers = sorted(set(a.expectations({})[1]))
hits = 0
for name in ("run1.stdout", "run1.stderr", "run2.stdout", "run2.stderr", "probe.stdout", "probe.stderr",
             "imports.txt", "imports.stderr"):
    for line in open(f"{out}/{name}", errors="replace"):
        for m in markers:
            if m in line:
                hits += 1
                print(f"HIT {name}: [{m}] {line.rstrip()[:300]}")
print("markers:", len(markers))
print("disarm hits:", hits)
EOF
echo "driver done" > "$OUT/done"
chown -R "${HOST_UID:-1000}:${HOST_GID:-1000}" "$OUT"
