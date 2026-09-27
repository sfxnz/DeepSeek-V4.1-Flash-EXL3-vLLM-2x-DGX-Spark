#!/usr/bin/env bash
# compute-sanitizer over the det mHC kernels, one tool per call (each run stays under the 10-min
# lock budget). Run inside the serve image by spark2.sh with SANITIZER=1, which mounts the host's
# /usr/local/cuda-13.0/compute-sanitizer read-only:
#   SANITIZER=1 kernel_study/mhc_det/spark2.sh <tag> kernel_study/mhc_det/san.sh <tool>
#   memcheck / racecheck / synccheck: kernel-name filter mhc_det, T 1..16 (sanitize.py)
#   initcheck: no filter and PYTORCH_NO_CUDA_MEMORY_CACHING=1 (every torch.empty is a fresh
#   cudaMalloc, torch kernels count as writers), T 1/5/8/9/16 plus the two canaries.
# Reports land in results/2026-09-25-kernels/mhc-det/sanitizer/<tool>.txt.
S=/usr/local/cuda-13.0/compute-sanitizer/compute-sanitizer
OUT=/repo/results/2026-09-25-kernels/mhc-det/sanitizer
tool="$1"
mkdir -p "$OUT"
case "$tool" in
  memcheck|racecheck|synccheck)
    extra=""
    [ "$tool" = racecheck ] && extra="--racecheck-report all"
    $S --tool "$tool" $extra --report-api-errors no --kernel-name kns=mhc_det --print-limit 200 \
       python3 /repo/kernel_study/mhc_det/sanitize.py > "$OUT/$tool.txt" 2>&1
    rc=$?
    grep -v "^dsv41-patch" "$OUT/$tool.txt" | grep -E "ERROR SUMMARY|RACECHECK SUMMARY|Invalid|Race|azard|Barrier|ok:" | head -8
    ;;
  initcheck)
    PYTORCH_NO_CUDA_MEMORY_CACHING=1 $S --tool initcheck --report-api-errors no --print-limit 60 \
      python3 /repo/kernel_study/mhc_det/sanitize.py --init > "$OUT/initcheck.txt" 2>&1
    rc=$?
    grep -v "^dsv41-patch" "$OUT/initcheck.txt" | grep -E "ok:|CANARY|ERROR SUMMARY|Uninitialized|^=========     at " | sort | uniq -c | head -30
    ;;
  *) echo "usage: san.sh memcheck|racecheck|synccheck|initcheck"; exit 2 ;;
esac
echo "== $tool rc=$rc"
exit $rc
