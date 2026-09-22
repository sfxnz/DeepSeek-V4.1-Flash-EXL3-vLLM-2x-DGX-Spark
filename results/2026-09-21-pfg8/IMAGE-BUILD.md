# G8 IMAGE BUILD — dsv41-flash-exl3-sm121:canonical-g8 (2026-09-22)

Status: **BUILT + VERIFIED (CPU-side) 2026-09-22 01:36 BST** — GPU bit-exact
gate scheduled post-04:00 (pack rebuild still running).

## What this image is

`canonical-e12` + G8-aware kernels (BOOT-CHAIN-AUDIT components 6/7/8):
rebuild of `exllamav3_ext` and `vllm_exl3_c` from the SAME pinned refs
(EXLLAMAV3_REF=5be88657, VLLM_EXL3_REF=d3cfd39) with the PF-G8 port applied
by two new patch scripts, CPU-only (no GPU in the build container; run
alongside the pack rebuild, nice + cpuset to keep host footprint modest).

## Build command (exact)

```bash
cd ~/projects/ai-lab/recipes/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark/docker
nohup nice -n 15 docker build --cpuset-cpus 4-19 -f Dockerfile.g8 \
  -t dsv41-flash-exl3-sm121:canonical-g8 . > /tmp/pfg8-image-build.log 2>&1
```

`docker/Dockerfile.g8` = `FROM canonical-e12` → clone pins → patch chain
(aarch64 patch → pfg8_kernels_exllamav3.py → pip build → import assert
`pf_g8_set` → widen chain → pfg8_kernels_vllm.py → pip build → import assert)
→ `pfg8_loader_reindex.py` baked in (dormant) → sitecustomize (fat gate) →
py_compile checks. canonical-e12 is NOT overwritten (rollback intact).

## What the patches do (docker/patch/)

- `pfg8_kernels_exllamav3.py` — component 6 + two extra readers found by the
  port audit:
  - `exl3_gemm_inner.cuh`: the EXACT harness-twin B-load remap (verified:
    patched file == `kernel_study/gemv_bench/build_prefill/
    exl3_gemm_inner_pf.cuh` modulo the flag plumbing; 20/20 GPU bit-exact
    proof carries over). Covers `exl3_gemm` + `exl3_moe` + `exl3_mgemm`.
  - `exl3_gemv_kernel.cuh`: QTIP small-m GEMV reader — NOT in the audit's
    component list but REACHABLE at K=2 for every m<=8 call (heuristic:
    `if (K == 2) return size_n <= 8192 ? 0 : 1;` — always eligible), i.e.
    the LinearEXL3 decode fallback would misread G8 without it. DEC5 pattern
    adapted to 8-tile pack groups.
  - host entries `exl3_gemm.cu`/`exl3_gemv.cu`: G8 dims branch (stock derives
    `K=B.size(2)/16`, `size_n=B.size(1)*16` — both wrong on a G8 trellis) +
    `pfg8::apply()` once; mgemm n-slice mode fails loud under G8.
  - `exl3_moe.cu`: apply-once.
  - `bindings.cpp`: `pf_g8_set`/`pf_g8_get` exports + `pf_g8_build` marker.
  - `modules/quant/exl3.py`: `LinearEXL3.K` derive for G8 last-dim 128*K;
    reconstruct branch (stock-layout decoder) never taken under G8.
- `pfg8_kernels_vllm.py` — component 7 + 8:
  - `p2b_moe.cu run_gemv_tile` (the decode reader; CFG=1 WNT=4 → pack group
    = group>>1) and `p2b_batched.cu p2b_run_gemv_tile_2` (WNT=2 → group>>2):
    DEC5 group-major addressing, word-stream identical to stock.
  - fat-GEMM `exl3_fat_gemm[_scatter]`: fail-loud TORCH_CHECK under G8
    (audit R3). PRIMARY gate is loader-side (sitecustomize sets
    `VLLM_EXL3_FAT_THRESHOLD=2^30` under DSV41_LOAD_PF_G8=1 unless the
    operator set it) — no expert is ever routed fat; all rows go through the
    G8-aware `exl3_moe` kernel. Belt and braces.
  - `bindings.cpp`: same exports/marker.
  - NOTE: `pf_g8`/`pfg8::` in vllm_exl3_c come from the patched exllamav3
    headers via EXL3_EXT_INCLUDE (ONE registry per .so; building against an
    unpatched exllamav3 fails to compile — the desired guard).

## Flag mechanics (audit component: env gating)

`DSV41_LOAD_PF_G8` read ONCE per process (`pfg8::env_value`, static-cached);
`pfg8::apply()` (std::call_once at first kernel dispatch) pushes the value
into every registered TU-local `static __constant__ int pf_g8` via
`cudaMemcpyToSymbol`. Env unset → 0 → constants initialized 0 at module load,
zero writes, zero per-step cost, stock layout path 100% untouched
(bit-identical arithmetic to the unpatched build). `pf_g8_set(v)` (pybind)
forces + pushes for A/B testing.

## Rollback

`IMAGE=dsv41-flash-exl3-sm121:canonical-e12 ./serve.sh` (untagged, still
present); boot-g8.sh on stock = results/2026-09-21-gatherv2/boot-k3c-pf-gv2.sh.

## Validation status

| Check | State |
|---|---|
| Patch scripts lint + idempotent on pinned trees (cb2 sha bf8df786… == image; vllm-exl3-patched == post-widen d3cfd39) | PASS (host, 2026-09-22) |
| Patched `exl3_gemm_inner.cuh` == proven twin modulo plumbing | PASS (diff verified) |
| Container patch apply (both repos, build log) | PASS |
| `import exllamav3_ext; pf_g8_set` assert | PASS ('exllamav3 g8 ok', build log) |
| `import vllm_exl3_c; pf_g8_set` assert | PASS ('vllm_exl3 g8 cuda ok', build log) |
| `strings` pf_g8 marks (boot-g8 guard): exllamav3_ext 3580, vllm_exl3_c 17 | PASS (guard needs >=1) |
| CPU import, flag default 0, markers pfg8-port-v1 both .so | PASS |
| env DSV41_LOAD_PF_G8=1: flag reads 1 both .so; loader 'already present'; fat gate sets VLLM_EXL3_FAT_THRESHOLD=2^30 | PASS |
| env unset: threshold untouched, loader dormant | PASS |
| GPU bit-exactness vs stock kernels (`results/2026-09-21-pfg8/pfg8_bitexact_gate.py`: exl3_gemm m∈{64..512}, exl3_gemv + p2b_fused_moe m∈{1,2,4,8}, gate/up + down shapes, torch.equal) | SCHEDULED post-04:00 (GPU window after pack rebuild exits) |
| Boot smoke 17*19=323 + A/B e2e | after image + pack both ready |

## Post-04:00 GPU validation (exact commands)

```bash
# only after the rebuild containers exit (docker ps shows no GPU containers)
docker run --rm --network none --entrypoint bash \
  dsv41-flash-exl3-sm121:canonical-g8 -c \
  'strings /usr/local/lib/python3.12/dist-packages/exllamav3_ext.*.so | grep -c pf_g8; \
   strings /usr/local/lib/python3.12/dist-packages/vllm_exl3_c.*.so | grep -c pf_g8'
# then, with GPU:
docker run --rm --gpus all --network none \
  -v $PWD/results/2026-09-21-pfg8:/g -w /g \
  dsv41-flash-exl3-sm121:canonical-g8 python3 /g/pfg8_bitexact_gate.py
```

## Build result

- Image tag: `dsv41-flash-exl3-sm121:canonical-g8`
- Image ID: `sha256:79b4d0c572a187902ec18db6f174afb9bfd1181532541a771f0248e57426182e`
- Built: 2026-09-22 01:36 BST (build wall ~6.5 min: exl3 183.7s + vllm-exl3 289.4s + wiring)
- Build log: `/tmp/pfg8-image-build.log` (host); both in-build import asserts green.
- canonical-e12 untouched (rollback intact).
- boot-g8.sh now points at canonical-g8 (guard passes: 3597 combined pf_g8 marks).
