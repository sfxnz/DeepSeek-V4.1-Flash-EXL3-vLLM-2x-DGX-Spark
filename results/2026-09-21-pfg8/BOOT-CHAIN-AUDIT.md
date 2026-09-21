# G8 BOOT-CHAIN AUDIT — 2026-09-21 (CPU-only dispatch, repo @ 2931012)

Scope: everything required to SERVE `snapshots/2.0bpw-mcg-g8` (writing now on
both ranks, WATCH.md). Rule from REBUILD-PLAN.md: *a G8 pack on a stock image
misreads trellis* — this audit names every component that must be in place,
its state, and what was wired/staged this session.

## Component table

| # | Component | State | Evidence |
|---|-----------|-------|----------|
| 1 | G8 pack `2.0bpw-mcg-g8` | building (10/20 shards spark1 at audit time) | WATCH.md; shard-3 layout+data PASS (`shard3-verify.md`: w1/w3 `(18,320,256)`, w2 `(40,144,256)`, reconstruct relerr parity 0.3775-0.3779 vs stock 0.3771-0.3775) |
| 2 | Quantize-time fold `DSV41_PACK_PF_G8=1` (`tools/quantize_experts_exl3.py` `_pfg8_fold`) | **in-image, ACTIVE for the rebuild** (env is per-process; image itself unchanged) | commit 3242152; WATCH.md; shard-3 folded shapes |
| 3 | Loader re-index `docker/patch/pfg8_loader_reindex.py` (TP narrow-dim swap under `DSV41_LOAD_PF_G8=1`) | **staged → WIRED this session** (sitecustomize installs it at boot; fail-loud on anchor mismatch) | patch idempotent, anchors verified byte-exact against image (sha256 `afc6da3e…` of image `vllm_exl3/exl3.py` == `kernel_study/cb2_vllm_exl3/exl3.py`); chain-apply + py_compile PASS |
| 4 | `DSV41_LOAD_PF_G8` env passthrough | **was MISSING → wired this session** | run.sh: added to `env_args` (`-e DSV41_LOAD_PF_G8=${DSV41_LOAD_PF_G8:-0}`) AND to the spark2 ssh worker-forward line; `bash -n` OK; `kit/render.py --check` OK (var is hand-wired like the other DSV41 levers, outside the generated block) |
| 5 | sitecustomize wiring for the loader patch | **was MISSING → wired this session** | `docker/patch/sitecustomize.py` new block: env-gated install, prints `dsv41: pfg8 loader re-index installed/already present`, `os._exit(1)` on failure (never silently serve unpatched); py_compile PASS |
| 6 | Prefill GEMM G8-awareness (exllamav3 `exl3_gemm` B-load remap, runtime `pf_g8`) | **NOT in image — harness twin only** (`kernel_study/gemv_bench/build_prefill/exl3_gemm_inner_pf.cuh`, k-stride `TILEBLOCKS_K*8*16*bits`, bug 9f71b6e fixed) | `strings` on image `exllamav3_ext.*.so`: 0 `pf_g8` hits; REBUILD-PLAN item 3 "NOT yet written into the image build" |
| 7 | p2b decode G8 reader (`vllm_exl3_c` p2b kernels) | **NOT in image — bench proof only** (`bench5.cu` DEC5 group-major pattern, bit-exact +6.0% cold, RESULTS.md R9) | `strings` on image `vllm_exl3_c.*.so`: 0 `pf_g8` hits; no docker/patch script ports it; `widen_p2b_*` patches are perf-only, layout-stock |
| 8 | Fat-GEMM / reconstruct fallback readers (`exl3_fat_gemm.cu[cuh]`, `ext.reconstruct`) | **stock-layout, NOT G8-aware, no twin exists** | `kernel_study/vllm-exl3-patched/csrc/exl3_fat_gemm.cuh` unmodified upstream; reached when experts exceed `VLLM_EXL3_FAT_THRESHOLD=256` rows or distinct_suh |
| 9 | Boot script | **staged this session** | `results/2026-09-21-pfg8/boot-g8.sh` — copy of boot-k3c-pf-gv2.sh + `SNAPSHOT_SHA=2.0bpw-mcg-g8` + `DSV41_LOAD_PF_G8=1` + kernel-readiness guard (aborts if no `pf_g8` symbols in either .so) + 48-shard pack guard + PRE-BOOT CHECKLIST in comments; `bash -n` OK |
| 10 | Rollback pack `2.0bpw-mcg` | **intact** | 48/48 shards present, shard-3 mtime 2026-09-10 15:10 (untouched by the G8 build, which writes only into `2.0bpw-mcg-g8`) |

## Loader re-index math — verified against shard-3 layout at TP=2

Fold `view(kt, nt/8, 8*w).permute(1,0,2)`: gate/up stock `(320,144,32)` →
`(18,320,256)` ✓ matches shard3 w1/w3; down stock `(144,320,32)` → `(40,144,256)` ✓
matches shard3 w2. TP=2: gate/up G8 dim0 = 18 groups → 9/rank = 72 n-tiles ==
stock dim1 144/2 (groups never straddle ranks: 8-tile runs stay whole ✓);
down G8 dim1 = 144 k-tiles → 72/rank == stock dim0 144/2 ✓. Both splits whole,
`_narrow_tp` divisibility holds (18%2=0, 144%2=0). Patch swaps col dim 1→0 and
row dim 0→1 only when `DSV41_LOAD_PF_G8=1`; default exact stock. **Math verified
numerically this session** (script in scratch, all True).

## Decode-side bit-exactness — RISK (named, unresolved)

- PREFILL GEMMs: bit-exact PROVEN for the exact algorithm that must be ported
  (GPU 20/20 `torch.equal`, `PREFILL-GATE-RESULT-2026-09-21.log`) — but on the
  **harness twin** (`exl3_gemm_inner_pf.cuh` generated from cb2 sources), not
  on kernels compiled into an image. Port-to-image must preserve the exact
  remap (k-stride!) and re-run an m≤16 decode-shaped + m≥64 prefill-shaped
  `torch.equal` against the stock image kernel before serve.
- p2b DECODE GEMV: bit-exact proven only in `bench5.cu` (DEC5 pattern,
  standalone). **There is NO bit-exact proof for the image `vllm_exl3_c` p2b
  path on G8 layout, and no port of the reader exists at all.** This is the
  largest correctness gap: decode serves through p2b (`VLLM_EXL3_MOE_KERNEL=native`).
- Fat-GEMM path (experts >256 rows prefill): reads trellis with stock
  addressing and has NO G8 twin — if fat experts occur in a prefill chunk on a
  G8 pack, garbage. Port must cover it or the loader must disable the fat
  kernel (`use_kernel=False` / raise threshold) for G8 packs.
- The boot smoke gate (17×19→323) catches gross misreads but is NOT a
  bit-exactness proof; REBUILD-PLAN item 4's A/B e2e (8/8 correctness, 3/3
  quality) remains mandatory before the G8 pack replaces 2.0bpw-mcg.

## Exact boot sequence (once image kernels land)

1. Rebuild ranks finish + `tools/assemble_pack.sh` (CODEBOOK=mcg REV=2.0bpw-mcg-g8)
   + `tools/permute_pack_group_major.py <pack>` dry-run verification.
2. Image rebuild dispatch ports pf_g8 into exllamav3_ext + vllm_exl3_c builds
   (runtime flag, default off) + covers/disables the fat-GEMM path; boot log
   markers to expect listed in boot-g8.sh checklist.
3. `bash results/2026-09-21-pfg8/boot-g8.sh` (guards: pf_g8 symbols ≥1,
   48 shards present). Loader install line expected in `docker logs` both
   ranks: `dsv41: pfg8 loader re-index installed|already present`.
4. Watch weight load for `EXL3 TP shard ... not divisible` / shape mismatch
   (would mean the loader swap is not engaged).
5. Smoke: '17 * 19 = ? Step by step, then answer.' → 323. Then the A/B e2e.

## Rollback (confirmed)

Stock pack `snapshots/2.0bpw-mcg` untouched (48/48, mtime 2026-09-10).
`./stop.sh` then `bash results/2026-09-21-gatherv2/boot-k3c-pf-gv2.sh`
(SNAPSHOT_SHA defaults to 2.0bpw-mcg; DSV41_LOAD_PF_G8 unset → loader dormant
→ stock addressing). The run.sh/sitecustomize additions are default-off
no-ops for non-G8 boots (env defaults to 0; install block only runs under
DSV41_LOAD_PF_G8=1).

## Validation performed this session (CPU-only, no GPU containers)

- Loader patch chain-apply on image-snipped `vllm_exl3/exl3.py`: anchors match,
  markers at L464/L477, py_compile OK, idempotent re-run OK.
- Full boot patch chain in sitecustomize order on image snips
  (indexer_workspace → census → fast → prefetch v3 → cpu-hash → gather v2 →
  defer → pfg8): all install, all py_compile OK, hooks coexist in
  model_runner.py (cpu-hash @1693/2013, defer @2026 — post-cpu-hash as
  required by the defer chain).
- sitecustomize wiring simulated with env set on a fresh image snip: installs,
  prints marker, idempotent; py_compile OK.
- `bash -n` boot-g8.sh + run.sh; `kit/render.py --check` clean (generated
  blocks untouched).
