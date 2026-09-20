# PREFILL GATE — group-major pack permutation no-regression proof

Status: **harness built + CPU-validated (nvcc compile OK, sm_121a, no-GPU
container); GPU run pending a maintenance window.**

## What this gates

The group-major trellis pack permutation is the last p2b decode lever
(RESULTS.md round 9/10: +6.0% cold decode, bit-exact, `bench5.cu` DEC5).
It is gated on a proof that the **prefill** side does not regress: the
exllamav3 prefill readers re-index the pack, and E8 taught us that
decode-shaped wins can silently cost prefill. This harness measures the
exllamav3 `exl3_gemm` tiled kernels (the prefill MoE path) on stock vs
group-major layout at prefill row counts.

## Layout under test

**PF-G8**: trellis `[k/16, n/16, 16·K] → [n/128, k/16, 128·K]` — groups of
8 n-tiles major, per-k-block rows 512 B contiguous (K=2). Chosen over the
decode-side G4 (`[group of 4][k][128]`, driver5 DEC5) because 8 % 4 == 0:
every aligned 4-tile run stays contiguous for the p2b decode warp stream,
so **one permuted pack serves both readers**. The GEMM patch is a pure
B-load address remap (`load_b_gl` / `gl_b_ptr` / k-stride in
`exl3_gemm_kernel_inner`); math order is untouched → bit-exact vs stock.

## Harness

- `make_bench_prefill.py` — generates the patched kernel headers +
  `bench_prefill.cu` from `kernel_study/cb2` (= image
  `exllamav3_ext/quant` sources) with a runtime `__constant__ int pf_g8`
  layout flag; both variants live in one compiled module.
- `build_prefill.sh` — CPU-only build path: throwaway **no-GPU container**
  from `dsv41-flash-exl3-sm121:latest` (the serve image's pip cu13 crt
  headers break its own nvcc stubs; the bench image has the full toolkit).
  Never JIT on the host while the serve is resident (UMA RAM), never in the
  live serve container (GPUs attached). **Verified: BUILD OK.**
- `driver_prefill.py` — check + sweep. m ∈ {64,128,256,512} rows/expert;
  matrices gate/up `[m,5120]×[5120,2304]` (GEMM shapes 2,3) and down
  `[m,2304]×[2304,5120]` (shapes 2,3,4); TP=2 local dims, 384 local
  experts, 37 routed layers for the layer estimate. Reports us/GEMM and
  total prefill-layer estimate per layout.

Chunked prefill at 8192 tokens / 384 local experts ≈ 21 rows/expert on
average; fat experts take far more — the sweep measures the fat tail that
dominates layer time.

## Bit-exactness

Same logical weights in both layouts (G8 is a pure storage reindex).
`--mode check` runs every (matrix, m, shape) through the SAME kernels with
only B-load addresses differing and requires `torch.equal` — the driver5
criterion. Bit-exactness at m≤16 additionally follows structurally: the
kernel is identical, and at m>16 the m-loop slices into ≤16-row inner
calls whose scheduling is layout-independent.

## How to run in a maintenance window

1. Serve down (`./stop.sh`), GPUs free.
2. Host: `kernel_study/gemv_bench/build_prefill.sh` (~3 min; only needed
   if the built `.so` is absent — the extension cache in the image persists).
3. In a GPU-enabled container of `dsv41-flash-exl3-sm121:latest` (or host
   torch if the serve is down):
   `python3 kernel_study/gemv_bench/driver_prefill.py --reps 5`
4. Expected runtime: check ~1 min + sweep ~4 m × 5 shapes × 2 layouts ×
   (50 iters × ~0.3-3 ms) ≈ 5-10 min total.

## Gate criteria (numeric)

For each matrix and each m ∈ {128, 256, 512}:

- **PASS**: median us/GEMM(PF-G8) ≤ 1.05 × median us/GEMM(stock) at every
  shape, i.e. group-major never runs >5% slower; and
  `--mode check` is BIT-EXACT at every (matrix, m, shape); and
  the aggregate layer estimate is not >2% worse (noise band for summed
  medians).
- **FAIL**: any (matrix, m ∈ {128,256,512}, shape) median slower by >5%,
  or any check mismatch.
- m=64 is advisory (below it the QTIP-style GEMV path usually takes over);
  it does not veto.

Decision rule feeding the arm:

- **PASS** → schedule the permutation arm: fold PF-G8 into
  `tools/quantize_experts_exl3.py` (`pack_trellis` output view) at the
  next pack build, re-index p2b (DEC5 pattern, G8-aware) and the
  exllamav3 gemm/gemm-inner readers, rebuild image, A/B e2e
  (8/8 correctness + 3/3 quality + prose decode/prefill bands).
- **FAIL** → close the permutation lever permanently; keep p2b stock
  layout. Decode-side +6.0% does not justify a prefill regression
  (E8 lesson); record in RESULTS.md.

## Pack-permutation apply path

`tools/permute_pack_group_major.py` (dry-run by default, `--apply` writes
a permuted sibling copy with per-shard sha256 + mapping manifest).

Measured plan: 46 080 trellis tensors across 48 shards; permuted copy =
358.1 GB (same-size re-layout). Free space on the pack filesystem is
currently ~609-653 GB, so a one-off copy WOULD fit — but it leaves the
machine at ~88% disk and requires loader-side re-indexing anyway.

**Recommendation: quantize-time permutation** — fold the G8 view into
`tools/quantize_experts_exl3.py` right after `pack_trellis(...)` (a single
`.view(kt, g, 8*w).permute(1,0,2).contiguous()` on the packed trellis;
zero extra disk, zero extra GPU time, and the rebuild is already required
to produce a new revision). Use the copy script only as verifier/fallback
if an existing pack must be converted without a requant.
