# Step 2b feasibility: upstream `exl3_moe_coop` (cb=1 MCG) vs p2b

Status: note only. Nothing is built, no image changed.

## What exists

- The vendored tarball `docker/cooperative/upstream/exllamav3.tar.gz`
  (exllamav3 @02aef45) already ships cb=1 MCG coop:
  - `quant/exl3_moe_coop.cu:243` has `cb = mcg ? 1 : (mul1 ? 2 : 0)`.
  - `comp_units/exl3_moe_coop_inst_k1..k8.cu` instantiate cb 0, 1 and 2.
  - No 126-commit plugin bump is needed to test it.
- The kernel groups slots by expert. Each run of up to `ROWS=8` slots becomes
  the rows of one m16 MMA, so a run's weights are read once.
- At bsz>1 it takes **3 launches**: `rot`, then A (gate/up), then B (down).
  Completion counters replace grid.sync. The final slot sum runs in a fixed
  order, so the kernel is deterministic but **not bit-exact vs p2b**.
- The activation clamp matches p2b: `g = min(g, L)`, `u = clamp(u, -L, L)`
  (`exl3_moe_coop_kernel.cuh:104-105`). V4.1 uses L=10.
- Pointer tables are `int64` tensors of `data_ptr()` (`tests/test_moe_coop.py:42`),
  the same layout p2b uses. A kernel_study arm can share the weights in
  `kernel_study/p2b_srcsort/driver.py`.
- The live image pins exllamav3 5be8865 (`docker/Dockerfile:18`), which has no
  `exl3_moe_coop`. The local `docker/cooperative` build is Mia's goal50 C ABI:
  - hardcoded `<K,2,...>` (MUL1; `native/cooperative_moe.cu:20-23`);
  - `runtime.py:layer_eligible` reads `_exl3_mul1` / `_exl3_mcg`, which our
    plugin never sets (it sets `_exl3_codebook_flags`), so it is inert on
    canonical-e12.

## Cheapest probe (recommended first; no image build)

A standalone kernel_study A/B, coop vs the p2b chain, at m=4, cold:

1. Stage the tarball's `exllamav3_ext` into `kernel_study/coop_cb1/build/`:
   - `quant/exl3_moe_coop.cu`
   - `quant/comp_units/exl3_moe_coop_inst_k2.cu`
   - `quant/exl3_devctx.cu`
   - the headers they include (`util.h`, `util.cuh`, `compat.cuh`,
     `exl3_gemv_kernel.cuh`, `hadamard_inner.cuh`, `codebook.cuh`, ...)

   Put only this tree on the include path. Do not mix it with the image's
   5be8865 headers.
2. Write a pybind shim that binds `exl3_moe_coop`. The upstream declaration
   is `bindings.cpp:259`.
3. Size the scratch per `tests/test_moe_coop.py:128-134`:
   - `gu_g` / `gu_u` / `act_out`: `[smax,1,I]`
   - `d_out`: `[smax,1,H]` f32
   - `ctr`: `smax*(I/128) + 8*(H/128) + 2*smax + 3` int32, zeroed
   - `had_g` / `had_u`: `[slots,H]`
   - `sel` is int64 `[4,6]`
4. Add a `coop` arm to the driver with the same weights and routings.
   - Correctness: compare the f32 output against p2b with a tolerance
     (max abs diff on one-hot weights at < 1e-2 relative). Bit-exact is
     impossible because the MMA-over-rows accumulation order differs.
   - Timing: cold and warm at dup {0, 25, 50}%, alternating arms.

Cost:

- about 0.5-1 day of code;
- CPU compile of 3 TUs in a no-GPU container (like `sass_identity.sh`),
  ~5-10 min;
- GPU run with the serve down, ~10 min.

## Serve probe (only if the harness passes)

Two routes:

- **(a) Bump exllamav3 to 02aef45 in `docker/Dockerfile`.**
  - Re-verify the `patch_exllamav3_aarch64.py` and `pfg8_*` anchors on the
    newer tree.
  - Wire an exl3.py patch: for m <= 8, call `ext.exl3_moe_coop` instead of
    `p2b_fused_moe`, env-gated like `DSV41_P2B_SRC_SORT`.
  - Preallocate the scratch and zeroed counters before graph capture.
    Counter re-zeroing under breakable CUDA graphs is unverified.
  - Full image build ~45-60 min; boot ~15-20 min; then L.A.I.L n=10, prose,
    correctness 8/8, NLL.
- **(b) Re-instantiate `docker/cooperative` with `<K,1,...>`.**
  - Relax `layer_eligible` to `_exl3_codebook_flags`, and convert ids to
    int64.
  - Rebase `Dockerfile.coop` onto canonical-e12. It is FROM Mia's image
    today.
  - About 1-2 days, with a similar build and boot cost.

## Expected gain and risk

- **Gain comes only from fewer weight bytes.**
  - p2b already streams at ~76-78% of UMA peak cold (206-212 GB/s).
  - The "-1.2 ms barrier floor" is unsupported. `evidence/p2b-nocoop` lost 5%
    with 9 launches instead of 8 grid.syncs.
  - Gain is therefore about dup x 0.78 x 22.4 ms/step. That is ~0.4 ms (<1%)
    at the random-routing dup of 2.3%, and ~4.4 ms (~6.6% of a 66 ms step)
    at 25%.
- **Risk is high on sm_121.**
  - MMA-over-same-expert-rows is the widen_p2b_mma design class, which lost
    10.2 vs 15.1 tok/s on GB10 (register pressure).
  - The upstream speedups (1.58-1.73x) are against stock `exl3_moe` on sm80,
    not against p2b. p2b is already ~1.23x over `exl3_moe` at m=4.
  - The coop tile is `WK=16` (512-thread blocks) with `PF=4`. That is p2b's
    CFG=0 shape. GB10 moved p2b to CFG=1 (256 threads, `WK=8`, `PF=2`) for
    occupancy, and a 512-thread block fits at most 3 per SM
    (`evidence/p2b-cfg1.md`). Deeper prefetch rings were also rejected
    (flags.md PF 2/4/8 sweep).

## Gate

Build the harness only if both hold:

- census mean dup >= 10% (`tools/moe_census.py`);
- src-sort (2a) leaves more than 5% of p2b time on the table relative to the
  dup x 0.78 ceiling.

Boot only if coop is >= 5% faster cold at m=4 at the census dup, and the
one-hot tolerance check passes. Then accept under ARMS.md step 6 (ABAB
boots, ms/step at matched acceptance, noise-aware gate, quality quick and
full), with smoke 323. Not against a fixed number from another day.
