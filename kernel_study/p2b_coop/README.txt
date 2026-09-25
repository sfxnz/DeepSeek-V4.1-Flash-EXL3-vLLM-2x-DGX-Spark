p2b coop microbench (moe-expert-dedup step 2b, DSV41_P2B_COOP)
===============================================================

What
----
docker/patch/widen_p2b_coop.py adds a SORT=2 instantiation of the live p2b
kernel (K=2 MCG only). It reads each unique expert's trellis once per call:

- Every block builds the src-sorted (row, expert) order (p2b_sort_build from
  widen_p2b_srcsort.py) and cuts each run of equal expert into chunks of <= 8
  members (p2b_coop_build).
- Gate/up and down run one tile per (chunk, group[, gate|up]). Lane group
  r = lane / 4 loads member r's activation into MMA row r. p2b already issues
  the m16n8k16 MMA with only row 0 live, so the MMA count, registers and the
  B prefetch ring are unchanged; one B stream now feeds up to 8 rows. This is
  the upstream exl3_gemv MMODE 1 row layout, not widen_p2b_mma.py (that one
  looped a separate MMA + fp32 accumulator set per row and lost to spills).
- The final slot sum is fixed-order, no atomics: SORT=2 is deterministic.
  Same products as p2b (__fmul_rn), so one-hot routing weights give the same
  bits; full weights differ only by fp32 slot-sum order before the fp16 store.
- Same launch, grid, scratch and host checks as p2b: one cooperative launch,
  no allocation or host sync added (CUDA-graph capture as today).
- Off (unset / not "1"), K != 2, cb != 1 or m*K > 64: SORT=0/1, whose machine
  code is byte-identical to canonical-e13 (sass_identity.sh).

Why not the upstream exl3_moe_coop (docker/cooperative/upstream, 02aef45)
- 3 launches with completion counters (re-zeroing under graphs unverified),
  a 512-thread WK=16 PF=4 tile (GB10 moved p2b to 256 threads for occupancy),
  fp32 output, its own scratch/ABI; needs an exllamav3 bump or a standalone
  build of 3+ TUs.
- Its SILU clamp is min(silu(g), L) (MOE_COOP_ACT_SILU); p2b/vLLM clamp g
  before silu. SILU_OAI has alpha 1.702 and (u + 1). Neither is p2b's math.
- The coop-e1 image (docker/cooperative) hardcodes <K, 2> (MUL1) and cannot
  serve the MCG pack.

Files
-----
make_bench.py     no GPU. build/chain_srcsort.cu (live chain), build/chain_coop.cu
                  (+ coop, what Dockerfile.e14 compiles), build/bench_coop.cu
                  (runtime set_coop(0|1) + occupancy() binding, no env).
sass_identity.sh  no GPU. One --network none --memory 8g --cpus 4 container on
                  canonical-e13: compiles the three TUs for sm_121a, requires the
                  12 SORT=0/1 kernels byte-identical and exactly one new kernel
                  <2,1,2>, reports inner-loop spills (host cuobjdump), and checks
                  the pin chain against the image's vllm_exl3_c.
text_identity.py  the SORT=0/1 identity check (reuses ../p2b_srcsort).
hot_loops.py      inner MMA loop stats from cuobjdump -sass.
driver.py         GPU, serve DOWN: checks + capture + cold/warm timing.

CPU result (2026-09-25, results/2026-09-24-review/moe-dedup/coop-sass-identity.txt)
- 12/12 SORT=0/1 kernels IDENTICAL; new <2,1,2> 86912 B.
- <2,1,2>: 64 registers, 40 B stack, spill 56/140 B (= p2b <2,1,0>),
  17668 B static smem (p2b 2048). Inner loops: gate/up 591 instr, 3 LDL
  (p2b 565, 1); down 582 instr, 0 LDL (p2b 578, 1).
- The pin chain equals the image's vllm_exl3_c for all 12 p2b kernels.

GPU window (serve DOWN on both nodes; exclusive GPU)
----------------------------------------------------
0. Pre-flight. From the serving worktree (perf-review-0924):
     ./stop.sh
     docker ps --format '{{.Names}} {{.Image}}'; ssh spark2 docker ps   # no GPU containers
     free -h
1. CPU re-check (optional, ~75 s):
     cd /home/sfxnz/projects/ai-lab/recipes/.worktrees/r2-coop
     kernel_study/p2b_coop/sass_identity.sh
2. Microbench m=4 (DSpark-3 verify, one sequence). JIT-builds bench_coop.cu
   (~1-2 min), allocates ~1.7 GiB of random trellis:
     docker run --rm --gpus all --network none --memory 16g --cpus 8 \
       -v "$PWD:/repo" -w /repo --entrypoint python3 \
       dsv41-flash-exl3-sm121:canonical-e13 kernel_study/p2b_coop/driver.py \
       --out results/2026-09-24-review/moe-dedup/coop-microbench-m4.json
3. Microbench m=8 (two sequences x 4 verify rows):
     same command with  --m 8 --out results/2026-09-24-review/moe-dedup/coop-microbench-m8.json
   Census windows at m=8 are 8 consecutive tokens of one request, an upper
   bound on the overlap of two independent sequences; read the synth rows too.
4. Exit code 1 = a check failed. Report check_pass, occupancy, and per source
   cold/warm reduction_pct and saving_ms_per_step_cold.

Gates (microbench -> serve arm)
- check, every source: onehot_bitexact (expected true; if false, onehot_max_ulp
  <= 1 and an explanation), full_max_rel <= 1e-3, coop_repeat_bitexact,
  outputs_finite. capture.replay_equals_eager true.
- occupancy_blocks_per_sm coop == p2b (expected 4 and 4).
- dup0 cold reduction_pct >= -2 (nothing to share must not cost).
- Serve arm only if census cold reduction_pct >= 5 (feasibility gate).

Expected (not measured)
- Unique ratio at the census: 1 - 0.2987 = 0.70 (m=4). Streaming share of p2b
  cold time ~0.78 (feasibility note), so cold coop ~ p2b x (1 - 0.30 x 0.78)
  = ~0.77 x p2b: ~525 -> ~405 us per call at m=4.
- Per step: 40 routed layers x ~120 us = ~4.9 ms of a ~63 ms verify step,
  ~7-8% (upper bound 0.2987 x 21.7 ms = 6.5 ms). At matched acceptance that is
  ~+8% tok/s (L.A.I.L ~34 -> ~36.5), more at c=2 where m=8.
- Risks: wave quantization (m=4 census: 612 gate/up + 1360 down items on
  ~192 blocks vs 864 + 1920), the 16 KB coop reduction buffer (smem per SM
  ~75 KB at 4 blocks), and 2 extra L1-resident spill loads per gate/up
  iteration. The microbench measures all three.

Serve arm (after the gates pass)
--------------------------------
Build (needs network for apt + git clone; CPU nvcc, ~2-5 min; run with the
serve down or >= 14 GiB MemAvailable):
  cd /home/sfxnz/projects/ai-lab/recipes/.worktrees/r2-coop
  docker build -f docker/Dockerfile.e14 -t dsv41-flash-exl3-sm121:review-e14 docker
  docker save dsv41-flash-exl3-sm121:review-e14 | ssh spark2 docker load
  for h in "" "ssh spark2"; do $h docker run --rm --network none --entrypoint bash \
    dsv41-flash-exl3-sm121:review-e14 -c 'grep -c DSV41_P2B_COOP /usr/local/lib/python3.12/dist-packages/vllm_exl3_c*.so'; done
ABAB per ARMS.md, one lever, same image for A and B (A = SORT=0, the canonical
code, byte-identical):
  A: ./stop.sh && AUDIT=strict IMAGE=dsv41-flash-exl3-sm121:review-e14 ./run.sh
  B: ./stop.sh && AUDIT=strict IMAGE=dsv41-flash-exl3-sm121:review-e14 DSV41_P2B_COOP=1 ./run.sh
  each boot: python3 smoke_chat.py (323); python3 smoke_vision.py;
             tools/four_numbers.sh --arm coop-{A1,B1,A2,B2}; tools/disarm_scan.sh
             (no "decode lever p2b_coop" line on either rank)
  B boots also: python3 tests/quality_eval.py --quick \
                  --baseline results/2026-09-24-review/quality-baseline/quick.json --out <arm-dir>/quality_quick.json
                python3 tests/quality_eval.py --full \
                  --baseline results/2026-09-24-review/quality-baseline/full.json --out <arm-dir>/quality_full.json
                (numerics change by design: fixed slot-sum order)
  optional engagement proof: one profiler boot (tools/profile_window.sh) shows
  p2b_moe_batched_kernel<2, 1, 2> in B, <2, 1, 0> in A.
Rollback: ./stop.sh, then the usual canonical-e13 boot (unset DSV41_P2B_COOP).

Round 3 (2026-09-25, branch k3/coop-moe): dataflow coop kernel, DSV41_P2B_COOP=2
================================================================================

What
----
docker/patch/widen_p2b_dataflow.py (after widen_p2b_coop.py) adds p2b_coop_df_kernel<2, 1>:
the round-2 coop tiles scheduled as one atomic task list (gate/up tiles, then down tiles)
with last-finisher epilogues and per-unit readiness counts, 2 grid barriers instead of 7,
3 blocks/SM (80 registers, spill-free MMA loops), L2 prefetch 2 k-slices past the register
ring, the A fragment one k-slice ahead, the first tile prefetched in the prologue, svh
prefetch and an unrolled top-6 output pass. Bit-identical to DSV41_P2B_COOP=1 for any routing
weights (one-hot: to p2b). docker/Dockerfile.e15 builds review-e15 (chain + coop + dataflow).

Harness (all under kernel_study/p2b_coop, force-added: kernel_study/ is gitignored)
-----------------------------------------------------------------------------------
make_bench_r3.py   CPU. build_r3/chain_r3.cu (= the e15 TU), bench_r3.cu (runtime variant
                   switch = DSV41_P2B_COOP value, bench-only read-ceiling kernels),
                   bench_r3_ts.cu (%globaltimer phase stamps + per-warp tile accounting).
bench_r3.py        GPU. Real layer weights from /hf (the pack), TP-sharded like vllm_exl3,
                   census routing of the same layer (source census) or of all 40 routed layers,
                   layer-stratified (census_all: the per-step mix). Modes: check (one-hot bitwise
                   vs p2b, bitwise vs =1, full-weight tolerance, repeat, fp64 reference with --ref),
                   capture (graph replay == eager), stress (race hunt vs =1, eager + graph),
                   time (cold flush / warm, alternating arms, median p10 p90, GB/s, % floor,
                   paired per-call savings mean +- 95% CI and x 40 layers per step),
                   phases, stream (tile-pattern/contiguous read ceiling), bw / ld / gemm
                   (plain read, load flavour and cuBLAS GEMV bandwidth probes).
prebuild_r3.sh     CPU. JIT-builds both bench modules in the serve image (no GPU in the window).
ptxas_r3.sh        CPU. ptxas resources + SASS + hot-loop spills of every K=2 MCG kernel.
sass_identity_r3.sh CPU. e14 TU vs e15 TU: SORT=0/1/2 machine code byte-identical (13/13),
                   exactly one new kernel.
spark2.sh / gpu_run.sh  sync the worktree to spark2, flock spark2's GPU lock, refuse to run
                   with a foreign GPU process (pmon), sample clocks every 250 ms, run in the
                   serve image, sync results back. PROFILE=1 runs Nsight Compute instead.
ncu_target.py      target process for ncu (flushed calls on the real layer).

Reproduce (spark1, serve down or not: spark2's GPU is used)
  kernel_study/p2b_coop/prebuild_r3.sh
  kernel_study/p2b_coop/spark2.sh final-validate kernel_study/p2b_coop/bench_r3.py \
      --mode check,capture,stress --variants 0,1,2 --ms 1,3,4,6,8 --check-routings 12 --ref \
      --replays 32 --stress-calls 1000 --out results/2026-09-25-kernels/coop-moe/final-validate.json
  kernel_study/p2b_coop/spark2.sh final-time kernel_study/p2b_coop/bench_r3.py \
      --mode time,phases,stream --variants 0,1,2 --ms 1,3,4,6,8 --sources census,dup0 \
      --iters 300 --phase-calls 40 --out results/2026-09-25-kernels/coop-moe/final-time.json
  kernel_study/p2b_coop/spark2.sh fix-tlayers kernel_study/p2b_coop/bench_r3.py \
      --mode time --variants 0,1,2 --ms 1,3,4,6,8 --sources census_all,census --iters 800 \
      --out results/2026-09-25-kernels/coop-moe/fix-tlayers.json
  kernel_study/p2b_coop/sass_identity_r3.sh
  Round-4 final suite (it 16): final4-validate (--mode check,capture,stress --check-routings 16 --ref
  --replays 48 --stress-calls 2000), the same on --layer 33 --tp-rank 1 without --ref, final4-time
  (--mode time,phases,stream --sources census,census_all,dup0 --iters 800), real_ext_check.py; the
  exact commands are the "# cmd:" lines of results/2026-09-25-kernels/coop-moe/runs/final4-*.log.

Results: results/2026-09-25-kernels/coop-moe/ (iterations.txt = every iteration with numbers;
summary.json = the final table from the round-4 final suite (final4-*.json): time_layer20 = per call
on layer 20, projected_ms_per_step_40_layers = the per-step projection from all 40 layers with its
three replicates, stream_ceiling = read ceilings of the same bytes, phases_v2 = barrier and prologue
stamps, round4_exploration = the rejected fixed-cost variants).

Serve arm (DSV41_P2B_COOP=2), not run yet
-----------------------------------------
Gates already met (microbench, re-run at the end of round 4: iterations.txt it 16): one-hot
bit-exact vs p2b, bitwise = DSV41_P2B_COOP=1, full weights <= 1 fp16 ulp vs p2b (max rel <= 4.9e-4),
fp64-reference error equal to p2b's, deterministic, graph replay bitwise (48 per m), 2000-call race
hunts per m bitwise, all on layer 20 rank 0 and layer 33 rank 1; the real vllm_exl3_c from the e15
recipe = the bench bitwise; SORT=0/1/2 SASS identical; no-sharing routing not slower (-7.2% at m=4).
1. Build (network for apt + git clone; CPU nvcc ~2-5 min; serve down or >= 14 GiB MemAvailable):
     cd /home/sfxnz/projects/ai-lab/recipes/.worktrees/k3-coop-moe
     docker build -f docker/Dockerfile.e15 -t dsv41-flash-exl3-sm121:review-e15 docker
     docker save dsv41-flash-exl3-sm121:review-e15 | ssh spark2 docker load
     for h in "" "ssh spark2"; do $h docker run --rm --network none --entrypoint bash \
       dsv41-flash-exl3-sm121:review-e15 -c 'grep -c "p2b coop dataflow kernel engaged" \
       /usr/local/lib/python3.12/dist-packages/vllm_exl3_c*.so'; done
2. ABAB per ARMS.md, one lever, the same image for A and B (A = env unset: canonical-e13's
   p2b SORT=0 code, byte-identical), spark1's GPU exclusive (no chromium GPU process):
     A: ./stop.sh && AUDIT=strict IMAGE=dsv41-flash-exl3-sm121:review-e15 ./run.sh
     B: ./stop.sh && AUDIT=strict IMAGE=dsv41-flash-exl3-sm121:review-e15 DSV41_P2B_COOP=2 ./run.sh
   each boot: python3 smoke_chat.py (323); python3 smoke_vision.py;
              tools/four_numbers.sh --arm coopdf-{A1,B1,A2,B2}; tools/disarm_scan.sh
   B boots: the strict audit must find "dsv41: p2b coop dataflow kernel engaged (DSV41_P2B_COOP=2)"
            on head and worker and no "p2b coop dataflow lever is OFF" / "decode lever
            p2b_coop_dataflow" line; then
            python3 tests/quality_eval.py --quick --baseline results/2026-09-24-review/quality-baseline/quick.json \
                --out <arm-dir>/quality_quick.json
            python3 tests/quality_eval.py --full --baseline results/2026-09-24-review/quality-baseline/full.json \
                --out <arm-dir>/quality_full.json
            (numerics differ from p2b only by the fixed fp32 slot-sum order, <= 1 fp16 ulp)
   Engagement proof (one boot): tools/profile_window.sh shows p2b_coop_df_kernel<2, 1> (grid 144)
   in B and p2b_moe_batched_kernel<2, 1, 0> in A.
3. Expected: p2b 23.2 ms/step (c=1 profile) -> ~16.4 ms (-6.8 ms of ~63: ~+12% tok/s at matched
   acceptance; -1.8 to -1.9 ms/step vs DSV41_P2B_COOP=1); c=2 -15.2 to -15.3 ms/step. Basis: census
   routing of all 40 routed layers drawn layer-stratified, mean of the paired per-call savings x 40,
   three independent 800-iteration runs: m=4 6.77 / 6.76 / 6.81 (+- 0.16), m=8 15.27 / 15.17 / 15.34
   (+- 0.28) ms/step (fix-tlayers.json, review r4, final4-time.json). The earlier -7.4 / -16.2 were
   layer 20's median saving x 40, biased high: layer 20 shares more experts (dup 0.344 at m=4) than
   the 40-layer mean (0.2987). Accept on ms/step at matched acceptance (L.A.I.L n=10 and bench
   c=1/c=2) with the ARMS.md noise gate; quality quick+full within baseline bands.
Rollback: ./stop.sh, canonical-e13 boot with DSV41_P2B_COOP unset.

Round 4 fix pass (2026-09-25): the review's missed opportunities, measured; nothing promoted
--------------------------------------------------------------------------------------------
iterations.txt it 15 (results x1-x6). One bench-only exploration kernel derived from the production
text, every knob compiled in and passed as a kernel argument, so all arms share machine code and a
control arm isolates each knob. Bitwise gates pass for every arm; none clears the promotion bar
(>= 0.5% of the call at m=4 and at m=8, nothing slower elsewhere): 32-column tail tiles (bitwise,
slower: per-tile fixed work dominates), whole-K tail prefetch and a deeper prologue prefetch (no
change), early Hadamard loads (+0.74% at m=4, +0.36% at m=8). Read ceiling of the same bytes: a
flat 16-B sweep 233 / 237 GB/s at m=4 / 8, the tile pattern 224 / 231; closing that gap needs a
layout change (flags.md Round 9, group-major trellis), not a kernel change.

Next lever after the DSV41_P2B_COOP=2 serve ABAB: fold the wrapper glue (not built)
  Serve trace (review r4: c=1, rank 1, 10,000 p2b calls): top-k end -> p2b start takes 73.8 us
  through 16 routing-stream kernels (map_topk_to_local, clamp, int32 / fp16 casts, valid mask,
  bf16 -> fp16 x) that overlap the side-stream shared expert and end ~7 us after it; after p2b,
  fp16 -> fp32 -> bf16 and the shared-expert add take 6.5 us. Bound ~10 us/layer, ~0.4 ms/step at c=1.
  1. Output: write bf16 from the output pass (bf16_rn(float(half_rn(s))) is the bits of the
     fp16 -> fp32 -> bf16 chain) and hand it from _apply_native_fused_moe to apply_exl3_experts.
     Removes 2 of the 3 post-kernel launches on the same stream; no scheduling change.
  2. Input: fold the int64 -> int32 clamp and valid mask, the fp32 -> fp16 weight cast and the
     bf16 -> fp16 x cast into the prologue (the same conversions). Keep p2b ordered after the shared
     expert (an explicit join before the launch): the 144-block cooperative grid uses 61,440
     registers/SM and cannot co-reside with b12x CTAs, so launching it while the shared expert runs
     could park the shared down GEMM behind the whole p2b call.
  Measure per call with a profile boot (shared-down end -> p2b start, p2b end -> add end), ABAB
  against DSV41_P2B_COOP=2 alone (ARMS.md: one boot = one lever).
