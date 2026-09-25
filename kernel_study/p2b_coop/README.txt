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
