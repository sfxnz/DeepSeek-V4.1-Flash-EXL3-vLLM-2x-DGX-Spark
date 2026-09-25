# Decode levers: microbenches and campaign arms

This directory covers three env-gated decode levers. All three are off by default. The GPU campaign decides whether any of them ship. Each one saves roughly 0.3-0.9 ms from a ~64-68 ms DSpark-3 verify step, which is about 1% and below the L.A.I.L noise floor. The decision metric is therefore the kernel's ms/step from a profiled window, with e2e numbers used only to check for regressions.

| Lever | Env | Code | Removes (trace3 baseline, `results/2026-09-24-review/decode-levers/baseline-trace3-kernels.json`) |
|---|---|---|---|
| woa-scale-prepack | `DSV41_WOA_PREPACK=1` (needs image `e12-woa-prepack`) | `docker/patch/fix_o_proj_woa_fp8.py` stage 2 | `transpose_and_pack_fp32_into_ue8m0` grid (22,4,1): 43.2 calls/step x 19.1 us = 0.83 ms/step |
| mhc-prenorm-splits | `DSV41_MHC_DECODE_SPLITS=N` (N >= 2) | `docker/patch/decode_levers.py` | part of `sm120_tf32_hc_prenorm_gemm` grid (16,1,1): 86.4 calls/step x 22.1 us = 1.91 ms/step |
| sparse-markov | `DSV41_DSPARK_SPARSE_MARKOV=1`, `DSV41_DSPARK_SPARSE_MARKOV_TOPK=256` | `docker/patch/decode_levers.py` | Markov `cutlass_80 ... 32x32_128x1` grid (8,505): 3.0 x 283.8 us = 0.86 ms/step; adds one topk + fill_; the 3 x 24 us argmax stays |

## Files

- `bench_woa_prepack.py` runs the wo_a `fp8_einsum` twice, once with the fp32 weight scale and once with the scale pre-packed by vLLM's `transform_sf_into_required_layout`. It checks that the two outputs are bitwise equal (M = 3, 4, 8, five seeds each) and times each variant in a CUDA graph. Exit 1 means some output was not bitwise equal.
- `bench_mhc_prenorm.py` times `tf32_hc_prenorm_gemm` on its own and the full `mhc_pre_delayed_tilelang` (GEMM plus the fused norm kernel, which reduces the partials serially) at T = 4, 6, 8 and splits 16, 24, 32, 40, 48. It also prints the max |diff| against 16 splits, since split-K changes the summation order.
- `trace_kernels.py` reads a torch-profiler trace and reports per-step sums for the kernels above, grouped by grid. It is stdlib only and streams the ~2 GB JSON in about 20 s using ~0.5 GB RSS.

## Serve-down microbenches (both GPUs free; `./stop.sh` first)

```bash
cd /home/sfxnz/projects/ai-lab/recipes/.worktrees/rv-decode-levers   # or the merged checkout
out=results/2026-09-24-review/decode-levers
docker run --rm --gpus all --ipc host --network none -v "$PWD":/w -w /w \
  --entrypoint python3 dsv41-flash-exl3-sm121:canonical-e12 \
  kernel_study/decode_levers/bench_woa_prepack.py --json $out/woa_prepack.json
docker run --rm --gpus all --ipc host --network none -v "$PWD":/w -w /w \
  --entrypoint python3 dsv41-flash-exl3-sm121:canonical-e12 \
  kernel_study/decode_levers/bench_mhc_prenorm.py --json $out/mhc_prenorm.json
docker run --rm --gpus all --ipc host --network none -v "$PWD":/w -w /w \
  --entrypoint python3 dsv41-flash-exl3-sm121:canonical-e12 \
  kernel_study/decode_levers/bench_mhc_prenorm.py --no-pdl --json $out/mhc_prenorm_nopdl.json
```

Gates:
- woa: `BITWISE PASS` and `saving_us` > 0 at M = 3 and 4. The audit measured 6.8 us/call in a graph.
- mhc: pick the N with the lowest `pre_us` at T = 4 (the pre time includes the fused-norm reduce). Carry it to a boot arm only if `pre_us_16 - pre_us_best` >= 5 us/call, which is about 0.4 ms/step at ~86 calls/step, and `maxabs_vs_16` <= 1e-2 on the bf16 layer input.

## Boot arms (bench-honesty protocol, ARMS.md)

Build the woa image on both nodes first. It is a thin layer on canonical-e12.

```bash
docker build -f docker/Dockerfile.woa-prepack -t dsv41-flash-exl3-sm121:e12-woa-prepack docker
ssh spark2 'rm -rf /tmp/dl-docker' && scp -q -r docker spark2:/tmp/dl-docker
ssh spark2 docker build -f /tmp/dl-docker/Dockerfile.woa-prepack -t dsv41-flash-exl3-sm121:e12-woa-prepack /tmp/dl-docker
```

The arms are A (baseline), W (woa), M (mhc N), S (sparse-markov), and C (the combination of every lever that passed alone). One lever per boot, ABAB against A, at least 2 boots per arm, `./stop.sh` between boots. Example:

```bash
./stop.sh
IMAGE=dsv41-flash-exl3-sm121:e12-woa-prepack DSV41_WOA_PREPACK=1 ./run.sh      # W
DSV41_MHC_DECODE_SPLITS=40 ./run.sh                                             # M (N from the microbench)
DSV41_DSPARK_SPARSE_MARKOV=1 DSV41_DSPARK_SPARSE_MARKOV_TOPK=256 ./run.sh       # S
```

Engagement check on both ranks, right after ready:
- W: `docker logs dsv41-flash-exl3 2>&1 | grep -c '\[woa-prepack\] packed'` must be 43 on spark1 and on spark2 (40 target + 3 draft, the same count as the `[woa-requant]` lines). `grep -c REJECTED` must be 0.
- M: `dsv41: MHC prenorm split-K forced to N` appears. After ready, no new JIT compile shows up in the logs.
- S: `dsv41: DSpark sparse Markov (gathered top-k) k=256` appears, and there is no `decode lever sparse-markov FAILED`.

Per boot:
1. `python3 smoke_chat.py && python3 smoke_vision.py`, then `tests/correctness.sh`. Correctness must be 8/8.
2. `tools/four_numbers.sh --arm <arm>-<boot>`. This covers the 9-run prose median, pp 8k/32k, MemAvail on both nodes, and the ms/step cells once the bench-honesty package is merged.
3. `python3 tools/measure_lail_prose.py --runs 10` (L.A.I.L n=10, which also reports acceptance).
4. Quality quick: `python3 tests/quality_eval.py --quick --baseline <A json>` (quality-harness package). ΔNLL must be <= +0.01. W is bit-exact, so its NLL must match A.
5. Kernel trace: boot with `EXTRA_ARGS='--profiler-config {"profiler":"torch","torch_profiler_dir":"/tmp/dsv41-traces"}'`, run `results/2026-09-21-trace3/profile_window.sh`, `docker cp` the trace out, and run `python3 kernel_study/decode_levers/trace_kernels.py <trace> --json <out>`. Use a separate trace boot so the profiler does not skew the timed cells.

Keep rules, per lever:
- W: every (22,4,1) `woa_pack` call is gone from the trace, trace kernel ms/step is not higher than A, correctness is 8/8, and the NLL is identical. The change is bit-exact, so a tok/s win is not required. Result: default-on candidate.
- M: `mhc_prenorm_gemm` + `mhc_pre_norm_fused` ms/step drops by >= 0.4 ms, correctness is 8/8, ΔNLL <= +0.01, and ms/step at matched acceptance is not worse than A beyond the A/B boot spread.
- S: `markov_gemm` is gone from the trace, and the CLI acceptance_len (L.A.I.L n=10 and prose) is inside A's boot-to-boot band. Any drop >= 1% cancels the ~1% gain, so the result is REVERT. Correctness must be 8/8. The output distribution is exact by construction, because greedy/rejection verify is unchanged.
- C: only the levers that passed alone. The gate is ms/step at matched acceptance, noise-aware as in ARMS.md step 6.
