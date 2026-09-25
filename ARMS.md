# ARMS.md — campaign arm discipline

One boot = one lever. Every arm produces the four numbers via
`tools/four_numbers.sh --arm NAME` and a verdict appended to `flags.md`,
`results/RESULTS.md`, and the `recipes` skill ledger.

## Arm loop

1. **Pre-flight (both nodes)** — `docker ps --format '{{.Names}} {{.Image}}'`
   must show only `dsv41-flash-exl3`; `ssh spark2 docker ps` same.
2. **Stop both nodes**: `./stop.sh` (reads `.run-state/worker_host`, ssh's
   to spark2).
3. **Boot with ONE env lever** (one knob changed vs baseline; full argv/env
   recorded — paste the exact `FOO=1 ./serve.sh` line into the arm notes):
   ```bash
   DSV41_ENGRAM_PREFETCH=1 ./serve.sh     # example lever
   ```
   run.sh refuses unsafe configs (`FORCE_UNSAFE_QUANT/ENGRAF/CTX` guards);
   if an A/B legitimately needs an override, set the guard for that boot
   only and record it — never commit it as a default.
4. **Boot floors** (read `free -h`, NEVER nvidia-smi):
   - abort if MemAvailable **< 12 GiB at boot** (before smoke);
   - abort if MemAvailable **< 8 GiB after smoke** (`smoke_chat.py` +
     `smoke_vision.py`).
5. **Four numbers** (serialized, ~20-25 min estimated; re-measure on the
   first campaign run):
   ```bash
   tools/four_numbers.sh --arm <name>          # prose 9x median, pp_warm/pp_novel 8k/32k,
                                               # MemAvailable both nodes, L.A.I.L 3x,
                                               # prose_long c=1/c=2, warm-prefix,
                                               # env digest per rank, page cache
   ```
   Number 3 (MoE/attention ms/layer at shipped chunk) has no live probe —
   fallback is boot knobs `DSV41_STEP_CENSUS=1` / `DSV41_ENGRAM_CENSUS=1`
   (decode-side only; E0 prefill-flush caveat); real profiling is a
   separate gated step. `four_numbers.sh` does not run e2e: run
   `benches/e2e.py` separately on the same boot (step 6 requires exit 0).

   **Order on a fresh boot** (the Round 34 FULL and DECODE protocols; keep
   it the same on every arm so the cells compare like with like):
   `smoke_chat.py` + `smoke_vision.py`, then the fresh L.A.I.L
   (`tools/measure_lail_prose.py --runs 10`), then `bench_decode.py`, then
   `benches/micro.py`, quality, and C2-STRESS last. The L.A.I.L goes before
   `bench_decode.py` because bench_decode's default `--concurrency 1 2`
   sends c=2 traffic. A c=1 cell that has to run before the L.A.I.L passes
   `--concurrency 1`.
6. **Promote rule**. The old text here ("9-run prose median beats
   baseline") was never what decided an arm. Rounds R16-R33 used +3% on
   the pooled L.A.I.L t=0.2 median (flags.md sections Round 16, Round 21,
   Round 23, Round 30, Round 32 and Round 33). That gate sits inside boot-to-boot
   noise: the identical config measured 27.40/26.13/28.01
   (`results/2026-09-21-cpuhash/VERDICT.md`), and lm_head went from REVERT
   at +2.5% (R30) to KEEP at +5.9% (R33). An arm promotes only when all of
   these hold:
   - **ABAB boots**: baseline A and arm B each boot at least twice,
     interleaved A, B, A, B, with `four_numbers.sh` on every boot. No
     comparison against a baseline from another day or boot sequence.
   - **Primary metric**: `prose_median_ms_per_step`, and
     `median_ms_per_step` of the prose_long c=1 cell (ms per verify step;
     lower is better; the reference is the A boots of the same ABAB
     sequence, not a number from another day). Use it only at matched acceptance: the A and B
     per-boot `median_run_acceptance_len` must overlap. If they do not,
     the arm changed drafting; gate on prose_long c=1 tok/s instead with
     the same noise rule.
   - **Noise-aware gate**: noise = the larger boot-to-boot spread (max-min
     of per-boot medians) of A and of B. Promote when the B-vs-A median
     improvement exceeds that noise AND every B boot beats the A median.
     A fixed +3% is not a gate.
   - **Non-inferiority**: prefill `pp_warm` and `pp_novel` 8k/32k,
     prose_long c=2 aggregate, and warm-prefix `hit_fraction_of_expected`
     not worse than A by more than their own boot-to-boot spread; boot
     floors (step 4) hold; `benches/e2e.py` exits 0.
   - **Honest cells**: prose_long `natural_finish_reason` is `length`
     (post-EOS fraction 0), `serve_env_ranks_match` is true, and
     `lever_disarmed` is false (`tools/disarm_scan.sh` found no
     LOG_DISARMED line in either rank's docker logs; a lever can turn
     itself off at runtime, after the post-ready audit ran).

   **Quality gate (required before any KEEP)**: after the four numbers,
   serialized and never interleaved with perf capture, the quick eval must
   pass against the stored baseline:
   ```bash
   python3 tests/quality_eval.py --quick \
     --baseline results/2026-09-24-review/quality-baseline/quick.json \
     --out results/<arm-dir>/quality_quick.json      # exit 0 = pass
   ```
   It gates prefill numerics (NLL), decode numerics (decode-vs-prefill
   probe and the golden flip hazard against the A/A control), tool calls,
   needle 8k/32k, c=2 and vision. A lever that changes numerics by design
   (a new pack, lm_head, a kernel format) also runs `--full` against
   `full.json`. A failed gate means no KEEP, even when the perf win is real.
7. **Free the box between arms**: `./stop.sh`, confirm no GPU containers on
   either node, then boot the next arm. Never stack levers on a promoted
   arm's boot without re-running the full four numbers.
8. **Verdicts**: append to `flags.md` (per-round note), `results/RESULTS.md`
   (row with evidence path), and the `recipes` skill ledger. Record per-boot
   medians, the spread and the ms/step deltas, not only the pooled number.
   Reference cells at close (R33): frozen prose c=1 39.60 tok/s and
   L.A.I.L prose 33.23 tok/s. Both force `ignore_eos`: the frozen prose
   prompt stops at 78 of 200 tokens
   (`results/2026-09-24-review/bench-honesty/smoke-prose.txt`), so most of
   that cell is post-EOS text. They stay for continuity only. Round 34
   reference (s13, the round-34 defaults, `four_numbers.sh`): prose 40.37
   at 63.03 ms/step, prose_long c=1 33.13 at 62.88 ms/step, prose_long c=2
   45.94 aggregate, pp_novel 8k/32k 808.8/800.2, pp_warm 759.6/766.8,
   L.A.I.L 3x 33.98 (`results/2026-09-24-review/campaign/s13-promote-final/`).
   Warm-prefix hits depend on the prompt length: a repeat of N tokens
   missed entirely when N ran only 10-58 tokens past the last 128-token
   boundary, and hit at 68-127 (results/RESULTS.md round 34).
   `tools/warm_prefix.py` now pads the prompt so that tail is at least 80
   tokens (`tail_tokens` in its summary). Read a 0.0 in older captures with
   that in mind.

## Exact restore sequence

Current serve (round 34) = `dsv41-flash-exl3-sm121:canonical-e13` on both
nodes with the round-34 defaults (`DSV41_ENGRAM_WILLNEED=1`,
`DSV41_STREAM_FEED=1`, `DSV41_WOA_PREPACK=1`,
`DSV41_DSPARK_SPARSE_MARKOV=1`), launched with `AUDIT=strict ./run.sh` from
the perf-review-0924 worktree. Until that branch merges, the main checkout
keeps the R33 defaults and boot-lm.sh below restores the R33 config. After
the merge, boot-lm.sh exports only its own names, so it would inherit the
round-34 lever defaults on e12, and WOA would log `lever is OFF`. Use the
"R33 reproduction" command further down instead.

```bash
cd /home/sfxnz/projects/ai-lab/recipes/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark
./stop.sh                                  # both nodes (stops spark2 via ssh)
bash results/2026-09-22-endgame2/boot-lm.sh   # R33 config (canonical-e12, lmhead pack), old code
# readiness: python3 smoke_chat.py && python3 smoke_vision.py
# then: tools/four_numbers.sh --arm <label> to re-confirm baseline cells
```

- `serve.sh` → `exec ./run.sh`; on the head with `ORCHESTRATE=auto` (default)
  and `NNODES=2`, run.sh starts the **worker on spark2 first** (scps itself +
  `docker/patch` to /tmp there, launches worker container with the full env
  forwarded), sleeps 25s for NCCL, then starts head rank 0 and `wait_ready`
  (`/health` + `/v1/models`; allow up to 60 min, plan ≥20).
- Kernel arms rollback: `IMAGE=dsv41-flash-exl3-sm121:canonical-e12 DSV41_WOA_PREPACK=0 ./serve.sh`
  (on e12 the WOA lever logs `the lever is OFF` and a strict audit fails).
- Round-34 levers off on the new code (the s3 B1 config):
  `IMAGE=dsv41-flash-exl3-sm121:canonical-e12 DSV41_ENGRAM_WILLNEED=0 DSV41_STREAM_FEED=0 DSV41_WOA_PREPACK=0 DSV41_DSPARK_SPARSE_MARKOV=0 ./run.sh`.
  An empty value keeps the default, so use `0`.
- R33 reproduction after the merge. boot-lm.sh stays as recorded evidence and
  is not edited. Run it with the four round-34 levers off, plus `WARMUP=0`
  because the R33 run.sh sent no post-ready warmup:
  ```bash
  DSV41_ENGRAM_WILLNEED=0 DSV41_STREAM_FEED=0 DSV41_WOA_PREPACK=0 \
    DSV41_DSPARK_SPARSE_MARKOV=0 WARMUP=0 bash results/2026-09-22-endgame2/boot-lm.sh
  ```
  In a run.sh dry run (tests/run_sh_harness.py), this gives the same container
  env and vllm argv as the levers-off `./run.sh` line above, apart from the
  port. The argv matches the s3 B1 capture. The env matches the R33 env
  captured in s2 (`campaign/s2-old-fresh/serve_env_spark1.txt`) except for
  three new names that do nothing on a clean boot:
  `DSV41_PATCH_STRICT=1`, `DSV41_DENSE_DG_SMALLM=0` and
  `DSV41_ENGRAM_WILLNEED_MIN_ROWS=512`. The patch code is the merged code. For
  the R33 code as well, boot from a worktree at `45d3303`. `AUDIT` only reads
  logs, so it does not change the config.
- Patch-script changes (`docker/patch/engram_*`) take effect on restart
  without an image rebuild (patch dir is volume-mounted read-only).
