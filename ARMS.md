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
6. **Promote rule**. The old text here ("9-run prose median beats
   baseline") was never what decided an arm. Rounds R16-R33 used +3% on
   the pooled L.A.I.L t=0.2 median (flags.md:308 R16, :432, :488, :672
   R30, :715, :725 R33). That gate sits inside boot-to-boot
   noise: the identical config measured 27.40/26.13/28.01
   (`results/2026-09-21-cpuhash/VERDICT.md`), and lm_head went from REVERT
   at +2.5% (R30) to KEEP at +5.9% (R33). An arm promotes only when all of
   these hold:
   - **ABAB boots**: baseline A and arm B each boot at least twice,
     interleaved A, B, A, B, with `four_numbers.sh` on every boot. No
     comparison against a baseline from another day or boot sequence.
   - **Primary metric**: `prose_median_ms_per_step`, and
     `median_ms_per_step` of the prose_long c=1 cell (ms per verify step,
     ~64-68 ms today; lower is better). Use it only at matched acceptance: the A and B
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
     (post-EOS fraction 0) and `serve_env_ranks_match` is true.

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
   that cell is post-EOS text. They stay for continuity only. prose_long, pp_novel,
   c=2 and warm-prefix have no baseline until the first ABAB boot records
   one.

## Exact restore sequence

Current serve = canonical image `dsv41-flash-exl3-sm121:canonical-e12` on
both nodes, launched via `serve.sh` from this repo on spark1 (head). To
restore after any arm:

```bash
cd /home/sfxnz/projects/ai-lab/recipes/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark
./stop.sh                                  # both nodes (stops spark2 via ssh)
./serve.sh                                 # defaults = boot-lm.sh config (canonical-e12, lmhead pack)
# readiness: python3 smoke_chat.py && python3 smoke_vision.py
# then: tools/four_numbers.sh --arm <label> to re-confirm baseline cells
```

- `serve.sh` → `exec ./run.sh`; on the head with `ORCHESTRATE=auto` (default)
  and `NNODES=2`, run.sh starts the **worker on spark2 first** (scps itself +
  `docker/patch` to /tmp there, launches worker container with the full env
  forwarded), sleeps 25s for NCCL, then starts head rank 0 and `wait_ready`
  (`/health` + `/v1/models`; allow up to 60 min, plan ≥20).
- Kernel arms rollback: `IMAGE=dsv41-flash-exl3-sm121:canonical-e12 ./serve.sh`.
- Patch-script changes (`docker/patch/engram_*`) take effect on restart
  without an image rebuild (patch dir is volume-mounted read-only).
