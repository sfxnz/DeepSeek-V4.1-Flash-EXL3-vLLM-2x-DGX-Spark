# VERDICT — 2026-09-21 Round 19: CPU-side Engram hash (Round 18 attribution fix) — REVERT (patch stays dormant in repo)

Baseline: k3c 28.76 median L.A.I.L (results/2026-09-21-capture-pf). Target 35+.

## Result

| arm | L.A.I.L decode_c1 | verdict |
|-----|-------------------|---------|
| Boot 1: k3c + DSV41_ENGRAM_CPU_HASH=1 (single lever) | warmup 26.25 (cold); then 28.87 / 28.92 / 28.35 → **median 28.87** — but the patch had SELF-DISARMED at warmup (stock path, see below) | disarmed-boot parity only |
| Boot 2 (the ONE diagnostic retry, two-stage warmup): k3c + CPU_HASH=1 | self-disarmed at warmup stage B (PREDICT) — no fast-path steps ran | **REVERT** (never engaged) |
| Restore: boot-k3c.sh (stock) | confirm e0eb7fcad0ee / f92df6d85eaa / fc2ead22b96d = 27.40 / 26.13 / 28.01 → median 27.40 (k3c boot-to-boot spread; best-of-boot medians remains 28.76) | serve left UP, smoke 323 ✓, 0 cpu-hash lines, MemAvail 25/27 |

35 NOT crossed — the fast path never executed live; both boots fell back to
stock before the sync could be removed.

## What was built (commits 03c59e1 + ea8bfe8, dormant default-off)

- `docker/patch/engram_cpu_hash.py` + sitecustomize wiring + run.sh env
  forwarding (head `-e` AND worker ssh line) for `DSV41_ENGRAM_CPU_HASH`.
- Post-propose runner hook snapshots step outputs to pinned mirrors on a side
  stream; off-thread worker reconstructs the next step's chunk (prefetch-v3
  rule), mirrors `_hash_ids_kernel` bit-exactly (numpy int64) into
  `hash_host`, preads/dequants all tables into `rows_host`; `stage()` then
  only H2Ds — GPU hash launch + D2H + `hashes_ready.synchronize()` deleted
  from the main thread. Batch-signature check each step; stock fallback for
  prefill/oversized/surprise batches.
- Offline validation (`experiments/dsv41-opt/patches/validate_cpu_hash.py`):
  300 fuzz cases bit-exact vs an independent scalar kernel transcription;
  chain-apply on scratch snip copies; full-chain dry-run against the real
  image (all patches apply, py_compile OK); gates bash -n / py_compile /
  render --check / 225 unittests OK.

## Premise correction (important for the next attempt)

The Round-18 note's premise "input ids are host-resident before
prepare_inputs" is **false** in this build: `input_batch.input_ids` is a GPU
buffer written by `combine_sampled_and_draft_tokens` from GPU-only
`last_sampled_tokens` / `draft_tokens`. Any D2H of the hash inputs queues
behind the previous step's graph — which is exactly why the sync costs
~14 ms. The implementable form is prediction-based (what was built), and the
prediction is where it failed.

## Engagement evidence (both boots)

- Wiring ENGAGED both boots, both ranks: `dsv41: engram cpu-hash stager
  installed` + `runner hooks installed` + `armed (layers=2 ngram=4 heads=8
  span=12 depth=3)` (spark1 TP0/TP1).
- Boot 1: single-stage warmup → `DISABLED: mirror mismatch vs GPU kernel at
  warmup` (both ranks) — conflated mirror+predict.
- Boot 2 (two-stage): **stage A (MIRROR) PASSED on live data** — the CPU
  hash of the ACTUAL ids is bit-exact vs the real GPU kernel. **Stage B
  (PREDICT) FAILED**, dump:
  `row 1: aid=85 apos=21 | pid=14 ppos=21`, rows 1-3 mismatch, row 0
  (bonus) always matches. I.e. the draft-token rows fed at the next step are
  NOT the tokens `propose()` returned (positions all match; ids differ).
  Root cause of the ~33% prefetch-v3 intersect is now named: only the bonus
  row of the v3 next-chunk rule is real; the draft rows are re-derived
  somewhere between `propose()` and the next step's
  `combine_sampled_and_draft_tokens` (candidate: `set_draft_tokens` /
  draft_tokens_handler, or dspark re-sampling from anchored positions).
- Fallback PROVEN live both boots: serve healthy, smoke 17×19=323 ✓, and
  disarmed-boot L.A.I.L = 28.87 median ≈ 28.76 baseline (dormant patch is
  free).

## Numbers vs baseline

| quantity | baseline k3c | cpu-hash boots |
|---|---|---|
| L.A.I.L decode_c1 median | 28.76 | 28.87 (disarmed parity, n=3) |
| quick prose / acc | 34.89 / 2.26 | not run (arm never engaged) |
| prefill 8k/32k, MemAvail post-32k | per capture-pf VERDICT | not run (arm never engaged) |

## Serve state left UP

Stock `results/2026-09-21-capture-pf/boot-k3c.sh` on :8000 (k3 + captures
[1,3,4,6,8], no cpu-hash — env not set). Confirm smoke + L.A.I.L run below.

## OOM floor log

| boot | pre s1/s2 | post-smoke s1/s2 | aborts |
|------|-----------|------------------|--------|
| cpu-hash boot 1 | 116/117 | 25/27 | none |
| cpu-hash boot 2 (retry) | 116/117 | (smoke run; logs in 10-boot log) | none |
| restore k3c | 117/117 | below | none |

stop.sh + docker ps verified empty on spark1 AND spark2 before every boot;
no host CUDA JIT; no trace parsing.

## Gap to 35 + next lever

Still 6.24 tok/s (28.76 best). The 13-14 ms/step sync pool is intact and the
attribution stands. Next attempt should fix the PREDICTION, not the hash:
dump `draft_tokens_handler`-side state for one step to find where the draft
ids change post-propose (set_draft_tokens writes req_states.draft_tokens —
verify the next step's combine reads THAT buffer, not speculator.draft_tokens;
if so, snapshot `input_batch`-adjacent draft state at the commit hook
instead of propose time). The mirror is proven bit-exact live — once the
prediction is right, the whole arm is ready to engage unchanged.
