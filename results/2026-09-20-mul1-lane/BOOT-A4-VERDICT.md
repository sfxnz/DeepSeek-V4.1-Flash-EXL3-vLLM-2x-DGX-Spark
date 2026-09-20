# VERDICT — mul1-lane Boot A4 (LANGUAGE_MODEL_ONLY=1 text-only, k=3) — lane PARKED PERMANENTLY

Date: 2026-09-20 · Run dir: `../` · Artifacts: this dir · Raw boot log: `04-boot-full.log`
Single lever vs Boot A: `LANGUAGE_MODEL_ONLY=1` — replicates her documented measurement condition (text-only server, HER-README.md:512-513). `.env` also returned `DSPARK_TOKENS=3 / SPEC_METHOD=dspark` with `DSPARK_DRAFT_SAMPLE` unset (A3's lever rejected; her default draft sampling). Launch line verified in-container (`00-launch-line.txt`): `--language-model-only` present, `--speculative-config {"method":"dspark","num_speculative_tokens":3}` with NO `draft_sample_method`, vision flags (`--mm-encoder-tp-mode`, `--limit-mm-per-prompt`, `--skip-mm-profiling`) absent. APIServer parsed `speculative_config={'method':'dspark','num_speculative_tokens':3}`.

## Table — A4 vs A vs A2 vs A3 vs MCG (median, port 8888)

| Metric | **A4: text-only k=3** | A: vision-on k=3 | A2: spec OFF | A3: k=3 greedy-draft | MCG (K4) |
|---|---|---|---|---|---|
| Prose decode c=1 tok/s (9 runs, greedy) | **16.45** | 16.35 | 23.64 | 16.80 | 34.69 |
| Prose 400-tok pass (3 runs, her token budget) | **20.30** | — | — | — | — |
| Cold prefill 8k tok/s | **592.5** | 593.7 | 587.9 | 597.4 | 690 |
| Cold prefill 32k tok/s | **821.8** | 711.7 | 845.6 | 697.5 | 733 |
| LAIL prose tok/s (t=0.2) | **26.79** | 25.63 | 23.46 | 24.92 | 26.39 |
| Micro tg 8k / 32k tok/s | **36.8 / 33.9** | 33.3 / 33.3 | 23.1 / 23.3 | 32.9 / 31.4 | — |
| acceptance_len prose greedy | **1.33** | 1.32 | n/a | 1.36 | ~2.8+ |
| acceptance_len prose 400-tok greedy | **1.69** | — | — | — | — |
| acceptance_len LAIL t=0.2 | **2.15** | 2.08 | n/a | 2.09 | — |
| Micro tg acc 8k / 32k | **2.89 / 2.92** | — | n/a | 2.67 / 2.36 | — |
| MemAvail after (s1/s2 GiB) | **6.2 / 7.7** | 5.3 / 6.7 | 8.8 / 10.0 | 5.2 / 6.7 | 22.3 / 23.8 |

## Decision rule — BOTH FAIL → mul1 lane PARKED PERMANENTLY

Rule: prose ≥ 24 AND acceptance ≥ 2.0 → conditions were the difference, lane REOPENED (Boot B unblocked).
Result: prose **16.45** (< 24), acceptance **1.33** (< 2.0).

**Conclusion: the measurement-conditions hypothesis (DRAFT-ROOT-CAUSE.md #3) is REFUTED.**

- Text-only vs vision-on moved greedy prose by **+0.1 tok/s** (16.35 → 16.45) and acceptance by +0.007 (1.32 → 1.33) — identical, within noise. The vision tower was never the cost at c=1 greedy.
- Her exact 400-token budget lifts prose to 20.30 tok/s at acceptance 1.69 — acceptance grows with generation length (long-horizon drafting pays off; same family as tg 8k/32k at 2.89/2.92), not with text-only. Still < 24 at double our token budget.
- Her 28 tok/s k=3 prose figure is **not reproducible on this pack under any measurement condition tested** (A/A3/A4 × vision-on/off × 200/400 tokens × default/greedy draft sampling). The residual gap to her figure is a harness/pack-revision difference outside our levers.
- What A4 does confirm: the pathology is exactly as DRAFT-ROOT-CAUSE.md concluded — **greedy short-prompt verify acceptance ~1.3 is structural to the 4-bit EXL3 MTP drafter (`mtp_bits: 4`)**, while sampled decoding (LAIL t=0.2: 2.15, micro tg: 2.9) drafts fine. Speculative decode is a net LOSS at c=1 greedy on this pack (16.45 vs A2's 23.64 spec-off).

**Lane disposition: PARKED PERMANENTLY.** `SPEC_METHOD=none` (A2: 23.64 greedy prose, 845.6 @32k prefill) stands as the mul1 lane's best greedy config on record; do not ship k=3 on this pack. Boot B (coop) stays BLOCKED on this lane. Named re-open mechanism: **re-quantize the pack with a source-precision drafter** (`mtp_experts: source` à la the MCG pack — whose greedy acceptance 2.77/2.87 with the same harness is the pack-level control — or coolbho3k's direction, mia-analysis/ANALYSIS.md:98). No boot-level lever below pack-requantization exists; four boots (A/A2/A3/A4) have exhausted the space.

## Boot / smoke

- Boot clean ~10 min to ready: 39/39 shards both nodes, health OK, zero NVRM/Xid lines, floors 6.2/7.7 GiB after the full 32k sweep (structural k=3 regime, above the 4.5 GiB abort line, no drift).
- Smoke: `17 × 19 = 323` ✓ (finish stop).

## Files

`00-launch-line.txt` · `00-smoke.txt` · `01-prose-decode.log` · `01b-prose-400tok.log` · `02-micro.log` · `03-lail-prose.log` · `04-boot-full.log` · `05-mem-after.log` · `06-docker-inspect-head{,-cmd,-env}.*` · `summary.json` · this `VERDICT.md`

## Restore (post-run)

MCG serve (`dsv41-flash-exl3-sm121:canonical-e12`, 2.0bpw-mcg pack) restored via saved `results/2026-09-20-nccl/boot-arm.sh`: ready on :8000 (~17 min), docker inspect verified — image, `--max-num-batched-tokens 8192`, dspark k=5 + its known-good `draft_sample_method:greedy`, all 9 armed envs (NCCL_BUFFSIZE/LL128/PROTO/MAX_NCHANNELS, DSV41 drop-page-cache/indexer-prefill-factor/prefill-empty-cache×2, VLLM_SPARSE_INDEXER_MAX_LOGITS_MB). Floors 25/26 GiB. Smoke: step-by-step prompt → **323** ✓ finish stop (her `17 * 19 =` phrasing; terse "number only" returns 361 — known phrasing artifact). L.A.I.L perf jobs on restored serve: warmup `a0afa5f21537` = 26.75 (discarded), **real `c33b06334e36` = 25.34 decode median, agg 24.55, ttft 0.324** — consistent with the MCG lane (26.12/26.39).

