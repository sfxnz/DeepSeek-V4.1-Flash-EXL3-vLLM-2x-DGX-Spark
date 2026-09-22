# VERDICT — mul1-lane Boot A3 (k=3 + draft_sample_method=greedy) — lever REJECTED

Date: 2026-09-20 · Run dir: `../` · Artifacts: this dir · Raw boot log: `04-boot-full.log` · start.sh diff: `00-startsh-diff.txt`
Single lever vs Boot A: `DSPARK_DRAFT_SAMPLE=greedy` → `--speculative-config {"method":"dspark","num_speculative_tokens":3,"draft_sample_method":"greedy"}`. Launch line + APIServer parsed `speculative_config` both verified to carry the key (see `00-launch-line.txt`, `04-boot-full.log`); `DSPARK_DRAFT_SAMPLE=greedy` confirmed in `docker inspect` head env (`06-docker-inspect-head-env.json`).

## Table — A3 vs A (k=3 default-draft) vs A2 (spec-off) vs MCG (median, port 8888)

| Metric | **A3: k=3 greedy-draft** | A: k=3 default | A2: spec OFF | MCG (K4) |
|---|---|---|---|---|
| Prose decode c=1 tok/s (9 runs, greedy) | **16.80** | 16.35 | 23.64 | 34.69 |
| Cold prefill 8k tok/s | **597.4** | 593.7 | 587.9 | 690 |
| Cold prefill 32k tok/s | **697.5** | 711.7 | 845.6 | 733 |
| LAIL prose tok/s (t=0.2) | **24.92** | 25.63 | 23.46 | 26.39 |
| Micro tg 8k / 32k tok/s | **32.9 / 31.4** | 33.3 / 33.3 | 23.1 / 23.3 | — |
| acceptance_len prose greedy | **1.36** | 1.32 | n/a | ~2.8+ |
| acceptance_len LAIL t=0.2 | **2.09** | 2.08 | n/a | — |
| draft_acceptance_rate prose / LAIL | **0.118 / 0.362** | 0.108 / — | n/a | — |
| MemAvail after (s1/s2 GiB) | **5.2 / 6.7** | 5.3 / 6.7 | 8.8 / 10.0 | 22.3 / 23.8 |

## Decision rule — BOTH FAIL → mul1 lane stays PARKED

Rule: acceptance ≥ 2.0 AND prose ≥ 24 → drafting fix confirmed, lane reopened, Boot B unblocked.
Result: greedy-verify acceptance **1.36** (< 2.0) and greedy prose **16.80** (< 24).

**Conclusion: the draft-sampling hypothesis is REFUTED.** Forcing the drafter to greedy sampling moved greedy-verify acceptance from 1.32 → 1.36 — within noise. The pathology is not "stochastic drafts rejected under greedy verify"; it is deeper in the dspark draft path on this 2.9bpw pack. Evidence pattern unchanged from A: greedy acceptance ~1.3 while temp-0.2 acceptance holds at 2.09 and micro tg at 2.4–2.8 — the draft agrees with sampled decoding but is systematically wrong under greedy verify. Next suspect per plan: **draft-wo-a-slices patch** (a-slices exclusion in draft weights), then draft-model quality at 2.9bpw / k tuning.

Boot B (coop) remains BLOCKED on this lane. A2's `SPEC_METHOD=none` stays the mul1 lane's best greedy config on record.

## Boot / smoke

- Attempt 1 aborted pre-verdict: launch line lacked `draft_sample_method` — the var wasn't in start.sh's container env allowlist (`serve_env` loop + `-e` list); patched allowlist + `-e DSPARK_DRAFT_SAMPLE=...` and rebooted (diff covers all three edit sites).
- Boot 2 clean: ~7 min to ready, health OK, 39/39 shards, zero NVRM, no abort criteria. Mem cruised 5.2–6.7 GiB (structural k=3 floors, same as A).
- Smoke: `323` ✓ (finish stop, 20 completion tokens).

## Files

`00-startsh-diff.txt` · `00-launch-line.txt` · `00-smoke.txt` · `01-prose-decode.log` · `02-micro.log` · `03-lail-prose.log` · `04-boot-full.log` · `05-mem-after.log` · `06-docker-inspect-head-cmd.json` · `06-docker-inspect-head-env.json` · `four-numbers.log` (combined raw) · `summary.json` · this `VERDICT.md`
