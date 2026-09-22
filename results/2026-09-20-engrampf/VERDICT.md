# VERDICT — Engram prefetch v2 (DSV41_ENGRAM_PREFETCH=1 + v2 patch): **NO-GO → REVERT**

Arm: engram-pf-v2. One lever on top of the KEEP'd nccl-set baseline:
`DSV41_ENGRAM_PREFETCH=1 DSV41_ENGRAM_CENSUS=1 DSV41_ENGRAM_PF_DUMP=1
DSV41_ENGRAM_PREFETCH_DEBUG=1` (all baseline envs unchanged, per
`results/2026-09-20-nccl/boot-arm.sh` + the four prefetch envs).

Wiring commit `befa570`: v2 patch staged into `docker/patch/engram_prefetch_v2.py`,
sitecustomize import swapped to v2 (+3rd census arg), run.sh forwards
`DSV41_ENGRAM_PF_DUMP` + `DSV41_ENGRAM_PREFETCH_DEBUG` on BOTH the head env
list and the worker ssh line (the run.sh:475 omission the RUNSH-NOTE flagged).
All gates green before commit (bash -n, py_compile, kit/render.py --check,
unittest discover). Install verified live on both ranks: "engram prefetch v2
stager installed", "runner hook installed", "census pf_hit accounting fixed
(v2)". Smoke 17*19 = 323, MemAvail healthy (25/27 GiB at boot, 22.08/24.17
after 32k prefill), zero NVRM/CUDA-error lines.

## GO/NO-GO criteria vs ENGRAAM-PREFETCH-V2.md §6

| Criterion | Gate | Measured | Pass? |
|---|---|---|---|
| [pf-pair] intersect ≥ 0.5×\|consumed\| | first-consumption self-check | **≈0** on 70/80 pairs (sum 88/72 rows over 40 pairs/rank; only 4 gens showed 12-20-row intersects) | **FAIL** |
| census pf_hit ≥ 50%, trending up | first census windows | **~0%** (98× pf_hit=0%, one 33%, one 8%, rest ≤7% of 105 windows w/ pairings) | **FAIL** |
| read_w avg < ~0.1 ms | census | **0.06–0.10 ms** (already at floor WITHOUT prefetch helping) | pass (moot) |
| L.A.I.L ≥ 28 tok/s @ acc 2.25–2.35 | four-numbers + real job | **25.23 cold / 26.79 warm / 26.05 L.A.I.L job** @ acc 2.26–2.39 | **FAIL** (parity with 26.39–26.55 baseline) |
| correctness spot-check | smoke | 323 ✓ | pass |
| no "prefetch disabled" line | logs | none (worker did NOT self-disable; it published every gen) | pass |

## Numbers — nccl-set baseline (2026-09-20-nccl) vs arm (2026-09-20-engrampf)

| Metric | nccl-set baseline | engram-pf-v2 | Delta |
|---|---:|---:|---|
| Prose decode c=1 median tok/s (9x) | 31.59 cold / **34.69 re-run** | **33.24** (single pass) | inconclusive single-pass inside the established 30–35 cold/warm band; not re-run (verdict already decided by pf_hit/intersect) |
| Cold prefill 8k tok/s | 690.4 / 687.3 | 694.8 | flat |
| Cold prefill 32k tok/s | 733.0 | 727.3 | −0.8% (noise) |
| L.A.I.L prose median tok/s | 26.55 / 26.12 / job 26.39 | 25.23 cold / **26.79 warm** / **job 26.05** | parity |
| DSpark acceptance len | 2.743–2.871 | 2.856 (prose), 2.26–2.39 (L.A.I.L) | noise |
| MemAvail after 32k prefill spark1/spark2 GiB | 22.29 / 23.84 | 22.08 / 24.17 | flat |

## Verdict rationale

This is the doc's named NO-GO branch: **"intersect ≈ 0 on the [pf-pair]
self-check (pairing still wrong)"** — with the diagnostic payoff the v2
self-check was built for: publish-side and consume-side set ids are BOTH
printed per pair, publishes demonstrably land BEFORE consumption
(`[pf-pub] gen=N` precedes `[pf-pair] gen=N` for every paired gen), yet
table_pred_id ≠ consumed_id on ~88% of pairs. The prediction itself is the
wrong-side divergence now: the CPU hash reproduces rows that the next gather
mostly does not read (only the first gen after a request boundary and one
mid-gen pair showed 12–20-row intersects ≈ 17–28% of |pred|). Combined with
read_w already at 0.06–0.10 ms on cold gathers (the warm floor), the
projected "gaps 11–13 ms → 2–3 ms" mechanism is not there to claim: the
gathers are not obviously NVMe-latency-bound at these shapes, or the
predicted row set targets the wrong tokens/positions. NOT the "pf_hit high
but tok/s flat" fadvise-latency branch (pf_hit never rose); the named next
lever (direct pread staging) is therefore NOT indicated by this run — the
row PREDICTION, not the delivery path, is the remaining unknown.

Revert executed: `./stop.sh` (both nodes verified clean), re-booted via
`results/2026-09-20-nccl/boot-arm.sh` (exact NCCL-arm envs, PREFETCH
unset→0), log `00-boot-revert.log`. The wiring commit `befa570` stays
(dormant: env-guarded, PREFETCH=0 by default; v2 module on disk unused).

## Evidence

- `results/2026-09-20-engrampf/four_numbers.json` + numbered logs (both L.A.I.L passes kept: `04-lail.log` 25.23 cold, `04-lail-rerun.log` 26.79 warm)
- `results/2026-09-20-engrampf/pf-evidence-spark1.log` (202 lines, 40 [pf-pair], 28 [pf-pub], census lines TP0)
- `results/2026-09-20-engrampf/pf-evidence-spark2.log` (196 lines, 40 [pf-pair], 22 [pf-pub], census lines TP1)
- `results/2026-09-20-engrampf/00-boot.log` — arm boot; `00-boot-revert.log` — NCCL restore
- L.A.I.L runner records: warmup `0c88f184d126` (25.42, discarded), real **`243a53d345ca` = 26.05 tok/s**
