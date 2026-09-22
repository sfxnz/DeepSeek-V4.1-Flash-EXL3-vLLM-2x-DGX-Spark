# VERDICT — mem-hygiene bundle: **KEEP**

Arm: mem-hygiene (ONE room lever: page-cache drop + indexer factor=1 + logits
cap 256 MB + prefill empty-cache floor). Image `dsv41-flash-exl3-sm121:canonical-e12`,
`MAX_NUM_BATCHED_TOKENS=8192` pinned to match baseline. Wiring commit `003d5db`.

Envs: `DSV41_DROP_PAGE_CACHE=1 DSV41_INDEXER_PREFILL_FACTOR=1
DSV41_PREFILL_EMPTY_CACHE_TOKENS=8192 DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB=2.5
VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256` (head `-e` list + run.sh worker ssh line).

## Numbers — baseline (2026-09-20-baseline-live) vs arm (2026-09-20-memhygiene)

| Metric | Baseline | mem-hygiene | Delta |
|---|---:|---:|---|
| Prose decode c=1 median tok/s (9x) | 34.53 | 30.04 first pass / **34.67 re-run** | +0.14 (re-run; within noise) |
| Cold prefill 8k tok/s | 693.3 | 690.4 | −0.4% (noise) |
| Cold prefill 32k tok/s | 715.0 | 718.2 | +0.4% (noise) |
| L.A.I.L prose median tok/s (3x) | 25.71 | 25.34 | −1.4% (within 25.2–26.3 band) |
| DSpark acceptance len | 2.837 | 2.743 | noise |
| MemAvail after 32k prefill, spark1 GiB | 12.98 | **18.80** | **+5.82** |
| MemAvail after 32k prefill, spark2 GiB | 14.72 | **21.05** | **+6.33** |

Prose note: the first in-harness pass (30.04) was taken minutes after first boot
on cold caches; an immediate 9x re-run (`01-prose-rerun.log`) medianed 34.67
(acceptance 2.852). L.A.I.L and prefill were already flat in the same harness,
so the ROOM success criterion (MemAvail clearly higher, perf within noise)
holds.

## Armed evidence (both ranks)

- head `docker logs dsv41-flash-exl3`, worker `ssh spark2 docker logs`:
  - `dsv41-indexer-workspace: applied (.../mla/indexer.py)` — BOTH ranks
  - `[dsv41-drop-page-cache] armed (...)` — BOTH ranks
  - `[dsv41-prefill-empty-cache] armed (threshold 8192 tokens, release when MemAvailable < 2.5 GiB)` — BOTH ranks
- Page cache actually dropped on BOTH ranks:
  - TP0 (spark1): `dropped page cache of 48 shard files: MemFree 13.07GiB -> 33.39GiB`
  - TP1 (spark2): `dropped page cache of 48 shard files: MemFree 10.25GiB -> 34.64GiB`
- Adaptive skip observed (intended): `[dsv41-prefill-empty-cache] skipped #1 (longest seq 16376, MemAvailable 18.83 GiB >= 2.5 GiB)` on both ranks.

## Floors

- Boot: spark1 22.7 / spark2 24.9 GiB MemAvailable (≥12 required) — pass.
- Post-smoke: spark1 22.2 / spark2 24.3 GiB (≥8 required) — pass.
- No NVRM / NV_ERR lines on either node.

## Call

**KEEP** — serve left UP with the five envs active. ~6 GiB/rank recovered
(indexer factor=1 at 1M ctx + fadvise page-cache drop + 256 MB logits cap)
with prefill, L.A.I.L and (re-run) prose all within noise. Rollback if needed:
`./stop.sh`, unset the five envs (patches are env-guarded no-ops), `./serve.sh`.

## Evidence paths

- `results/2026-09-20-memhygiene/four_numbers.json` (+ 00–05 logs, 01-prose-rerun.log)
- `results/2026-09-20-baseline-live/four_numbers.json`
- Wiring: commit `003d5db` (docker/patch/{drop_page_cache,indexer_workspace,prefill_empty_cache}.py + sitecustomize.py + run.sh env forwarding, head + :475 worker line)
