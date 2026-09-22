# VERDICT — 2026-09-21 Round 21: prefetch v3 re-arm (ordering-fixed) on k3c

Baseline: k3c 28.76 (Round 18/20 best; boot-to-boot spread ±1 tok/s).
Lever (single): `DSV41_ENGRAM_PREFETCH=1` on k3c boot — v3 code with the
Round-20 ordering fix (35e05fd, wait_stream hoisted out of side-stream
with-block). `DSV41_ENGRAM_CENSUS=1` for evidence; PF_DUMP / PREFETCH_DEBUG
OFF (minimal per-step overhead). Boot script: `boot-k3c-pf.sh` (diff vs
`../2026-09-21-capture-pf/boot-k3c.sh` = only the two env lines).

## Result: KEEP

| batch | runs (tok/s, sorted) | median |
|-------|----------------------|--------|
| 1 (n=5) jobs d84f55fc3c8f d45ed3dfa5d5 23dbb238b521 c037f2bf0a07 4576d9713e64 | 27.28 28.03 29.32 30.87 31.66 | **29.32** |
| 2 (n=5) jobs 7f5516e99708 8a509112b258 d124596efc03 83295e42b01d 5152a7f69eb4 | 28.32 28.51 30.99 31.09 31.87 | **30.99** |
| pooled n=10 | 27.28 28.03 28.32 28.51 29.32 30.87 30.99 31.09 31.66 31.87 | **30.10** |

vs 28.76 = **+4.6%** (KEEP gate ≥+3% → ≥29.62 pooled; both independent
medians also > 28.76 individually). Not ≥35: no CLI-twin claim, no 35
declaration. Warmup job 2ee75d59f526 discarded per protocol.

## Why it worked now (Round-16 NO-GO superseded)

- Install verified BOTH ranks: `engram prefetch v3 stager installed` +
  `runner hook installed` + `census pf_hit accounting fixed (v3)` +
  `v3 armed (ngram=4 heads=8 depth=3 layers=[0,1])`; capture sizes
  [1,3,4,6,8] confirmed in engine config.
- One-shot `[pf-pair]` self-check (gen=1, warmup): s1 intersect 32/48=0.67,
  s2 16/48=0.33 — pre-steady-state, but no longer the broken run's 0.29–0.33
  *steady* ratio.
- Census (the decisive evidence): pf_hit climbed to **100%** by call 64 and
  held 100% for the entire run on BOTH ranks (89 census lines/rank). Broken
  Round-16 run: pf_hit <50%, mean pair intersect ratio 0.325 (s1) / 0.292
  (s2). The ordering fix made prediction timely ⇒ timely pairs intersect
  fully ⇒ fadvise(WILLNEED) covers the rows actually consumed.
- read_w (pread wall per stage call): warmup 4.01 ms (pf_hit 66%) →
  steady-state **0.05–0.07 ms** both ranks (from ~14 ms/step GPU idle pool
  attribution in Round 18). read_s 0.02–0.03 ms, dequant 0.01 ms.
  The page-cache overlap IS converting to stage-time reduction.

## Interpretation

Round-20 conclusion refined: the idle pool WAS page-cache-bound after all —
the Round-16 prefetch failure was purely the stale-id prediction racing the
draft-graph replay, not a wrong overlap thesis. +1.34 tok/s pooled (28.76 →
30.10) from collapsing stage pread latency; the remaining ~5 tok/s gap to 35
sits in the residual gather path (dequant/H2D serialization) and spec
acceptance, not in pread waits.

## Serve state left UP

Current boot (boot-k3c-pf.sh, prefetch ON, census ON) left UP on :8000 —
this IS the best config. MemAvail at close: 25/27 GiB (floor 8).

## OOM floor log

| point | spark1 avail | spark2 avail | abort? |
|-------|--------------|--------------|--------|
| pre-boot (after stop.sh, both empty) | 116 GiB | 116 GiB | no |
| post-boot | 25 GiB | 27 GiB | no (floor 12) |
| post-smoke | 25 GiB | 27 GiB | no (floor 8) |
| post-batch-1 | 25 GiB | 25 GiB | no |
| post-batch-2 / close | 25 GiB | 27 GiB | no |

stop.sh + `docker ps` verified empty on spark1 AND spark2 before boot.
One aborted launch attempt (nohup orphaned when wrapper session exited at
~15 s, worker-only on spark2) — cleaned via stop.sh before the real boot;
not a boot of record. No host CUDA JIT, no trace parsing. Smoke gate:
`17 * 19 = ? Step by step, then answer.` → 323 ✓ (05-smoke.json).

## Artifacts

- boot-k3c-pf.sh, 00-boot-k3c-pf.log, 05-smoke.json
- 10-lail-warmup.json, 11-lail-real.ndjson (batch 1), 12-lail-real-b2.ndjson (batch 2)
