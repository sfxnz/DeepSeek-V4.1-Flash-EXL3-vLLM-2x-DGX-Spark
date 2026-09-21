# VERDICT — 2026-09-21 capture-match + prefetch v3 arms

Baseline: k=3 @ 27.27 median L.A.I.L (jobs 1e542db2889d / e398e2358cf3 / 9a3d37e2308f;
final-boot confirm 65933ed0ddd7 = 27.0). Target: 35+ tok/s.

| arm | L.A.I.L median (c=1 decode) | acc_len | quick prose c=1 | MemAvail s1/s2 | pf intersect / pf_hit | call |
|-----|------------------------------|---------|------------------|----------------|------------------------|------|
| ARM 1 k3c (capture sizes [1,3,4,6,8]) | **28.76** (runs 28.76 / 26.20 / 28.94; jobs ba2b5d9f2654 / 094bb42c84ac / d04d844d5b97) | ~2.26 (prose bench 2.62 c1) | 34.89 | 26/27 GiB | n/a | **KEEP** (+5.5% vs 27.27; gate ≥28.1) |
| ARM 2 k3c+pf3 | 28.69 (runs 28.70 / 27.87 / 28.69; jobs 374f5bc756bc / 06f4a253e44d / 983ef4d132a8) | 2.62 (prose bench) | 34.14 | 26/27 GiB | intersect ≥0.5× consumed: 20% s1 / 15% s2 (mean ratio 0.325 / 0.292) | **NO-GO** (−0.2% vs ARM1; gate ≥29.62; intersect gate failed) |
| ARM 3 | — skipped: no v3-doc single flag cleared the bar after ARM 2 NO-GO; time spent on restore | | | | | skip |

## ARM 1 — capture-size match (KEEP)

- Lever: `COMPILATION_CONFIG` env override in `boot-k3c.sh` — cudagraph capture
  sizes [1,5,6,10,12] → [1,3,4,6,8] (k3 c=1 verify batch is 4; 4 padded to 5 before).
- Verified in engine log: `cudagraph_capture_sizes: [1, 3, 4, 6, 8]` (both
  APIServer non-default args and EngineCore compilation_config).
- L.A.I.L n=3 median 28.76 vs 27.27 = **+5.5%** (KEEP gate ≥+3% → ≥28.1).
- Quick prose c=1 greedy: 34.89 tok/s (vs 35.55 k3-baseline quick prose).

## ARM 2 — engram prefetch v3 (NO-GO, reverted)

- Commit FIRST: e70c4b1 `feat(tables): engram prefetch v3 — post-propose
  publish, bonus+draft anchored prediction` (staged patch copied to
  docker/patch/engram_prefetch_v3.py + sitecustomize import v2→v3; gates:
  bash -n, py_compile, kit/render.py --check, unittest 225 OK).
- Boot log: `dsv41: engram prefetch v3 stager installed` +
  `runner hook installed`; [pf-pub]/[pf-pair] on BOTH ranks, 40 pair-rows/rank,
  |pred|=48/pair-row (expected 72/table on A=1 gens was NOT met — observed 48).
- Delivery proven again ([pf-pub] precedes [pf-pair] every gen, both ranks),
  but prediction recall is still short: mean intersect ratio 0.325 (s1) /
  0.292 (s2); ≥0.5× gate met on only 20%/15% of pair-rows. pf_hit <50%.
- L.A.I.L median 28.69 ≈ ARM 1 (−0.2%), below the +3% gate (29.62).
- Reverted serve-side: DSV41_ENGRAM_PREFETCH=0 (restored boot-k3c.sh boot).
  Patch + commit stay in the repo (inactive unless env enables it).

## Serve state left UP

`results/2026-09-21-capture-pf/boot-k3c.sh` on :8000 — k3 + capture sizes
[1,3,4,6,8], no prefetch. Final-boot smoke 17×19=323 OK, MemAvail 25/26 GiB,
0 pf lines (prefetch confirmed off).

## OOM floor log

| boot | pre-boot MemAvail s1/s2 | post-smoke s1/s2 | abort lines |
|------|--------------------------|-------------------|-------------|
| ARM1 k3c (boot-k3c.sh) | 117/117 GiB | 26/27 GiB | none |
| ARM2 k3c-pf3 (boot-k3c-pf3.sh) | 117/117 GiB | 26/27 GiB | none |
| restore k3c (boot-k3c.sh) | 117/117 GiB | 25/26 GiB | none |

All boots: ./stop.sh + docker ps verified empty on spark1 AND spark2 before
each boot; free -h checked before every boot and after every smoke/L.A.I.L
job; no CUDA JIT / host allocations outside the container.

## Gap to target

Best 28.76 vs 35 target → **6.24 tok/s short** (82.2% of target). Remaining
levers: acceptance shaping at t=0.2 (ARM 3 candidate, untested), deeper
prefetch recall (v4 would need a different anchor stream).
