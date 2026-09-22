# VERDICT — 2026-09-21 Round 23: engram gather v2 (preadv run batching) on k3c+pf

Baseline: Round-21 best **30.10** pooled n=10 median (k3c +
`DSV41_ENGRAM_PREFETCH=1` + `CENSUS=1`, boot-k3c-pf.sh). Target 35+.

Lever (single): `DSV41_ENGRAM_GATHER_V2=1` added to the identical env set —
boot script `boot-k3c-pf-gv2.sh` (diff vs `../2026-09-21-pfrearm/boot-k3c-pf.sh`
= one env line). Code: `docker/patch/engram_gather_v2.py` (commit 5507311),
wired via sitecustomize after the census chain, env forwarded in run.sh
(head `-e` + worker ssh line).

## What v2 is (Round-22 attribution → fix)

Round 22 attributed ~7.5 ms/step of GPU idle to HOST EXECUTION of the
`engram.stage` gather loop: 51 rows/call × (per-row pread syscall +
per-row dequant link + pinned staging), with the data already page-cached
(read_w 0.1 ms, pf_hit 100%).

Pre-implementation probe on spark1 (hot cache, real row geometry 256B/8B,
R=51): stock `_read_rows` chunk-pool cost **0.636 ms/call**, of which the
pool submit/result overhead alone was 0.105 ms; the SAME 102 per-row
preads inline with NO pool cost 0.053 ms. **The pool dispatch was the
loop cost, not the syscalls.**

v2 (env-gated, default off, inside `DiskEngramTable.gather_dequant` — so
it runs inside the round-11 parallel per-table stage workers unchanged):
- ONE `os.preadv` per contiguous run of file-adjacent rows (scatter
  iovecs land each row directly in its output slot; duplicate row ids
  read once + in-memory copy; row-aligned partial-read resume);
- dequant chain VERBATIM from the stock body (bit-exact by construction);
- H2D stays the existing ONE pinned copy per table per step (stage()
  contract untouched);
- one-shot self-check: first call ALSO runs stock and compares raw bf16
  BITS (fp8 NaN bytes make float-equality lie — found in validation);
  any error anywhere disarms v2 permanently with ONE warning line and
  the stock body serves — the serve never crashes.

## Offline validation — ALL PASS (validate_gather_v2.py, in e12 image)

- py_compile; chain-apply prestage→census→v2 on scratch snip copies;
  idempotent re-apply; py_compile patched module.
- Bit-exactness on REAL table geometry (safetensors pack, F8_E4M3
  [N,256] / F8_E8M0 [N,8], 8-byte header framing): stock vs v2
  tensor-for-tensor — fuzz R=51 mixed-owned, crafted contiguous runs +
  duplicates, all-unowned row-0 ×51, single row, fuzz R=300: all
  bit-identical (int16-bitview compare).
- Partial-preadv resume (capped-100B transport): byte-identical rows
  (this caught and fixed a REAL resume bug — follower rows of a
  multi-iov run were corrupted by a row_bytes-based skip after the
  first partial; fixed to consume n bytes off the iov list front).
- Error path: forced multi-iov preadv failure → one DISABLED line, stock
  fallback result correct, flag disarmed, no crash.
- Micro-benchmark (R=51, hot cache): **stock 0.82-0.85 ms/call vs v2
  0.12-0.19 ms/call = 4.2-6.9× per call** (both paths same table, flag
  toggled per call; 400 iters after 50 warm). preads == runs
  (one preadv/run), runs ≤ 2R.

## Boot + engagement evidence (boot 1 of cap 3)

- stop.sh + docker ps empty on spark1 AND spark2 pre-boot; MemAvail
  116/117 GiB.
- Install line BOTH ranks: `dsv41: engram gather v2 installed
  (DSV41_ENGRAM_GATHER_V2)` after `census instrumented` + `prefetch v3
  stager installed`.
- **Self-check BOTH ranks: `engram gather v2 self-check bit-exact
  (r=120)`** (prefill staging call; live-data stock-vs-v2 bit compare).
- Smoke `17 * 19 = ?` → **323** ✓ (05-smoke.json). MemAvail post-smoke
  25/27 GiB (floor 8).
- Steady-state `[gv2-census]` BOTH ranks: rows/call=48,
  **runs/call=96 = preads/call=96** (one preadv per run; stock made 96
  per-row preads + 12 pool futures per call → pool dispatch eliminated),
  read=1.3-1.8 ms/32calls window. ZERO `DISABLED` lines.
  Note: the stock `[engram-census]` read_w/read_s lines are silent on
  this boot because v2 returns before the instrumented stock body —
  gv2-census IS the replacement census (by design, same env gate).
- pf_hit accounting fix (v3) installed; no pf-pair anomaly lines.

## L.A.I.L (warmup discard + n=5 + independent n=5, decode prose c=1)

| batch | jobs | runs (tok/s, sorted) | median |
|-------|------|----------------------|--------|
| warmup (discarded) | 2fe85d9214b0 | 33.09 | — |
| 1 (n=5) | e910d913fe49 44588940f04f 0942ffae2c3d e0a0561e073f 796443ece907 | 29.47 29.48 32.11 32.76 33.72 | **32.11** |
| 2 (independent n=5) | d0fd4d6cf165 1ddbd3e364c7 cbd848c8e02c 8e2f28d54ce2 3e218b5a6435 | 30.17 30.34 30.63 32.22 32.90 | **30.63** |
| pooled n=10 | | 29.47 29.48 30.17 30.34 30.63 32.11 32.22 32.76 32.90 33.72 | **31.37** |

vs 30.10 = **+4.2%** → KEEP gate (≥ +3%, i.e. ≥31.0 pooled) **PASSED**;
both batch medians ≥30.1. Consistent with the warm restore-health
confirm band from Round 22 (30.5-33.9).

**35 status: NOT crossed.** Two-medians rule fails (32.11 / 30.63, not
both ≥35); no CLI-twin claim.

## Repo prose 9-run + acceptance

- `bench_decode.py --phase prose --runs 9` (200 tok): agg rates 33.72 /
  35.70 / 35.77 / 35.84 / 36.80 / 37.49 / 37.67 / 38.03 / 39.47 →
  **median 36.80 tok/s** (vs k3c-era ~33.8-35.2), median acceptance_len
  **2.55**, draft acceptance rate 0.517.
- `tools/measure_lail_prose.py --runs 9` (512 tok, L.A.I.L math):
  median_lail_tok_s **30.71**, recipe 30.65, TTFT 0.32 s, acc 2.12,
  no collapse (repeat_ratio ~0.002). Acceptance in the normal band —
  the gain is not an acceptance artifact.

## Prefill + memory (cold, 3 runs each; 21-micro-prefill.log / 22-mem-post32k.log)

- Cold prefill 8k: 166.2 / 259.1 / 321.0 (run 1 is a cold-start outlier
  in this harness; warm band ≈ 260-320).
- Cold prefill 32k: 493.1 / 712.9 / 785.6 (warm runs in the historic
  727-733 band; run 1 cold).
- Decode-after-32k tg: 26.0-33.5 @8k, 30.0-32.4 @32k, acc 2.06-2.36.
- **MemAvail immediately post-32k: spark1 21 GiB / spark2 23 GiB**
  (historic band 21-25; above the 8 GiB floor).

## Result: KEEP

- Pooled 31.37 vs 30.10 (+4.2%, gate +3%). v2 ENGAGED (self-check lines
  both ranks, gv2-census preads=runs evidence, zero DISABLED).
- Serve left UP on this config (boot-k3c-pf-gv2.sh, :8000) — new best.
- Code dormant-by-default (DSV41_ENGRAM_GATHER_V2=0 ships off).

## Interpretation

The Round-22 thesis held but the mechanism was refined by measurement:
the 7.5 ms/step "gather loop" cost was dominated by chunk-pool
dispatch/coordination (0.105 ms overhead per 8-future batch × per-table
× per-step, plus lock/GIL churn), not by syscall count (102 inline
preads cost 0.05 ms hot). v2 keeps the syscall count (hash rows are
uniform over 384M rows — runs are length 1) but removes the pool hop
and the per-call alloc/dequant chaining, recovering ~0.5-0.65 ms/call
× 2 tables ≈ 1.0-1.3 ms/step of host execution. +1.27 tok/s pooled is
consistent with ~1 ms/step recovered at ~72 ms steps. The remaining
idle (~2.4 ms sub-1-ms fragments + residual) is elsewhere; 35 needs
acceptance or device-side gains (unchanged conclusion).

## OOM floor log

| point | spark1 avail | spark2 avail | abort? |
|-------|--------------|--------------|--------|
| pre-boot (stop.sh, both empty) | 116 GiB | 117 GiB | no |
| post-boot | 25 GiB | 27 GiB | no (floor 12) |
| post-smoke | 25 GiB | 27 GiB | no (floor 8) |
| post-batch-1 | 25 GiB | 27 GiB | no |
| post-batch-2 | 25 GiB | 27 GiB | no |
| post-bench close | see 22-mem-post32k.log | — | no |

Boot cap: 1 of 3 used. No host CUDA JIT, no trace parsing, bounded dumps.

## Artifacts

- boot-k3c-pf-gv2.sh, 00-boot-gv2.log, 05-smoke.json
- 10-lail-warmup.json, 11-lail-real.ndjson (b1), 12-lail-real-b2.ndjson (b2)
- fetch_jobs.py, lail_runs.sh, validate_gather_v2.py (copy; source of
  truth /home/sfxnz/projects/experiments/dsv41-opt/patches/)
- 20-prose-9run.log, 21-micro-prefill.log, 22-mem-post32k.log
