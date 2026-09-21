# VERDICT — 2026-09-21 Round 24: engram stage defer (off-thread next-step gather)

Baseline: Round-23 best **31.37** pooled n=10 median (k3c + `DSV41_ENGRAM_PREFETCH=1`
+ `CENSUS=1` + `DSV41_ENGRAM_GATHER_V2=1`, boot-k3c-pf-gv2.sh). Target 35+.

Lever (single): `DSV41_ENGRAM_DEFER=1` added to the identical env set —
boot script `boot-k3c-pf-gv2-defer.sh` (diff vs
`../2026-09-21-gatherv2/boot-k3c-pf-gv2.sh` = one env line). Code:
`docker/patch/engram_defer.py` (commits ea871aa + b2c8ad0 + dc548ef),
wired via sitecustomize after the gather-v2 block, env forwarded in
run.sh (head `-e` + worker ssh line).

## What defer is (Round-18 option 2, made safe by 35e05fd ordering)

ONE persistent worker thread per rank gathers the ENTIRE next step's
rows off the prepare_inputs critical path:

- post-propose enqueue (same site as the cpu-hash hook) snapshots
  sampler/draft outputs to pinned mirrors on a side stream ordered
  behind ALL main-stream work (wait issued while the MAIN stream is
  current — the Round-19/20/35e05fd lesson), records `snap_ev`;
- the worker (preadv/numpy — GIL released) reconstructs the next chunk
  (the bit-exact prefetch-v3 rule), CPU-hashes it with the cpu-hash
  numpy mirror, gathers every table via the v2 preadv path into
  DOUBLE-BUFFERED per-table pinned slots (its own buffers; the
  critical-path hash_host/rows_host are never touched), waits `h2d_ev`
  (bounds pinned-slot reuse), H2Ds into a second device staging set on
  the side stream (wait_stream BEFORE the with-block), records `h2d_ev`;
- stage() (after the stock GPU hash+sync, still needed by the graph):
  gen+batch-signature match -> ONE DtoD per table into staged_rows,
  ZERO gather work on the critical path; else the exact sync v2 path
  (identical behavior — never slower). Bounded 2 s join, queried
  events, first 4 hits bit-verified against the sync gather (int16
  bitview — fp8 NaN patterns make float equality lie); ANY anomaly
  disarms permanently with ONE warning line.

Boot-2 finding (fixed in dc548ef): an adaptive-verification micro-step
calls the post-propose hook with a MALFORMED snapshot
(`sum(qsl[1:]) != num_tokens`, e.g. qsl=[0,4,4]/num_tokens=2).
Enqueueing it churned `_df_gen`, made the good worker abandon (gen
check) and made stage() reject every good ready as "superseded" ->
0% hits, silent (census only printed on hits). Fixes: (a) enqueue
validates qsl monotonic + sum == num_tokens and skips malformed
snapshots; (b) bounded `[defer-miss]` reason logs + attempt-cadence
census (prints even on all-miss windows); (c) seen-gen double-consume
guard.

## Offline validation — ALL PASS

- `validate_defer_chain.py` (in results dir): canonical chain
  prestage->census->fast->v3->cpu-hash->v2->defer applies cleanly on
  the REAL snip sources; anchors, hooks, stage(input_batch=) wiring,
  sync fallback intact, idempotent re-apply, py_compile all targets.
- `validate_defer_fn.py`: threaded harness exec'ing the REAL patched
  v2 `_gv2_read_runs` + the REAL defer methods (predict/hash/try_stage/
  disarm) against real safetensors geometry (F8_E4M3 [N,256] /
  F8_E8M0 [N,8], sequential packing):
  - v2 preadv read + dequant vs numpy reference bit-exact (x3 trials;
    the numpy e4m3 decoder itself was verified against the image's
    torch over all 256 bit patterns — fp8_probe.py, canonical-e12);
  - T1 happy path 30 steps: served rows == sync reference EVERY step
    (uint32 bitview; catches NaN-bit diffs), warm 0->4, hits=30;
  - T2 varying accept 1..k+1, 2 requests: bit-exact every step;
  - T3 forced scheduler surprises: fallback serves sync rows, stays
    ARMED (miss != disarm), misses counted;
  - T4 hung worker (30 s sleep): stage falls back after bounded wait,
    serve continues, total hang cost <0.1 s;
  - T5 worker exception: ONE `DISABLED -> sync v2 path` line, sync
    serves forever after;
  - T6 warm-verify mismatch (corrupted hash): ONE DISABLED line,
    sync serves;
  - T7 GIL: worker launches no GPU work; stream ops are ordering-only;
    gather is the v2 preadv path.
  (Two harness fixture bugs were found and fixed during bring-up —
  primes that pushed every hash out of vocab [all-zero outputs,
  vacuous pass] and NaN!=NaN compares; the SHIPPED code was unchanged
  by both.)

## Live engagement (boot 3 of cap 3)

- Install + armed lines BOTH ranks; `gather v2 self-check bit-exact`
  both ranks; smoke `17 * 19` -> **323** OK (05-smoke3.json);
  MemAvail 24/26 GiB (floors 12/8 respected at every checkpoint).
- **`engram defer ACTIVE (warm-verified bit-exact x4)`** spark1.
- Steady state `[defer-census]` (spark1 TP0, L.A.I.L prose decode):
  hits 21-32 / window(32), **hit% 65.6-100, typical ~88**, late=0
  always, worker 4.6-7.7 ms/prediction (step ~72 ms — comfortable
  overlap). spark2 similar (46.9-96.9, late=0). ZERO `DISABLED` lines
  on either rank. Fallback rate = miss fraction ~12% steady
  (req-finish/schedule surprises + occasional micro-steps), 0% late.

## L.A.I.L (warmup discard + n=5, decode prose c=1)

| batch | jobs | runs (tok/s, sorted) | median |
|-------|------|----------------------|--------|
| warmup (discarded) | 2 jobs | — | — |
| 1 (n=5) | 047ef4266d1d ffa7b80dbc3e 2bb3e408af3c 23b628e729e3 97f25c3cc03a | 28.73 29.72 30.26 30.28 30.55 | **30.28** |

Median 30.28 < 32 -> second independent batch NOT triggered (rule:
run batch 2 only if median >= 32). KEEP gate: pooled >= 32.3 — FAIL.
(vs 31.37 baseline = **-3.5%**, inside the run-to-run band 29.5-33.7.)

**35 status: NOT crossed.** (Two-medians rule moot at this level.)

## Result: REVERT env lever (DSV41_ENGRAM_DEFER back to 0)

Protocol: engaged-but-flat -> census/defer-hit evidence says defer-hit
~88-100% with late=0 and tok/s flat. **The stage() host gather was NOT
the remaining wall** — moving ~7 ms/step of gather fully off the
critical path (verified engaged) does not move L.A.I.L tok/s. The
Round-22 residual idle attribution must be re-examined: the ~2.4 ms
sub-1-ms fragments + whatever the DtoD+event-wait now costs sit
elsewhere (scheduler/step overheads, kernel launch tails), not in the
gather. Record: idle is elsewhere.

Code stays dormant (default off, DSV41_ENGRAM_DEFER=0 ships). Serve
restored UP on the Round-23 best config (boot-k3c-pf-gv2.sh, defer
env absent), smoke 323 OK.

## OOM floor log

| point | spark1 avail | spark2 avail | abort? |
|-------|--------------|--------------|--------|
| pre-boot-1 (both empty) | 116 GiB | 117 GiB | no |
| post-boot-1 | 25 GiB | 26 GiB | no (floor 12) |
| post-smoke-1 | 25 GiB | 26 GiB | no (floor 8) |
| pre-boot-2 (both empty) | 117 GiB | 117 GiB | no |
| post-boot-2 | 24-25 GiB | 26-27 GiB | no |
| pre-boot-3 (both empty) | 116 GiB | 117 GiB | no |
| post-boot-3 / post-smoke-3 | 24 GiB | 26 GiB | no |
| post-batch-1 | 24 GiB | 26 GiB | no |
| restore-best post-boot | in 03-restore-best.log | — | no |

Boot cap: 3 of 3 used (boot 3 doubled as the measurement boot; the
restore boot is the Round-23 script, not a defer lever boot). No host
CUDA JIT, no trace parsing, bounded dumps.

## Artifacts

- boot-k3c-pf-gv2-defer.sh, 00-boot-defer.log, 01-boot-defer2.log,
  02-boot-defer3.log, 05-smoke{,2,3}.json
- 11-lail-real-b1.ndjson (boot-3 rows tagged batch1; the file also
  carries 5 stale boot-1 rows — jobs d4fffd770bcf 28e64a8f5131
  bac2e7a600ad 1fa08843292b 99edcd9ebde1 from the disarmed boot-1
  serve — analysis used the 5 boot-3 job ids above)
- validate_defer_chain.py, validate_defer_fn.py, fetch_jobs.py,
  lail_runs.sh
- source of truth: /home/sfxnz/projects/experiments/dsv41-opt/patches/
  (engram_defer.py + both validators + fp8_probe.py)
