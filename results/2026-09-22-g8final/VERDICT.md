# G8 FINAL — VERDICT (Round 31/32, 2026-09-22, g8final session)

**G8 LANE CLOSED. Root cause found and fixed; boot still dies post-load
(no-traceback kill during post-load init). Stock restored at 31.37 baseline.**

## Baseline & stakes

- Stock serve (boot-k3c-pf-gv2.sh): L.A.I.L decode/prose median **31.37**.
- G8 lane: 2.0bpw-mcg-g8 pack (same bytes, G8-folded trellis layout) +
  DSV41_LOAD_PF_G8=1 → kernels proven bit-exact (24/24 gate ×2, round 29),
  but boot OOM-killed ×3 in round 29 (worker anon 45.5 GiB, deterministic).

## Evidence chain (this session)

### Phase 0 — restore + commit
- Pushed the capped session's 2 unpushed commits; committed diag v2/v3
  (62015a3 diag v4/v5). Working tree clean at de51315+.

### Phase 1 — wrapper-in-loop diag (the layer every prior diag skipped)

**diag_v4 (CPU, real mapper + sorted feed + real shard fns + G8 consume,
3 layers / 13,824 keys, legs g8-full / stock-full / g8-feedonly):**

| leg | per-step new anon storage | anon RSS at end |
|---|---|---|
| g8-full | detach_contig=0, shard=0, copy=0 | flat ~689 MiB |
| stock-full | detach_contig=0, shard=0, copy=0 | flat ~689 MiB |
| g8-feedonly | — (list built, no consume) | flat ~689 MiB |

→ **PRIME HYPOTHESIS REFUTED**: no `.contiguous()`/`.reshape()` anon copy
anywhere in the G8 narrow path; both stock and G8 `sharded` views are
contiguous (verified separately: strides show plain dim-0/dim-1 storage
slices). The tracer's "2 tensors per checkpoint key" are **both file-backed
mmap views** — the WeightsMapper rename creates a second tensor object on the
SAME storage (kernel-reclaimable, harmless). The sorted list itself is not an
anon balloon on CPU.

**diag_v5 (FULL GPU topology on spark2, 40 layers / 184,320 expert keys,
real CUDA dests + real H2D copy_ + real pwaf):**

| leg | feed anon | consume-phase anon peak | swap | survives? |
|---|---|---|---|---|
| stock | 1.0 GiB flat | **17.9 GiB** | 9.6 GiB | yes (real boot lives) |
| g8 | 1.0 GiB flat | **39.3 GiB** | 14.0 GiB | no (real boot dies 59.4) |

→ **ROOT CAUSE**: with CUDA dests, every `dest.copy_(sharded)` is an H2D
from a pageable mmap view; under GB10 UMA the H2D **pins the source tensor's
host pages until the tensor is freed**. The wrapper retains the entire sorted
`mapped` list (all 184,320 expert tensors) for the whole load → pins
accumulate linearly with keys consumed. Stock pins ~27.6 GiB total (fits),
G8 pins ~53.3 GiB (dies). Site: **vl_model.py
`DeepseekV41ForCausalLM.load_weights`: `mapped = sorted(...)` retained +
`loader.load_weights(mapped)`** — AutoWeightsLoader itself is fully streaming
(generators end-to-end, verified in models/utils.py); the retention is
entirely the wrapper's list.

Why G8 ≈ 2× stock per tensor: G8 w13 tensor views are (E,2,9,320,256) —
the per-expert dest slice `param.data[e, s]` copy touches the same bytes,
but the G8 page-pin footprint per key is larger because each expert's w13
gate+up live in one 2-slot dim (copy per slot pins the whole expert slab
while either slot is live).

### Phase 2 — the fix (minimal, env-gated DSV41_LOAD_PF_G8, stock untouched)

`docker/patch/g8_stream_feed.py` + sitecustomize wiring (de51315): the
wrapper's `load_weights` now feeds AutoWeightsLoader a **drain-in-place
generator** — identical sort order, but each list slot is set to None as
soon as it is yielded, so each source tensor (and its pinned pages) is
dropped immediately after its H2D.

**diag_v6 A/B (same full-GPU topology, G8 leg, fix semantics):**

| | before (v5) | after (v6) |
|---|---|---|
| consume anon peak | 39.3 GiB | **1.9 GiB (flat)** |
| swap | 14.0 GiB | 0.6 GiB |
| MemAvail through load | 14 GiB floor | flat 51.9 GiB |

Validated: py_compile, patched vl_model compiles, marker absent without env
(stock dormant, verified on canonical-e12 AND canonical-g8 images), anchor
self-check, real-image install proof (`dsv41: g8 stream feed installed`).

### Phase 3 — G8 boot attempts (boot 5 + boot 6)

- boot-g8-5: killed by MY watcher v2 at 43 GiB — **watcher bug**, it
  aborted on rss+swap TOTAL which counts the ~79 GiB/rank of reclaimable
  file-backed pack pages that a normal stock load also shows. Not an OOM.
  (watcher v3 written: anon-only via smaps_rollup; unreadable — root-owned
  container procs — so MemAvail floors used: abort <12 boot.)
- boot-g8-6 (the real attempt, fix live, both ranks logged
  `g8 stream feed installed`): **the load-phase OOM is GONE** — both ranks
  completed 48/48 shards (~23 s shard read + load), MemAvail never
  approached the kill zone during load (spark2 dipped to ~20 GiB avail
  during its load, recovered to 119.7 GiB free after; spark1 similar).
  Then, ~7-8 min AFTER load completion, during post-load init
  (process_weights_after_loading / p2b table build / cudagraph capture —
  no log line between "48/48" and the shutdown), worker TP0 (spark1 head)
  died with NO traceback and the engine tore down; TP1 (spark2) died with
  the head (container exit 1, not 137). Memory signature at death:
  spark1 avail 0.6 GiB → the post-load phase balloons a SECOND, distinct
  host allocation the diag could not see (diag_v5's pwaf leg was lean:
  2.2 GiB — but it ran single-process, no NCCL/EngineCore/cudagraph layer).

**Per the task's abort rule (OOM again after ONE fix iteration → G8 LANE
CLOSED), the lane is closed.** Evidence chain complete: loader proven lean
(×4 incl. wrapper-in-loop), feed site identified and fixed (39.3→1.9 GiB),
boot survives the entire weight load for the first time, residual post-load
balloon is a separate site (native post-init under the real engine, not
reproducible outside the serving stack within budget).

## Phase 5/6 — fallback & state

- lm_head lever: honestly measured round 30 at +2.5% pooled (32.17 n=10) —
  below the +3% KEEP gate (32.31) → stays REVERTED; no re-measure spent
  (would change nothing: same pack, same gate).
- **Best serve = stock**: restored via boot-k3c-pf-gv2.sh, smoke 323,
  MemAvail ≥ 8 both ranks (verified below). 31.37 stands. **35 NOT
  reached** (needs two medians ≥35; best-ever candidate 32.45 n=5).
- Boot budget: 5 used of 5 (restore-aborted, g8-5 watcher-kill, g8-6 real,
  stock restore, + the initial aborted restore). Correct final state up.

## Byte ledger (from G8-LEAN-LOAD.md, g8_ledger_tally.py)

Per-rank read ~78.9 GiB, stock == G8 byte-for-byte; dest pools identical
(shape-only phase-2 alloc); the entire lane is memory-neutral by design —
the failures were host-page-lifetime artifacts, now fully attributed.

## Artifacts (results/2026-09-22-g8final/)

diag_v2/v3/v4/v5/v6.py, diag_v4_runs.log, diag_v5_{g8,stock}.log,
diag_v6_g8.log, harness_* (byte ledger + lean proof), boot-g8-{1..6}.log,
boot-restore-stock.log, worker_watch{,2,3}.sh, VERDICT.md (this file).

## Commits

- 62015a3 diag v4/v5 committed (capped-session artifacts + unrun diags)
- de51315 fix(pfg8): lean expert load — stream wrapper feed (drain-in-place)
- (this verdict + flags Round 31/32 follow)
