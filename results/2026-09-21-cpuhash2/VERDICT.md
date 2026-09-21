# VERDICT — 2026-09-21 Round 20: CPU-hash stream-ordering fix — ENGAGED, REVERT on L.A.I.L

Baseline: k3c 28.76 median L.A.I.L (results/2026-09-21-capture-pf). Target 35+.
Boot cap: 2 of 3 used (diagnostic boot skipped — static analysis was conclusive).

## Diagnosis (BOOT 1 replaced by source audit)

The Round-19 named candidate (combine reads
`req_states.draft_tokens` written by `set_draft_tokens` at the commit hook,
"snapshot point wrong") is **refuted from source** (image `canonical-e12`):

- `model_runner.py:1980`: `self.req_states.draft_tokens[input_batch.idx_mapping]
  = draft_tokens` — scatters the EXACT tensor `propose()` returned (the
  enqueue hook's `draft_tokens` argument) into req_states. Same data.
- `input_batch.py:445` `_combine_sampled_and_draft_tokens_kernel`: next-step
  ids = `last_sampled_tokens[req_state_idx]` (bonus row 0) +
  `req_states.draft_tokens` rows 1..k. Verbatim read of what we snapshot.
- `spec_decode/utils.py:22` `DraftTokensHandler.set_draft_tokens`: with no
  structured-output requests it sets `draft_tokens_np = None` and RETURNS —
  it never touches any GPU buffer (structured-output scheduler validation
  only). Red herring.
- Ordering question answered: the commit hook (model_runner.py:1690) runs on
  the host thread BEFORE `model_state.prepare_inputs` (line 1733) →
  `EngramDiskStager.stage()` — so a commit-hook snapshot would be TOO LATE
  for the same step's CPU hash. The enqueue (post-propose) point is correct.

**Actual root cause (our own patch, commit 35e05fd)**: in all three
side-stream blocks of `engram_cpu_hash.py` (enqueue snapshot, warmup stage-A
mirror, steady-state canary) — and identically in `engram_prefetch_v3.py:170` —

```python
with torch.cuda.stream(side):
    side.wait_stream(torch.cuda.current_stream())   # NO-OP: waits ITSELF
```

Inside the with-block `current_stream()` IS the side stream, so the wait is a
self-wait no-op: the pinned D2H copies were UNORDERED w.r.t. main-stream work
queued so far — including the dspark draft CUDA-graph replay that writes
`speculator.draft_tokens` (`dflash/speculator.py:484` returns the persistent
buffer; replay = `query_cudagraph_manager.run_fullgraph`). The snapshot raced
the replay and read the PREVIOUS generation's drafts. This explains every
Round-19 symptom exactly: bonus row always matched (sampler output written
earlier in the step), draft rows stale-but-plausible (pid=14 = stale id,
positions all matched = host-computed), mirror stage-A passed (prefill
boundary, queue drained). Also explains the historic ~33% prefetch-v3
intersect (same no-op in its enqueue).

Fix: hoist `side.wait_stream(main)` BEFORE the with-block (3 sites cpu_hash +
1 site prefetch_v3), commit 35e05fd, gates all green (bash -n, py_compile,
render --check, 225 unittests, offline validate_cpu_hash ALL PASS).

## Engagement evidence (the fix boot)

Boot `results/2026-09-21-cpuhash2/boot-k3c-cpuhash-fix.sh` (k3c +
DSV41_ENGRAM_CPU_HASH=1, commit 35e05fd in docker/patch):

- Wiring engaged both ranks: `engram cpu-hash stager installed` + `runner
  hooks installed` + `armed (layers=2 ngram=4 heads=8 span=12 depth=3)`
  (TP0 spark1 / TP1 spark2).
- Smoke 17×19 → 323 ✓.
- **`dsv41: engram cpu-hash ACTIVE (mirror+predict bit-exact x4;
  prepare_inputs D2H+event sync removed)` — BOTH ranks.** The Round-19
  PREDICT failure is gone; the prediction rule (prefetch-v3 chunk rule) was
  always right, the snapshot was racy.
- Fast path live: `engram cpu-hash fast-path steps=500` (TP0) — hundreds of
  steady-state steps took the no-sync path; no canary DISABLED line, no
  fallback warnings. The `hashes_ready.synchronize()` is genuinely SKIPPED
  on engaged steps (fast-path counter only increments when the stock
  hash+D2H+sync block is skipped).

## L.A.I.L (n=5 + independent n=5, decode prose c1)

| batch | runs (tok/s) | median |
|---|---|---|
| 1 | 27.29 / 28.19 / 28.86 / 26.84 / 30.68 | 28.19 |
| 2 (independent) | 27.22 / 25.91 / 26.32 / 27.62 / 29.14 | 27.22 |
| **all 10** | | **27.46** |

vs k3c baseline 28.76 (and k3c boot-to-boot medians 27.40/26.13/28.01 in
Round 19). **35 NOT crossed; median moved the wrong way.**

## Verdict: REVERT (patch stays in repo, dormant, default-off)

The sync removal is real (engagement + fast-path counters) but does NOT
convert to wall-clock: the off-thread worker's pread/dequant gather + the
per-step async canary (an extra side-stream GPU hash launch every step)
evidently cost about as much host/GPU contention as the 13-14 ms sync saved.
Both independent medians < 28.76 baseline ⇒ revert the env lever; the code
fix (correct stream ordering) stays dormant in the repo for any future arm
that reuses the snapshot machinery (prefetch-v3's fixed ordering also makes
a future DSV41_ENGRAM_PREFETCH=1 re-arm actually testable).

The 13-14 ms/step idle pool attribution (Round 18) stands, but "delete the
sync via CPU-side prediction" is now measured END-TO-END and does not
recover it at L.A.I.L decode shapes. Next lever must attack the pool
differently (e.g. overlap/deferral of the gather, or shrink the pool's
producer) — not re-predict the hashes.

## Serve state

Restored stock k3c via `results/2026-09-21-capture-pf/boot-k3c.sh`
(boot log `30-boot-k3c-restore.log`), no CPU-hash env (`DSV41_ENGRAM_CPU_HASH=0`
in container env; hooks installed but dormant — 0 armed/ACTIVE lines).
Smoke 323 ✓; L.A.I.L n=3 confirm: 30.04 / 29.37 / 29.38 → **median 29.38**
(k3c boot-to-boot spread; best-of-boot medians stays 28.76).

## OOM floor log

| boot | pre s1/s2 | post-smoke s1/s2 | post-LAIL s1 | aborts |
|---|---|---|---|---|
| fix boot (CPU_HASH=1) | 115/117 | 25/25 | 25 | none |
| restore k3c | 116/117 | below | below | none |

stop.sh + docker ps verified empty on spark1 AND spark2 before every boot;
no host CUDA JIT; no trace parsing.

## Jobs

warmup discard 395474dd2e35; batch1 a8928b73e09f 020dc0224287 a28d19a54ea2
91adc6e1e762 8f1e2348c28b; batch2 f491b1604dee 756ab4896882 29c2a9bd0b31
e4d6f9222032 f9bac738bd4b; restore confirm f40b5a671270 0fadce8058f5
21b08df76311.
