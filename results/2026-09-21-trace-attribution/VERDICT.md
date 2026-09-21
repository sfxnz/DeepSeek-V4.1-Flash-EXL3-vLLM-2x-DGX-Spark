# VERDICT — 2026-09-21 trace attribution + SOFTMAX_VERIFY A/B (round 18)

Baseline: campaign best 28.76 L.A.I.L (k3c = k3 + capture sizes [1,3,4,6,8]
+ greedy, `results/2026-09-21-capture-pf/boot-k3c.sh`). Target: 35+ tok/s.

## BOOT 1 — TRACE ATTRIBUTION on stock k3c (see ATTRIBUTION.md)

- Serve: k3c byte-identical + `--profiler-config
  {"profiler":"torch","torch_profiler_dir":"/tmp/dsv41-traces"}` (this is
  what mounts `/start_profile`//stop_profile in this vLLM build — the
  round-5/7/11 "profiler replica" precedent, now captured as
  `boot-k3c-trace.sh` + `profile_window.sh`).
- Window: ONE L.A.I.L prose request (512 tok, t=0.2, c=1), 20.57 s,
  5.02M events, 638k kernels. Trace (119 MB gz) kept locally under
  `traces/` (gitignored), sha256 24f51860fad84767…
- Result: device busy ~71.4 ms/step (profiler-inflated), pure GPU idle
  ~14.2 ms/step across 256 gaps ≥ 1 ms — and **99.9% of that idle is ONE
  stack**: `engram.py EngramDiskStager.stage → hashes_ready.synchronize()`
  (a hard cudaEventSynchronize in `prepare_inputs`, waiting on a tiny D2H
  hash copy queued behind the previous step's graph). Classes #2 (eager
  sampler region) / #4 (sampler softmax) / #5 (indexer) closed at <0.15%
  of idle; #3 is the mechanism's tail, not an owner.
- **Winning patch class: #1 — per-step host critical path in
  prepare_inputs, anchor = the Engram stage hard sync.** Named fix (NOT
  implemented this session, per plan): compute the next step's token
  hashes on CPU (input ids are host-resident before prepare_inputs; a
  verified CPU port of `_hash_ids_kernel` exists in the prefetch-v3
  work) → delete the D2H + `hashes_ready.synchronize()` entirely;
  fallback shape = overlap/defer the NVMe gather to a side thread with
  the event waited only at replay. Expected recovery ≈ the whole
  13-14 ms/step pool ⇒ step ~78.5 → ~64-67 ms ⇒ ~34.5-37 tok/s at
  acc 2.26 — 35 is reachable from this one fix alone.
- Note: in-container trace export dropped host MemAvail to 3 GiB (<8
  floor) → serve stopped before parsing; both ranks back to 117 GiB
  after stop. Lesson logged: stop or defer big host work after any
  profiler export while serve is resident.

## BOOT 2 — SOFTMAX_VERIFY A/B on k3c: REVERT (env-off; not a keeper)

- Boot: `boot-k3c-softmax.sh` = boot-k3c.sh + only lever
  `DSV41_DSPARK_SOFTMAX_VERIFY=1`. Wrap confirmed engaged at boot
  (`dsv41: DSpark greedy propose, softmax verify` in engine log).
  Knob resolution verified boot-fixed (sitecustomize wraps
  DSparkSpeculator.__init__/_sample_logits at import; env read once) —
  it can never be folded into another boot's env.
- Smoke 323 ✅; MemAvail post-smoke 25/27 GiB.
- L.A.I.L via :8765/api/bench/perf (X-Lail-Token), warmup discarded
  (job 60d83e9d46c8), real n=3: **27.14 / 26.46 / 28.56 → median 27.14**
  (jobs 1f0a5885dbd0 / 3cad5d567a2b / fb60db5b21fb).
- Repo harness cross-check: median 27.86 tok/s, **acc_len 2.20**,
  draft acceptance 0.399.
- Gate ≥ 29.62 (≥+3% vs 28.76): **FAILED (−5.6%)**. The k5-era
  acceptance gain did NOT transfer to k3c: at k3 the verify batch is 4
  rows (not 6), the softmax-q correction matters less, and the greedy
  propose's full-vocab `draft_logits` buffer adds per-step work.
  **Verdict: REVERT** — knob stays documented-but-dead
  (`DSV41_DSPARK_SOFTMAX_VERIFY=0` default). Campaign best remains
  28.76 k3c.

## BOOT 3 — restore best serve (k3c stock)

`results/2026-09-21-capture-pf/boot-k3c.sh` rebooted; left UP on :8000.
Final smoke 323 ✅, MemAvail 25/27 GiB, 0 softmax-verify lines in engine
log (wrap confirmed absent). This is the campaign-best config.

## OOM floor log

| boot | pre-boot s1/s2 | post-smoke s1/s2 | post-L.A.I.L s1/s2 | abort |
|---|---|---|---|---|
| 1 k3c+profiler (attempt a: ignore_frontend crash, aborted at smoke) | 116/117 | — | — | none |
| 1 k3c+profiler (attempt b) | 116/117 | 26/27 | 3* post-export → serve stopped, restored 117/117 | none |
| 2 k3c+SOFTMAX_VERIFY | 116/117 | 25/27 | 25/27 | none |
| 3 k3c restore | 116/117 | 25/27 | 25/27 | none |

*the 3 GiB reading was host page-cache pressure from docker cp of the
119 MB trace + in-container export, not a leak; stop restored full memory.

All boots: ./stop.sh + docker ps empty on BOTH ranks before each boot.

## Gap to 35

Best 28.76 → 35 target: **6.24 tok/s short**. The measured idle pool
(~13-14 ms/step, single owner) is worth ~+6 tok/s by itself at unchanged
acceptance — the CPU-hash patch class is the single remaining lever that
spans the gap.
