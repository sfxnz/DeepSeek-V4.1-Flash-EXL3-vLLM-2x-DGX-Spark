# ATTRIBUTION — Round 25 arbitration trace: the 31.37 config (gather-v2 engaged)

**Question**: is 35 tok/s at the L.A.I.L workload's acceptance reachable by
recovering GPU idle, or is the wall device kernels? Decision rule from the
protocol: 35 @ acc 2.3 needs step ≤ 65.7 ms; if un-inflated device-busy alone
is ≥ 65.7 ms → device-bound, no idle-recovery A/B; if device-busy ≤ 62 ms AND
idle ≥ 7 ms with a single env-reachable owner ≥ 4 ms → run that one A/B.

Boot: `boot-k3c-pf-gv2-trace.sh` (the live-best env exactly + ONLY the torch
profiler flag, same pattern as Round 22's trace boot). One L.A.I.L prose
request (61 prompt + 512 completion, streaming) profiled via
`profile_window.sh`; stop → 15 s flush → **docker cp BEFORE stop.sh** (both
Round-18/22 lessons held; cp took 0.1 s but host MemAvail dipped to 2 GiB
transiently — logged below, recovered to 117/117 after stop). Trace:
`traces/dp0_pp0_tp0_dcp0_ep0_rank0.1790020768177050814.pt.trace.json.gz`
(119 MB gz, NOT committed per gitignore convention).

**Step count = 249** (SSE content chunks on the profiled request = 249 →
acc 2.056; cross-checked: p2b_moe n=9880, topk n=9920 ⇒ ~247-248 model
steps + prefill spillover). All per-step numbers below divide by 249.

## (1) Idle distribution NOW (vs Round 18 / Round 22)

| owner (per step) | R18 (k3c) | R22 (k3c+pf, 30.10) | **R25 NOW (k3c+pf+gv2, 31.37)** |
|---|---|---|---|
| stage() hash-event wait-behind-work (blocked IN cudaEventSynchronize) | ~13.0 ms | ~0.0 ms | **~0.0 ms** (0.7 ms/window total) |
| stage() post-sync MAIN-thread gather loop (per-row pread+dequant+pool) | (in 13.0) | **~7.5 ms** | **GONE** — gather now runs in the 2 parallel per-table stage workers (`_fast_stage_one` 2×3.6 ms, `_gather_dequant_v2` 2.85 ms/call, 96 preadvs @ 20 µs) |
| off-thread prefetch pool worker: 191 `posix_fadvise`/step (WILLNEED, 11 µs each) | — | — (not resolved; inside R22's "post-sync tail") | **~3.1 ms of ≥1 ms-gap time sits under these frames** (766 of 1208 ms; tid 1640, NOT the main thread; pf_hit 100 % ⇒ pure waste, but code-gated not env-gated) |
| 0.5–1 ms fragments | ~1.2 | ~2.4 | **0.02 ms** (6 gaps/window) |
| sub-0.5 ms dust | (not counted) | (not counted) | **2.29 ms** (569.6 ms/window — now counted per protocol) |
| **total idle** | **~14.2 ms** | **~9.9 ms** (≥1 ms pool only) | **7.15 ms** (4.87 ≥0.5 ms pool + 2.29 dust) |

Gap structure changed completely: R22 was 237 gaps of 5–10 ms (one per step,
median 7.5); NOW it is 73×5–10 ms + 173×2–5 ms, median 4.69 ms — the old
per-step 7.5 ms block is split/shifted, consistent with gv2 moving the gather
into the parallel workers and the residual being scheduler/launch + off-thread
GIL interference + replay boundaries.

## (2) Device-busy, un-inflated

- Device busy (union of kernel timestamps, profiler ON): 16 247 ms / 249 =
  **65.25 ms/step** (90.1 % duty). Kernel-time sum is 70.61 ms/step — the
  7.6 % excess is real multi-stream concurrency (NCCL/compute overlap),
  not inflation; the union number is the honest device-busy.
- Profiled wall = 72.40 ms/step. Un-profiled step from L.A.I.L live tok/s:
  at 29.5–33.7 tok/s and acc 2.06–2.55 the true step is **65.5–69.3 ms**
  (31.37 median ⇒ ~68 ms; repo 9-run median 36.8 ⇒ ~66 ms at acc 2.55).
- **Inflation estimate: profiler costs ~3–5 ms/step (~5 %)** — the profiled
  wall (72.4) sits ~4 ms above the live-implied step (~68). Since kernel
  timestamps come from the GPU-side clock, device-busy 65.25 is itself
  essentially un-inflated; the inflation lives in the CPU-side launch path
  (which widens gaps, not kernels).
- Device-busy composition (kernel-sum/window): p2b MoE 5.59 s (**22.4 ms/step**),
  dense GEMM (flashinfer b12x) 4.38 s (**17.6 ms/step**), AR NCCL 1.14 s
  (**4.6 ms/step**), draft/aux (bf16 GEMM, prenorm, sparse-MLA, topk,
  elementwise ~0.2 s each) ≈ 3.3 s (~13 ms/step with the rest), prefill
  residue 1.33 s. Matches the protocol's prior (33–34 MoE + 19–20 dense +
  6 AR + 3–4 draft) once the profiled-window's lower acc is accounted.

## (3) THE ARBITRATION

- 35 tok/s @ acc 2.3 needs step ≤ **65.7 ms**.
- Device-busy alone = **65.25 ms/step** ≥ 65.7 minus noise — and at the
  window's acc 2.06 the requirement is even harder (~70.9 ms needs ≤62.4).
- Even PERFECT recovery of all 7.15 ms of idle (impossible: it is diffuse —
  no single owner ≥4 ms that is env-reachable; the largest coherent pool,
  the off-thread fadvise storm at ~3.1 ms/step, is a code patch — cap the
  fadvise fan-out, pf_hit is 100 % — not an env lever, and it is not even
  on the critical-path thread) gives 65.25 vs 65.71 required: **zero margin**.
- **VERDICT: device-bound.** 35 tok/s at this acceptance is NOT reachable by
  idle recovery on this config. The wall is the kernels: p2b MoE 22.4 ms +
  dense GEMM 17.6 ms + AR 4.6 ms + draft/aux ~13 ms (per profiled window).
  No A/B boot was run (gate conditions not met on both prongs). Next levers
  are code/kernel-side (p2b/dense GEMM work) or acceptance (draft/verify).

## Numbers cross-check

- busy 65.25 + idle-pool 4.87 + dust 2.29 = 72.41 = profiled wall 72.40 ✓
- R22 same-accounting: busy 65.30 + pool 7.61 + dust 2.27 = 75.18 ≈ R22's
  75.18 span ✓ (idle pool shrank 7.61 → 4.87 while busy held ~65.3 —
  gv2's +1.27 tok/s came out of host time, exactly as claimed in Round 23.)

## OOM floor log

| point | spark1 avail | spark2 avail | abort? |
|---|---|---|---|
| pre-boot (stop.sh, both empty) | 117 GiB | 117 GiB | no |
| post-boot (trace) | 25 GiB | 27 GiB | no (floor 12) |
| post-smoke (323 ✓) | 25 GiB | 26 GiB | no (floor 8) |
| during docker cp (transient) | **2 GiB** | — | logged: transient page-cache dip during cp, recovered post-stop; both ranks 117/117 after stop (same phenomenon as Round 22 boot 1) |
| post-stop (parse on idle host) | 117 GiB | 117 GiB | no |

Boot cap: 1 of 3 used for the trace; boot 2 = restore of
`boot-k3c-pf-gv2.sh` (best serve), below. No host CUDA JIT; bounded dumps;
trace 119 MB < 2 GB cap.

## Artifacts

- `boot-k3c-pf-gv2-trace.sh`, `profile_window.sh`, `00-boot-trace.log`,
  `05-smoke.json`, `10-profile-window.log`
- `parse-gap-split.txt` (sync split), `parse-post-sync-attribution.txt`
  (gap-owner attribution), `parse-idle-summary.txt` (canonical summary),
  `parse-gaps-stacks.txt` (stack dumps), `parse-reference-trace2.ipynb`
- parsers copied from Round 22 / kernel_study (`parse_gap_split.py`,
  `parse_gaps.py`, `parse_gaps_stacks.py`)
