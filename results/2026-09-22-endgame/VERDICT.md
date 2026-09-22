# ENDGAME VERDICT — 2026-09-22 (session started ~01:30 BST)

## Outcome: **G8 LANE ABORTED at the GPU bit-exact gate (Phase 2). Serve
restored to stock live-best (31.37 tok/s baseline config). 35 tok/s NOT
reached this session — levers undelivered, not measured.**

## Phase verdicts

| Phase | Verdict |
|---|---|
| 0 — pack rebuild wait | **DONE.** Both ranks Exited (0): r1 04:12, r2 04:10 BST, 20/20 expert shards each (46,080 trellis tensors total, `done experts=23040 elapsed≈21900s` per rank). Zero error/fail/traceback lines in either rebuild log. |
| 1 — assemble | **DONE (spark1 side).** Local pack 48/48 shards, 334 GB (du), index rebuilt (`tensors=188245`, 48 files in weight_map). Layout verification: **46080/46080** routed-expert trellis tensors have the folded G8 shapes — (18,320,256)×30720 gate/up + (40,144,256)×15360 down — across all 40 expert shards (shard-header scan, both ranks' outputs). `permute_pack_group_major.py` dry-run parses the pack (plan: 358.1 GB, 48 shards, 46080 trellis tensors). spark2 push was killed at ~8% during the abort (see Phase 2). |
| 2 — GPU bit-exact gate | **FAIL → ABORT G8 LANE.** See below. |
| 3 — boot G8 | NOT RUN (aborted upstream). |
| 4 — measure G8 | NOT RUN. |
| 5 — lm_head stack | NOT RUN (G8 pack never served; lm_head lever depends on the G8 pack per plan). |
| 6 — fadvise cap | NOT RUN. |
| 7 — close | Stock serve restored + smoke-verified; VERDICT.md (this file) written; flags/commit below. |

## Phase 2 — the gate, in full

The staged `pfg8_bitexact_gate.py` did not match the shipped image's kernel
signatures (pybind `exl3_gemm(A,B,C,suh,A_had,svh,shape,mcg,mul1,sms)`, and
`pf_g8_set` drives only the DEVICE constants while the host-side shape checks
and `LinearEXL3.K` derive read `DSV41_LOAD_PF_G8` via a static-cached
`pfg8::env_value()`). It was rewritten (kept in
`results/2026-09-21-pfg8/pfg8_bitexact_gate.py`, two-process design):

- ref process (env unset): stock-layout trellis, stock kernels via the REAL
  serving entry (`vllm_exl3.exl3.make_linear_exl3` → `LinearEXL3.forward`,
  with `AUTO_RECONSTRUCT_THRESHOLD` forced high so m=256/512 stay on the
  trellis kernels — discovered the staged script compared against
  `reconstruct_hgemm` for m>144, an algorithm not under test).
- g8 process (`DSV41_LOAD_PF_G8=1`): same seeded inputs, G8-folded trellis,
  env-gated ported kernels; `torch.equal` per point.
- Determinism control: two ref runs bit-identical (16/16 SAME) → harness
  sound; same-layout GEMV has no split-K nondeterminism.

Results (4 independent G8 runs, logs in this dir):

| Point | run1 | run2 | run3 | run4 |
|---|---|---|---|---|
| exl3_gemm gate_up m=64..512 | PASS×4 | PASS×4 | PASS×4 | PASS×4 |
| exl3_gemm down m=64..512 | PASS×4 | PASS×4 | PASS×4 | PASS×4 |
| exl3_gemv gate_up m=1,2,4,8 | PASS×4 | PASS×4 | PASS×4 | PASS×4 |
| exl3_gemv down m=1 | PASS | PASS | **FAIL** | **FAIL** |
| exl3_gemv down m=2 | PASS | PASS | **FAIL** | **FAIL** |
| exl3_gemv down m=4 | **FAIL** | **FAIL** | **FAIL** | **FAIL** |
| exl3_gemv down m=8 | PASS | PASS | PASS | PASS |
| p2b_fused_moe m∈{1,2,4,8} | not reached — **CUDA illegal memory access** in the harness p2b call (crashed in the stock-flag reference leg) | | | |

Interpretation (recorded, not debugged — per mandate):
- `exl3_gemv` down-shape (NT=144, the shape where 144 is NOT a multiple of
  the G8 warp-stream width that gate_up's 320 hides) mismatches
  **non-deterministically** at m∈{1,2,4} — same inputs, different wrong
  answers across runs → a race/edge in the ported QTIP small-m GEMV reader
  (`exl3_gemv_kernel.cuh` DEC5 adaptation), not a layout math error (m=8 and
  all GEMM points are exact).
- The p2b in-process A/B harness itself could not complete (illegal memory
  access) — the decode fused-MoE path therefore remains **unproven** in-image,
  consistent with BOOT-CHAIN-AUDIT's named R2 risk.

Mandate rule: ANY mismatch = ABORT G8 lane, restore stock, report; do not
debug kernels in this session. Executed.

## Abort actions taken

1. Killed the assemble's spark2 push at ~8% (~28/358 GB partial on spark2).
2. `docker rm` both rebuild containers (Exited(0) corpses) on both ranks.
3. Rollback pack verified intact pre-boot: 48/48 shards `2.0bpw-mcg` both
   ranks.
4. Restored serve: `bash results/2026-09-21-gatherv2/boot-k3c-pv2.sh`
   (canonical-e12, SNAPSHOT 2.0bpw-mcg, k3c+PREFETCH+CENSUS+GATHER_V2).
   Ready 04:57 BST; `/health` green both ranks.
5. Smoke gate: '17 * 19 = ? Step by step, then answer.' → **323** ✓.

## L.A.I.L table

| Config | L.A.I.L median tok/s | prose | acc | correctness |
|---|---|---|---|---|
| baseline k3c+pf+census+gv2 (stock, restored) | **31.37** (pooled n=10, prior session) | prior session | prior session | prior session (8/8) |
| G8 | not measured (aborted at bit-exact gate) | — | — | — |
| +lm_head | not measured | — | — | — |
| +fadvise cap | not measured | — | — | — |

35 tok/s status: **NOT MET** (no configuration exceeded 31.37 this session;
the two-medians rule was never invoked because no candidate booted).

## OOM log (free -h, MemAvail GiB)

| When | spark1 | spark2 |
|---|---|---|
| 04:18 post-rebuild (pre-gate) | 117 | 117 |
| 04:40 abort cleanup, GPU idle | 116 | 117 |
| 04:58 post-restore-boot + smoke | 25 | 27 |

Never below 12 pre-boot or 8 post-smoke. No OOM events.

## Engagement evidence

- G8 loader/kernels: never booted — no engagement markers exist by design
  (aborted before Phase 3).
- Restored serve engagement: boot log `role=head/worker`, snapshot
  `2.0bpw-mcg` both ranks, health green, smoke 323.
- Gate evidence logs: `gate-ref.log`, `gate-g8-run{1..4}.log` in this dir.

## What is left on the table (precise)

1. Fix `exl3_gemv_kernel.cuh` G8 reader for the down shape (NT=144) at
   m∈{1,2,4} — the non-deterministic mismatch signature points at the
   DEC5 8-tile pack-group addressing at a non-multiple-of-warp-stream NT.
   `exl3_gemm` (all m) and gate_up GEMV are already exact.
2. Prove `p2b_fused_moe` G8 in-image (the harness crash must be understood —
   ptr-array convention or scratch requirement may differ from the staged
   call; the IMAGE-BUILD patch claims the kernel is ported, but it was never
   proven here).
3. Re-run the gate (two-process script is ready and debugged), then resume
   Phase 3+ as planned. The G8 pack is INTACT on spark1 (48/48, verified
   layout) — only the spark2 copy is partial; re-run
   `CODEBOOK=mcg REV=2.0bpw-mcg-g8 DST=... bash tools/assemble_pack.sh`
   (resumable rsync) when the lane resumes.

## Serve state at session end

UP: stock live-best config (`results/2026-09-21-gatherv2/boot-k3c-pf-gv2.sh`,
canonical-e12, 2.0bpw-mcg, k3c spec-3 + captures [1,3,4,6,8] + PREFETCH=1 +
CENSUS=1 + GATHER_V2=1). Health green, smoke 323.
