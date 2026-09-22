# ENDGAME-2 VERDICT — 2026-09-22 (session ~05:30–08:00 BST, continues endgame r28)

## Outcome: **G8 KERNELS PROVEN BIT-EXACT (gate ×2 ALL PASS after harness fix);
G8 SERVING lane dead at the boot floor (spark2 worker OOM during weight load,
not a kernel defect); lm_head fallback blocked by an in-image key-routing bug
(lever was never integration-tested). Serve restored to stock live-best
31.37 config. 35 tok/s NOT reached.**

## Root cause — the Round-28 "NT=144 nondeterministic gemv race" was a HARNESS ARTIFACT

The gate script (v2) had three bugs; none were kernel defects:

1. **mcg=None → the QTIP gemv never ran.** v2 built LinearEXL3 with
   `make_linear_exl3(trellis, suh, svh, None, None)` → host `cb=0` →
   `exl3_gemv_cfg()` returns -1 for K=2 (`K != 4 && cb == 0` guards the
   codebook-specific decode paths) → every "gemv m≤8" point silently ran the
   **autotuned regular GEMM** (exl3_gemm.cu falls through to
   CoopKernelAutotuner). The autotuner TIMES CANDIDATES ON THE REAL B TENSOR
   (coop_autotune.cu measure_stage, trimmed-mean over timed rounds); stock and
   G8 layouts stream differently (that is G8's entire purpose), so near-tie
   shapes flipped winners between the two processes → different split-k
   accumulation order → the observed sub-ULP nondeterministic mismatches
   (max ~1.5e-4 on ~94% of elements = global order noise, NOT tile-local
   garbage). Corroboration: the autotune key hashes `MAX(size_m, 2)` — m=1 and
   m=2 share one entry and always failed/passed together (r28 runs 1-2 PASS
   both, runs 3-4 FAIL both), and m=8 (different key, different candidate
   margin) passed 4/4. The serving pack is 2.0bpw-**mcg** (cb=1): decode m≤8
   takes the deterministic QTIP gemv path (`K==2 → cfg by size_n only`) — the
   path v2 never tested.
2. **p2b illegal memory access = wrong-length scale tables.** v2 passed
   2304-long tensors for gu/uu/dv which the kernel indexes with
   `(w*128) % hidden` (% 5120) → OOB reads (crashed in the STOCK leg —
   layout-independent).
3. **p2b pointer tables too short**: 1-entry tables with ids=[0,1] read
   past the end → garbage expert pointers (second illegal-access source).
   Plus torch.equal fails on identical-NaN pairs (random-trellis overflow).

**CPU ownership-map proof** (`results/2026-09-22-endgame2/ownership_map_g8.py`,
run output in this dir): models the exact G8 branch addressing of
exl3_gemv_kernel.cuh (CFG0 WNT=2 WK=16) and p2b run_gemv_tile (CFG1 WNT=4
WK=8) for both per-rank shapes — k-chunks divide exactly (no partial warp),
every logical word loaded exactly once by its owning group, bounds exact-fit,
block-stride coverage total, G8 offsets bit-identical to the quantize-time
fold. **No ownership race exists.** Suspects (a)/(b)/(c) from the Round-28
triage are all void.

## Gate result (Phase B) — canonical-g8 image, unchanged

`results/2026-09-21-pfg8/pfg8_bitexact_gate.py` v3: mcg marker passed (real
serving path), shared EXLLAMAV3_TUNE_CACHE between legs (identical GEMM
candidate choice → pure addressing A/B), correct p2b scale/pointer tables,
NaN-aware bit-equality.

| Run | exl3_gemm ×8 | exl3_gemv ×8 (incl. down m=1,2,4!) | p2b_fused_moe ×4 | verdict |
|---|---|---|---|---|
| 1 | PASS | PASS | PASS | **PASS** |
| 2 (fresh ref+cache) | PASS | PASS | PASS | **PASS** |

Logs: `gate-run1-g8c.log`, `gate-run2-g8.log` (this dir). The G8 kernel port
is bit-exact on every reachable reader, deterministically, twice.

## Phase C — pack

- spark2 push: serial rsync collapsed to ~22MB/s (11ms RTT single-stream);
  rewrote as 4 parallel streams (`parallel_push.sh`) → ~400MB/s, done in ~7min.
- 48/48 shards both ranks, spot md5 (shards 3/23/43) MATCH.

## Phase D — G8 boot: DEAD at the boot floor (5 attempts)

1. spark2 missing canonical-g8 image → docker save|load (fixed, not a boot).
2. `w13_trellis dest (320,72,32) != loaded (9,320,256)` — loader re-index
   phase-1 swapped the narrow dim but `create_weights` still allocated
   stock-shaped dest. Fixed in `docker/patch/pfg8_loader_reindex.py`
   (phase-2 alloc, commit 2bddbc3): w13 (E,2,9,320,256), w2 (E,40,72,256),
   stock path untouched (env-gated).
3. torch.empty nested-tuple TypeError — splat fix (commit 06670cf).
4+5. **spark2 kernel OOM-killed VLLM::Worker_TP during weight load**
   (journalctl: anon-rss ~45.5GB, killed at 06:22, 06:38, 07:09 BST), even
   after pack cache-warming (330GB dd warm) and clean MemAvail 117GiB
   pre-boot. The G8 loader path on spark2 balloons anonymous memory during
   `load_weights` (stock boots never did this; the G8 dest alloc doubles
   trellis Parameter allocation while the folded tensors are also material
   — host-side, before the GPU copy). Not debugged further: boot cap 5 hit.

The G8 lane is dead **in this image's loader**, not in its kernels. The
kernels are proven; the loader needs a memory-lean G8 load (e.g. narrow
before alloc, or stream per-expert) — a new dispatch.

## Phase E/F — lm_head MXFP8 fallback: BLOCKED (in-image key routing)

- Fallback pack `2.0bpw-mcg-lmhead-mxfp8` built on both ranks (symlink copy +
  re-encoded model-00043: head.weight → lm_head.weight mxfp8 + e8m0 scale,
  682MB; index rewritten). G8 twin `2.0bpw-mcg-g8-lm` also staged (unused).
- Two harness-side fixes landed in `docker/patch/lmhead_mxfp8.py`
  (commit ba0675d): module path `vllm.models.deepseek_v4_1.nvidia.model`
  (the draft's `vllm.model_executor.models.*` does not exist in this build →
  sitecustomize silently skipped the hook), and direct `flashinfer.mm_mxfp8`
  import (no vllm_flashinfer wrapper in this build). `install()` verified
  True in-image.
- **Remaining blocker (in-image, not fixable from the patch dir):** the
  checkpoint key `lm_head.weight` does not route to the model parameter —
  loader raises `ValueError: There is no module or parameter named
  'lm_head' in DeepseekV41ForCausalLM` (real param is
  `language_model.lm_head.weight`; the WeightsMapper rename assumption from
  the c32be56 draft does not hold in this build). The lever's own tests were
  written against a FAKE vllm package, so this was never caught. Needs its
  own dispatch with a real-image integration test.

## Phase G — serve restored

`results/2026-09-21-gatherv2/boot-k3c-pf-gv2.sh` (canonical-e12,
2.0bpw-mcg, k3c+PREFETCH+CENSUS+GATHER_V2) — the 31.37 baseline config.
Health green, smoke 323 verified below.

## L.A.I.L table

| Config | L.A.I.L median tok/s | correctness |
|---|---|---|
| stock k3c+pf+census+gv2 (restored) | **31.37** (pooled n=10, prior session; config identical) | 8/8 (prior) |
| G8 | not measurable (boot floor) | gate 24/24 ×2 |
| +lm_head | not bootable (key routing) | — |

35 tok/s: **NOT MET** — no candidate served this session; two-medians rule
never invoked. (L.A.I.L bench not run: no candidate booted.)

## OOM log (free -h MemAvail GiB; full log `oom-log.md` this dir)

| When | spark1 | spark2 |
|---|---|---|
| 05:36 pre-stop (stock serve up) | 25 | 27 |
| 05:37 post-stop | 116 | 117 |
| 05:39 post-gate | 115 | — |
| 05:56 pre-boot-g8 | 117 | 117 |
| 07:05 pre-boot-g8-5 (warmed) | 117 | 117 |
| 07:15 pre-boot-lm | 116 | 117 |
| 07:30 pre-restore | 117 | 117 |

Never below 12 pre-boot / 8 post-smoke on the host. BUT: spark2 **kernel OOM
kill** of the worker during G8 weight load ×3 (see Phase D) — the floors did
not catch it because host MemAvail stayed high until the kill.

## Commits (branch cursor/mul1-p2b-prefill-95c3)

- 8c4d372 gate v3 + ownership-map proof
- 2bddbc3 loader phase-2 G8 dest alloc
- 06670cf alloc splat fix
- ba0675d lmhead import fixes + push/boot scripts + this dir

## What's left on the table (precise)

1. G8 kernels: DONE (proven bit-exact ×2). The remaining defect is a
   host-side memory blow-up in the G8 weight-load path on the worker rank —
   profile `Exl3MoEMethod._load_exl3`/`create_weights` under DSV41_LOAD_PF_G8
   (fold materialization + doubled Parameter allocation), make the load
   stream per-expert, then boot: gate evidence carries over unchanged.
2. lm_head: fix the checkpoint-key routing (lm_head.weight →
   language_model.lm_head.weight) with a REAL-image integration test; the
   numerics/swap code is already correct (offline tests green).
3. Optional +DSV41_ENGRAM_FADVISE_CAP=24 leg once either lever boots.
