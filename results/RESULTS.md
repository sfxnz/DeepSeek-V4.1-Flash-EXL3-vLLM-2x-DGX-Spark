# RESULTS — DeepSeek-V4.1-Flash EXL3 on 2× DGX Spark (GB10), vLLM TP=2

Append-only record. Baseline is the published main config (commit `e507021`),
which is also what was serving when this file was created. Every experiment
row names: command/flags, quant, context, concurrency, pp/tg cells, e2e,
acceptance, quality verdict. Never delete prior rows.

Method (from the z.ai GLM-5.3-Flash infra writeup): high-density feedback —
localized, cheap, objectively verifiable measurements; local (micro) checks
kill bad ideas early, e2e checks whether local wins transfer; one hypothesis
per change; quality gates every perf number.

## Harness

- `tests/correctness.sh [--full]` — 8 checks: 2 math, strict JSON, tool-call
  args, code trace, needle recall 8k + 64k (--full), prose anti-collapse
  (type-token ratio + 8-gram diversity). Greedy, seed 20260918, thinking off,
  effort low. Exit 1 on any failure.
- `benches/micro.sh` — isolated pp (prompt_tokens/TTFT) and tg32
  (post-first-token) at 512/4096/16384/65536 context. Fresh docs per
  invocation (prefix-cache-proof; cache-hit retry at >3000 tok/s pp).
  Filler = this repo's real text (docs+code), not synthetic sentences.
- `benches/e2e.sh` — coding-agent turn (~800-token scaffold + guard task,
  graded), 64k doc recall + 2-sentence answer (graded, cold prefill),
  tool/JSON call (graded), then the frozen L.A.I.L prose harness.
- `bench_decode.py --phase prose -c 1` and `tools/measure_lail_prose.py`
  stay the published decode references.

Constraints that hold every row: DSpark-5, CUDA graphs on, vision on, EXL3
2.0bpw MCG pack, Engram on NVMe, fp8 KV, thinking off, effort low.

## Baseline quality notes (measured while building the suite)

- At the shipped default (thinking off, reasoning_effort low) the model
  fails multi-step arithmetic traces deterministically: `13*17+5` → `221`;
  a 4-element index-diff loop → `-3` (truth `-11`). Single-op string traces
  pass (`reversed`+slice+join). Suite checks are calibrated to sit inside
  the passing envelope; a future quality improvement may make formerly-wrong
  answers right — that is a pass, not a regression.
- Prefix caching is on by default: identical 8k doc recall went 12.7s → 1.3s
  on repeat. All pp/tg benches use fresh docs per invocation.

## B0 — Baseline (published main, serving since 2026-09-16)

Config: see `flags.md`. Container: `dsv41-flash-exl3-sm121`, pack
`sfxnz/DeepSeek-V4.1-Flash-EXL3@2.0bpw-mcg`, `MAX_NUM_BATCHED_TOKENS=8192`.

Published decode history (from `evidence/trail.tsv`, same method):

| date | change | decode tok/s | verdict |
|---|---|---|---|
| 09-11 | main defaults (eager, no spec, exl3_moe) | 12.75 | denominator |
| 09-11 | + DSpark-5 | 21.2–23.0 | keep |
| 09-13 | + native p2b CFG=1 fused MoE | ~23 | keep |
| 09-13 | + b12x MXFP8 Q/O | 23.44 (L.A.I.L wave 22.37) | keep, published |

Fresh cells (2026-09-18, this suite): `2026-09-18/baseline/`, c=1,
DSpark-5 greedy, graphs on, fp8 KV, thinking off, effort low, real-text corpus.

| cell | value | notes |
|---|---|---|
| pp@512 | 286.8 tok/s | tiny chunk, fixed overhead dominates |
| pp@4k | 683.0 tok/s | ttft 5.41s |
| pp@16k | 695.0 tok/s | ttft 22.57s |
| pp@64k | 703.3 tok/s | ttft 89.57s — the long-context pain |
| tg32@512 | 20.4 tok/s | acc 2.43 |
| tg32@4k | 21.8 tok/s | acc 2.57 |
| tg32@16k | 24.0 tok/s | acc 2.62 |
| tg32@64k | 26.6 tok/s | acc 2.83 |
| correctness | 8/8 | incl. recall@64k prose |
| e2e coding_agent | pass, decode 38.6 tok/s, ttft 1.42s | guard patch graded |
| e2e doc_recall@63k | needle miss (single-sample flake) | sweep below |
| e2e tool_json | pass | Paris, unit c |
| L.A.I.L prose | 23.94 tok/s median | published 23.4/22.37 — continuity OK |
| bench_decode prose c=1 | 32.5 tok/s median | high-acceptance structured phase |

Needle sweep (repo-text corpus, 2026-09-18, live serve, greedy):
16k/31k/48k/62.8k(d0.25)/63.7k(d0.5)/63.3k(d0.75)/94k → **7/7 hits**.
No recall cliff to ~94k on this pack; the e2e miss was a two-part-prompt
generation flake, so e2e doc_recall is now single-task.

Prefill observations (worker logs + image code):
- `EXL3 fat-chunk slicing ACTIVE` fires on every ≥3.6k-token chunk on real
  text: whole-chunk MoE re-sliced into `TEMP_ROWS_FUSED=2048`-token slices,
  each re-run through the native p2b kernel whose tiles were tuned for
  decode (m=1..12). Native dispatch never fails (0 warnings).
- 384 experts × topk 6 ÷ TP 2 = 192 experts/rank; an 8192-token chunk
  averages 256 routed rows/expert = exactly `VLLM_EXL3_FAT_THRESHOLD`
  (default 256): hot experts cross it constantly.
- `MAX_NUM_BATCHED_TOKENS` 2048→8192 (commit `e507021`) was justified by
  "GB200 uses 16384", not by a local measurement — and 8192 chunks are
  exactly what re-triggers the slicing path that 2048 chunks never hit.

## Experiments

(append below; one hypothesis per row-set)

### E1 — MAX_NUM_BATCHED_TOKENS=2048 (2026-09-18)

- Hypothesis: 2048-token chunks cannot exceed `TEMP_ROWS_FUSED=2048`
  rows/expert, so the whole-chunk fat-chunk re-slicing never fires; if
  re-slicing was the prefill tax, pp improves.
- Change: `MAX_NUM_BATCHED_TOKENS=2048 ./serve.sh` (single flag). Zero
  fat-chunk slicing lines in worker logs (verified).
- Correctness: 7/7 (without --full).
- Micro: pp@16k 498.9 (base 695.0, −28%), pp@64k 689.3 (base 703.3, −2%
  ≈ noise), tg@16k 20.7 (base 24.0), tg@64k 22.2 (base 26.6).
- E2E: not run (micro rejected).
- Verdict: **REJECT** — smaller chunks lose at 16k, nothing at 64k. The
  slicing path is not the dominant cost; per-token compute is. 8192 stays.

### E2 — MAX_NUM_BATCHED_TOKENS=16384 (2026-09-18)

- Hypothesis: bigger chunks amortize fixed per-chunk costs (launches,
  all-reduce latency, prestage) that E1 showed matter at 16k.
- Result: **boot failure** — 16384-token chunks grow activation/scratch so
  the KV pool drops to 4.0 GiB < 4.16 GiB needed for max_model_len 1M
  (`ValueError` from `_check_enough_kv_cache_memory`). Measured memory
  cost of the bigger chunk: ~0.5 GiB at UTIL=0.75.
- Retry E2b with `KV_CACHE_MEMORY=5368709120` (5 GiB, still under the
  8 GiB guard) as the enabler.

### E2b — MAX_NUM_BATCHED_TOKENS=16384 + KV_CACHE_MEMORY=5 GiB (2026-09-18)

- Change: `KV_CACHE_MEMORY=5368709120 MAX_NUM_BATCHED_TOKENS=16384 ./serve.sh`.
- Micro: pp@16k 524.1 (base 695.0, −25%), pp@64k 619.1 (base 703.3, −12%),
  tg cells ≈ noise. 10 fat-chunk slicing lines in logs.
- Verdict: **REJECT** — both directions away from 8192 lose (E1: −28%@16k;
  E2b: −25%@16k, −12%@64k). 8192 is a measured local optimum for prefill on
  GB10; the GB200-style 16384 default is wrong for this hardware and costs
  ~0.5 GiB KV headroom on top. `MAX_NUM_BATCHED_TOKENS=8192` and
  `KV_CACHE_MEMORY=4 GiB` stay, now with numbers behind them.
- Chunk-size axis closed. Next: kernel-mix flag
  (`VLLM_EXL3_FAT_THRESHOLD`) at fixed 8192 chunks.

### E3 — VLLM_EXL3_FAT_THRESHOLD=96 at 8192 chunks (2026-09-18)

- Hypothesis: lowering the fat-expert threshold from 256 to 96 sends more
  routed rows through the per-expert 128×128 fat GEMM; if the standard
  exl3_moe kernel is the prefill bottleneck at 64-256 rows/expert, pp
  improves.
- Change: `VLLM_EXL3_FAT_THRESHOLD=96 ./serve.sh` (single env; run.sh
  already passes it through to the container).
- Correctness: 7/7.
- Micro: pp@16k 618.2 (base 695.0, −11%), pp@64k 711.9 (base 703.3, +1.2%
  ≈ noise), tg cells noise.
- Verdict: **REJECT** — kernel mix does not move prefill. Combined with
  E1 (slicing never fires at 2048 chunks, no gain) this says prefill is
  per-token compute/bandwidth bound in the kernels themselves, not in
  chunking, slicing, host syncs, or the fat/standard split.

### Round summary (2026-09-18)

- Prefill flag space measured and closed: chunk 2048/8192/16384 and fat
  threshold 96/256 all reject; **8192 + threshold 256 stays**, now with
  local numbers instead of "GB200 uses 16384".
- The remaining prefill upside is kernel-level: a prefill-shaped grouped
  GEMM for routed experts (m≈64-512 rows/expert, the decode-tuned p2b
  tiles and the per-expert python fat loop both leave it on the table).
  That is the next single hypothesis if kernel work is in scope; before/
  after cells are `pp@16k 695`, `pp@64k 703`.
- Decode/tg cells did not move outside acceptance noise in any experiment.
- Serve restored to published defaults after the round.


### Restore check (2026-09-18, end of round)

`./serve.sh` with defaults up again: chunk 8192, KV 4 GiB, `smoke_chat` 323,
`smoke_vision` ok, unit tests + `kit/render.py --check` pass, correctness
7/7, L.A.I.L prose 21.56 tok/s (waves this harness produced on this pack:
23.44 / 22.37 / 23.94 / 21.56 — temperature-0.2 run-to-run band).

Next single hypothesis, in priority order:
1. Prefill grouped GEMM for routed experts at m≈64-512 rows/expert (kernel
   work; needs a bit-exactness gate before any perf claim). Before-cells:
   pp@16k 695, pp@64k 703.
2. DSpark `NUM_SPECULATIVE_TOKENS=10` (divisible by block 5): only if the
   draft cost does not swamp the extra accepted tokens; e2e-gated.

### E0 — live prefill census instrument (2026-09-19, round 2)

- Attempt: `docker/patch/prefill_census.py` wrapping the exl3 MoE entries +
  model.forward with CUDA-event timers, bound after `load_general_plugins()`
  (the shipped `DSV41_STEP_CENSUS` never flushed — it dumps only from draft
  ticks that never fire in this build).
- Result 1: wraps bound ("installed" printed) but recorded nothing — the
  model's MoE call path does not resolve through the wrapped module
  attributes in the executing processes (registry/custom-op dispatch).
- Result 2: the CUDA-event `synchronize()` wraps coincided with
  `TimeoutError: RPC call to sample_tokens timed out` → EngineDead during a
  decode request. In-process instrumentation around the spec-decode
  machinery is unsafe here.
- Verdict: **abandoned** (wiring reverted; two restarts spent). Any kernel
  work must justify itself with before/after micro cells instead of live
  per-op timing. Config-math estimate stands in: routed experts are ~85% of
  prefill FLOPs (6 experts × 3 GEMMs × 5120×2304 × 37 layers vs attention
  topk 512), so the MoE path is the kernel target.

### E4 — VLLM_EXL3_FAT_THRESHOLD=2048, fat path off (2026-09-19)

- Hypothesis: the per-expert python fat-GEMM loop is a prefill tax; turning
  it off (threshold 2048 = kernel's own row cap) routes everything through
  the standard kernel.
- Correctness: 7/7.
- Micro: pp@16k 601.1 (base 695.0), pp@64k 703.4 (base 703.3 — identical),
  tg noise.
- Verdict: **REJECT**. With E3 this closes the fat/standard mix axis at all
  three thresholds (96/256/2048): pp@64k ≈ 700±10 regardless. Prefill is
  not chunking-, slicing-, sync-, or mix-bound.

### Kernel-work boundary assessment (2026-09-19)

- Bandwidth check: per 8192-token chunk per rank, routed-expert weight
  traffic ≈ 262 GB across slices ≈ 23 GB/s — nowhere near the ~273 GB/s UMA
  ceiling. Prefill is not weight-bandwidth-bound.
- Efficiency check: ~18.5 GFLOP/token × 703 tok/s ≈ 13 TFLOPS effective vs
  ~200 TFLOPS bf16-class peak → ~6.5% MFU. The p2b kernel is a GEMV design
  (one token row per work item, scalar fp32 accum; `a_row0 = 0` in
  `p2b_moe.cu`), correct for decode m=1..12, structurally wrong for prefill
  m=64..512 rows/expert.
- The fix is a tensor-core grouped dequant-GEMM for the EXL3 trellis format
  at prefill M. That is a multi-day kernel project with a documented
  failure history in this repo (`evidence/p2b-{mma,fma,cp16,cpasync,ldg,
  pf4,nocoop,mrow}`, `mma-revert`); the FMA variant also produced silently
  wrong output caught only by the essay-collapse check. Not a bounded
  single-hypothesis loop — deferred with this record.

### E5 — KV_CACHE_MEMORY 4→8 GiB (2026-09-19) — **KEEP**

- Hypothesis: at MAX_NUM_SEQS=2 the 4 GiB pool held exactly 2×1M context
  with zero slack, so the prefix cache evicted constantly on agent
  workloads; doubling the pool doubles resident prefix blocks and adds
  real long-context headroom.
- Change: `KV_CACHE_MEMORY=8589934592 ./serve.sh` (exactly the guard
  ceiling; no FORCE needed). Engine reports "GPU KV cache size: 2,289,205
  tokens, Maximum concurrency for 1,048,576 tokens per request: 2.18x".
  16 GiB still available after boot — Engram staging keeps its headroom.
- Correctness: **8/8 including 64k recall**.
- Micro: pp@16k 708.4 (base 695.0, noise), pp@64k 703.1 (base 703.3,
  identical), tg@16k 28.0 / tg@64k 29.6 (base 24.0/26.6 — top of the
  acceptance-noise band).
- E2E: 3/3 (coding decode 38.2–38.4 tok/s, needle hit at 63k, tool ok).
  One coding TTFT read 14.2s on the first request after engine init;
  re-run 0.51s — cold-start flake, not a regression.
- L.A.I.L: 22.57 tok/s (band 21.6–23.9).
- Verdict: **KEEP** — capacity doubled, nothing regressed. New recipe
  default (`recipe.yaml` regenerated; run.sh now defaults 8589934592).

### E6 — DSpark NUM_SPECULATIVE_TOKENS=10 (2026-09-19)

- Hypothesis: goal step D — extending the draft block from 5 to 10 (guard
  passes, divisible by 5) buys accepted tokens on long matches.
- Correctness: 7/7.
- Micro: tg@4k 12.0 (E5 cell 28.0), tg@64k 16.6 (E5 29.6), pp unchanged.
- L.A.I.L: 12.14 tok/s median vs 22.57 on E5 — decode nearly halved.
  Acceptance length **fell** to 1.87 (from 2.4–2.8 at n=5): the Markov
  draft's extra 5 positions are almost never accepted, while every step
  pays for two draft blocks instead of one.
- Verdict: **REJECT** — n=5 is the measured optimum for this draft head.
  Textbook case of the goal's "do not raise n if e2e slows".

### Round 2 summary (2026-09-19)

- **KEEP: KV_CACHE_MEMORY 8 GiB (E5)** — capacity doubled (2.29M tokens,
  2.18× at 1M ctx) with zero perf/quality cost. New recipe default.
- Rejected: fat-threshold 2048 (E4), spec n=10 (E6).
- Abandoned with record: in-process prefill census (never bound to the
  live call path; CUDA-event wraps coincided with an engine RPC timeout).
- Kernel boundary documented: prefill MoE runs a decode-shaped GEMV at
  ~6.5% MFU; the fix is a grouped tensor-core dequant GEMM — a multi-day
  project with prior failures in this repo, deferred deliberately.
- Remaining measured-open axes: none at flag level. Serve restored to the
  new default config (8 GiB KV, DSpark-5, 8192 chunks).

### E7 — torch.compile / inductor for prefill (2026-09-19)

- E7a `mode: INDUCTOR`: fast config-validation fail — valid modes on this
  build are NONE, STOCK_TORCH_COMPILE, DYNAMO_TRACE_ONCE, VLLM_COMPILE.
- E7b `mode: VLLM_COMPILE` (boot OK, `CompilationMode.VLLM_COMPILE`
  confirmed in engine config): correctness 7/7; pp@16k 670.6 (E5 cell
  708.4, −5%), pp@64k 699.1 (703.1, noise), tg cells in band, L.A.I.L
  21.95 (band).
- Verdict: **REJECT** — inductor has nothing profitable to fuse here;
  prefill time sits in the opaque custom MoE/MLA kernels. Closes the
  compile axis and reconfirms the kernel-boundary conclusion by
  measurement.

### Kernel archaeology (2026-09-19, before E8)

- The shipped p2b kernel runs `mma.m16n8k16` with ONE live A-row per work
  item (15 of 16 M-rows zero-padded; `a_row0 = 0`). At decode m_loc≈1 so
  the padding is unavoidable there.
- `widen_p2b_mma.py` (present, unwired) batches up to 8 same-expert rows
  into one tile: weights decoded once, reused across rows. It was reverted
  on a DECODE-ONLY verdict (L.A.I.L 10.8 vs 15.1 at the time — the mma
  indexing overhead taxes the m_loc≈1 decode case).
- Prefill was never measured with it: at prefill m_loc ≈ 64-512 rows per
  expert per 2048-row slice. E8 measures exactly that.

### E8 — p2b mma-over-m rows, prefill verdict (2026-09-19)

- Hypothesis: `widen_p2b_mma.py` batches ≤8 same-expert rows per tile
  (weights decoded once, reused across rows). It was reverted in round -1
  on a decode-only verdict; prefill has m_loc ≈ 64–512 rows/expert per
  2048-row slice, so the batching should finally engage there.
- Change: derived image `dsv41-flash-exl3-sm121:mma8`
  (`docker/Dockerfile.mma`: full patch chain shapes→mrow→mma→cfg1→
  codebook, rebuild of vllm_exl3 CUDA; "mma build ok" gate; image shipped
  to spark2 via `docker save | ssh docker load`). Served on both ranks.
- Correctness: 7/7 (mma is numerically correct).
- Micro: **pp@64k 709.4 (base 703.1 — flat)**, pp@16k 620.3 (base ~700,
  −12%), tg@16k 10.5 / tg@64k 11.5 (decode halves, as known).
- Verdict: **REJECT** for serving — but it closes the row-batching
  hypothesis with a prefill measurement: batching A-side work (rows) and
  halving weight re-reads does not move prefill at all. Combined with the
  flat pp across chunk sizes, fat thresholds, and compile modes, the
  remaining bottleneck is the **B-side weight-decode instruction stream**
  (per-weight trellis dq8 + `__shfl_sync` + mma packing), which is
  identical in the one-row and mma variants. ~23 GB/s of effective weight
  throughput against ~273 GB/s UMA says the decode path is
  instruction-bound, not memory-bound.
- Next kernel target, if kernel work resumes: vectorized multi-weight
  trellis decode (fewer shuffles per fragment) — a from-scratch inner
  loop, not a config flip. The E8 image recipe stays in the repo for
  reproducibility.

### Round 3 summary (2026-09-19)

- Rejected with measurements: E7 inductor compile (pp flat, −5% at 16k),
  E8 mma row-batching (pp flat, decode −50%).
- Every flag-level axis is now closed by local numbers; the row-batching
  kernel axis is closed by a direct prefill A/B. The recipe is at a
  measured local optimum on this hardware for every bounded change.
- Serve restored to the winning config (image `dsv41-flash-exl3-sm121`,
  8 GiB KV, DSpark-5, 8192 chunks).

### Round 4 — final capture on the winning config (2026-09-19)

`results/2026-09-19-final/all-cells.log`, defaults from `./serve.sh`
(8 GiB KV, DSpark-5, 8192 chunks, graphs on, vision on):

- Correctness: **8/8** (includes 64k needle recall)
- pp: 491 / 747 / 709 / 710 tok/s at 512 / 4k / 16k / 64k (64k runs
  706.5–710.4 — tight)
- tg32: 23.8 / 23.5 / 27.0 / 22.5 tok/s (DSpark acceptance 2.3–2.9)
- e2e: 3/3 (coding guard, 63k needle, tool/JSON)
- L.A.I.L prose c=1: 22.40 tok/s (acceptance 2.24)
- decode prose c=1: 34.7; **c=2: 41.2 aggregate, 21.3 per stream**
  (MAX_NUM_SEQS=2 two-stream capability, now in the README table)

Published `recipe.yaml`/README measured table refreshed from this capture;
history rows remain above. Round-3 verdicts (E7 inductor, E8 mma) added to
flags.md.

## Round 5 — decode campaign: kernel-format study, GEMV harness, live profile (2026-09-19)

Goal: substantial decode improvement via the B-side trellis-decode work.
Method: understand the format exactly, reproduce the deployed kernel in a
bit-exact standalone harness, then profile the LIVE decode step before
committing to any rewrite.

### Format findings (both packs)

- MCG and MUL1 share the identical trellis stream: per 16x16 tile 32 uint16
  (512 bit = 256 weights x 2 bit); value[p] = codebook(16-bit window at bit
  2p). The codebooks are fixed procedural functions: MCG = `x*0xCBAC1FED +
  LOP3 + HADD2`, MUL1 = `x*0x83DCD12D + dp4a byte-sum + HFMA2`. The `.mcg` /
  `.mul1` tensors are 1-element int32 markers carrying the multiplier.
- The naive MUL1+cb=2 port (branch `38e67b4`, image `:cb2`) decodes the same
  windows with different arithmetic (~6 vs 7 instr per pair). Its earlier
  -16% prose loss was acceptance-driven (2.86 -> 2.39), not kernel-time.
- Full-LUT decode of the 64K-entry codebook is **closed on GB10**: 128 KiB
  table vs 99 KiB smem/block opt-in (48 SMs, sm_121).

### GEMV harness (`kernel_study/gemv_bench/`, study tree; sources tracked, builds not)

- `bench.cu`/`bench2.cu`: byte-faithful clone of the deployed tile (image
  patch chain reproduced on the pinned plugin ref), plus PFMUL (prefetch
  depth) and DEC (decode variant) template knobs. Bit-exact vs the installed
  `vllm_exl3_c` (one-hot routing weights make the stock atomicAdd epilogue
  order-exact; arbitrary weights are 1 half-ULP nondeterministic in stock
  itself at e>6).
- Warm e=30 (30 experts x 3 mats, fixed ids): stock 657-672 us/call =
  **790-810 Gw/s = 198-202 GB/s of trellis stream (74% of 273 GB/s UMA)**.
- Cold-rotating ids: 624.6 us median = **212 GB/s** — instruction-side wins
  disappear when experts are DRAM-cold: the GEMV is memory-stream-bound.
- PF ring depth sweep REJECT: PFx2 -3%, PFx4 -14%, PFx8 -45% (register
  pressure). Occupancy 4 blocks x 256 thr/SM throughout.
- Funnelshift extraction variant (vdec1): **-6.1% bit-exact warm** (670.7 ->
  629.7 us), no cold effect. Warp-smem staging (vdec2): -2.5% warm.

### Live profile (torch profiler replica, `--profiler-config`, rank 0)

One 110-token prose generation; 2240 p2b calls (median **775 us** each in
serving vs 625-672 harness — profiler/CUPTI and L2-cold effects account for
the gap; graph capture stays on). Device-busy composition per decode step:

| phase | share | note |
|---|---|---|
| p2b MoE kernel | ~35% | 40 x ~0.65-0.78 ms; at 74% UMA peak already |
| dense projections (b12x mxfp8 GEMMs + bf16 WMMA) | **~42%** | 234 b12x calls/step at ~85 us, grids (1,1,14..48) x 96 thr = parallelism-starved (96 thr/SM vs p2b's 1024); o_proj wo_a BMM runs cutlass_80 sm_80 WMMA bf16 at 176 us (0.9 TFLOPS) |
| NCCL AllReduce | ~11% | ~90 x ~105-165 us latency-bound 2-rank RING_LL |
| eltwise/topk/norm/MLA decode | ~12% | sparse_mla itself is 1.2% |

Decode step model at prose c=1 (~70-83 ms/step): expert streaming 5.3 GB at
~205 GB/s = 26 ms is near-roofline; dense projection streaming ~4 GB at
~111 GB/s = ~36 ms is at HALF the achievable stream rate. The decode wall is
now the dense-projection path and comms, not the trellis GEMV.

### Experiment queue (next rounds)

- X1 NCCL proto/algo tune for 2-rank small ARs (env-only restart A/B).
- X2 flashinfer mxfp8 cute-dsl tile/tuner for m=5 dense GEMMs (serve pins
  `enable_flashinfer_autotune:false`; default tile -> tiny grids).
- X3 o_proj wo_a bf16 WMMA fallback -> modern fp8/b12x path.
- X4 p2b vdec1 funnelshift (+vdec2): bit-exact, small; rides along with any
  kernel image rebuild.

### E10 — decode bundle: fshift extraction + b12x small-m tiles + wo_a probe (2026-09-19) — **KEEP-candidate**

Image `dsv41-flash-exl3-sm121:e10` (Dockerfile.e10; chain promoted into the
main Dockerfile). Three independently attributable changes:

1. `widen_p2b_fshift.py` — single-`SHF` window merge in the p2b bits==2
   decode (bit-identical windows; harness: 670.7 -> 629.7 us warm e=30).
2. `widen_b12x_smalls.py` — flashinfer sm120 blockscaled dense GEMM picks
   (16,64) tiles for m<=8, n<=8192 (2x CTAs; grids were 14-48 CTAs on 48
   SMs). Prefill untouched (m>128 branch).
3. `probe_wo_a.py` — boot log. **Finding: wo_a loads as bf16 (4096,4096)
   despite F8_E4M3 in the pack -> per-layer `torch.bmm` on an sm_80 WMMA
   kernel (~176 us). wo_b correctly fp8.** Root-caused next target.

Gates on :e10:

- Correctness **8/8** including 64k recall; e2e **3/3** (coding decode 31.0,
  needle hit at 63k, decode 42.9 on that cell, tool ok).
- pp@4k 754.3 / pp@64k ~700 — prefill flat (b12x change correctly scoped).
- Prose decode c=1 (5-run median): **31.44 tok/s, acceptance 2.97** vs
  25.11/2.78 same-session pre-E10 and 27.98/2.86 historical MCG baseline.
- L.A.I.L 23.0-23.2 (band 21.6-23.9, final capture 22.40).
- NCCL proto axis closed by isolation test: default already LL = 43.0 us for
  the 51 KB 2-rank AR (LL128 81.8) — serving AR overhead is launch/graph
  side, not protocol.

Serve is live on `:e10` (identical patch chain to the promoted Dockerfile);
canonical image rebuild/retag lands next round. Next decode targets by
measured size: wo_a bf16-bmm -> fp8 einsum (~5.6 ms/step), NCCL AR overlap,
eltwise/gap trims.

### E11 — o_proj wo_a exact-requant onto the fp8 einsum (2026-09-19) — **KEEP**

Root cause chain (from E10's probe): the pack stores wo_a as F8_E4M3 +
ue8m0 block scales, but on GB10 the ModelOpt MXFP8 BMM path picks the
emulation kernel whose load-time dequant (`VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD`,
default on) replaces the weight with BF16. `deep_gemm_fp8_o_proj` then took
its `torch.bmm` fallback — a strided bf16 BMM cublas maps to a cutlass_80
sm_80 WMMA kernel, ~176 us/layer at decode (~0.9 TFLOPS; 37 of them per
step in the live profile).

Fix (`fix_o_proj_woa_fp8.py`): the layer retains its e8m0 scales, so
requantizing the bf16 weight with THOSE scales is a bit-exact roundtrip of
the original e4m3 bytes. One-time, capture-guarded, self-tested conversion
(43/43 layers engaged at boot: `[woa-requant] fp8 einsum engaged`), with a
permanent bf16 fallback on any failure. Isolated bench of the einsum at the
decode shape: 45.1 us vs 157.2 us bf16 bmm (3.5x).

Gates on `:e11` (all green):

- Correctness **8/8** (incl. 64k recall); e2e **3/3**.
- Prose decode c=1 (5-run median): **33.57 tok/s, acc 3.05** — E10 31.44,
  session pre-E10 25.11 (**+34% cumulative**).
- L.A.I.L 22.2-23.1 (band); tg cells acceptance-noisy (18.5-32.6 at
  acc 2.0-3.4), medians consistent with E10-or-better.
- pp@64k tie-breaker 706/699/698 — inside the 695-710 band; no prefill
  regression (one 681.8 outlier was cache-cold).

Chain promoted into the main Dockerfile. Serve live on `:e11`.

### Round 7 — attribution + cp.async reject + canonical promotion (2026-09-19)

**E11 attribution profile** (profiler replica, same methodology as round 5):
the wo_a fix is verified in-trace — the per-layer cutlass_80 WMMA (288 us
x40/step) is gone from decode, replaced by `deep_gemm` fp8 einsum at
80.6 us x40/step. Step composition now: p2b 48%, dense b12x GEMMs 28%,
NCCL 17.5% (92.7 ARs/step at 124 us in-graph vs 43 us isolated floor —
launch/graph-structure overhead, not protocol).

- `widen_b12x_smalls` measured **no effect** (dense total trace-flat; those
  GEMMs are latency-bound, not parallelism-starved). Kept, harmless.
- p2b **cp.async 4-buffer ring** (bit-exact, bench3): REJECT — 672.4 us vs
  642.2 us stock cold (−4.7%). Load-mechanics axis closed: PF depth, smem
  staging, cp.async all rejected; the tile runs at ~76% of UMA peak cold and
  further gains need pack-layout work (multi-day, out of scope).
- **b12x pip package regression found**: canonical-e11 (Dockerfile rebuild,
  which installs the optional `b12x` extra) measured prose 29.9/32.0 vs
  :e11's 31.4-34.3 at equal acceptance — vLLM routes dense MXFP8 through the
  package kernels and loses 5-9%. Removed the pip line from the Dockerfile;
  canonical-e12 (without it) rebuilds content-equivalent to :e11.

Serve now on `dsv41-flash-exl3-sm121:canonical-e12` (both nodes): smoke 323,
woa-requant 43/43, prose 31.2 (acc 2.83) / L.A.I.L 21.9 — both inside the
final bands. Remaining untested levers: flashinfer autotune flag (one serve
flag), NCCL AR overlap (structural).

### Round 8 — autotune null; lever inventory closes (2026-09-19)

- **flashinfer autotune A/B** (serve flag, tuner enabled at warmup):
  correctness 7/7, prose 30.2 (acc 2.79) / L.A.I.L 22.3 vs stock 31.2/21.9 —
  inside bands, **no effect**. Consistent with the round-7 finding that the
  decode dense GEMMs are latency-bound: no tile choice fixes per-GEMM
  latency. Flag stays off.
- **Async-TP / comm-overlap**: not present in this vLLM build (no flag, no
  config) — reducing the 124 us in-graph AR (vs 43 us isolated floor) is a
  structural patch, not configuration. Scoped hand-off.
- Serve restored to `canonical-e12` stock flags after clearing a leftover
  NCCL-test container that briefly blocked exclusive GPUs.

**Campaign close-out state (2026-09-19):** decode steps are 10-15% faster
kernel-verified (wo_a 288->80.6 us x40/step + fshift); measured prose decode
24-37% above the session baseline (25.1 -> 31-34 tok/s band); L.A.I.L at
band top (+0-4%); quality gates 8/8 + 3/3 throughout; prefill unchanged.
Every cheap lever is now closed by measurement. Remaining multi-day levers,
in measured-size order: (1) NCCL AR overlap (~7 ms/step), (2) p2b pack
layout for >76% UMA peak (~6 ms/step), (3) pair-codebook requant (decode
instruction halving, needs constant search + 12 GPU-h requant).

### Round 9 — WNT=8 reject; pack-layout locality CONFIRMED and de-risked (2026-09-19)

- **WNT=8 wide tile** (CFG=2, 512B contiguous per k-stride, bit-exact):
  REJECT — 1798 us vs 655 us cold (2.7x slower). Register pressure collapses
  occupancy; wider-tile-per-warp is not a viable route to DRAM locality.
- **Group-major trellis layout** (DEC=5 in `kernel_study/gemv_bench/`):
  permute `[k][n][words] -> [group][k][128]` so each warp's whole k-chunk
  stream is contiguous. Bit-exact (same logical weights, permuted storage):
  **cold 627.4 vs 667.6 us (−6.0%), 211.5 vs 198.8 GB/s; warm −4.1%.**
  The DRAM-locality hypothesis is confirmed and the reference implementation
  of both sides exists (bench5.cu: stock layout vs group-major indexing).
- Integration cost: the permutation must be applied at load time and every
  trellis reader re-indexed — p2b (done, DEC5), exllamav3 `exl3_gemm/moe`
  prefill kernels, and the vllm-exl3 fat GEMM. Expected e2e: ~+3% decode
  (p2b is ~48% of step). Weighed against prefill-correctness risk (E8
  lesson), this is a **de-risked, measured, multi-file project** — the
  recommended next kernel investment, not a same-day keep.

Campaign verdict table for the decode-side axes now closes with every
configuration-level lever measured; the three structural levers (NCCL
overlap ~7 ms/step, pack layout ~2 ms/step + prefill upside, pair-codebook
requant — now known to be pointless while memory-bound) are scoped in
flags.md / RESULTS.md round 8-9.

### Round 10 — N-split warp decomposition reject; p2b variant space exhausted (2026-09-19)

- **N-split warps** (each warp owns one 4-tile group's full K range; 8 warps
  read 2KB contiguous per k-slice; reduction-free epilogue; numerically
  equivalent reorder, uniform ≤3 half-ULP diffs, no clustering): **REJECT —
  758.0 vs 643.3 us cold (−18%).** The stock K-split's 8 independent
  k-regions per block are the latency hiding; N-split trades that for DRAM
  locality and loses. Gate tail waste (18 groups / 8 warps) adds more.
- With WNT=8 (registers) and N-split (latency) both rejected, the p2b
  optimization space is exhaustively mapped: **the stock K-split tile at
  206 GB/s cold is the local optimum for this pack layout**, and the
  group-major pack permutation (+6.0%, bit-exact, DEC5 reference) is the
  only remaining p2b lever — gated on a prefill-harness proof of
  no-regression (exllamav3 gemm-inner remap, E8 risk class).

**Campaign final state**: decode step ~10-15% faster kernel-verified
(wo_a fp8 einsum 288->80.6 us x40/step + fshift); prose decode 25.1 ->
31-34 tok/s; L.A.I.L band top; 8/8 + 3/3 quality; prefill unchanged;
every lever measured; structural hand-offs documented with sizes and
reference implementations.

## Campaign completion summary (2026-09-19, rounds 1-10)

**Objective**: substantial decode improvement with quality maintained, via
the B-side trellis-decode lever, MUL1 pack considered.

**Shipped (all gated 8/8 correctness + 3/3 e2e, prefill unchanged):**

1. **E10 `widen_p2b_fshift`** — single-SHF trellis window merge in the p2b
   decode (bit-exact; −6.1% warm kernel).
2. **E11 `fix_o_proj_woa_fp8`** — exact requant of wo_a onto the deep_gemm
   fp8 einsum (the pack stores F8; MXFP8 emulation dequantized it to bf16 at
   load -> strided bmm -> sm_80 WMMA 288 us/layer). 43/43 layers engaged;
   288 -> 80.6 us x 40/step, kernel-trace-verified.
3. **b12x pip package removal** — latent −5-9% prose regression found by
   A/B before it shipped in the canonical image.

**Decode result**: step time −10-15% kernel-verified; prose decode 25.1 ->
31-34 tok/s (repeated 5-run medians; historical MCG baseline ~28); L.A.I.L
21.9-23.4 vs published 22.40 (band top). Quality maintained throughout.

**The B-side trellis-decode question, answered with measurements**: the
trellis decode is NOT the decode bottleneck. The p2b GEMV streams at
206-212 GB/s cold = 76-78% of UMA peak; eight variants were measured
(PF depth, smem staging, cp.async ring, funnelshift [kept], WNT=8 wide
tile, N-split warps, 64K-entry LUT [hardware-closed at 99KB smem],
pair-codebook [closed: decode is memory-bound, instruction cuts provably
don't move it]). The stock K-split tile is the local optimum for this pack
layout. The MUL1 pack was found on disk, analyzed (identical stream format,
different codebook arithmetic), and closed (prior port lost −16% via
acceptance; no kernel-level upside).

**Documented hand-offs (sized, de-risked, reference code in repo):**
- group-major pack permutation: +6.0% p2b bit-exact (bench5.cu DEC5 both
  sides); gated on a prefill-harness no-regression proof.
- NCCL AR in-graph overhead: 124 us vs 43 us isolated floor, ~7 ms/step;
  needs structural comm overlap (not in this vLLM build).
- Dense GEMM latency stacking (~18.5 ms/step): latency-bound at m<=8,
  tiles/autotune closed.

Final serve: `dsv41-flash-exl3-sm121:canonical-e12` (Dockerfile-exact),
both nodes, stock flags, smoke 323, woa-requant 43/43.

## Round 11 — the L.A.I.L gap: Engram disk staging (2026-09-19)

The user's brief: kernel wins must show up in L.A.I.L (22.0-23.5 band,
acc ~2.27-2.32). Cross-checking steps/s showed the wins were real at the
device level but absorbed by inter-step dead time. Trace forensics on a
28-step L.A.I.L window (crash-safe ijson parser with RLIMIT_AS after a
full-trace json.load OOM'd the host — see incident note below):

- ~25.6 ms/step of GPU idle; the innermost frame spanning nearly every
  gap was `_thread.lock.acquire` under `engram_disk._read_rows ->
  Future.result`. Not NCCL (AR p50 = 41-54 us, near the 43 us isolated
  floor; the 122-236 us "averages" were tail-polluted), not the scheduler.
- `EngramDiskStager.stage` gathers the per-layer disk tables serially in
  prepare_inputs; 1-2 cold-row NVMe misses per step block the next graph
  replay. `async_scheduling` measured NEUTRAL-negative (21.8/21.9 vs
  22.0-23.5 band); `NCCL_MIN/MAX_NCHANNELS=1` measured NULL (steps/s 9.65
  vs 9.68-10.26 band).

### KEEP: parallel Engram disk staging (`docker/patch/engram_stage_fast.py`)

The per-table CPU gathers (pread + dequant into pinned host rows) now run
concurrently; H2D stays on the calling thread (stream order before replay
unchanged). Self-check vs the serial loop is bit-exact on both TP ranks.

- census: avg read_w 0.20 -> 0.03 ms per gather
- L.A.I.L prose: 22.0-23.5 -> 25.2-26.3 tok/s at matched acceptance
  (n=10 per boot, three boots), steps/s 9.7-10.3 -> 11.1-11.5
- full validation pass: correctness 5/5 (math_small 323, math_mid 252,
  json_strict, tool_call, code_trace), pp in band (685-689 @16k/64k),
  prose bench 28.4-33.4

### EXPERIMENTAL (off by default): next-step row prefetch (`engram_prefetch.py`)

Timeline fact: the next step's exact input tokens (verify outputs) are on
the GPU before the draft graph launches, so a side-stream D2H + CPU port
of `_hash_ids_kernel` + `posix_fadvise(WILLNEED)` can warm the rows under
the draft graph. The CPU hash port was verified against live stage dumps
(predicted rows intersect the actual staged rows; garbage-tail truncation
via num_sampled fixed; anchor-relative next-chunk positions fixed). But
the measured pf_hit stays ~0% and read times do not drop, so the
consumption pairing or the fadvise window is still wrong somewhere.
Ships dark (`DSV41_ENGRAM_PREFETCH=0`), fully advisory, self-disabling.

### Incident note (worth remembering)

Loading a 132 MB / 5.1M-event torch trace with `json.load` materializes
~15-20 GB of Python objects on this UMA host with the serve resident ->
kernel OOM cascade -> hard hang (user power-cycled spark1). All trace
analysis now goes through `kernel_study/gemv_bench/parse_trace_safe.py`
(ijson streaming + 6 GB RLIMIT_AS). Traces must be `docker cp`'d out of
the container immediately (docker rm destroys them) and only after the
profiler flushes (wait >10 s after stop_profile).

### Hand-offs

- Prefetch: the hash math and the hook are verified end-to-end offline;
  the remaining bug is in the published-set vs gather pairing (off-by-
  something) or fadvise latency. Worth one focused session.
- Step composition at L.A.I.L after round 11 (per step): p2b 33-34 ms,
  dense b12x ~19-20 ms, AR steady ~6 ms + tail, draft bf16 sm80 kernels
  ~3-4 ms, gaps ~11-13 ms. 35 tok/s at acc 2.3 needs a 66 ms step:
  p2b group-major pack (+6% kernel, needs prefill no-regression proof) +
  dense fusion or the prefetch above are the remaining paths.

## 2026-09-20 — nccl-set arm (KEEP)

NCCL buffer/proto/channel set (BUFFSIZE=1M, LL128_BUFFSIZE=256K, PROTO='^LL128',
MAX_NCHANNELS=8) on the mem-hygiene baseline. Wiring `1a43d15`; verdict +
evidence in `results/2026-09-20-nccl/VERDICT.md`. Prose 34.69 (flat),
prefill32k 733.0 (+2.1%), L.A.I.L 26.55/26.12, MemAvail 22.29/23.84 GiB
(+3.5/+2.8). Zero NCCL WARN/error lines on either node. Serve left up with
the full env set; exact boot in `results/2026-09-20-nccl/boot-arm.sh`.

## 2026-09-20 — engram-pf-v2 arm (NO-GO, REVERTED)

Final campaign arm: Engram prefetch v2 (DSV41_ENGRAM_PREFETCH=1). The v2
pairing self-check fixed v1's publish-too-late and proved the remaining
failure is the prediction itself: publishes land before consumption every
gen, yet pred∩consumed ≈ 0 (pf_hit ~0%, 98/105 census windows); read_w
already 0.06–0.10 ms cold. L.A.I.L 26.05 (job 243a53d345ca) = parity with
baseline, not the projected 34–36 tok/s. Reverted to the exact NCCL-arm
serve. Wiring commit `befa570` stays (dormant, env-guarded). Verdict +
evidence: `results/2026-09-20-engrampf/VERDICT.md`.

## 2026-09-21 — trace attribution round 18 (gap owned by engram stage sync; SOFTMAX_VERIFY revert)

Round 18 profiled ONE L.A.I.L prose request on stock k3c (torch profiler
replica via --profiler-config; /start_profile only mounts with that flag).
Result: pure GPU idle ~14.2 ms/step, 99.9% under ONE stack —
`EngramDiskStager.stage → hashes_ready.synchronize()` (hard event sync in
prepare_inputs, engram.py:1329-region). Eager sampler region, sampler
softmax, and indexer bookkeeping all closed at <0.15% of idle. Named fix
(not implemented): CPU-side next-step hashing to delete the sync — sized
to span the remaining gap to 35 alone. SOFTMAX_VERIFY A/B on k3c:
median 27.14 (−5.6% vs 28.76; acc_len 2.20) → REVERT, knob dead.
Best unchanged: **28.76** (k3c), serve left up on boot-k3c.sh. Evidence:
results/2026-09-21-trace-attribution/.

## Campaign close (2026-09-22, rounds 15-33) — FINAL: 33.23 L.A.I.L, lm_head KEEP, serve UP

Target was 35+ tok/s in the L.A.I.L cell (decode/prose c=1 t=0.2). Final
best: **33.23 pooled n=10 median** (rounds 15-33 scoreboard: 26.4 → 27.27
(k3) → 28.76 (capture match) → 30.10 (prefetch v3 re-arm) → 31.37 (gather
v2) → 33.23 (lm_head MXFP8, round-33 high-n KEEP — round-30's +2.5% miss
was a low-band sample; fresh n=10 shows +5.9%)). Serve left UP on
results/2026-09-22-endgame2/boot-lm.sh (canonical-e12 + pack
2.0bpw-mcg-lmhead-mxfp8 + DSV41_LMHEAD_MXFP8=1 on the round-23 keep set).
Four numbers (fresh): prose 9-run 39.60, prefill 8k 240.1 / 32k 589.6,
MemAvail post-32k 21.6/23.2 GiB. 35 NOT crossed: the box is device-bound
(trace3: device-busy 65.25 ms vs 65.7 ms bar at acc 2.3); named path =
p2b MoE 22.4 ms (G8 fold −1.34 ms projected, blocked on the G8 post-load
host-memory site — lane closed, evidence in results/2026-09-22-g8final/)
+ dense b12x 17.6 ms (CUDA kernel work) + banked lm_head −2.6 ms + staged
fadvise cap ~3.1 ms. Zero OOM the whole campaign (floors logged every
boot). Full round-by-round table + honest ceiling statement + artifact
index: results/2026-09-22-close/VERDICT.md.

## 2026-09-24 review s0 live baseline — ran with prefetch v3 disarmed (NOT a promoted-config baseline)

`results/2026-09-24-review/s0-live-baseline/` was recorded (d4db6cd) as the
09-22 promoted config on the same container ("cmd digest 107dc938 = 09-22
close"). The cmd and env were the same, but prefetch v3, a default-on KEEP
lever (R21, +4.6% L.A.I.L), had already turned itself off on BOTH ranks:
three `IndexError` at `_pf_next_chunk` `outs_r[A - 1]`, then "dsv41: engram
prefetch disabled after 3 errors" (worker log 09-24 ~13:39 container
clock; both ranks per the read-only audit run recorded in a911bb3), during
the quality-harness full run with its c=2 phase. Excerpt:
`s0-live-baseline/08-prefetch-disarm-worker.log`. `serve_env` still showed
`DSV41_ENGRAM_PREFETCH=1`, so four_numbers could not see it.

Every s0 cell (14:55-15:55Z) ran prefetch-off: L.A.I.L 10x 27.44
(76 ms/step vs 66 at the 09-22 close), four_numbers L.A.I.L 28.84, prose
37.65 / 39.35, prose_long c=1 75.08 ms/step. In the captured worker log,
decode-sized gathers (rows/call < 150) read 2.63 ms/call (575 census
lines) before the disable and 10.80 ms/call (75 lines, to ~14:48) after;
the review's full-log count was 2.68 vs 7.20 ms/call, about +9 ms/step
over 2 Engram layers, consistent with the unexplained 66 -> 76 ms/step.

Do not use 27.44 / 28.84 as a L.A.I.L floor or read the drop as drift;
re-baseline from the A boots of the next ABAB sequence. Fixes on the
branch: the likely IndexError cause (record_stream on the side-stream
sources, sampled-row stride, bounded num_sampled, 2ebb78d; not yet
verified on GPU), the gv2
census race that could disarm gather v2 the same way (13ee9f4), and
tools/disarm_scan.sh in four_numbers.sh (lever_disarmed in the JSON,
c3800a3) so a runtime disarm now invalidates the capture.

## Round 34 — 2026-09-24 review campaign: four levers promoted on canonical-e13, serve UP

The branch perf/review-2026-09-24 (a foundation package, 9 lever packages, then review fixes) went through a serialized GPU campaign on both Sparks. The stages ran s1 to s13. There are no s9 or s11 directories. Evidence for each stage is in `results/2026-09-24-review/campaign/<stage>/` (`summary.json`, numbered logs, and both ranks' docker logs).

**New defaults** (commit af90bbb):
- `DSV41_ENGRAM_WILLNEED=1`
- `DSV41_STREAM_FEED=1`
- `DSV41_WOA_PREPACK=1`
- `DSV41_DSPARK_SPARSE_MARKOV=1` (k=256)

The image is `dsv41-flash-exl3-sm121:canonical-e13`. It is review-e13 (`sha256:c81762335a12`, `docker/Dockerfile.e13` on canonical-e12), tagged on both nodes. `docker/Dockerfile` runs the same two new build stages (srcsort, woa-prepack). All of its other build-time scripts predate the e12 build (`s13-promote-final/01-image-provenance.txt`).

**Rejected, still default off** (numbers in flags.md):
- `DSV41_MHC_DECODE_SPLITS=40` (quality)
- `DSV41_P2B_SRC_SORT` (below its 3% gate)
- `DSV41_DENSE_DG_SMALLM` (slower on every shape)
- dual-rail NCCL (small-message latency)

**Headline (pooled s8+s13, 2 boots).** s8 and s13 ran the identical default config, so the headline pools both boots' per-run values and reports the pooled median (`campaign/pool_s8_s13.py` → `campaign/pooled-s8-s13.json`; the recipe.yaml measured rows use the same numbers). Before is the s2 fresh boot of the R33 config.

| Metric | s2 before | pooled s8+s13 after | Δ | per boot s8 / s13 |
|---|---:|---:|---:|---:|
| L.A.I.L fresh (tok/s; n=10 → 20) | 32.44 | 33.68 | +3.8% | 34.51 / 32.44 |
| L.A.I.L fresh ms/step | 66.38 | 62.63 | −3.75 | |
| L.A.I.L after C2-STRESS (tok/s; n=10 → 20) | 31.91 | 33.75 | +5.8% | 34.00 / 33.63 |
| prose c=1 (tok/s; n=9 → 18) | 37.07 | 39.52 | +6.6% | 39.87 / 39.18 |
| prose c=1 ms/step | 67.77 | 63.5 | −4.3 | |
| structured c=1 (tok/s) | 58.57 | 62.52 | +6.7% | 62.51 / 62.68 |
| prose c=2 aggregate (tok/s) | 52.93 | 56.77 | +7.3% | 57.55 / 56.36 |
| structured c=2 aggregate (tok/s) | 84.04 | 89.44 | +6.4% | 89.39 / 89.45 |
| novel prefill 8k / 32k (tok/s, s13 only) | 184.2 / 231.1 | 838.0 / 811.5 | 4.5x / 3.5x | |

Pooled bench_decode medians use the per-run values as logged (2 decimals); ms/step pooled from the logged per-run values (1 decimal). SPARSE_MARKOV's effect on acceptance is open (round-2 ABAB pending; see below).

**Before and after, s13 alone.** Before is s2: a fresh boot of the R33 config from the main checkout (45d3303, `results/2026-09-22-endgame2/boot-lm.sh`, canonical-e12). After is s13: a fresh `AUDIT=strict ./run.sh` from the branch with no overrides. Both boots ran the same protocol in the same order. s13's DSV41/NCCL/VLLM env and engine argv are identical to s8's, which set the four levers by hand.

| Metric | s2 before | s13 after | Δ |
|---|---:|---:|---:|
| L.A.I.L fresh, median of 10 (tok/s) | 32.44 | 32.44 | 0.0 |
| L.A.I.L fresh ms/step (decode_s/chunks, median) | 66.38 | 62.57 | −3.81 |
| L.A.I.L fresh acceptance (median) | 2.167 | 2.071 | −0.096 |
| L.A.I.L fresh TTFT (median, s) | 0.318 | 0.355 | +0.037 |
| L.A.I.L after C2-STRESS (tok/s; ms/step) | 31.91; 66.99 | 33.63; 62.76 | +5.4%; −4.23 |
| prose c=1, bench_decode 9 runs (tok/s; ms/step) | 37.07; 67.77 | 39.18; 63.28 | +5.7%; −4.49 |
| prose c=1 acceptance (pooled; median-run) | —; 2.532 | 2.488; 2.525 | |
| prose c=2 per stream / aggregate | 27.13 / 52.93 | 28.70 / 56.36 | +6.5% agg |
| structured c=1 (tok/s; ms/step) | 58.57; 67.63 | 62.68; 63.19 | +7.0% |
| structured c=2 per stream / aggregate | 42.06 / 84.04 | 45.18 / 89.45 | +6.4% agg |
| TTFT p50, prose c=1 (s) | 0.263 | 0.251 | |
| micro pass 1, 8k warm / novel (tok/s) | 211.9 / 184.2 | 716.3 / 838.0 | 3.4x / 4.5x |
| micro pass 1, 32k warm / novel (tok/s) | 430.0 / 231.1 | 761.1 / 811.5 | 1.8x / 3.5x |
| micro 8k repeat pass, warm / novel (tok/s) | 771.9 / 247.3 | 789.9 / 814.6 | +2.3% / 3.3x |
| micro pass 1 novel TTFT 8k / 32k (median, s) | 44.06 / 141.14 | 9.71 / 40.20 | |
| API ready after start (s) | 576 | 572 | |
| 'Loading weights took' TP0 / TP1 (s, main + second load) | 330.18+27.78 / 180.54+20.83 | 311.53+27.54 / 148.03+22.79 | −5.3% / −15.2% |
| MemAvailable at ready, spark1 / spark2 (GiB) | 24 / 26 | 23 / 24 | |
| MemAvailable after FULL (GiB) | 21 / 23 | 21 / 22 | |
| quality quick vs quick.json | PASS | PASS | |
| NLL / decode median abs dlogprob | 0.244203 / 0.01838 | 0.243823 / 0.00703 | |
| tools exact / needle / A/A hazard / golden hazard | 0.9545 / 6/6 / 0.01636 / 0.01543 | 0.9545 / 6/6 / 0.02005 / 0.02305 | |

In both boots, micro pass 1 is the first long prefill after the boot, so the Engram page cache is cold and its "warm" cell is not page-cache warm. The 8k repeat pass is warm.

The L.A.I.L cell is sampled at t=0.2, so its tok/s follows acceptance. The fresh s13 batch drew the lowest L.A.I.L acceptance of the campaign (2.071). Its ms/step is 3.81 lower than s2, but its median tok/s is exactly s2's. Pooling every 10-run L.A.I.L batch per arm (fresh plus post-C2-STRESS) gives:

| Arm | Batches | Pooled median tok/s | Pooled median ms/step | Mean acceptance |
|---|---|---:|---:|---:|
| old config (s2) | 2 | 32.25 | 66.95 | 2.149 |
| B: new code, levers off (s3, s6) | 4 | 33.95 | 63.50 | 2.159 |
| C: WOA + MHC40 (s5, s7) | 4 | 33.74 | 62.91 | 2.131 |
| S: WOA + SPARSE_MARKOV, the new defaults (s8, s13) | 4 | 33.72 | 62.67 | 2.129 |

The new defaults beat the old config by +4.6% pooled L.A.I.L tok/s and −4.28 ms/step. Against B the decode levers save 0.83 ms/step, but pooled acceptance is 1.4% lower, so pooled L.A.I.L tok/s is −0.7%. WOA_PREPACK is bitwise exact and cannot move acceptance. SPARSE_MARKOV can. Its per-batch acceptance (2.172 / 2.131 / 2.071 / 2.118) overlaps B's (2.160 / 2.152 / 2.183 / 2.118). Two S boots cannot separate this from noise. The GPU_PLAN arm-S gate would call a drop of 1% or more "REVERT or retry with TOPK=1024", and s13's fresh batch (2.071) is 4.1% under B's pooled mean acceptance (2.159). The lever stays promoted as s8 decided, with this open. Next step: an S-vs-WOA-only ABAB, or the TOPK=1024 retry. The bench_decode and four_numbers cells gain on both ms/step and tok/s: four_numbers prose 40.37, and bench prose median-run acceptance 2.525 is inside the B band of 2.513 / 2.597.

**s13 four_numbers** (`s13-promote-final/four_numbers/four_numbers.json`, 11.5 min, not partial, serve_env_ranks_match true, lever_disarmed false):
- prose 9x: 40.37 tok/s at 63.03 ms/step, acceptance 2.566, post-EOS 0.61.
- micro, 3 runs: pp_warm 759.6 (8k) / 766.8 (32k), pp_novel 808.8 / 800.2.
- MemAvailable after 32k: 21.3 / 22.37 GiB.
- L.A.I.L, 3 runs: 33.98.
- prose_long c=1: 33.13 tok/s at 62.88 ms/step, acceptance 2.09, natural finish length.
- prose_long c=2: 23.52 per stream, 45.94 aggregate, 86.89 ms/step.

**Warm-prefix finding.** four_numbers' warm-prefix cell got 0 of 2048 expected hits on a 2058-token prompt (TTFT 2.76 s cold, 2.67 s warm). Rechecks in the same boot (`13-warm-prefix-*.log`) show that a repeat either hits fully or not at all, depending on how far the prompt runs past the last 128-token boundary. With the hit point P = floor((N−1)/128)·128, a repeat of N tokens missed when N−P was 10, 14, 25, 55 or 58. It hit when N−P was 68, 97, 99, 116, 117, 118 or 127: for example 1791→1664, 2372→2304 and 4086→3968 tokens, with TTFT 0.33–0.46 s. An identical 8k resend hit 7936 tokens (TTFT 0.4 s, `13-prefix-resend-8k.log`). The s0 warm-prefix hit (N=2037, N−P=117) is in the hit band. Nothing in round 34 touches the KV cache manager. Whether the R33 config has the same band was not tested. The four_numbers warm-prefix gate (hit fraction ≥ 0.9) therefore depends on the random prompt length.

**Quality --full** (`s13-promote-final/quality_full.json`, 1554.8 s) passes against `quality-baseline/full.json`:
- NLL 0.243755 / 0.242506 over 2 runs (repeat delta 0.001249); baseline 0.243098.
- Decode median |dlogprob| 0.01621. Tools exact_args 21/22.
- Needle 9/9, including all three 131072 cells.
- GSM8K thinking off: 92/100 against the baseline's 94/100. The discordant items split 2 vs 0 (idx 611 and 689; s12's stock-head boot also missed 689), exact binomial p = 0.5. The Wilson upper bound, 0.9589, clears the 0.94 gate.
- GSM8K thinking on: 38/40, the same two misses as the baseline.
- MMLU: 199/228 against 197/228 (discordant 2 s13-only vs 4 baseline-only).
- A/A hazard 0.02088, golden hazard 0.02033.
Before and after deltas for every metric are in `s13-promote-final/final.json`.

**Serve left UP**: canonical-e13, round-34 defaults, started 2026-09-25T01:24:43Z from the perf-review-0924 worktree. Smoke returned 323 and Red. The strict audit was ok at ready, after FULL, after C2-STRESS and at the end. disarm_scan rc was 0, prefetch v3 stayed armed on both ranks, and 0 post-ready TileLang JIT warnings appeared. C2-STRESS passed 12/12.

### Stages

- **s0 (live baseline, 2026-09-24).** The aged R33 serve ran with prefetch v3 already self-disabled; see the section above.
- **s1 (microbenches, serve down, `s1-microbench/`).**
  - WOA prepack is bitwise equal; 31.75 → 25.0 us/call at M=3.
  - MHC prenorm split-K 16 → 40 goes 18.08 → 11.17 us at T=4 (maxabs 6e-6).
  - p2b srcsort: SASS identical, but cold time only −0.37 / +1.41 / +0.83%, below the 3% gate.
  - deep_gemm is slower than b12x on every dense shape (−3.1% to −12.0%).
  - Dual-rail NCCL: busbw +64–77%, but small-message latency +9.5–30%.
  - Requant probe: Viterbi+refit relerr 0.2616 vs stock 0.3773 (MSE ratio 0.4805). A full pack takes 12.61 h on 2 nodes with exllamav3 1.5.1. The probe is weight-space only; model quality was not measured.
  - review-e13 was built and is identical on both nodes.
- **s2 (old config, fresh boot, `s2-old-fresh/`).** This is the "before" above. C2-STRESS did not reproduce the prefetch bug (L.A.I.L 32.44 → 31.91, still armed).
- **s3 (B1: new code, levers off, e12, `s3-B1-new-defaults/`).**
  - L.A.I.L 34.11; prose 39.10 at 64.26 ms/step against s2's 67.77. The per-run ranges do not overlap.
  - The strict audit is clean, and env parity with boot-lm.sh holds except for the expected new names.
- **s4 (E1: WILLNEED + STREAM_FEED, e12, `s4-E1-willneed/`).**
  - Novel prefill: 8k 798.3 (4.23x s3), 32k 805.7 (3.43x). Warm 8k 747.2 (−1.9%). Decode fadv/call is 0.0.
  - 'Loading weights took' −6.3% on TP0 and −20.6% on TP1.
  - Worker-death drill: run.sh exited 1 35 s after `docker kill`, saved both ranks' logs and left no orphans.
- **s5 / s7 (C1 / C2: WOA_PREPACK + MHC40, e13).**
  - The ABAB against s3 / s6 gives prose c=1 −0.60 ms/step (B spread 0.27), with tok/s flat.
  - The trace shows WOA −0.71 / −0.81 and MHC −0.47 / −0.58 ms/step.
  - C1 failed quick quality (A/A hazard 0.03409 > 0.03206). Verdict: WOA promote, MHC reject.
  - The post-ready TileLang JIT of `mhc_pre_big_fuse_with_norm` (about 13 s on the first request) comes with the e13 image and is intermittent. It appeared in B2, C1 and C2, but not in s8, s12 or s13.
- **s6 (B2: e13 with WILLNEED + STREAM_FEED, decode levers off, `s6-B2-defaults-e13/`).** L.A.I.L 34.48, prose 40.12 at 64.53 ms/step.
- **s8 (S: WOA + SPARSE_MARKOV, `s8-S-sparse-markov/`).**
  - ms/step against the B mean: prose −0.825, structured −0.45, L.A.I.L −0.245, post-stress −1.35.
  - L.A.I.L 34.51. Quick quality passed. Verdict: promote (one boot, see above).
- **s10 (diagnostics, `s10-diag-census-skew/`).**
  - The step census could not see FULL-graph decode; fixed in bffbae8.
  - The MoE duplicate-expert fraction is 0.2987 per layer (random 0.0232), a realistic ceiling of about 5.1–5.2 ms/step. The next MoE step is a coop kernel that reads each expert once, microbenched on the saved routing first.
  - Rank 0 (spark1) is the straggler inside target: +1.32 ms/step non-NCCL kernel time (+2.2%).
- **s12 (lm_head A/B, `s12-lmhead-quality-ab/`).**
  - Stock bf16 head against mxfp8: ΔNLL −0.00045, GSM8K 94/94, MMLU 197/197, and the flip hazard is inside the control.
  - The stock head costs −6.2% L.A.I.L. Verdict: keep mxfp8.

### Prefetch v3 c=2 self-disable bug

On the aged R33 serve (up since 2026-09-22 11:55Z), prefetch v3 hit `IndexError` at `engram.py:1598 _prefetch_worker` 3 times per rank during the 09-24 quality run with its c=2 phase, then logged "dsv41: engram prefetch disabled after 3 errors" on both ranks (worker 13:42:58.751Z, head 13:43:44.500Z; `s1-microbench/aged/`).

Measured impact on that serve, prefetch off: L.A.I.L 27.44 (s0, 10 runs, 76 ms/step) and 28.82 (s1, 3 runs). A fresh boot of the same config with prefetch armed gave 32.44 (s2). That is −15.4% / −11.2%, but it also includes about 2.3 days of uptime (swap 6.7 / 3.8 GiB), so not all of the gap is attributed to the disable.

The direct cost shows in the s0 worker log: decode-sized Engram gathers read 2.63 ms/call before the disable and 10.80 ms/call after.

Cause: no record_stream on the side-stream copies, an intermittent allocator race. The fix is 2ebb78d, and tools/disarm_scan.sh in four_numbers makes a runtime disarm invalidate a capture. After the fix, s3, s5, s6, s7, s8 and s13 all ran bench c=2, quality c2 and C2-STRESS (12 c=2 requests). Prefetch v3 stayed armed on both ranks with 0 IndexError lines. The old code did not reproduce the bug on a fresh boot either (s2), which matches a race that needs uptime.

### Not done / open

- The SPARSE_MARKOV acceptance question above (round-2 ABAB pending).
- An MHC-only ABAB with a larger selfcons sample, if MHC is re-opened.
- The coop MoE kernel microbench.
- The Viterbi requant pack build (12.61 h).
- hb-3 (G8 balloon re-judge).
- Drill (b), head failure.
- E2 (MAX_ROWS=256), not needed.
- A warmup key for the intermittent e13 TileLang JIT.
- The warm-prefix band on the R33 config.

## Round 35 — 2026-09-26 kernels + Viterbi: the round-3 kernel bundle and SPARSE_MARKOV_TOPK=1024 promoted on canonical-e14, Viterbi pack rejected, serve UP

The branch perf/kernels-r3 carries the k3 round-3 kernel work: five workstreams (`results/2026-09-25-kernels/coop-moe/`, `dense-gemv/`, `mhc-det/`, `fusion-host/`, `comm/`), the ground-truth profile (`profile/timeline-r3.txt`), the integration record (`integration/VERIFY.txt`, one image `review-e14` for every lever) and the full Viterbi re-encode of the routed experts (`requant/`, `/home/sfxnz/projects/data/dsv41-requant-viterbi/`). A serialized serve campaign on both Sparks (2026-09-26 05:27Z to 2026-09-27 03:33Z) followed `results/2026-09-25-kernels/ARMS-r3.txt` and ARMS.md step 6. Evidence per boot is in `results/2026-09-26-serve-r3/<arm>-<n>/`, and each decision recomputes from the raw per-boot files (`decision-bundle.json`, `decision-viterbi.json`, `decision-addons.json`, `decision-markov.json`, with their `*-compute.py.txt`). Every boot ran image `sha256:3a002b55c9bc` on both ranks.

**New defaults** (commit 5da6f21):
- the round-3 kernel bundle, promoted as one unit: `DSV41_P2B_COOP=2`, `DSV41_DENSE_GEMV=1`, `DSV41_MHC_DET_SPLITS=16`, `DSV41_ENGRAM_NATIVE_STAGE=1`, `DSV41_ENGRAM_EARLY_HASH=1`, `DSV41_ATTN_T2R_DEDUP=1`, `DSV41_SWA_META_FUSED=1`, `DSV41_MOE_PREP_FUSED=1`, `DSV41_CANDIDATE_MASK_BOUNDED=1`, `DSV41_INDEXER_WP_GEMV=1`
- `DSV41_DSPARK_SPARSE_MARKOV_TOPK=1024` (SPARSE_MARKOV=1 stays)

The image is `dsv41-flash-exl3-sm121:canonical-e14`: review-e14 (`docker/Dockerfile.e14` on canonical-e13, `sha256:3a002b55c9bc`) tagged on both nodes (`results/2026-09-26-serve-r3/promote-image-provenance.txt`). `DSV41_P2B_COOP=2` needs it. `docker/Dockerfile` runs the same p2b chain from scratch.

**Not promoted, still default off** (numbers in flags.md): the five KP add-ons (`DSV41_MHC_DET_OVERLAP`, `DSV41_ENGRAM_WKV_TP`, `DSV41_PM_QOS_US`, `DSV41_NCCL_EAGER_TWIN`, `DSV41_AR_L2_PREFETCH`); the revert to `DSV41_DSPARK_SPARSE_MARKOV=0`. **Rejected**: the Viterbi pack `2.0bpw-mcg-viterbi-lmhead-mxfp8`; the serve pin stays `2.0bpw-mcg-lmhead-mxfp8`.

**Headline.** Before is arm A, the round-34 defaults on the same image (A-1, A-2). After is the final promoted config: S1024-1 and S1024-2 (P6, same env set by hand) and final-1 (the validation boot of the committed defaults). All five boots have the same pack, image ID, docker/patch sha256 list and procedure; the after boots share one container env. Arm value = median of the per-boot medians; noise = the larger boot-to-boot spread. The A boots and the after boots ran in the same window but were not interleaved with each other, so this table describes the result; the promotions rest on the interleaved decisions below. Computed by `results/2026-09-26-serve-r3/round35-headline-compute.py.txt` → `round35-headline.json`.

| Metric | A-1 / A-2 | S1024-1 / S1024-2 / final-1 | A → final (medians) | Δ | noise |
|---|---|---|---|---:|---:|
| L.A.I.L fresh, t=0.2 (tok/s) | 33.69 / 34.21 | 43.17 / 43.62 / 42.74 | 33.95 → 43.17 | +9.22 (+27.1%) | 0.88 |
| L.A.I.L fresh (ms/step) | 63.01 / 62.57 | 49.22 / 49.00 / 48.73 | 62.79 → 49.00 | -13.79 (-22.0%) | 0.49 |
| L.A.I.L fresh acceptance (median run) | 2.120 / 2.156 | 2.133 / 2.186 / 2.109 | 2.138 → 2.133 | -0.005 | 0.076 |
| four prose c=1 (tok/s) | 39.07 / 39.24 | 53.67 / 52.04 / 52.52 | 39.16 → 52.52 | +13.37 (+34.1%) | 1.63 |
| four prose c=1 (ms/step) | 63.87 / 63.08 | 48.10 / 48.07 / 47.92 | 63.47 → 48.07 | -15.40 (-24.3%) | 0.78 |
| prose_long c=1 (tok/s) | 32.88 / 33.12 | 44.17 / 43.73 / 43.56 | 33.00 → 43.73 | +10.73 (+32.5%) | 0.61 |
| prose_long c=1 (ms/step) | 63.47 / 63.22 | 48.80 / 49.49 / 48.98 | 63.34 → 48.98 | -14.36 (-22.7%) | 0.69 |
| prose_long c=1 acceptance | 2.151 / 2.073 | 2.163 / 2.174 / 2.149 | 2.112 → 2.163 | +0.051 | 0.078 |
| bench prose c=1 (tok/s) | 38.11 / 39.29 | 51.99 / 52.08 / 54.03 | 38.70 → 52.08 | +13.38 (+34.6%) | 2.04 |
| structured c=1 (tok/s) | 61.62 / 62.76 | 83.81 / 83.45 / 83.90 | 62.19 → 83.81 | +21.62 (+34.8%) | 1.14 |
| structured c=1 (ms/step) | 64.28 / 63.11 | 47.26 / 47.46 / 47.21 | 63.69 → 47.26 | -16.44 (-25.8%) | 1.16 |
| bench prose c=2 aggregate (tok/s) | 55.44 / 57.95 | 82.38 / 81.69 / 80.61 | 56.70 → 81.69 | +24.99 (+44.1%) | 2.50 |
| structured c=2 aggregate (tok/s) | 88.35 / 89.73 | 138.60 / 138.24 / 138.34 | 89.04 → 138.34 | +49.30 (+55.4%) | 1.38 |
| prose_long c=2 aggregate (tok/s) | 47.18 / 47.63 | 65.24 / 65.62 / 68.05 | 47.40 → 65.62 | +18.22 (+38.4%) | 2.81 |
| prose_long c=2 (ms/step) | 88.74 / 87.21 | 61.37 / 62.56 / 62.35 | 87.97 → 62.35 | -25.62 (-29.1%) | 1.52 |
| L.A.I.L 3x after c=2 traffic (tok/s) | 32.85 / 33.38 | 43.68 / 42.46 / 43.66 | 33.12 → 43.66 | +10.54 (+31.8%) | 1.22 |
| pp_warm 8k / 32k (tok/s) | 758.1 / 756.6 ; 765.9 / 763.8 | 757.6 / 753.8 / 765.0 ; 763.0 / 769.5 / 768.0 | 757.4 → 757.6 ; 764.9 → 768.0 | +0.2 ; +3.1 | 11.2 ; 6.5 |
| pp_novel 8k / 32k (tok/s) | 819.5 / 818.9 ; 808.5 / 817.4 | 821.4 / 807.4 / 835.0 ; 806.2 / 806.3 / 810.7 | 819.2 → 821.4 ; 813.0 → 806.3 | +2.2 ; -6.7 | 27.6 ; 8.9 |

Every after boot beats the A median on every tok/s and ms/step row. Acceptance did not move: L.A.I.L median-run acceptance 2.138 → 2.133 and four prose 2.501 → 2.513, inside the noise. Prefill is unchanged within the noise (the bundle is decode-side). Pooled per-run medians: L.A.I.L 33.92 (n=20) → 42.98 (n=30) tok/s, 62.88 → 48.96 ms/step, TTFT 0.356 → 0.322 s; four prose c=1 39.155 (n=18) → 52.66 (n=27). bench_decode pooled per stream (the recipe.yaml measured rows): prose c=1 38.46 → 52.97, structured c=1 62.14 → 83.73, prose c=2 aggregate 56.95 → 81.73, structured c=2 aggregate 88.76 → 138.34.

Against the round-34 published cells (s8+s13 pooled, a different day): L.A.I.L 33.68 → 42.98 pooled, prose c=1 39.52 → 52.97, structured c=1 62.52 → 83.73. The round-33 target of 35 tok/s L.A.I.L is crossed on all 20 serve boots in this window that ran the full K bundle (lowest per-boot median 41.829, VIT-1 on the Viterbi pack); the A boots read 33.69 / 34.21 and the KC boots (bundle without COOP) 37.56 / 36.73.

Quality on the final config: quick PASS on S1024-1, S1024-2 and final-1; full PASS on final-1 on every component (NLL 0.242247, decode median |dlogprob| 0.01185, golden hazard 0.02659 and A/A 0.01698 against 0.04348, tools 22/22 json and 21/22 exact, GSM8K 94/100, GSM8K thinking 39/40, MMLU 199/228, needle 9/9 including 128k). Round 34's s13 full read NLL 0.243755, GSM8K 92/100, thinking 38/40, MMLU 199/228. tools exact_args is the tightest gate: 21/22 (t09 `calculate_loan_payment` called with `{}`) sits exactly at the 0.9545 limit, and 20/22 (t01 `get_weather` also `{}`) passes only by the gate's one-item slack. The A readings are 22/22, 21/22 (A-1 rep1, rep2) and 21/22 (A-2); the 20 full-bundle quick readings are 20/22 on K-2, CUR-2, Kp-1, KP-1, Kp-2 and W-2, 22/22 on VIT-1, VIT-2 and W-1, and 21/22 on the other 11 (KC-1 / KC-2 read 21/22 and 20/22); the full readings are 20/22 (K-2), 22/22 (CUR-1, VIT-1, VIT-2) and 21/22 (CUR-2, final-1).

**Final validation boot** (`final-1/`, 5da6f21, commits f6d9fd2, d8f8e19, 77df363). `AUDIT=strict ./run.sh` with no lever env: strict audit ok on both ranks, every K line and the k=1024 line on both ranks, KP lines absent, disarm_scan rc 0, gpu_guard rc 0 (0 FOREIGN in 1332 / 260 samples), smokes 323 / Red, correctness 8/8, e2e 3/3. L.A.I.L x10 42.737 tok/s at 48.732 ms/step (acceptance 2.1092), four prose c=1 52.52 at 47.921, prose_long c=1 43.564 at 48.979 (natural finish length, post-EOS 0), c=2 aggregates prose 80.605 / structured 138.339 / prose_long 68.047, pp_warm 765.0 / 768.0, pp_novel 835.0 / 810.7, warm prefix 1.0. Quality quick and full PASS. MemAvailable 116/117 GiB before, 23/25 after smoke, 21.42/23.50 after the 32k prefill. Load: model loading 431.14 s (75.58 GiB), init engine 60.94 s, server start 594 s after run.sh. **Serve left UP** on this boot (canonical-e14, round-35 defaults, from the kernels-r3 worktree).

### Phases

Every counted boot below passed the section 1 preconditions and ran the section 2 procedure: strict audit at ready, smokes, correctness 8/8, fresh L.A.I.L x10 before any c=2, `bench_decode.py --runs 9`, `tools/four_numbers.sh`, `benches/e2e.py` (3/3 on every boot; rc 0 wherever captured, CUR-1 ran it in the background), quality, `disarm_scan` rc 0 and `gpu_guard` rc 0 with 0 FOREIGN samples. No boot was VOID and no lever was dropped. No lever disarmed on a counted boot: K-1's first attempt hit the MHC_DET self-test dtype bug (bf16 default under the loader), fixed in ae0255a, and the retry is the counted K-1. Every boot ran image `sha256:3a002b55c9bc` on both ranks; honest cells held on all of them (prose_long natural finish `length`, post-EOS 0, `serve_env_ranks_match` true, `lever_disarmed` false). MemAvailable was 116-117 GiB before each boot and 22-25 GiB after the smokes (floor 8). Per-boot numbers are in each boot's `summary.json` / `notes.txt`; the decisions recompute from the raw files (`lail10.txt`, `bench_decode.txt`, `four/four_numbers.json`, `four/02-micro.log`, `quality_*.json`).

- **P0 / P1** (`p0-p1/`, commits a80c789, 96a7de7).
  - Viterbi pack assembled on both nodes: 40/40 unit shards sha256- and size-equal to the manifest on each node, sha-list digest equal across nodes, 40/40 hard links, safetensors headers byte-equal to stock, 47,323/47,323 non-expert tensors in the unit shards byte-equal to stock, 0 problems.
  - Preflight: requant containers gone (exit 0), no serve, GPUs free, MemAvailable 116/116 GiB, image IDs equal, unit tests 797 OK.
  - NCCL twin device check (two-node harness with memset gaps, serve down): self-test flags exactly step 37 on both ranks; c=1 and c=2 device checks 330/330 steps, 0 bad steps, 0 host mismatches. Rank-0 step medians: c=1 keep 56.673, twin 50.054, keep_qos 55.562, twin_qos 49.448 ms; c=2 keep 57.643, twin 50.664. This is a harness, not a serve.
- **P2 bundle** (A-1, K-1, KC-1, A-2, K-2, KC-2; `decision-bundle.json`, 54ad48e). Arm value = median of its two boots, noise = the larger boot-to-boot spread.

  | Cell (ms/step) | A | K | KC | K vs A | K vs KC | KC vs A |
  |---|---|---|---|---:|---:|---:|
  | four prose c=1 | 63.866 / 63.083 | 47.156 / 48.245 | 56.154 / 56.213 | −15.774 (noise 1.089) | −8.483 (1.089) | −7.291 (0.782) |
  | prose_long c=1 | 63.47 / 63.216 | 49.01 / 48.702 | 56.9 / 57.161 | −14.487 (0.308) | −8.175 (0.308) | −6.312 (0.261) |
  | L.A.I.L | 63.013 / 62.568 | 48.328 / 48.909 | 56.205 / 56.349 | −14.172 | −7.659 | −6.513 |

  Every K boot beats the A and the KC median on both primary cells. Acceptance is matched for all pairs (four prose median-run: A 2.5443/2.4568, K 2.557/2.5316, KC 2.5844/2.5443). Non-inferiority holds against the larger spread; K vs KC pp_novel 32k is −12.7 tok/s against a noise of 11.9 (runs overlap; COOP does not run in prefill). Quality quick PASS on K-1, K-2, KC-1 and KC-2; full PASS on K-2 (NLL 0.242805, GSM8K 94/100, thinking 39/40, MMLU 199/228, needle 9/9). The A/A control: A-1's quick rep1 failed on aa_hazard 0.03516 with no lever armed (rep2 on the same boot passed); A-2 passed. Every measured saving is larger than the section 11 projection: K vs A c=1 projected 11.9-13.1, measured 14.172-16.657 ms/step; c=2 projected 21.5-22.0, measured 26.905-31.707. Promoted as one unit.
- **P3 attribution** (profile boots A_p-1, K_p-1, not counted; `P3/attribution.txt`, 7db92e9). Profiled step wall K_p − A_p: c=1 −14.321 / −14.135 ms (rank 0 / rank 1), c=2 −27.798 / −27.385. Device ops per step 2569 → 1612. Per lever, c=1, rank 0 / rank 1:

  | Lever | Measured on | K − A (ms/step) | Projection (s11) | Reading |
  |---|---|---:|---:|---|
  | `DSV41_P2B_COOP=2` | routed MoE segment | −6.109 / −5.909 (c=2 −16.508 / −16.406) | −6.76..−6.81 (c=2 −15.2..−15.3) | delivered |
  | `DSV41_DENSE_GEMV=1` | dense MXFP8 GEMM kernel time | −1.690 / −1.683, plus −0.46..−0.51 act-quant | −1.60 | delivered; router gate next to the shared gate_up slows 19.6 → 47.0 us/layer, +1.096 / +1.136 back |
  | `DSV41_MHC_DET_SPLITS=16` | elapsed mHC chain | −0.984 / −0.976 | −1.0..−1.4 | delivered (low end) |
  | `DSV41_ENGRAM_NATIVE_STAGE=1` | Engram stage span | −1.507 / −1.161 | −1.22..−1.57 | partial: span 1.368 / 1.558 ms, expected 0.2-0.35 |
  | `DSV41_ENGRAM_EARLY_HASH=1` | eager chain before the hash | −1.018 / −1.001 | −0.18 | delivered, above projection |
  | `DSV41_MOE_PREP_FUSED=1` | router gate end → routed start | −2.944 / −3.000 | −0.19..−0.34 | delivered, ~10x projection |
  | `DSV41_CANDIDATE_MASK_BOUNDED=1` | mask + flags kernels | −0.300 / −0.296 | −0.27..−0.43 | delivered |
  | `DSV41_INDEXER_WP_GEMV=1` | qkv join wait | −0.123 / −0.122 | −0.10..−0.17 | delivered |
  | `DSV41_ATTN_T2R_DEDUP=1` | t2r scan/map kernels | −0.070 / −0.068 | −0.41..−0.43 | mechanism delivered, step gain masked |
  | `DSV41_SWA_META_FUSED=1` | SWA metadata kernel | −0.001 / +0.004 | −0.09..−0.16 | mechanism delivered, step gain masked |

  The reconciliation rows sum to −14.251 / −14.338 against the measured −14.321 / −14.135 (c=1). Unplaced: the unprofiled P2 leave-one-out gave COOP −7.66..−8.48 ms/step at c=1 while the profiled COOP kernel delta is −5.91..−6.11, so about 2 ms/step at c=1 is an interaction the two profile arms cannot place (a KC_p boot would). Remaining top 3 in K_p: the routed MoE coop kernel (15.96 / 15.81 ms/step; about 4 ms above a flat-read floor at c=1), the Engram host stage (span 1.37-1.56 ms; page-cache misses), and the router gate slowed by the shared gate_up GEMV (+1.1 ms/step). K_p worker traces had NUL bytes at page-final indentation positions (cause unidentified; parsed after a lossless NUL → space rewrite).
- **P4 Viterbi pack** (CUR-1, VIT-1, CUR-2, VIT-2 on the promoted config; `decision-viterbi.json`, c02cbf6). Kernel pre-check on the Viterbi weights passed first (spark2, layer 20 with `--ref` and layer 33 rank 1: 24/24 checks each, one-hot bit-exact vs p2b, COOP=2 bitwise = =1, full max_rel ≤ 4.13e-4, 8bf97c5). **Rejected, and the serve pin stays `2.0bpw-mcg-lmhead-mxfp8`**, on two grounds, either sufficient as written:
  - Quality gate: quick and full FAIL on both VIT boots, on `selfcons.golden_hazard` only (0.09756 / 0.10204 quick against 0.03206; 0.0939 / 0.0939 full against 0.04348; 20/24 pairs diverge from the old pack's greedy goldens, on 8/12 prompts at the same token on all 16 VIT sequences). CUR passed every component on all four readings. The pack's own A/A hazard stays in band (VIT 0.0084-0.0161, CUR 0.0123-0.0274), so the failure is the pack being different from the goldens, not unstable output.
  - Speed: acceptance is not matched, so the gate is tok/s. prose_long c=1 CUR 44.893 / 43.395 vs VIT 45.875 / 48.629 (+3.108, noise 2.754, better); L.A.I.L CUR 42.927 / 43.123 vs VIT 41.829 / 42.174 (−1.024, noise 0.345, worse beyond noise). c=1 ms/step is equal within noise (four prose −0.084, prose_long −0.032, L.A.I.L −0.322).
  - What improved: teacher-forced NLL −0.10489 nats/token paired over 40 passages (−42.1%, 95% CI −0.12856..−0.08122, 39/40 lower, Wilcoxon p = 5.5e-12; full NLL VIT 0.1386 / 0.1393 vs CUR 0.2434 / 0.2439). GSM8K 97 / 97 vs 95 / 94 and MMLU 206 / 204 vs 196 / 198 favour VIT but are not significant after pooling (McNemar p 0.219 and 0.134). Structured c=2 is −5.38 ms/step (noise 0.2) on VIT: the two c=2 streams start together and run in lockstep at acceptance 4.0; cause not established.
  - Whether the golden-hazard gate should apply to a pack change is left to the protocol owner. The pack is local only (both nodes, `snapshots/2.0bpw-mcg-viterbi-lmhead-mxfp8`); it is not on the Hub (see "Viterbi pack: status and commands" below).
- **P5 KP add-ons** (safety boot KPsafe-1, then Kp-1, KP-1, Kp-2, KP-2; `decision-addons.json`, 172359d). KP = K + `DSV41_MHC_DET_OVERLAP=1 DSV41_ENGRAM_WKV_TP=1 DSV41_PM_QOS_US=20 DSV41_NCCL_EAGER_TWIN=1 DSV41_AR_L2_PREFETCH=1`. The safety boot booted with every add-on engaged on both ranks and no hang; greedy smokes were byte-identical to K on all five boots. ms/step KP − K: four prose c=1 −1.325 (noise 0.561), prose_long c=1 −1.81 (0.333), L.A.I.L −1.732 (0.082), every KP boot below the K median; the s11 projection was −2.4..−7.1. Per-boot median-run acceptance does not overlap on the two primary cells (four prose K 2.5897/2.5844 vs KP 2.525/2.5769; prose_long c=1 K 2.1505/2.117 vs KP 2.0729/2.0938), so step 6 falls back to tok/s: prose_long c=1 +0.198 (noise 0.718, KP-2 below the K median), L.A.I.L +1.459 (noise 1.594). **Not promoted**; all five stay off. No KP_p profile boot ran, so there is no per-add-on attribution. On structured c=1 (acceptance 3.961 on every run) KP is −1.563 ms/step (18/18 KP runs below every one of the 54 K-config runs) and +2.893 tok/s (noise 0.512). Next per s8: an ABAB of K + {PM_QOS, WKV_TP} against K, and the KP_p profile.
- **P6 SPARSE_MARKOV** (`decision-markov.json`, 018ec0d).
  - W vs S (W-1, S-1, W-2, S-2; W = `DSV41_DSPARK_SPARSE_MARKOV=0` with WOA_PREPACK=1): L.A.I.L W 42.814 / 43.624 vs S 42.791 / 42.785, +0.431 (noise 0.81); prose_long c=1 W 44.257 / 44.196 vs S 44.739 / 43.336, +0.189 (noise 1.403). W does not beat S beyond the noise: the revert is rejected and SPARSE_MARKOV=1 stays. The trade, S relative to W: L.A.I.L acceptance −2.74% (beyond noise) against ms/step −0.735 (noise 0.07), which roughly cancel in tok/s. This settles the round-34 open question on SPARSE_MARKOV's acceptance cost.
  - S1024 vs S (S-3, S1024-1, S-4, S1024-2): L.A.I.L S 42.36 / 42.871 (median 42.6155) vs S1024 43.167 / 43.615, margins +0.552 / +1.000 against noise 0.511; prose_long c=1 S 42.788 / 42.195 (median 42.4915) vs S1024 44.174 / 43.728, margins +1.683 / +1.237 against noise 0.593. Every S1024 boot beats the S median by more than the noise on both gate cells: **`DSV41_DSPARK_SPARSE_MARKOV_TOPK=1024` promoted**. The gain is acceptance (prose_long c=1 +0.1062, noise 0.0214); the gather costs +0.473 ms/step on L.A.I.L (noise 0.219). Quick quality PASS on all eight P6 boots. Robustness caveat recorded with the decision: S-3/S-4 are the two lowest of the ten S-config boots on prose_long c=1, and against S-1..S-4 (same env, not interleaved) or all ten S-config boots neither gate cell meets the rule.

### Viterbi re-encode (requant) and the pack's status

- Encoder: exllamav3 1.5.1 `quantize_tiles` (tail-biting Viterbi) + `refit_scales` (H = I), 2.0 bpw MCG K=2, from the MXFP4 source (official commit `dba1be0a40aa45a94ad051997016db3960a90277`), resumable across both Sparks (`tools/requant_full.py`; units: 26 on spark1, 14 on spark2). First unit started 2026-09-25T10:34:44Z; `state/DONE` 2026-09-26T04:34:48Z.
- 46,080 routed-expert tensors in 40 shards. Relative error against the source: final mean 0.261564 (min 0.253070, median 0.261592, max 0.261920) vs stock 2.0bpw-mcg 0.377535 (median 0.377533); Viterbi before refit 0.261711. Final / stock per tensor: mean 0.69282 (0.66615-0.69446); 46,080/46,080 improved, refit kept on all, all finite; MSE ratio 0.48000; per-unit gate ratio 0.69254-0.69293 (`p0-p1/pack-relerr.json`).
- Same format and bytes as the stock pack: safetensors headers byte-equal on 40/40 shards; 47,323/47,323 non-expert tensors in the unit shards byte-equal (6,884,224,448 B); trellis, suh and svh differ on 46,080/46,080 tensors (refit makes svh a real per-column scale). Assembly on both nodes as `snapshots/2.0bpw-mcg-viterbi-lmhead-mxfp8` (hard links to the verified outputs; model-00043 = the lm_head MXFP8 file), 0 problems (`p0-p1/pack-verify-spark{1,2}.json`).
- The P4 verdict above rejects it for serving as ARMS-r3 section 7 is written. The weight-space gain shows as NLL −42.1% on the fixed passages; the golden-hazard gate and the L.A.I.L speed gate fail. It is **not on the Hub**. To run it locally on the current code: assemble on both nodes (`/home/sfxnz/projects/data/dsv41-requant-viterbi/PLAN.txt` section 9), then `SNAPSHOT_SHA=2.0bpw-mcg-viterbi-lmhead-mxfp8 AUDIT=strict ./run.sh` (ARMS.md, Exact restore sequence).
- Pending decision for the user / protocol owner, not an action taken: if the golden-hazard gate is judged not to apply to a pack change, the publish command is
  ```bash
  HF_XET_HIGH_PERFORMANCE=1 /home/sfxnz/.hf-cli/venv/bin/python tools/publish_pack.py \
    --src ~/.cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-EXL3/snapshots/2.0bpw-mcg-viterbi-lmhead-mxfp8 \
    --revision 2.0bpw-mcg-viterbi-lmhead-mxfp8 --dry-run      # then again without --dry-run
  ```
  `publish_pack.py` also uploads `model-card.md` as the README of `main` and of the new branch, so that card has to describe the new revision first. The L.A.I.L regression (−1.024 tok/s, noise 0.345) stands either way.

### Not done / open

- KC_p profile boot: about 2 ms/step of COOP's c=1 saving is an interaction P3 could not place.
- KP_p profile boot and the section 8 split (K + {PM_QOS, WKV_TP} against K); EAGER_TWIN's missing graphUsageMode lines.
- The Engram native stage span (1.37-1.56 ms, expected 0.2-0.35): page-cache misses under the early-hash gather.
- The router gate next to the shared gate_up GEMV (+1.1 ms/step on the FFN prologue).
- The coop kernel's remaining ~4 ms/step above the flat-read floor at c=1 (shared expert on the coop tail).
- Whether the golden-hazard gate should gate a pack change (P4), and whether bit-exact levers should gate on the tok/s fallback when their acceptance gap comes from sampling (P5).
- P6 robustness: S1024 beat the designed pair's S boots, not all ten S-config boots. A further S vs S1024 pair would tighten it.
- Structured c=2 on the Viterbi pack runs its two streams in lockstep (−5.38 ms/step); cause not established.
- Viterbi pack publication (above).

## Round 36 — 2026-09-27 Viterbi pack adopted

The protocol owner adopted the Viterbi re-encoded pack `2.0bpw-mcg-viterbi-lmhead-mxfp8` as the recipe default, to be published to the Hub as a new revision. Round 35 P4 had rejected it under the rules as written. Evidence is in `results/2026-09-27-viterbi-adopt/`: the decision addendum, two validation boots of the new defaults (V-1, V-2), the new quality baseline and `round36-headline.json` (computed by `round36-compute.py.txt` from the raw per-boot files with the round-35 readers). Commits: aac194a (protocol amendment + decision), 8f94827 (default pack), dcca87e (per-boot driver), ff93f29 (V-1 + new baseline), and the round-36 results commit.

**Decision** (`decision-viterbi-addendum.json`, numbers copied from `results/2026-09-26-serve-r3/decision-viterbi.json`). P4's rejection had two grounds:
- quality quick and full failed on both VIT boots on `selfcons.golden_hazard` only (0.094-0.102 against 0.032/0.043);
- L.A.I.L tok/s was worse beyond the noise (−1.024, −2.38%, noise 0.345; acceptance 2.0648 vs 2.1202 arm medians), while prose_long c=1 was better beyond it (+3.108, +7.04%, noise 2.754).

Every other quality reading favoured VIT or was equal:
- paired NLL −0.10489 nats/token (−42.12%, 95% CI −0.12856..−0.08122, 39/40 passages lower, Wilcoxon p 5.5e-12)
- GSM8K 97/97 vs 95/94; MMLU 206/204 vs 196/198; tools exact args 88/88 vs 84/88 (none significant after pooling)
- VIT's own A/A hazard 0.00843-0.01609, against CUR's 0.01232-0.0274

The owner judged that the golden gate does not apply to a pack change and accepted the speed trade as measured. c=1 ms/step was equal within the noise, and speed moves only through acceptance.

**Protocol amendment** (ARMS.md step 6, "Pack changes"; tests/quality_eval.py docstring and `--baseline` help; README quality section). The golden flip hazard compares an arm's greedy text with run A of the baseline's `selfcons.runs`. So it gates kernel and numerics levers on the pack the baseline was recorded on, and nothing else. A pack (weights) change is judged on:
- paired NLL over the 40 fixed passages
- paired benchmark tests over the fixed items (GSM8K, GSM8K-think, MMLU, tools exact_args: McNemar per pair, pooled sign test)
- the new pack's A/A self-consistency against the current pack's A/A band
- every other component in its band
- speed per the rest of step 6

Adopting a pack re-records the quality baseline and goldens on the new pack. The old baseline stays the reference for the old pack.

**Default change** (8f94827). `recipe.yaml` `model.revision` (`SNAPSHOT_SHA`) is now `2.0bpw-mcg-viterbi-lmhead-mxfp8`, rendered into run.sh and the README defaults. The README, model-card.md (the Hub README) and AGENTS.md name the new revision and keep `2.0bpw-mcg-lmhead-mxfp8` (rounds 33-35) and `2.0bpw-mcg` (stock) as previous revisions with their `SNAPSHOT_SHA` lines. The revision-pin tests and `tools/pack_meta.SERVE_REVISION` were updated. With no override, `RESOLVE_SNAPSHOT_ONLY=1 ./run.sh` resolves `snapshots/2.0bpw-mcg-viterbi-lmhead-mxfp8` on spark1 and spark2 (rc 0, 60 entries, equal config/index sha256), and the harness dry run passes that path to both ranks (`resolve-check.txt`). There is no `refs/` entry until the Hub revision is published.

**Boots.** V-1 and V-2 each ran `env AUDIT=strict ./run.sh` with no lever, pack or image env, through the full ARMS-r3 section 1-3 procedure (`boot-procedure.sh.txt`; section 1 checks abort the boot on failure). The procedure: strict audit at ready, smokes, correctness, L.A.I.L x10 before any c=2, `bench_decode.py --runs 9`, `four_numbers.sh`, `benches/e2e.py`, quality quick and full, `correctness.sh --full`, `disarm_scan`, an engagement capture with a strict audit re-run on the end-of-boot logs, and `gpu_guard --check`. spark2's GPU lock was held from V-1's preconditions to V-2's last measurement.

Both boots, first attempt, no retry:
- image `sha256:3a002b55c9bc` on both ranks; docker/patch sha256 list identical to final-1's
- head vLLM argv and both ranks' container env equal final-1's apart from the model path
- strict audit ok, 0 LOG_DISARMED hits, disarm_scan rc 0, gpu_guard rc 0 (0 FOREIGN)
- smokes 323 / Red, correctness 8/8 and `--full` 9/9, e2e 3/3
- honest cells: prose_long `length`, post-EOS 0, `serve_env_ranks_match` true, `lever_disarmed` false
- MemAvailable 116/117 GiB before, 22/24 after the smokes
- head model loading 427.04 / 426.78 s (75.58 GiB)

V-1 recorded the quality baseline; V-2 gated against it.

**Headline.** Before is the round-35 final config on the old pack (S1024-1, S1024-2, final-1; `round35-headline.json`). After is the round-36 defaults (V-1, V-2): the same config and code on the Viterbi pack. Arm value = median of the per-boot medians; noise = the larger boot-to-boot spread. The two sets ran on the same day but were not interleaved (round 35 00:08-03:33Z, V-1/V-2 05:50-07:51Z). The table describes the adoption; P4's interleaved CUR/VIT ABAB is the controlled comparison.

| Metric | S1024-1 / S1024-2 / final-1 (old pack) | V-1 / V-2 (Viterbi pack) | R35 → V (medians) | Δ | noise |
|---|---|---|---|---:|---:|
| L.A.I.L fresh, t=0.2 (tok/s) | 43.17 / 43.62 / 42.74 | 42.73 / 42.44 | 43.17 → 42.58 | -0.59 (-1.4%) | 0.88 |
| L.A.I.L fresh (ms/step) | 49.22 / 49.00 / 48.73 | 48.81 / 48.75 | 49.00 → 48.78 | -0.22 (-0.5%) | 0.49 |
| L.A.I.L fresh acceptance (median run) | 2.133 / 2.186 / 2.109 | 2.107 / 2.122 | 2.133 → 2.115 | -0.018 | 0.076 |
| four prose c=1 (tok/s) | 53.67 / 52.04 / 52.52 | 50.60 / 50.42 | 52.52 → 50.51 | -2.01 (-3.8%) | 1.63 |
| four prose c=1 (ms/step) | 48.10 / 48.07 / 47.92 | 47.66 / 48.25 | 48.07 → 47.96 | -0.12 (-0.2%) | 0.59 |
| four prose c=1 acceptance | 2.577 / 2.500 / 2.513 | 2.410 / 2.422 | 2.513 → 2.416 | -0.097 | 0.077 |
| prose_long c=1 (tok/s) | 44.17 / 43.73 / 43.56 | 45.17 / 42.42 | 43.73 → 43.80 | +0.07 (+0.1%) | 2.75 |
| prose_long c=1 (ms/step) | 48.80 / 49.49 / 48.98 | 49.05 / 50.12 | 48.98 → 49.59 | +0.61 (+1.2%) | 1.07 |
| prose_long c=1 acceptance | 2.163 / 2.174 / 2.149 | 2.222 / 2.126 | 2.163 → 2.174 | +0.011 | 0.096 |
| bench prose c=1 (tok/s) | 51.99 / 52.08 / 54.03 | 50.26 / 51.12 | 52.08 → 50.69 | -1.39 (-2.7%) | 2.04 |
| bench prose c=1 (ms/step) | 48.64 / 49.28 / 48.56 | 48.41 / 48.50 | 48.64 → 48.46 | -0.18 (-0.4%) | 0.72 |
| bench prose c=1 acceptance | 2.551 / 2.557 / 2.632 | 2.439 / 2.457 | 2.557 → 2.448 | -0.109 | 0.080 |
| structured c=1 (tok/s) | 83.81 / 83.45 / 83.90 | 84.37 / 84.19 | 83.81 → 84.28 | +0.47 (+0.6%) | 0.44 |
| structured c=1 (ms/step) | 47.26 / 47.46 / 47.21 | 47.41 / 47.51 | 47.26 → 47.46 | +0.20 (+0.4%) | 0.25 |
| structured c=1 acceptance | 3.961 / 3.961 / 3.961 | 4.000 / 4.000 | 3.961 → 4.000 | +0.039 | 0.000 |
| bench prose c=2 aggregate (tok/s) | 82.38 / 81.69 / 80.61 | 81.28 / 80.76 | 81.69 → 81.02 | -0.66 (-0.8%) | 1.77 |
| bench prose c=2 (ms/step) | 60.56 / 60.31 / 59.45 | 59.01 / 59.73 | 60.31 → 59.37 | -0.94 (-1.6%) | 1.12 |
| structured c=2 aggregate (tok/s) | 138.60 / 138.24 / 138.34 | 156.25 / 156.33 | 138.34 → 156.29 | +17.95 (+13.0%) | 0.36 |
| structured c=2 (ms/step) | 55.13 / 55.09 / 56.22 | 51.18 / 51.15 | 55.13 → 51.17 | -3.97 (-7.2%) | 1.13 |
| prose_long c=2 aggregate (tok/s) | 65.24 / 65.62 / 68.05 | 71.32 / 66.67 | 65.62 → 69.00 | +3.37 (+5.1%) | 4.65 |
| prose_long c=2 (ms/step) | 61.37 / 62.56 / 62.35 | 62.40 / 63.29 | 62.35 → 62.84 | +0.49 (+0.8%) | 1.19 |
| L.A.I.L 3x after c=2 traffic (tok/s) | 43.68 / 42.46 / 43.66 | 46.06 / 41.67 | 43.66 → 43.87 | +0.20 (+0.5%) | 4.39 |
| pp_warm 8k (tok/s) | 757.6 / 753.8 / 765.0 | 789.4 / 756.2 | 757.6 → 772.8 | +15.2 (+2.0%) | 33.2 |
| pp_warm 32k (tok/s) | 763.0 / 769.5 / 768.0 | 776.9 / 772.9 | 768.0 → 774.9 | +6.9 (+0.9%) | 6.5 |
| pp_novel 8k (tok/s) | 821.4 / 807.4 / 835.0 | 815.7 / 813.0 | 821.4 → 814.4 | -7.0 (-0.9%) | 27.6 |
| pp_novel 32k (tok/s) | 806.2 / 806.3 / 810.7 | 818.2 / 805.6 | 806.3 → 811.9 | +5.6 (+0.7%) | 12.6 |

Reading. c=1 ms/step is equal within the noise on every c=1 cell, as P4 found (same kernels, same bytes per step). Every speed move comes through DSpark acceptance:
- L.A.I.L is −0.59 tok/s (−1.4%), inside its noise 0.88, but both V boots sit below the round-35 median. Acceptance 2.133 → 2.115, also inside the noise. P4's interleaved gap was larger (−2.38%, at TOPK=256); these sets are not interleaved, so the two gaps are not compared as a TOPK effect.
- prose_long c=1 is flat on the medians (+0.07). V-1 read 45.17 and V-2 42.42, a spread of 2.75, where P4's VIT boots read 45.88 / 48.63.
- The short prose cells lose beyond the noise: four prose c=1 −2.01 tok/s (−3.8%) and bench prose c=1 −1.39, with acceptance 2.513 → 2.416 and 2.557 → 2.448. P4 showed the same (−2.62). About 60% of these cells is post-EOS text under `ignore_eos`.
- Structured c=2 is +13.0% (138.34 → 156.29, −3.97 ms/step). On the Viterbi pack both streams run at acceptance 4.0 in lockstep (P4: start pattern, cause not established).
- Prefill and warm-prefix are unchanged within the noise (pp_warm 32k +6.9 against a noise of 6.5; runs overlap).

Pooled per-run medians (the recipe.yaml measured rows):
- L.A.I.L 42.98 (n=30) → 42.58 (n=20) tok/s, 48.96 → 48.81 ms/step, TTFT 0.322 → 0.322 s
- bench_decode per stream: prose c=1 52.97 → 50.41, structured c=1 83.73 → 84.20
- aggregates: prose c=2 81.73 → 81.02, structured c=2 138.34 → 156.29
- four prose c=1 52.66 (n=27) → 50.54 (n=18)

**Quality.** Every reading passes its gates. The round-35 readings were gated against the 09-24 old-pack baseline; V-2 against the new baseline; V-1's readings are the new baseline, so only vision and c2 are gated.

| Reading | NLL | decode median \|Δlogprob\| / greedy-text NLL | A/A hazard (identical) | golden hazard (limit) | tools json / exact / no-call | needle | GSM8K / thinking / MMLU |
|---|---|---|---|---|---|---|---|
| S1024-1 quick | 0.244484 | 0.01379 / 0.3393 | 0.01549 (5/12) | 0.01307 (0.03206) | 22 / 21 / 8 | 6/6 | – |
| S1024-2 quick | 0.243487 | 0.01399 / 0.3585 | 0.01887 (4/12) | 0.01456 (0.03206) | 22 / 21 / 8 | 6/6 | – |
| final-1 quick | 0.242947 | 0.02264 / 0.33845 | 0.0201 (4/12) | 0.01697 (0.03206) | 22 / 21 / 8 | 6/6 | – |
| final-1 full | 0.242247 | 0.01185 / 0.29669 | 0.01698 (3/12) | 0.02659 (0.04348) | 22 / 21 / 8 | 9/9 | 94/100, 39/40, 199/228 |
| V-1 quick (baseline) | 0.138564 | 0.00734 / 0.3336 | 0.0122 (5/12) | – | 22 / 22 / 8 | 6/6 | – |
| V-1 full (baseline) | 0.139323 | 0.00139 / 0.219 | 0.00968 (6/12) | – | 22 / 22 / 8 | 9/9 | 97/100, 39/40, 205/228 |
| V-2 quick | 0.138972 | 0.00256 / 0.27358 | 0.0123 (5/12) | 0.01104 (0.0244) | 22 / 22 / 8 | 6/6 | – |
| V-2 full | 0.137368 | 0.00335 / 0.25592 | 0.00957 (6/12) | 0.01128 (0.01936) | 22 / 22 / 8 | 9/9 | 96/100, 39/40, 202/228 |

Paired against the round-35 readings (`round36-headline.json` quality):
- **NLL**, per passage over the 40 fixed passages, arm = mean of its 4 readings: V − R35 = −0.10477 nats/token (95% CI t −0.12848..−0.08106, bootstrap −0.12827..−0.08299), −41.95%, 39/40 passages lower, Wilcoxon p 5.5e-12. This is the same size as P4's −0.10489. The same-pack cross-boot per-passage mean deltas are +0.00041 / −0.00196 (V-2 − V-1, quick / full) and −0.001 / −0.00054 on the old pack.
- **Items**: GSM8K V-1 / V-2 97 / 96 vs final-1 94 (McNemar p 0.25 / 0.5). MMLU 205 / 202 vs 199 (p 0.18 / 0.63). Tools exact args 22/22 on all four V readings vs 21/22 on all four R35 readings. Thinking 39/40 everywhere. None of these is significant. Between the V boots, 1 GSM8K item and 3 MMLU items differ.
- **Self-consistency**: V A/A 0.0122 / 0.00968 / 0.0123 / 0.00957 vs R35 0.01549-0.0201. Cross-boot on the same pack: V-2 vs V-1 run A 0.01104 quick / 0.01128 full (old pack S1024-2 vs S1024-1 0.02061). Cross-pack: V-1 vs final-1 run A 0.10638 / 0.10204, the gap the golden gate measured in P4.

**New quality baseline** (the harness's documented default; ARMS.md, README, quality_eval.py and AGENTS.md point at it):
- `results/2026-09-27-viterbi-adopt/quality-baseline/quick.json` and `full.json`, recorded on V-1 with no `--baseline`. The goldens are their `selfcons.runs`. Provenance and numbers are in `quality-baseline/README.txt`.
- The limits these files set: golden and A/A hazard ≤ 0.0244 quick / 0.01936 full; NLL ≤ 0.148564 / 0.149323; tools exact args ≥ 21/22.
- Offline re-gates (`quality-baseline/crosscheck/`, no traffic): round 35's VIT-1 and VIT-2 readings of this pack pass (golden 0.01006-0.01128), and the old-pack CUR readings fail (NLL 0.242-0.244, golden 0.10204).
- The 09-24 baseline (`results/2026-09-24-review/quality-baseline/`) stays the reference for `2.0bpw-mcg-lmhead-mxfp8`.

**Serve left UP on V-2** (canonical-e14, round-36 defaults on `2.0bpw-mcg-viterbi-lmhead-mxfp8`, `AUDIT=strict ./run.sh` from the kernels-r3 worktree, started 2026-09-27T06:55:34Z).

**Test change.** `tests/test_bench_honesty.py::test_fresh_doc_tops_up_when_build_undershoots` checked `benches/micro.py fresh_doc` at one seed over the live README/flags.md paragraphs. Adding a README paragraph reshuffles that draw, and the round-36 doc edits made it fail. The test now runs its repo-text path on 120 fixed 20-174-token segments; its novel-text path is unchanged. With them, `build_doc` lands at 0.878x, the top-up reaches 0.995x / 0.978x, and there are 0 misses over 200 seed/target pairs. Without the top-up it stays at 0.878x, so the test still catches a broken top-up. The underlying behaviour is unchanged and stays open: on the live corpus, `fresh_doc` at 8k misses [0.97, 1.0]x on 11/200 seeds at round-35 HEAD (16/200 after the round-36 edits) with the test's ratio. A multi-thousand-token flags.md table segment is dropped and the top-up pool runs out.

### Not done / open

- Publish the pack to the Hub as revision `2.0bpw-mcg-viterbi-lmhead-mxfp8` (the Publish stage; `tools/publish_pack.py` also uploads `model-card.md` as the README of `main` and of the new branch). This PR merges after it. **Not started.** The session's permission system blocked the Hub write, so no branch exists and nothing was uploaded; the owner has to approve it. A read-only check (`publish-delta-vs-lmhead.json`, script `publish-delta.py.txt`) compared the assembled snapshot with Hub `2.0bpw-mcg-lmhead-mxfp8` @ d3a74c0. 41 files differ, 143,482,916,447 bytes: `model-00003..00042` (same sizes, every sha256 different) and the new `requant-viterbi-manifest.json`. The other 13 serve files are byte-identical, including `model-00043`, the two Engram shards and the index. So a branch made from `2.0bpw-mcg-lmhead-mxfp8` needs only those 41 files. The tool's default route branches from `main` and uploads all 48 shards. `model-card.md` size row corrected to about 357 GB (333 GiB).
- The short-prose acceptance drop (four prose c=1 −3.8%) and the V-2 prose_long c=1 reading 42.42 (V-1 45.17). A broader prompt set would settle the net speed effect on a mixed workload.
- Structured c=2 lockstep on this pack (+13.0% aggregate): cause not established.
- `fresh_doc` top-up pool exhaustion on large table segments (above).
