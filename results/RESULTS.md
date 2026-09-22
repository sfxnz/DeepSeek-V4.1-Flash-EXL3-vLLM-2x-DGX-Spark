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

### GEMV harness (`kernel_study/gemv_bench/`, untracked study tree)

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
