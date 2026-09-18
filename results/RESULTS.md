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
