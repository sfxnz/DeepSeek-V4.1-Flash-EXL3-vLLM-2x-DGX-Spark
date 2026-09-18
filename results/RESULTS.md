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
