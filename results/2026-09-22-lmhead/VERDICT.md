# LM-HEAD MXFP8 VERDICT — 2026-09-22 (Round 30, session ~07:00–09:00 BST)

## Outcome: **KEY ROUTING FIXED (two bugs, both integration-caught); lm_head
## MXFP8 BOOTED and SERVED on the stock pack — but L.A.I.L pooled n=10 median
## 32.17 missed the keep gate (32.31 = 31.37×1.03) → REVERTED. Stock 31.37
## serve restored. 35 tok/s NOT claimed (two-medians rule never satisfied).**

## Key-routing root cause (two stacked bugs, both in-image)

The serving model root is the **VL wrapper** `DeepseekV41ForCausalLM`
(`vllm/models/deepseek_v4_1/nvidia/vl_model.py`), not the text model. Its
WeightsMapper maps HF names fully into the wrapper namespace; the child
(`DeepseekV41LLMForCausalLM`, prefix `language_model`) loads with a NO-OP
mapper (vl_model.py:177). The wrapper's suffix rule
`"head.weight" -> "language_model.lm_head.weight"` is applied via
`key.rsplit(suffix, 1)` — **prefix preserved** — so the staged
`lm_head.weight` key became `lm_language_model.lm_head.weight` (garbage),
and `lm_head.weight_scale` matched nothing (`\.scale$` regex wants a
literal dot; no suffix rule matches `_scale`) and hit the wrapper root
bare → the observed `ValueError: There is no module or parameter named
'lm_head' in DeepseekV41ForCausalLM`.

**Bug 2 (caught by boot 1, not by any offline test):** `ModelConfig.head_dtype`
is a PROPERTY in this build (config/model.py:1970) that returns the model
dtype (bfloat16) for generation models — never None. The round-29 disarm
`head_dtype is not None` always tripped → swap never ran → the routed
scale key then failed against the unswapped ParallelLMHead (`no module or
parameter named 'lm_head.weight_scale'`). The real dispatch
(logits_processor.py:143) uses `quant_method.apply()` whenever
`head_dtype == hidden_states.dtype` — which holds. Fix: disarm only when
`head_dtype != model dtype` (genuine float32 RL-parity override).

### Fixes (commits 791ef2b, 1a9def4)

`install()` now (a) patches BOTH mapper makers with regex rules
`^lm_head\.weight$ -> head.weight` (stock suffix rule then renames
canonically) and `^lm_head\.weight_scale$ -> language_model.lm_head
.weight_scale` — regexes run BEFORE the suffix pass so results cannot be
re-corrupted; inert for stock packs (no `lm_head.*` keys exist there);
and (b) disarms only on a genuine head-dtype override.

### Real-image integration test (the gap that bit round 29)

`results/2026-09-22-lmhead/integration_test.py`, run CPU-only inside
canonical-e12 via `docker run --rm --entrypoint python3` (repo + hub pack
dirs mounted ro): maps **every key of both real pack indexes** through the
real mapper and runs a real `AutoWeightsLoader` over a wrapper-shaped
tree. **15/15 PASS**: lm pack `lm_head.weight`/`.weight_scale` →
`language_model.lm_head.*`, stock pack mapping byte-identical, no
wrapper-root keys, scale tensor copied into the param. The test's first
run caught the `lm_language_model.lm_head.weight` corruption that pure
code-reading had missed (I initially concluded only the scale key needed
routing — wrong; the integration test is what proved it).

Offline: `tests/test_lmhead_mxfp8` 6/6 OK (now models the real
`vllm.models.*` layout, real mapper rule order, and the head_dtype
property); `tests/numerics_lmhead_mxfp8.py` PASS (top1 flip 3.1%
weight-only / 4.5% with act quant; greedy 95.5% agreement;
log in this dir).

## Boot + measure (boot 2, canonical-e12, pack 2.0bpw-mcg-lmhead-mxfp8)

- Boot healthy ~9 min. BOTH ranks: `dsv41: lm_head mxfp8 enabled (b12x,
  (64640, 5120))` (TP2 vocab shard; full head 129280×5120). No ValueError.
- Smoke: "17 * 19" step-by-step → **323** PASS.
- Correctness quick: **7/7 PASS** (math_small 323, math_mid 252, json,
  tool_call, code_trace, recall_8k, prose_sanity).
- Live argmax-flip check (greedy temp-0, 12 fixed prompts × 200 tok,
  stock vs lm): 3/12 completions token-identical; first-divergence word
  positions [0,1,3,4,9,11,15,16,20,24,29,67] over ~120-word completions →
  per-step flip ≈ 1–2% (upper bound 7.7% from the median first-flip;
  consistent with the offline single-forward 4.5% bound; greedy chains
  compound). No collapse, all coherent.
- Boot 1 (head_dtype disarm bug) = boot cap 4 → 3 used total.

## L.A.I.L (official perf API, runner=decode workload=prose, c=1)

Warmup discarded: 36.02 (job a64d907778f5).

| Set | n | decode medians (tok/s) | median |
|---|---|---|---|
| real1 | 5 | 31.72, 31.80, 32.45, 32.78, 35.58 | **32.45** |
| real2 | 5 | 34.32, 30.75, 33.69, 31.79, 31.89 | **31.89** |
| pooled | 10 | sorted: 30.75…35.58 | **32.17** |

CLI prose twin (tools/measure_lail_prose.py, n=9): median 32.35 tok/s,
recipe 32.28, acceptance_len 2.15, no collapse (cli_prose9.log).

- vs baseline 31.37: **+2.5%** — below the +3% keep gate (32.31) → REVERT.
- 35 claim: NOT made — requires two independent medians ≥35; best single
  median 32.45.
- Variance note: run-to-run spread 30.75–35.58 is wide (acceptance-length
  lottery at temp 0.2 dominates); the point estimate is a real but
  sub-gate gain, consistent with the −1.3–2.7 ms/step kernel expectation
  only partially surviving the acceptance-rate noise.

## Decision: REVERT; stock restored

`results/2026-09-21-gatherv2/boot-k3c-pf-gv2.sh` re-run
(boot-restore.log): serve UP on 2.0bpw-mcg, baseline config. Health
green; smoke 323 verified post-restore. lm pack + patch remain staged and
committed — the lever is now boot-clean and one flag away
(DSV41_LMHEAD_MXFP8=1 + SNAPSHOT_SHA=2.0bpw-mcg-lmhead-mxfp8) if a future
round wants to re-measure with more n.

## OOM log (MemAvail GiB)

| When | spark1 | spark2 |
|---|---|---|
| 08:20 pre-stop (stock up) | 19 | 24 |
| 08:20 post-stop | 110 | 117 |
| 08:22 pre-boot-lm (boot 1) | 116 | 117 |
| 08:32 post-smoke (lm serve) | 22 | 24 |
| 08:43 lm-serve (pre restore-stop) | 22 | 24 |
| 08:44 post-stop | 116 | 117 |

Never below 12 pre-boot / 8 post-smoke. No kernel OOM, no worker kills
(boot 1 died in-process at weight-load validation, not OOM).

## Jobs / artifacts

- L.A.I.L job ids: warmup a64d907778f5; real1 ae43c58cf1bf 6fc05a1f8c3a
  c8b0d5ebc0ae 5877b35c8837 0cec04637b04; real2 29240211cec7 d7d41b075ebd
  5408787df65d 9a36051753da 552b2f97b0be.
- boot1.log (head_dtype disarm + routed-scale ValueError), boot2.log
  (healthy), greedy_stock.json / greedy_lm.json, lail_real{1,2}.{json,log},
  cli_prose9.log, numerics.log, integration_test.py — all in this dir.

## Commits (branch cursor/mul1-p2b-prefill-95c3)

- 791ef2b fix(lmhead): checkpoint-key routing + real-image integration test
- 1a9def4 fix(lmhead): head_dtype is a property (always set) — disarm only
  on real override
- (this verdict + flags Round 30 follow)
