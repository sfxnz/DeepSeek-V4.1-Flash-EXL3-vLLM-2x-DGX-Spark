# Quality baseline: live serve, 2026-09-24

These are the first runs of `tests/quality_eval.py` against the serve that was live that day. The traffic was HTTP only; nothing was restarted, and nothing else ran against the serve at the same time.

Provenance, from both JSONs:

- pack `snapshots/2.0bpw-mcg-lmhead-mxfp8` (lm_head MXFP8 is on)
- image `dsv41-flash-exl3-sm121:canonical-e12`
- `flags_sha256` `443d9237a748fc7f` (container DSV41_/VLLM_/NCCL_ env plus cmd)
- DSpark k=3, `MAX_NUM_SEQS=2`

This is therefore the **lm_head-MXFP8 config**, not a stock-head reference.

| File | Mode | Wall time | Exit |
|------|------|-----------|------|
| `quick.json` / `quick.log` | `--quick` | 337 s (5.6 min) | 0 |
| `full.json` / `full.log` | `--full` | 1918 s (32 min) | 0 |

## Numbers

| Component | quick | full |
|-----------|-------|------|
| NLL, nats/token (19,828 tokens scored, BOS + 16-token skip) | 0.24316 | 0.24310 / 0.24318 (2 passes, repeat Δ 7.6e-5) |
| NLL, largest per-passage repeat Δ | – | 0.026 |
| Decode probe, median / p99 \|Δlogprob\| | 0.018 / 0.82 | 0.022 / 0.75 |
| Decode probe, prefill NLL of greedy text | 0.359 | 0.370 |
| Tools: JSON-valid / exact-args / no-call | 22/22, 21/22, 8/8 | 22/22, 21/22, 8/8 |
| Needle found (8k, 32k; full adds 128k) × depths 0.1/0.5/0.9 | 6/6 | 9/9 |
| Self-consistency: identical pairs / A/A hazard | 4/12, 0.016 | 4/12, 0.022 |
| c=2, vision | pass, pass ("Red") | pass, pass |
| GSM8K-100, thinking off | – | 94/100 (Wilson 95% 0.875-0.972) |
| GSM8K-40, thinking on (effort high) | – | 38/40 (0.835-0.986) |
| MMLU 4×57 | – | 197/228 = 0.864 (0.814-0.903) |

The one tool-call miss in both runs is `t09`: `calculate_loan_payment` is called with `{}` as its arguments. In the thinking-on arm, item 494 hit the 4096-token cap without an answer.

## What these runs show

- **NLL noise floor.** In `full`, the two passes differ by 7.6e-5 nats. A two-pass development run earlier the same day gave 0.24497 / 0.24245, a repeat Δ of 0.0025; that was its first traffic on novel text. Across all five passes the mean NLL spans 0.2424-0.2450. 3× the worst repeat Δ is 0.0076, which is below 0.01, so the 0.01-nat floor is the operative ΔNLL gate. `prompt_logprobs` requests read nothing from the prefix cache (`prefix_cache_hits_delta` 0).
- **The greedy A/A control diverges.** The same config, run twice, leaves only 4 of 12 greedy completions token-identical within 128 tokens. The per-token hazard is 0.016-0.022, and the first divergence can come as early as token 0-3. Cross-gating `quick.json` against `full.json` (`--result`) passes every gate. Its golden hazard is 0.023, about the same as the A/A hazard. R30 reported 9/12 stock-vs-lm completions diverging, with no control. That result is inside this A/A floor, so on its own it says nothing about lm_head MXFP8. The stock-head A/B is in the GPU plan: `DSV41_LMHEAD_MXFP8=0`, then `--result` against these files.
- **Decode and prefill disagree on the same tokens.** Re-scoring greedy-decoded tokens through prefill gives a median |Δlogprob| of about 0.02 but a p99 of about 0.8. The gates use the baseline median and greedy-text NLL, with slack. The greedy-text NLL moved 0.30 → 0.36 → 0.37 across three same-serve runs, which is why its slack is 0.15.
- **Needle timings are not a perf metric.** The filler's word bank is small, so Engram and page-cache state warm up after the first run: the 8k cells took 22 s on the first run and 10 s later.

## Gate usage

```bash
python3 tests/quality_eval.py --quick --baseline results/2026-09-24-review/quality-baseline/quick.json --out <arm>/quality_quick.json
python3 tests/quality_eval.py --full  --baseline results/2026-09-24-review/quality-baseline/full.json  --out <arm>/quality_full.json
```
