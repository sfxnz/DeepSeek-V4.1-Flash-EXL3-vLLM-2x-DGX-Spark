# VERDICT — mul1-lane Boot A (stock 2.9bpw image, k=3) vs MCG lane

Date: 2026-09-20 · Run dir: `../` · Artifacts: this dir · Raw boot log: `../results-bootA-boot3.log`

## Four numbers (median, port 8888, benches run directly with `--url`)

| Metric | mul1 stock (2.9bpw) | MCG lane (K4 pack) | Δ |
|---|---|---|---|
| Prose decode c=1 tok/s (9 runs) | **16.35** | 34.69 | −53% |
| Cold prefill 8k tok/s | **593.7** | 690 | −14% |
| Cold prefill 32k tok/s | **711.7** | 733 | −2.9% |
| LAIL prose tok/s | **25.63** | 26.39 | −2.9% |
| MemAvail after (s1/s2 GiB) | 5.3 / 6.7 | 22.3 / 23.8 | much lower |

This is the **PACK-layer comparison** (mul1 EXL3 codebook vs MCG K4 grouped kernels; same topology, same NCCL set). The mul1 pack halves decode throughput at c=1; prefill and LAIL prose are near parity. Spec-acceptance is visibly lower (prose acceptance_len 1.32 vs typical MCG ~2+; draft acceptance 0.108).

## Boot experience

- Attempt 1 failed: `--limit-mm-per-prompt {image:100}` — her `.env` value loses its double quotes when `source`d; this vLLM build requires strict JSON. Fixed: `LIMIT_MM='{"image":100}'` (single-quoted) in `.env`.
- Attempt 2 failed: worker env passes unset `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` as empty string → `int('')` crash. Fixed: set `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256` in `.env`.
- Attempt 3: clean boot, ~14 min, 39/39 shards both nodes, NCCL over f1/GID1, zero NVRM lines.

## Memory floors (anomaly — below the 8 GiB card)

Post-smoke idle MemAvailable: spark1 ≈ 5.5–5.6 GiB, spark2 ≈ 7.1 GiB, drifting ~5.0/6.7 after 32k prefill. Investigation: torch_reserved 103.4 GiB per GPU (weights + 2.5 GiB KV + CUDA graphs) is structural for this 2.9bpw pack; usage stable across 15 min of benching, container RSS ~5.7/4.3 GiB, no leak, no NVRM, cuda_free 6.7–7.4 GiB for activations. The MCG lane's higher floors (22.3/23.8) come from its mem-hygiene envs reclaiming page cache; this boot does carry the same hygiene envs (verified in docker inspect) but the pack's larger resident footprint dominates. Recorded as a real PACK-layer difference; boot did not OOM.

## Quality gate (first verbatim evidence for 2.9bpw — no quality evals exist)

- 17×19: `17 × 19 = 323` ✓ (finish stop)
- JSON sanity: `{"a":5,"b":"hello"}` ✓ (exact, only key order formatted)
- Code sanity: `def is_palindrome(s): return (t := ''.join(c.lower() for c in s if c.isalnum())) == t[::-1]` — correct logic ✓
- Vision (8×8 PNG, "how many pixels tall?"): answered `1024` — vision path executes end-to-end but the answer is wrong (expected 8). Verbatim, unjudged.

## Files

`00-models.json` · `01-prose-decode.log` · `02-micro.log` · `04-lail.log` · `serve_cmd.json` (docker inspect Cmd+Env of head) · `summary.json` · this `VERDICT.md`

## Restore (post-run)

MCG serve (dsv41-flash-exl3-sm121:canonical-e12, 2.0bpw-mcg pack) restored via saved boot-arm.sh: healthy on :8000, smoke 323 ✓, all armed envs + `--max-num-batched-tokens 8192` verified in docker inspect, floors 25.6/27.3 GiB. L.A.I.L perf job `ad2c0af86b03` (decode/prose/c=1) on the restored serve: decode median 26.12 tok/s, agg 25.24, ttft 0.338s — consistent with the MCG lane's lail 26.39. (First job `089a793293dd` fired during warmup tail: 22.39 — superseded.)
