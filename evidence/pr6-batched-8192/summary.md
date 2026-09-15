# PR 6 batched-tokens A/B — default stays 2048

Measured 2026-09-15 on spark1+spark2 at TP=2 by the Nsight baseline agent.
Still `sfxnz/DeepSeek-V4.1-Flash-EXL3` revision `2.0bpw-mcg`. MUL1 pack was not rebuilt.
Source: store `docs/deepseek-v41-pr6-spark-results.md`.

| Serve | Batched tokens | Decode prose c=1 | Prefill 12,712 | Prefill 3,182 |
|---|---:|---:|---:|---:|
| `main` `dcac67a` | 2048 | 27.98 tok/s | 797 tok/s | 719 tok/s |
| PR 6 `e507021` | 8192 + `--mm-encoder-tp-mode data` | 33.05 tok/s | 755 tok/s (−5%) | 797 tok/s (+11%, one batch) |

Decode 33.05 vs 27.98 overlaps run spread. Not a clean win.
Long prefill got worse. Do not ship 8192 as a proven prefill upgrade.

Also recorded on that serve: context 1,048,576; KV 1,273,440 tokens (`main`) vs 1,144,584 (PR 6); 1M concurrency 1.21x vs 1.09x.

Nsight on the current recipe (both Sparks): `p2b_moe_batched` ~22–23%; prefill `exl3_moe_kernel` ~15%; NCCL AllReduce ~11–15% with a 304 ms spark1 tail; b12x MXFP8 GEMM ~13%; cutlass WMMA ~10%; sparse MLA ~2%; Engram ~0%. HW counters blocked (`ERR_NVGPUCTRPERM`).

Call: revert the recipe default to 2048. Keep MUL1 rebuild tools, p2b `cb=2`, vision smoke, and `--mm-encoder-tp-mode data`. Next measured work is MUL1 + p2b `cb=2` A/B.
