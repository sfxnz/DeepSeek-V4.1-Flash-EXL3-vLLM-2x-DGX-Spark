# Baseline: current main defaults

Captured 2026-09-11T21:03:51Z against the live 12h `dsv41-flash-exl3` serve
(flags match `recipe.yaml`: `--enforce-eager`, `SPEC=none`, `QUANTIZATION=exl3`,
`DSV41_ENGRAM_DISK=1`, `--max-num-seqs 2`, fp8 KV 4 GiB).

Protocol: `python3 bench_decode.py --phase prose --concurrency 1 --runs 3 --max-tokens 200`
with thinking off, greedy, `ignore_eos` so 200 completion tokens actually emit.

| run | decode tok/s | TTFT s |
|---|---:|---:|
| 1 | 11.25 | 0.238 |
| 2 | 12.75 | 0.234 |
| 3 | 13.05 | 0.238 |
| **median** | **12.75** | **0.238** |

1.5× bar: **19.12 tok/s**.

Smoke: `323`. Both ranks `OOMKilled=false`. UMA remaining: spark1 available 24 GiB, spark2 26 GiB.

Engine: `fused_moe=exl3_moe` hidden=5120 intermediate_local=1152, CUDA graphs off,
FlashInfer AR disabled at TP=2 (PYNCCL), speculative_config=None.
