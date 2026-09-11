# H1: SPEC=dspark, NUM_SPECULATIVE_TOKENS=5, still eager

Flag-only. Same image `dsv41-flash-exl3-sm121`. Engram on disk. Adaptive verification left off.

| wave | median decode tok/s | TTFT p50 s | accept length | vs baseline 12.75 |
|---|---:|---:|---:|
| 1 | 23.03 | 0.367 | 3.19 | 1.81× |
| 2 | 21.23 | 0.365 | 2.98 | 1.67× |

Published cell is wave 2 (second independent 3-run median). Smoke `323` both waves. Both ranks `OOMKilled=false`. Remaining UMA: spark1 ~22 GiB available, spark2 ~24 GiB.

Draft sample method is `probabilistic` as already wired in `run.sh` for `SPEC=dspark`.
