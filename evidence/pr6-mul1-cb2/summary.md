# PR 6 MUL1 + p2b cb=2 A/B — serve stays MCG

Measured 2026-09-15 on spark1+spark2 at TP=2.
Pack: real `2.0bpw-mul1` (K=2) on `dsv41-flash-exl3-sm121:cb2`.
Score cell is prose decode only.
Source: store `docs/deepseek-v41-mul1-ab.md`.

| Cell | MCG baseline | MUL1 + cb=2 |
|---|---:|---:|
| Prose decode (c=1 median) | 27.98 tok/s | 23.52 tok/s |
| 12,712-token prefill | 797 tok/s | 792 tok/s |

MUL1 prose runs: 26.43 / 23.52 / 23.17.
Baseline: 26.34 / 27.98 / 31.77.
MUL1 median and two of three runs sit below the worst baseline run.

Call: do not switch the published serve pin to MUL1. Prefill is flat and does not change the call. Rebuild tools may stay. p2b `cb=2` may stay in the image so a later pack can use it. `SNAPSHOT_SHA` stays `2.0bpw-mcg`.
