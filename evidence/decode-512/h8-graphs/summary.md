# H8 CUDA graphs after Engram prestage

Frozen harness: `python3 tools/measure_lail_prose.py` (L.A.I.L prose c=1, 512 tokens, T=0.2).

| cell | median tok/s | accept_len | evidence |
|---|---:|---:|---|
| eager native p2b | 14.18 | 2.23 | `../h2-p2b/bench.txt` |
| FULL_AND_PIECEWISE | 15.29 | 2.34 | `bench.txt` |
| FULL_DECODE_ONLY | 15.32 | 2.34 | `bench-full.txt` |

Smoke `323` on both graph modes.

Capture (FULL_AND_PIECEWISE): 5 piecewise + 2 FULL + 2 DSpark FULL in 12 s, 0.18 GiB.
Capture (FULL_DECODE_ONLY): 2 FULL + 2 DSpark FULL in 11 s, 0.15 GiB.

Step time stayed ~152-153 ms vs ~160 ms eager. NVMe util during decode was 4-10%. Disk Engram is not the 150 ms.

Default after this cell: graphs on, disk rows staged in `prepare_inputs`.
