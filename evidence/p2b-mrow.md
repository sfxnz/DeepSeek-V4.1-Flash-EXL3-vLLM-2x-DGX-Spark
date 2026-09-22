# p2b true m-row

`docker/patch/widen_p2b_mrow.py` scales each fused-MoE phase's work list by
`m`. Same `grid.sync` count as m=1. Scratch is `{e, m, dim}`. GEMV tiles stay
one row; `A2`/`C` already point at that row. Not a serial
`for (row) { moe; grid.sync(); }` — that path measured 13.24 tok/s and was
reverted.

Apply after `widen_p2b_shapes.py`. Wired in the Dockerfile.

`widen_p2b_mma.py` (shared-B MMA over rows that hit the same expert) measured
10.2 tok/s vs 15.1. Not wired. Register pressure and grouping overhead.
