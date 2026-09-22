# SHARD 3 VERIFICATION — VERDICT: PASS (2026-09-21 22:34 BST)

First spark1 G8 shard `model-00003-of-00048.safetensors` (written 22:25:20,
~18 min incl. container startup; steady-state ~16.4 min/shard).

## Evidence

1. **Fold applied — layout level**: all 1152 trellis tensors have the folded
   shape w1/w3 `(18, 320, 256)` and w2 `(40, 144, 256)` — exactly
   `stock.view(kt, nt/8, 8*w).permute(1,0,2)` of `(320,144,32)`/`(144,320,32)`
   (checked via raw safetensors headers; `tools/permute_pack_group_major.py`
   `perm_axis`: nt=144→18 groups, kt=320, GW=256).
2. **Fold applied — data level (decisive)**: `verify_shard3_reconstruct.py`
   (in image, GPU): `ext.reconstruct_had_slice(unfold(G8 trellis), G8 suh/svh)`
   reproduces the source MXFP4 weights at relerr **0.3775-0.3779**, vs stock
   pack trellis **0.3771-0.3775** on the same experts (w1/w2/w3 of
   layers.0.ffn.experts.0). Worst ratio 1.0011 (normal 2-bit band, parity with
   stock). A wrong fold mapping explodes this to ~52.8 (measured control with
   mismatched basis) — it does not. Log: `shard3-reconstruct.log`.
3. **Shard sanity**: `tools/check_exl3_shard.py` OK (no leftover routed MXFP4,
   suffix counts trellis=suh=svh=mcg=1152, no mixed markers); tensor key set
   identical to stock shard (4638); `.mcg` marker bytes identical; file size
   3,585,021,280 B — byte-identical to stock shard 3 size.

## Why not bit-exact vs stock

The 2026-09-10 stock mcg pack is not bit-reproducible: greedy beam-16 encode
is deterministic in trellis words but the scale search (GPU SVD/Hessian path)
is not — a same-args re-encode produces suh with sign-flipped entries
(magnitudes identical, ~50% of entries) vs BOTH packs. Bit-exact-vs-stock is
therefore not a valid gate for this rebuild; the unfold+reconstruct check
above is the correctness gate, and it passes.

## Note for the assemble step

The G8 pack's suh/svh differ from stock run-to-run only by SVD sign; both are
valid bases paired with their own trellis — never mix scales across packs.

## Second failed-launch bug (postmortem, fixed before any GPU time was lost)

`bash -c "... $(seq ...)"` — seq's newline-separated output embedded raw
newlines into the `bash -c` string, so the first (22:01) launch would have
processed only shard 3 (remaining names executed as commands). Fixed with
`seq | tr '\n' ' '`; relaunched 22:06. Earlier 22:04 launch also superseded
(log redirection fix). The two dead attempts converted nothing GPU-visible
beyond partial shard-3 work (overwritten atomically by the final run).
