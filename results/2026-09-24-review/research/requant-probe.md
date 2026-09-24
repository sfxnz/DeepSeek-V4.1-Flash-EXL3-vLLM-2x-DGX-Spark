# requant: Viterbi re-encode vs the pack's greedy beam-16 (step 1 prep, 2026-09-24)

**Retitled per corrections:** "Re-encode routed experts with tail-biting Viterbi instead of greedy beam-16, with exllamav3 ≥ 1.5.1 and refit_scales. Calibrated Hessians come later."

A version bump alone does nothing. The pack path is `_quantize_fast`: meta-H q_fallback plus `skip_g_scale=True`, then `_quantize_greedy_tiles`. So d1feab126 (LDLQ drift in the g-scale search) is not on the path, and v1.5.1 gates refit behind `not q_fallback`. The lever is the encoder itself: greedy beam-16 has about 2.1x the tile MSE of Viterbi (commit 353eb28). The served experts measure relerr 0.3771-0.3775 against MXFP4 source (results/2026-09-21-pfg8/shard3-verify.md). If the 2.1x ratio holds, Viterbi should land near 0.377/√2.1 ≈ **0.26**.

## Where things are (read-only checks on spark1)

| what | path |
|---|---|
| MXFP4 source (routed experts + `.scale`) | `~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277/model-000NN-of-00048.safetensors` (shard 3: 7.39 GB) |
| served experts | `~/.cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-EXL3/snapshots/2.0bpw-mcg/` (root-owned). The live revision `2.0bpw-mcg-lmhead-mxfp8` symlinks `model-00003` to it (same inode 34081839); only `model-00043` differs. |
| routed-expert tensors | 46,080 (w1/w2/w3), 1,152 per shard over 40 shards (00003-00042) |
| quant flow | not `dsv41-exl3-nfs:local`, which is an Alpine nfs-utils server (63.6 MB). The flow is the serving image `dsv41-flash-exl3-sm121:canonical-e12` (exllamav3 1.4.9 @ 5be8865) with `--entrypoint bash`, running `tools/quantize_experts_exl3.py` (results/2026-09-21-pfg8/launch-rebuild-spark1.sh). |
| how the pack was encoded | `quantize_experts_exl3.py --codebook mcg --allow-partial --batch 8 --greedy --beam 16 --src … --dst … --only-files <shards>` → `_quantize_fast` → `_meta_h` (q_fallback) → `regularize(skip_g_scale=True)` → `_quantize_greedy_tiles(beam=16)` → `pack_trellis` |

## Probe

`tools/requant_probe.py` re-encodes N experts (default 2, i.e. 6 tensors; layers.0 experts 0-1 of shard 3). It uses the recipe's own `_quantize_fast` at 2.0bpw MCG K=2 and scores every arm with `ext.reconstruct_had_slice` against the same dequantized MXFP4 source, which is the verify_shard3_reconstruct.py method. Arms:

| arm | what |
|---|---|
| stock | served trellis + suh/svh |
| stock+refit | stock trellis, suh/svh refit (H = I); pack format and kernel unchanged, zero cost at inference |
| greedy16 | fresh greedy beam-16 encode; control that reproduces the pack's encoder |
| viterbi | `quantize_tiles` (the image's Viterbi kernel); no `--greedy` |
| viterbi+refit | viterbi plus refit |

Refit is v1.5.1 `refit_scales` (7e2e6b065) with H = I. In that case `(QQᵀ)∘I` is diagonal, so both alternating steps are per-row/column closed forms, which the unit tests check. On a v1.5.1 image the probe also runs upstream `refit_scales(W, Q, I, su, sv)` and records `<arm>_upstream` as a cross-check. `docker/Dockerfile.quant151` builds that image. It swaps exllamav3 to v1.5.1 (958ec93) on top of canonical-e12 and never serves. `patch_exllamav3_aarch64.py` dry-applies cleanly to v1.5.1 (checked on a clone). The trellis format and `reconstruct_had_slice` signature are unchanged; K became a float arg, which is still pybind-compatible.

## Wall time (history → full requant)

| encoder | s/tensor on GB10 | 1 GB10, 40 shards | 2 Sparks |
|---|---:|---:|---:|
| greedy beam-16 (measured 16.4 min / 1,152-tensor shard, REBUILD-PLAN.md) | 0.85 | ~10.9 h | ~5.5 h |
| Viterbi @ exllamav3 1.4.9 (7a083da / 353eb28: ~4.1 s) | 4.1 | **~52 h** | **~26 h** |
| Viterbi @ 1.5.1 (quantize_tiles rewrite, 03aed709c) | measured by the probe (`sec_per_tensor`) | probe prints | probe prints |

The probe prints `full_pack_h_1node` and `full_pack_h_2nodes` from the measured encode time. Only encode is timed; the ~1.1-1.2x IO/dequant overhead seen in the greedy history is not included.

## Run (serve down; exclusive GPU)

```bash
R=~/projects/ai-lab/recipes/.worktrees/rv-research    # or the merged checkout
# 1.4.9 (image as served)
docker run --rm --gpus all --ipc host --shm-size 16g --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages \
  -v $HOME/.cache/huggingface:/cache/huggingface:ro -v $R:/repo \
  --entrypoint python3 dsv41-flash-exl3-sm121:canonical-e12 \
  -S /repo/tools/requant_probe.py --experts 2 --json /repo/results/2026-09-24-review/research/requant-probe-149.json
# 1.5.1
docker build -f docker/Dockerfile.quant151 -t dsv41-quant151 docker/
docker run <same flags> dsv41-quant151 -S /repo/tools/requant_probe.py --experts 2 --json …/requant-probe-151.json
```

Gate (corrections (c)): the full rebuild goes ahead only if the `viterbi` (or `viterbi+refit`) mean relerr is ≤ 0.97x stock, i.e. at least a 3% drop. Expected: 0.26-0.27 vs 0.377. `stock+refit` alone is the cheapest variant; if it clears 3%, it is a same-trellis re-scale of the existing pack. Nothing here changes the serve pin. A full rebuild writes a new revision (e.g. `2.0bpw-mcg-vit`), keeps the model-00043 lm_head MXFP8 re-encode, and is gated on smoke 323, correctness --full, L.A.I.L n≥3 vs 33.2 and prose vs 39.6. The decode kernel is unchanged, so any speed change comes through acceptance only, which is unproven (the MUL1 acceptance loss was its 4-bit MTP drafter, flags.md:253-256).
