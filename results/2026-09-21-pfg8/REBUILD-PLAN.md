# PF-G8 PACK REBUILD PLAN (post-PASS; NOT started this session)

Gate PASSED 2026-09-21 (PREFILL-GATE-RESULT-2026-09-21.log): PF-G8 is
bit-exact 20/20 and within noise on prefill perf. This is the plan for the
next dispatch that owns the box for hours. DO NOT run from a live session.

## What must change before a G8 pack can serve

1. Pack: quantize-time fold (STAGED, dormant, commit this branch):
   `DSV41_PACK_PF_G8=1 tools/quantize_experts_exl3.py ...` — folds
   `.view(kt, nt/8, 8*w).permute(1,0,2).contiguous()` right after
   `pack_trellis()` (tools/quantize_experts_exl3.py, `pfg8-pack-fold`).
2. Loader: `docker/patch/pfg8_loader_reindex.py` (STAGED, dormant) applied to
   the image's vllm_exl3/exl3.py; serving env `DSV41_LOAD_PF_G8=1` swaps the
   TP narrow dims (gate/up dim0 = n-groups, down dim1 = k). Both TP=2 splits
   are whole: gate/up 144 n-tiles = 18 groups → 9/rank; down KT=144 → 72/rank.
3. Serving kernels G8-aware: exl3_gemm B-load remap exactly as proven in
   kernel_study/gemv_bench/build_prefill/exl3_gemm_inner_pf.cuh (pf_g8
   runtime flag; k-stride = TILEBLOCKS_K*8*16*bits — the 2026-09-20 bug was
   stride 0) + the p2b decode G8 reader (DEC5 pattern, groups of 8 ⊃ aligned
   4-tile runs). NOT yet written into the image build — separate dispatch.
4. Image rebuild + A/B e2e (8/8 correctness, 3/3 quality, prose
   decode/prefill bands) before the G8 pack replaces 2.0bpw-mcg.

## Measured converter shard rate (from the 2.0bpw-mul1 build, 2026-09-15)

- spark1, expert shards 3-22 (20 shards): first 13:30:08 → last 18:58:36
  = 5.47 h → **~16.4 min/shard**
- spark2, expert shards 23-42 (20 shards): first 13:29:48 → last 18:52:46
  = 5.38 h (parallel)
- Non-expert shards 1-2, 43-48: link-only, minutes.
- Per-trellis G8 fold adds one view+permute+contiguous on GPU per tensor —
  negligible vs the ~5.3 s/expert encode; rate assumed unchanged.

## Wall estimate

- Both ranks in parallel (the README split): **~5.5-6 h** wall + assemble
  (~10 min) + index rebuild. Single-rank fallback: ~11 h.
- New snapshot dir `snapshots/2.0bpw-mcg-g8` (or -mul1-g8); same total size.

## Disk headroom check (2026-09-21, measured)

- Pack filesystem (nvme0n1p2, both cache trees on it): 3.7 T total, 2.9 T
  used, **608-652 GB free**.
- New pack = 358.1 GB (46,080 trellis tensors across 48 shards; permuted
  dry-run plan re-measured this session). Quantize-time fold writes the G8
  layout directly — no second copy needed.
- 608 GB free vs 358.1 GB needed → fits with ~250 GB margin, but the box
  lands at ~93% full. **Delete or archive the superseded snapshot
  (2.0bpw-mul1, 358 GB — parked lane, fbcf4a7) first** to stay ~83%.
- Engram shards (~95 GiB x2) are untouched by G8 (not routed-expert trellis).

## Exact rebuild command sequence (separate dispatch, owns the box)

```bash
cd ~/projects/ai-lab/recipes/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark
./stop.sh                                   # both ranks; verify docker ps empty BOTH

# 0. (recommended) free 358 GB: archive/remove snapshots/2.0bpw-mul1

# 1. hardlink non-expert shards
DSV41_PACK_PF_G8=1 python3 tools/quantize_experts_exl3.py --codebook mcg --link-only \
    --dst ~/.cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-EXL3/snapshots/2.0bpw-mcg-g8

# 2. both ranks in parallel, inside the recipe image (exclusive GPU):
# spark1: shards 3-22
DSV41_PACK_PF_G8=1 python3 tools/quantize_experts_exl3.py --codebook mcg \
    --allow-partial --batch 8 --greedy --beam 16 \
    --only-files $(python3 -c "print(' '.join(f'model-{i:05d}-of-00048.safetensors' for i in range(3,23)))")
# spark2: shards 23-42 (same command, range(23,43))

# 3. merge + index
CODEBOOK=mcg bash tools/assemble_pack.sh

# 4. verify G8 layout + roundtrip on a sample shard
python3 tools/permute_pack_group_major.py <new-pack>   # dry-run plan/verifier

# 5. serving side (must already be in the image by then):
#    pfg8_loader_reindex.py applied + DSV41_LOAD_PF_G8=1 + G8-aware kernels;
#    boot, smoke '17 * 19 = ? Step by step, then answer.' → 323, then A/B e2e.
```

Note: the fold is env-gated per-process; docker exec passes it through. The
serving image does NOT read DSV41_PACK_PF_G8 — a G8 pack booted on a stock
image will produce garbage (trellis misread); the loader env + kernels must
land first (items 2-3 above), which is why the pack rebuild is sequenced
last and NOT started here.
