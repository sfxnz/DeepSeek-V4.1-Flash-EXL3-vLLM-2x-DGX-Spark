# vLLM v0.30.0 rebase: Phase A (CPU-only go/no-go, 2026-09-24)

**Verdict: NO-GO for a perf-motivated rebase now. Keep it as a maintenance item and re-run Phase A on the next release.**

- The anchor audit is mostly clean.
  - 16 of 22 wired patch steps report `changed` on v0.30.0 once `deepseek_v4_1` is retargeted to `deepseek_v41`. One of those, `apply_engram_disk`, applies only partially (see the table).
  - All `vllm.*`/`flashinfer.*` imports and monkeypatch targets in `docker/patch` resolve.
  - torch is identical, so the plugin ABI is unchanged.
- The breakage is concentrated in one place, the **Engram-on-disk chain**. Upstream refactored Engram (common base + `nvidia/engram.py` with its own async prefetch and DP shared memory). That is a real re-port, and `DSV41_ENGRAM_DISK=1` is mandatory (AGENTS.md).
- The decode upside on GB10 is at or below noise:
  - Mega-mHC is SM100-only.
  - The delayed-pre mHC is already in the pin.
  - DeepGEMM's SM120 kernel set is unchanged.
  - FlashInfer is still 0.6.18 (post1), not 0.7.

## Images (registry + local, no GPU)

| | pin (serving base) | v0.30.0 |
|---|---|---|
| tag | `deepseekv41-flash-0909` | `v0.30.0-aarch64` (= `v0.30.0` arm64) |
| digest | `sha256:d84a1232…` | `sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56` |
| vllm | 0.1.dev20904+g179dd0fa9 | 0.30.0 |
| torch | 2.13.0+cu130 | 2.13.0+cu130 |
| flashinfer | 0.6.18 | 0.6.18.post1 |
| DeepGEMM (vllm/third_party) | 2.6.1 | 2.8.0 |
| nvidia-nccl-cu13 | 2.30.7 | 2.30.7 |
| tilelang / triton | 0.1.12 / 3.7.1 | 0.1.12 / 3.7.1 |
| cutlass-dsl | 4.6.2 | 4.7.1 |
| model dir | `models/deepseek_v4_1` | `models/deepseek_v41` |

The registry has no `deepseekv41-*` tag newer than `-0909` and no v0.30.x patch release (633 tags listed, 2026-09-24). The pull was 9.7 GB compressed; disk had 469 GB free.

## Patch dry-run

`tools/rebase_patch_dryrun.py` runs inside a throwaway `--rm --network none` container of each image, with no `--gpus` and with `-S` so that no baked sitecustomize runs first. It:

1. replays the build-time patches (docker/Dockerfile + Dockerfile.e11) and the runtime file patches from sitecustomize.py, in order, recording failures instead of swallowing them;
2. scans every `*OLD*`/`*ANCHOR*` string constant of those patches against the pristine tree;
3. resolves every `vllm`/`flashinfer` import and `Class.attr = …` monkeypatch statically.

The baseline is the pristine pinned base (`@sha256:d84a…`). On it, all 22 steps report `changed`, 50/67 anchors hit, and the other 17 are chain anchors inserted by earlier patches. Full output is in `phaseA-dryrun-0909.{txt,json}` and `phaseA-dryrun-v030.{txt,json}`.

| step | pin | v0.30.0 (retargeted) | root cause |
|---|---|---|---|
| apply_engram_disk (build) | changed | changed, **but partial**: SIG_OLD, ALLOC_OLD, CTOR_OLD missing | Engram refactor. `_once()` short-circuits on the file marker after the first anchor lands, so misses are silent. This is a latent bug in the patch. |
| apply_engram_prestage (build) | changed | **error**: missing marker staged_rows (PREPARE_OLD, STAGED_OLD miss) | Engram refactor. Upstream now has `prepare_embeddings/_start_prefetch` on a prefetch stream. |
| engram_stage_fast | changed | **error**: stage() anchor missing | cascade from prestage (its anchors are prestage-inserted) |
| engram_prefetch_v3 | changed | **error**: stager init anchor missing | cascade from prestage |
| engram_cpu_hash | changed | **error**: fast-stage anchor missing | cascade from stage_fast |
| engram_defer | changed | **error**: stager init anchor missing | cascade from prestage |
| persistent_topk (sm120_page) | changed | **error**: dispatch not found | topk moved to `model_executor/layers/indexer_topk.py`. v0.30 adds `kernel_config.sparse_indexer_topk_backend`, so `--kernel-config '{"sparse_indexer_topk_backend":"per_row"}'` should replace this patch with no code. GPU-verify the SM120 path picks it. |
| the other 15 (b12x_smalls, probe_wo_a, fix_o_proj_woa_fp8, prefer_b12x_mxfp8, indexer_workspace, widen_mla_io2, widen_mla_tile32, kpool_persistent_topk, swa/attention image width, native_indexer_decode, indexer_adaptive, indexer_short_context, engram_stage_census, engram_gather_v2) | changed | changed | — (census and gather_v2 patch our own copied `engram_disk.py`) |

- **Anchors newly missing on v0.30.0** (present on the pin): 6. They are `apply_engram_disk` SIG_OLD, ALLOC_OLD and CTOR_OLD; `apply_engram_prestage` PREPARE_OLD and STAGED_OLD; and `sm120_page` PERSISTENT_TOPK_OLD. The 16 engram chain anchors cannot be checked until prestage is re-ported.
- **Anchors that moved files:** none apart from the expected retarget. `g8_stream_feed` INSTALL_OLD is no longer in `deepseek_v4/nvidia/vl_model.py`, but it still hits the V4.1 file it targets.
- **Imports and monkeypatch targets:** the same 5 unresolved names on both images, all of them created by our own patches (`engram_disk.DiskEngramTable`, `EngramDiskStager`, `Worker._dsv41_drop_page_cache`). No new breaks.
- **vllm-exl3 plugin:** its 24 `vllm.*` imports resolve the same on both bases. torch is 2.13.0+cu130 on both, so the plugin and exllamav3 wheels rebuild with the unchanged Dockerfile lines.
- **CPU import (no GPU):** `vllm.models.deepseek_v41.nvidia.{model,dspark}`, `common.engram`, `spec_decode.dspark.speculator` and `kernels.mhc` all import on v0.30.0 (they also import on the pin). `tests/test_lmhead_mxfp8.py` passes 6/6 in both images.
- **sm_121 arch:** both bases ship sm_120 SASS only. Counts are identical in `_C_stable_libtorch` (83) and `_moe_C` (26), and neither has sm_120 in `_flashmla_C`. v0.30 only adds sm_110. The arch situation matches the pin that already serves on GB10, so this is not a new risk. `TORCH_CUDA_ARCH_LIST=12.1a` applies only to our plugin builds.

## What v0.30.0 would bring to GB10 decode (corrections applied)

The step is ~64-68 ms per DSpark-3 verify step. Kernel shares (trace-attribution) are: p2b 31% (custom EXL3, which upstream does not touch), b12x dense 25%, NCCL 8.5%, wo_a 4.5% and mHC 3.4%.

- **Mega-mHC (#56962):** `is_mega_mhc_supported()` requires `is_device_capability_family(100)` (`nvidia/ops/mega_mhc.py:24`), so it is **0 on SM121**. The call falls back to `mhc_fused_post_pre_delayed_tilelang`.
- **Delayed pre (#56633):** `mhc_pre_delayed_tilelang` is **already in the pin** (0909 `nvidia/model.py:24`). What is new is the post+pre fusion on non-Engram layers (Engram layers keep a separate post). Its ceiling is part of the 3.4% mHC slice, realistically ≤ 1%.
- **DeepGEMM 2.8.0:** the SM120 impl set is identical to 2.6.1. `sm120_fp8_fp4_gemm_1d1d.cuh` changed by 69 lines and `sm120_tf32_hc_prenorm_gemm.cuh` by 9. Both are small shares (wo_a 4.5%, hc_prenorm ~2.6%), so the effect is unknown and small. The SM120 paged-MQA kernels exist in both versions (a 2-line diff).
- **FlashInfer:** 0.6.18.post1, not 0.7, so the b12x dense path (25%) and the sm121 sparse-MLA hang fix are **not in v0.30.0**.
- **#56441 / #56562:** host-side metadata and KV-only insert. Decode is device-bound (r33 VERDICT).
- **acceptance_estimator.py:** used only with `enable_adaptive_verification` (default False). The 6-token-verify/adaptive family is closed here (sitecustomize c1 note).

Expected decode gain is ≤ 1-2%, which is inside the 2-3% noise band. This is maintenance, not a perf item.

## Effort if pursued (Phase B)

| work | est. |
|---|---|
| Re-port Engram-on-disk onto `BaseParallelEngramEmbedding._allocate_weights` / nvidia `ParallelEngramEmbedding`, and decide how prestage/prefetch_v3/defer relate to upstream `_start_prefetch`. Re-verify 16 chain anchors. Fix the `_once` partial-apply bug. | 2-3 days |
| Replace persistent_topk with the kernel-config knob in run.sh/recipe.yaml | 0.5 day |
| Path retarget: 16-17 patch files hardcode `deepseek_v4_1` | 0.5-1 day |
| Rebuild image (same plugin lines; torch unchanged), boot, smoke 323 + vision, correctness --full, four numbers under ARMS.md step 6 (ABAB vs the current image, not vs 39.6 / 33.2) | ~4 h GPU (serve down) |

Re-run Phase A (about 1 minute per image once pulled) when any of these happens: a vLLM release ships FlashInfer ≥ 0.7, SM12x Mega-mHC lands, or the pin blocks a needed fix.

```bash
W=<worktree>; IMG=vllm/vllm-openai:<tag>
docker run --rm --network none -e NVIDIA_VISIBLE_DEVICES=void \
  -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages \
  -v $W/docker/patch:/opt/dsv41-patch:ro -v $W/tools:/probe:ro -v /tmp/out:/out \
  --entrypoint python3 $IMG -S /probe/rebase_patch_dryrun.py --retarget --json /out/dry.json
```
