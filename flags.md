# flags.md — every flag, env var, and patch in this recipe

What / why / measured effect for everything `serve.sh` (→ `run.sh`) sets and
everything the image patches at import time. Sources: `recipe.yaml` (source of
truth), `run.sh`, `docker/patch/sitecustomize.py`, `evidence/`, `results/`.

Effect column key: **measured** = number recorded in `evidence/` or
`results/RESULTS.md`; **required** = the serve fails or regresses without it;
**guard** = refuse-guard default, not a tuning knob.

## vLLM engine flags (run.sh)

| Flag / value | What | Why | Effect |
|---|---|---|---|
| `--tensor-parallel-size 2` `--nnodes 2` | One GB10 rank per Spark | 552B backbone at EXL3 2.0bpw needs both UMAs | **required** |
| `--distributed-executor-backend mp` | Multiproc within node | vLLM TP across 2 nodes via master-addr | **required** |
| `--max-model-len 1048576` | Full native window | V4.1 Flash ships CSA2 1M context; goal is context capacity | guard at 1M (`FORCE_UNSAFE_CTX`) |
| `--max-num-seqs 2` | Running requests | Occupancy measured for 2; DSpark draft + Engram staging leave no more | guard (`FORCE_UNSAFE_CTX`); `evidence/extra-topk-128` |
| `--max-num-batched-tokens 8192` | Chunked-prefill chunk | **measured local optimum** (2026-09-18): 2048 → pp@16k −28%, 16384 → pp@64k −12% and needs +1 GiB KV to even boot; see `results/RESULTS.md` E1/E2/E2b |
| `--kv-cache-dtype fp8` | KV cache storage | KV bytes/token halved vs bf16; CSA2 + indexer pages | measured vs auto in early packs |
| `--kv-cache-memory 8589934592` | 8 GiB KV pool | **measured keep (E5, 2026-09-19)**: pool grows to 2,289,205 tokens = 2.18× concurrency at 1M ctx (4 GiB held exactly 2×1M with zero prefix-cache slack); pp/tg/correctness/e2e all unchanged; 16 GiB still free for Engram staging | guard at 8 GiB (`FORCE_UNSAFE_CTX`) |
| `--gpu-memory-utilization 0.75` | UMA fraction | Leaves ~25 GiB for Engram staging + OS | **required** (0.9 OOMs boot) |
| `--block-size 64` | KV pages | FlashInfer SM120 DSV4 decode kernel is compiled for page 64 | **required** (32 breaks decode kernel) |
| `--quantization exl3` | Loader | Native MXFP4 experts ≈130 GiB/rank do not fit 121 GiB UMA | guard (`FORCE_UNSAFE_QUANT`) |
| `--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy"}'` | DSpark MTP drafting | Checkpoint ships 3 nextn layers + Markov head; 5 = block size | **measured**: 12.75→21.2 tok/s (1.67×), `evidence/h1-dspark` |
| `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,5,6,10,12],"custom_ops":["all"]}'` | CUDA graphs incl. draft | Eager decode wastes ~4 µs/launch × ~11 kernels/step | **measured**: see `evidence/c1-graphs234`, `evidence/vision-on-lail` |
| `--kernel-config '{"enable_flashinfer_autotune":false,"enable_jit_warmup":false}'` | Disable SM120 sparse-MLA autotune | Autotune feeds DeepGEMM paged-MQA which asserts `block_kv∈{32,64}` and crashes warmup | **required** for boot with graphs |
| `--tokenizer-mode/--tool-call-parser/--reasoning-parser deepseek_v41` | Native parsers | V4.1 chat template + tool format | **required** for tool-call eval |
| `--default-chat-template-kwargs '{"thinking":false,"reasoning_effort":"low"}'` | Thinking off by default | Recipe ships fast-answer default; caller can override per request | quality tradeoff — see `results/RESULTS.md` baseline notes |
| `--language-model-only` NOT set; `--mm-encoder-tp-mode data` | Vision on | V4.1 Flash is multimodal; data-parallel encoder | measured `evidence/vision-on-lail` (14.66 vs 15.1 LMO ≈ noise, kept vision) |
| `--trust-remote-code`, `--served-model-name deepseek-ai/DeepSeek-V4.1-Flash` | Identity | Clients expect the official id | — |

## Container / fabric env (run.sh)

| Env | Value | Why | Effect |
|---|---|---|---|
| `NCCL_IB_HCA=rocep1s0f1` (+`NCCL_NET=IB`, `NCCL_CROSS_NIC=1`, `NCCL_NVLS_ENABLE=0`, `NCCL_CUMEM_ENABLE=0`) | Pin one HCA | GB10 exposes 4 HCAs, 2 are DOWN; default selection hangs TP | **required** |
| `NCCL_SOCKET_IFNAME/GLOO_SOCKET_IFNAME/TP_SOCKET_IFNAME=enp1s0f1np1` | Pin TCP iface | Avoids routing over the management NIC | **required** |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Allocator | Fragmentation with 334 GB pack + graphs | avoids late-boot OOM |
| `TORCH_CUDA_ARCH_LIST=12.1a` `FLASHINFER_CUDA_ARCH_LIST=12.1a` | JIT arch | GB10 is SM121; wheels default misses some kernels | **required** for exl3 CUDA JIT |
| `VLLM_PLUGINS=vllm_exl3` + sitecustomize `load_general_plugins()` | Plugin load in every proc | EngineCore resolves `--quantization exl3` before plugins load | **required** |
| `VLLM_EXL3_MOE_KERNEL=native` | Fused p2b MoE | Native 5120×1152 CFG=1 tile vs generic `exl3_moe` | **measured**: ~23 vs 15 tok/s, `evidence/p2b-cfg1` |
| `VLLM_USE_BREAKABLE_CUDAGRAPH=1` | Breakable graphs | Graph capture survives spec-decode shapes on SM121 | **required** for graphs+DSpark |
| `DSV41_ALLOW_CUDA_GRAPHS=1` | Un-stub warmup | Re-enables `compile_or_warm_up_model` after DeepGEMM warmup stub | **measured**: graphs cell in RESULTS |
| `VLLM_ENGINE_READY_TIMEOUT_S=3600` | Boot patience | 334 GB pack load + graph capture | — |
| `DSV41_ENGRAM_DISK=1` | Engram on NVMe | 196B Engram pinned in UMA OOMs a Spark | guard (`FORCE_UNSAFE_ENGRAM`) |
| `LANGUAGE_MODEL_ONLY=0` | Vision weights loaded | See flag row above | — |
| `VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=256` | Adaptive verify profile | Spec verification profile context | default kept |
| `HF_XET_HIGH_PERFORMANCE=1` (download path) | Xet on | Engram shards ~95 GiB need xet | — |

## Refuse-guards (run.sh, all overridable)

`QUANTIZATION!=exl3` (`FORCE_UNSAFE_QUANT`), `DSV41_ENGRAM_DISK!=1`
(`FORCE_UNSAFE_ENGRAM`), `MAX_MODEL_LEN>1048576` / `MAX_NUM_SEQS>2` /
`KV_CACHE_MEMORY>8 GiB` / DSpark tokens not %5 (`FORCE_UNSAFE_CTX`). These
encode measured occupancy/memory limits, not style rules.

## Image patches — always on (docker/patch/sitecustomize.py)

| Patch | What | Why | Effect |
|---|---|---|---|
| `apply_engram_disk.py` + `engram_disk.py` | DiskEngramTable | Engram rows stay on NVMe, staged per step | **required** (memory) |
| `apply_engram_prestage.py` | Prestage rows in `prepare_inputs` | Hides NVMe pread behind compute; un-escapes lookback markers | **measured** decode cell |
| `prefer_b12x_mxfp8.py` | `mm_mxfp8 backend=auto` | SM120 b12x small-M tiles beat cutlass on Q/O GEMMs | **measured**: 23.44 vs ~23 (L.A.I.L wave 22.37), `evidence/b12x-mxfp8` |
| `widen_p2b_shapes.py` + `widen_p2b_cfg1.py` (in image) | Native p2b 5120×1152 CFG=1 | Stock kernel hardcodes hidden=4096 | **measured**: `evidence/p2b-cfg1` |
| `sm120_page.py` (page-64 coercion, indexer block sizes, extra-page 128, vision clamp, native indexer decode) | SM120 page-size fixes | FlashInfer DSV4 decode compiled for page 64; upstream reports 128 | **required** |
| `sm120_page.patch_persistent_topk` | Exclude family 120 from cooperative topk | persistent_topk oversubscribes at 2 rows TopK=512 | **required** (boot) |
| warmup stubs (`kernel_warmup = None`, `deepseek_v4_sparse_mla_attention_warmup = None`) | Skip DeepGEMM paged-MQA dummy forward | asserts block_kv∈{32,64} on this pack | **required** for boot |
| Exl3Config `weight_block_size` copy | Mapper picks `weight_scale` vs `weight_scale_inv` | Exl3Config keeps the field in `non_routed_quantization` | **required** (weights load) |
| `DSV41_DSPARK_MARKOV_SCALE` wrap | Scale Markov bias | experiment knob (default 1 = stock) | measured: ≠1 rejected (`evidence/` decode waves) |

## Compile / kernel experiments — verdicts (2026-09-19)

| Change | What | Verdict | Evidence |
|---|---|---|---|
| `--compilation-config mode=VLLM_COMPILE` | inductor compile of the model graph | **rejected**: pp@64k 699 vs 703 (flat), pp@16k −5%; nothing outside the opaque custom kernels to fuse | `results/RESULTS.md` E7 |
| p2b mma-over-m rows (`docker/Dockerfile.mma`, image `:mma8`) | batch ≤8 same-expert rows per mma tile | **rejected for serving**: pp@64k 709 vs 703 (flat), decode halves (m_loc≈1 pays the batching overhead) | `results/RESULTS.md` E8 |
| Chunk size 2048 / 16384 | chunked-prefill chunk | **rejected**: both directions lose; 8192 is the local optimum | E1/E2/E2b |
| `VLLM_EXL3_FAT_THRESHOLD` 96 / 2048 | fat-expert GEMM loop cutoff | **rejected**: pp@64k ≈ 700±10 at 96/256/2048 | E3/E4 |
| `NUM_SPECULATIVE_TOKENS=10` | deeper DSpark draft | **rejected**: L.A.I.L 12.1 vs 22.6; acceptance falls to 1.87 | E6 |

Kernel conclusion (measured, not guessed): prefill MoE time is invariant to
row batching, chunking, kernel mix, and compile mode — the wall is inside
the p2b weight-decode path itself (trellis `dq8` + shuffle stream at ~23 GB/s
effective vs ~273 GB/s UMA). The next kernel step is a vectorized multi-
weight trellis decode: a new inner loop, gated on the E8 image pipeline.
Do not retry row batching or threshold tuning; those axes are closed.

## Image/kernel env (vllm_exl3, read at import)

| Env | Default | What | Verdict |
|---|---|---|---|
| `VLLM_EXL3_MOE_KERNEL` | `native` (run.sh sets) | Native p2b fused MoE vs generic | **measured** ~23 vs 15 tok/s decode |
| `VLLM_EXL3_FAT_THRESHOLD` | 256 | Rows above which an expert takes the per-expert 128×128 fat GEMM loop instead of the standard kernel | **measured** (E3): 96 → pp@16k −11%, pp@64k noise; keep 256 |
| `VLLM_EXL3_PREFILL_SYNC` | unset | CPU/device sync workaround for 33..144-row prefill wedges | not needed at 8192 chunks; leave unset |
| `TEMP_ROWS_FUSED` | 2048 (const) | Fused-kernel per-expert row cap; chunks with a hotter expert are re-sliced | **measured** (E1): avoiding re-slice via 2048 chunks does not improve pp — not a bottleneck |

## Experiment knobs — present, default OFF, with verdicts

All are env vars consumed by sitecustomize; every one was measured on the
frozen decode bench and rejected unless noted. Keep them documented so the
same ideas are not retried blind.

| Env | What | Verdict | Evidence |
|---|---|---|---|
| `DSV41_STEP_CENSUS=1` | Per-step census (draft/target/Engram timings) | diagnostic only | `evidence/b12x-census` |
| `DSV41_INDEX_TOPK` | Clamp indexer topk | rejected as default | `evidence/extra-topk-128`, `evidence/indexer-native` |
| `DSV41_MHC_DECODE_SPLITS=1` | Collapse MHC prenorm split-K to 1 | rejected | `evidence/mhc-decode-splits` |
| `DSV41_ENGRAM_CACHE=1` | Host LRU for Engram rows | rejected (staging already prestage-hidden) | `evidence/engram-cache` |
| `DSV41_ENGRAM_FAST_STAGE=1` (default) | Parallel per-table Engram disk gathers | **KEEP**: kills ~13 ms/step of GPU idle; L.A.I.L 22.0-23.5 -> 25.2-26.3 | `results/2026-09-19-faststage` |
| `DSV41_ENGRAM_STAGE_THREADS=16` | Worker pool for the parallel stage | default measured good | round 11 |
| `DSV41_ENGRAM_PREFETCH=1` | fadvise next-step Engram rows from CPU-hash at postprocess | experimental, off: pf_hit ~0%, plumbing verified offline | round 11 |
| `DSV41_ENGRAM_CENSUS=1` | Per-gather read/dequant timing + pf_hit | diagnostic | round 11 |
| `--async-scheduling` | V1 async scheduler | neutral-negative on L.A.I.L (21.8/21.9 vs 22.0-23.5) | round 11 |
| `NCCL_MIN/MAX_NCHANNELS=1` | Force single NCCL channel | null (steps/s 9.65 vs 9.68-10.26); AR p50 already 41-54 us | round 11 |
| `DSV41_MHC_NO_DEEPGEMM=1` | TileLang GEMM for MHC prenorm | rejected | `evidence/mhc-tilelang-gemm` |
| `DSV41_DSPARK_DRAFT_TOPK=k` | Topk-mask draft logits | rejected | `evidence/dspark-draft-topk` |
| `DSV41_DSPARK_TAIL_NGRAM=1` (+`_POS`) | Prompt-lookup tail on drafts | rejected | `evidence/dspark-tail-ngram`, `evidence/ngram-overlay` |
| `DSV41_DSPARK_SOFTMAX_VERIFY=1` | Greedy propose, softmax verify | rejected | `evidence/dspark-softmax-verify` |
| `DSV41_DSPARK_REFINE_PASS=1` | Second draft pass on pass-1 fills | rejected | `evidence/dspark-refine-pass` |
| `DSV41_DSPARK_CONF_GATE=1` | Confidence-gated Markov bias | rejected | `evidence/dspark-conf-gate` |
| `DSV41_MLA_IO_WARPS=2` | 2 IO warps in DSV4 MLA decode | rejected (races mbarriers at 4) | `evidence/mla-io2` |
| `DSV41_MLA_CHUNKS_PER_BLOCK=k` | Bake MLA chunks_per_block | rejected | `evidence/mla-chunks-per-block` |
| p2b kernel variants (`widen_p2b_{mma,fma,cp16,cpasync,ldg,pf4,nocoop,mrow}.py`) | Alternate p2b kernels | all rejected or reverted; mma→mrow regression 13.24 | `evidence/p2b-*`, `evidence/mma-revert` |
| `widen_mla_{tile32,kv_buf,io2}.py`, `sm120_wo_a.py`, `c1_graph_safe_adaptive.py` | MLA/KV/graph variants | unwired (failed L.A.I.L or no win) | `evidence/mla-*`, `evidence/c1-*`, `evidence/sm120-wo-a*` |

## Bench/eval settings (frozen for comparability)

- Decode prose: `bench_decode.py --phase prose --concurrency 1 --runs 3 --max-tokens 200`, greedy, thinking off.
- L.A.I.L prose: `tools/measure_lail_prose.py` — 512 tokens, temperature 0.2, c=1.
- Correctness: `tests/correctness.sh [--full]` — temperature 0, seed 20260918, thinking off, effort low.
- Micro: `benches/micro.sh` — fresh docs per invocation (prefix-cache-proof), pp = prompt_tokens/TTFT, tg32 post-first-token.
- E2E: `benches/e2e.sh` — coding-agent turn, 64k doc recall+summary, tool/JSON, then L.A.I.L prose.
- `reasoning_effort` and `thinking` stay locked in every harness; never bench with the default-on thinking.

## Round 5 — decode campaign axes (2026-09-19)

| axis | verdict | evidence |
|---|---|---|
| p2b prefetch ring depth (PF 2/4/8) | **REJECT** | e=30 warm: 672 → 727/761/1359 us; register pressure |
| Full codebook LUT in smem | **CLOSED (hw)** | 128 KiB table > 99 KiB smem/block opt-in on sm_121 |
| vdec1 funnelshift extraction | **KEEP-candidate** | −6.1% bit-exact warm (670.7→629.7 us), 0 cold; ship with next kernel rebuild |
| vdec2 warp-smem staging | reject-ish | −2.5% warm, +3% cold |
| MUL1 pack as decode lever | **CLOSED** | same window format; −16% prose was acceptance-driven; kernel-time parity at best |
| decode-instruction rewrite as main lever | **DEPRIORITIZED** | cold-L2 serving case is memory-stream-bound at 74% UMA peak; ceiling ≤1.35x on 35% of step |

New open axes (from live profile): dense-projection stream efficiency
(b12x grids starve 48 SMs; o_proj wo_a on sm_80 WMMA), NCCL 2-rank AR
latency. See results/RESULTS.md round 5.

## E10 (2026-09-19)

| change | verdict | evidence |
|---|---|---|
| widen_p2b_fshift (funnelshift window merge) | **KEEP** | bit-exact; -6.1% p2b warm; correctness 8/8 |
| widen_b12x_smalls ((16,64) tiles m<=8 n<=8192) | **KEEP** | prefill flat 754/700; prose 25.1->31.4 same-session; gates pass |
| probe_wo_a (diagnostic) | keep | wo_a runtime dtype = bf16 (pack F8) -> sm_80 WMMA bmm: next target |
| NCCL_PROTO forcing | **CLOSED** | isolated 2-rank AR: default=LL 43.0us; LL128 81.8us |

## E11 (2026-09-19)

| change | verdict | evidence |
|---|---|---|
| fix_o_proj_woa_fp8 (exact requant, self-guarded) | **KEEP** | 43/43 layers engaged; prose 31.4->33.6; 8/8 + 3/3; pp band intact |
| wo_a bf16-bmm (emulation dequant) | root-caused | MXFP8 emulation kernel dequants at load; scales retained -> exact roundtrip |

## Round 7 axes (2026-09-19)

| axis | verdict | evidence |
|---|---|---|
| p2b cp.async 4-buffer ring (DEC3, bit-exact) | **REJECT** | cold e=30: 672.4 us vs stock 642.2 (−4.7%); kernel not load-starved |
| p2b load-mechanics axis | **CLOSED** | PF depth, smem staging, cp.async all rejected; kernel at ~76% UMA peak cold — further gains need pack-layout work |
| b12x (16,64) small-m tiles | **NO EFFECT** (kept, harmless) | trace-flat: dense GEMM total 990→1038 ms over same steps; those GEMMs are latency-bound, not parallelism-starved |
| wo_a fp8 einsum | **VERIFIED IN-TRACE** | WMMA 288 us x40/step gone; deep_gemm einsum 80.6 us x40/step |

## Round 8 (2026-09-19)

| axis | verdict | evidence |
|---|---|---|
| flashinfer autotune (serve flag) | **NO EFFECT / stays off** | prose 30.2 vs 31.2, L.A.I.L 22.3 vs 21.9 — bands; GEMMs latency-bound |
| async-TP / comm overlap | **NOT AVAILABLE** in build | no flag/config; structural patch required (hand-off) |

## Round 9 (2026-09-19)

| axis | verdict | evidence |
|---|---|---|
| WNT=8 wide tile (CFG=2) | **REJECT** | 1798 vs 655 us cold — registers kill occupancy |
| group-major trellis layout | **CONFIRMED +6% kernel, not yet integrated** | bit-exact; cold 627.4 vs 667.6 us; needs loader permute + all readers re-indexed (prefill risk) |
| pair-codebook requant | **DEPRIORITIZED→closed** | decode is memory-bound cold; instruction cuts provably don't move it |

## Round 10 (2026-09-19)

| axis | verdict | evidence |
|---|---|---|
| N-split warp decomposition (DEC6) | **REJECT** | 758.0 vs 643.3 us cold (−18%); K-split latency spreading wins |
| p2b variant space | **EXHAUSTED** | K-split+stock layout = local optimum; group-major permute is the sole remaining lever (+6%, needs prefill harness) |

## Round 14 — Engram prefetch v2 (2026-09-20): NO-GO, REVERTED

v2 patch (per-table immediate publish, gen-stamped union, exact fadvise spans,
paired census) wired at `befa570` (dormant while PREFETCH=0; also fixes the
run.sh:475 worker omission of PF_DUMP + adds PREFETCH_DEBUG fwd on both ranks).
Booted with PREFETCH=1 on the nccl-set baseline. Pairing self-check worked as
designed and pinpointed the failure: `[pf-pub]` lands before `[pf-pair]` every
gen (publish timing FIXED — v1's publish-too-late bug is dead), yet
table_pred_id ≠ consumed_id on ~88% of 80 pairs, intersect ≈0 (sum 88/72 rows
over 40 pairs/rank); census pf_hit ~0% (98/105 windows 0%), read_w already
0.06–0.10 ms cold. L.A.I.L 25.23 cold / 26.79 warm / job 26.05 = parity, not
≥28. Doc's "pairing still wrong" branch — but the divergence is now on the
PREDICTION side (rows hashed ≠ rows the next gather reads), not delivery;
direct-pread next lever NOT indicated. Serve reverted to the exact NCCL-arm
config (boot-arm.sh), wiring commit stays dormant. Evidence:
results/2026-09-20-engrampf/ (VERDICT.md, pf-evidence-spark{1,2}.log).

## Round 12 — NCCL AR-tail set (2026-09-20)

| axis | verdict | evidence |
|---|---|---|
| NCCL_BUFFSIZE=1M + LL128_BUFFSIZE=256K + PROTO='^LL128' + MAX_NCHANNELS=8 (on mem-hygiene baseline) | **KEEP** | prose flat (34.69 vs 34.67 re-run; first pass 31.59 = cold-cache artifact), prefill32k +2.1% (733.0), L.A.I.L 26.55/26.12 both above 25.2–26.3 band, MemAvail +3.5/+2.8 GiB/rank (pinned-buffer shrink 4.7→0.14 GiB realized), zero NCCL WARN lines both nodes. results/2026-09-20-nccl/ |

## Round 13 — mul1 lane A/B on 2x Spark (2026-09-20)

Mia-AiLab 2.9bpw pack on her stock GHCR image, our topology (spark2 CX7=f1/GID1,
NFS weights+engram over QSFP), k=3 dspark vs spec off:

| arm | prose c=1 | prefill 8k / 32k | LAIL t0.2 | MemAvail s1/s2 |
|---|---|---|---|---|
| k=3 dspark | 16.35 (acc 1.32, draft 0.108) | 593.7 / 711.7 | 25.63 | 5.3 / 6.7 |
| spec off | **23.64** | 587.9 / **845.6** | 23.46 | 8.8 / 10.0 |
| MCG lane ref | 34.69 | 690 / 733 | 26.39 | 22.3 / 23.8 |

Verdicts: pack+kernels healthy (spec-off matches her ~23; mul1 32k prefill
+15% over MCG). k=3 drafting actively harmful in our boots — draft/verify
pathology (her draft-wo-a-slices patch skipped). Mul1 lane parked at
SPEC_METHOD=none until a draft fix; coop Boot B deferred (its 40.2 ceiling
assumes healthy acceptance). Evidence: results/2026-09-20-mul1-lane/.

### Round 13a — mul1 A3: draft_sample_method=greedy (REFUTED)

Forcing greedy drafting moved greedy-verify acceptance 1.32 → 1.36 (noise);
prose 16.80. Sampling method is not the pathology. Signature stands: drafter
agrees with sampled decoding (2.08 @ t=0.2) but is systematically wrong under
greedy verify. Lane parked at SPEC_METHOD=none; Boot B (coop) blocked until a
draft fix. Evidence: results/2026-09-20-mul1-lane/BOOT-A3-VERDICT.md.

### Round 13b — mul1 A4: LANGUAGE_MODEL_ONLY=1 text-only arm — lane PARKED PERMANENTLY

Text-only server (her documented measurement condition) moved greedy prose by
+0.1 tok/s (16.35→16.45) and acceptance 1.32→1.33 — identical, within noise.
Her exact 400-token budget: 20.30 tok/s @ acceptance 1.69 (acceptance grows
with generation length, not with text-only). Her 28 tok/s k=3 is not
reproducible on this pack under any condition tested (A/A3/A4 × vision on/off
× 200/400 tok × default/greedy draft sampling). Root cause confirmed as pack
property: 4-bit EXL3 MTP drafter (`mtp_bits: 4`) — MCG's source-precision
drafter shows 2.77/2.87 with the same harness. Decision rule (prose ≥24 AND
acc ≥2.0) FAILED on both. mul1 lane PARKED PERMANENTLY at SPEC_METHOD=none
(23.64 / 845.6 @32k); Boot B stays blocked. Re-open mechanism: requantize the
pack with a source-precision drafter (`mtp_experts: source`, MCG-style).
A4 side-observations: text-only k=3 floors 6.2/7.7 GiB; micro tg 8k/32k
36.8/33.9 (acc 2.89/2.92 — best of the lane); LAIL t=0.2 26.79 (parity with
MCG lane). MCG restored + verified; LAIL real job c33b06334e36 = 25.34.
Evidence: results/2026-09-20-mul1-lane/BOOT-A4-VERDICT.md.

### Round 13c — p2b group-major prefill gate: FAIL (bit-exactness)

PREFILL-GATE.md harness (commit eb78a8b) run in the GPU window per spec:
`--mode check` MISMATCH at ALL 20 (matrix, m∈{64,128,256,512}, shape) combos;
control proves both layouts individually deterministic — the PF-G8 B-load
remap genuinely computes wrong outputs (max abs diff 415–678, ~99.998% of
elements differ). Advisory bench (wrong-result timing): G8 within ±1% of
stock nearly everywhere; layer estimate −0.1%..−2.4%. Verdict FAIL — do NOT
fold PF-G8 into quantize_experts_exl3.py; keep p2b stock layout. Lever
re-openable only via harness fix + fresh GPU gate re-run (the harness was
CPU-validated for compile only, never numerically on GPU). Evidence:
kernel_study/gemv_bench/PREFILL-GATE-RESULT-2026-09-20.log.

### Round 15 — MCG spec-k sweep: k=3 WINS the L.A.I.L cell (single-lever, same-day n=3)

First sweep of DSpark `num_speculative_tokens` on the MCG lane (k=5 was
inherited; %5 guard bypassed with FORCE_UNSAFE_CTX=1 — k=10 already rejected
in E6). Same day, same image/env, only k varied; L.A.I.L app bench n=3 per arm
+ CLI twin (512 tok, t=0.2) + greedy prose 9×:
k3 27.27 / 28.57 @ acc 2.25 / prose 35.55 @ 2.62 — k5 26.32 / 25.57 @ 2.30 /
33.03 @ 2.90 — k4 25.87 / 25.86 @ 2.18. k3 wins all three cells (+3.6% app,
+11.7% twin, +7.6% prose); k4 worst (deeper than t=0.2 acceptance justifies
AND pads verify 5→6); k7 skipped (flat acc-vs-depth + E6 collapse at k=10).
Mechanism: at t=0.2 acc≈2.3, draft slots 4–5 are dead weight; k3 verifies
4 tok/step and still pads 4→5 on the k5-era capture sizes [1,5,6,10,12] —
matched sizes [1,3,4,6,8] is the queued follow-up. Serve left UP at k=3
(boot-k3.sh in results/2026-09-20-ksweep/, smoke 323 ✓, confirm job
65933ed0ddd7 = 27.0). L.A.I.L job ids in VERDICT.md; contaminated job
1b2fa6c0fc52 discarded (ran concurrently with its warmup — app bench queue
bug in my driver script, fixed). Gap to 35: 7.7 tok/s. Evidence:
results/2026-09-20-ksweep/VERDICT.md.
