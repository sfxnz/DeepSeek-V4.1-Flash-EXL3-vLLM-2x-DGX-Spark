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
| `--max-num-batched-tokens 8192` | Chunked-prefill chunk | Raised 2048→8192 to cut TTFT on long prompts | **measured**: see `results/RESULTS.md` pp cells |
| `--kv-cache-dtype fp8` | KV cache storage | KV bytes/token halved vs bf16; CSA2 + indexer pages | measured vs auto in early packs |
| `--kv-cache-memory 4294967296` | 4 GiB KV pool | CSA2 ≈ 890 B/token → 4 GiB holds 1M×2 streams | guard at 8 GiB (`FORCE_UNSAFE_CTX`) |
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
