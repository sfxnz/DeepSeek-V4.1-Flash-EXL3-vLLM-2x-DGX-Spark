# Decode hillclimb hypotheses

Metric: frozen `bench_decode.py --phase prose --concurrency 1 --runs 3 --max-tokens 200`.
Baseline median **12.75 tok/s**. Keep if median ≥ **19.12** (1.5×) and smoke + no OOM.

## H1. SPEC=dspark, NUM_SPECULATIVE_TOKENS=5, still eager

Mechanism: V4.1 checkpoint ships DSpark (`dspark_block_size=5`, `num_nextn_predict_layers=3`,
`mtp.*` shards 44–46). Official image has `method=dspark`. V4 Flash DSpark was ~2.3× no-spec
on this fabric. Draft experts are `mtp_experts: source` (MXFP4, 128 experts × 3 layers) —
extra UMA, but remaining ~24 GiB may hold it.

Risk: OOM on MXFP4 draft experts; EXL3 loader mismatch on `mtp.*`; SM120 sparse-MLA hang
if adaptive verification is on (leave it off).

Image rebuild: no.

## H2. Native vllm-exl3 p2b fused MoE for 5120×1152

Mechanism: stock CUDA MoE hardcodes hidden=4096 and inter in {1024,2048}. V4.1 TP=2 is
hidden=5120, intermediate_local=1152 — both multiples of 128 (Hadamard) and 16 (tiles).
Kernel body is parameterized by `hidden`/`inter`. Measured on GLM 4096 K=2 m=1: 1.69× vs
exl3_moe. Requires compiling `vllm_exl3` CUDA (drop `VLLM_EXL3_NO_CUDA=1`) and widening
the TORCH_CHECK / Python gate.

Risk: illegal memory if tile math is wrong; image rebuild ~long; GB10 cooperative launch.

## H3. CUDA graphs (ENFORCE_EAGER=0) + un-stub compile_or_warm_up_model

Mechanism: eager disables graphs. sitecustomize stubs compile/warmup because DeepGEMM
paged-MQA warmup crashed. Graphs on DSpark draft are designed FULL. Expected 10–30% if
capture survives SM120 sparse MLA.

Risk: known crash class. Do not land if smoke dies.

## H4. NCCL instead of PYNCCL for TP=2 all-reduce

Mechanism: FlashInfer AR disabled at world_size=2; engine uses PYNCCL. Tony's 4× V4.1
recipe measured ~5 ms in 88 all-reduces/step. Smaller than MoE, not 1.5× alone.
