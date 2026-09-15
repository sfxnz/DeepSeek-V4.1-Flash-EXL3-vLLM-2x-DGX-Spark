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
PYNCCL is already libnccl. This H4 as written is a false fork.

## H5. Native p2b CFG=2 COLS=128, stay MCG cb=1

Mechanism: Nsight decode-hot slice is `p2b_moe_batched` (~22–23%) at 5120×1152.
CFG=1 is 80 down groups × 64 cols. Both dims divide 128. CFG=2 is WNT=8,
COLS=128, down groups 40, gate groups 9, same WK=8 / 256 threads /
`launch_bounds(256, 4)`. Halves GEMV work-list items. Register pressure at
WNT=8 is the fail mode. Do not drop minBlocks until a compile or boot log
shows spill.

Risk: spill drops occupancy; cooperative launch fails if four blocks cannot
reside. Image rebuild required.

## H6. Exact HCA, CROSS_NIC=0, Ring, two channels

Mechanism: Nsight AllReduce is ~11–15% plus a 304 ms spark1 tail. Decode AR
is about 10 KiB (5120×2). `CROSS_NIC=1` can hunt a DOWN HCA on GB10 (four
HCAs, two DOWN). Exact `NCCL_IB_HCA==$HCA` plus `CROSS_NIC=0` pins one rail.
Ring plus `MIN_NCHANNELS=1` / `MAX_NCHANNELS=2` matches two-rank small-message
AR. Do not set `NCCL_PROTO=LL`. Prefill ARs are 20–124 MiB.

Risk: Ring or channel pin can hurt prefill. Exact HCA stays required for boot.
Image rebuild not required.
