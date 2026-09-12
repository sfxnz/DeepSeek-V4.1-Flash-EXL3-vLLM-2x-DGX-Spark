# Serve DeepSeek-V4.1-Flash EXL3 on 2× DGX Spark

Serve an EXL3 pack of [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) across two NVIDIA DGX Spark (GB10) nodes at tensor-parallel 2.

The pack is [sfxnz/DeepSeek-V4.1-Flash-EXL3](https://huggingface.co/sfxnz/DeepSeek-V4.1-Flash-EXL3) at revision `2.0bpw-mcg`. Routed experts are EXL3 2.0 bpw (MCG). Engram stays on NVMe (`DSV41_ENGRAM_DISK=1`). Native MXFP4 and MXFP8 weights are about 511 GB and do not fit 2× Spark UMA.

Default thinking is off. If you omit `chat_template_kwargs`, V4.1 thinking is on at effort 50. A small `max_tokens` then returns empty `content`.

## Prerequisites

You need:

- Two DGX Spark nodes
- Passwordless SSH from the head to `WORKER_HOST` (default `spark2`)
- Exclusive GPUs on both nodes. Do not start the serve while another `--gpus all` container is up.
- About 340 GB free disk per node
- Hugging Face `hf` (or `huggingface-cli`) and Docker on both nodes

Pin `NCCL_IB_HCA`. GB10 exposes four HCAs and two are DOWN. Read unified memory with `free -h`. Never read VRAM from `nvidia-smi`.

The recipe defaults are `HEAD_IP=10.100.8.1`, `WORKER_HOST=spark2`, `IFACE=enp1s0f1np1`, and `HCA=rocep1s0f1`. If your fabric differs, export `HEAD_IP`, `WORKER_HOST`, `IFACE`, and `HCA` before `./run.sh`.

## Clone the recipe

The default branch is `main`.

```bash
git clone https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark.git
cd DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark
```

## Download the pack

Run the download on both nodes. Keep Hugging Face xet enabled. Do not set `HF_HUB_DISABLE_XET`. The revision is about 334 GB.

```bash
unset HF_HUB_DISABLE_XET
export HF_XET_HIGH_PERFORMANCE=1
hf download sfxnz/DeepSeek-V4.1-Flash-EXL3 --revision 2.0bpw-mcg
```

`hf download` writes `refs/2.0bpw-mcg` and `snapshots/<commit>/`. `./run.sh` resolves that layout. An assembled pack at `snapshots/2.0bpw-mcg` still works. If both exist, `run.sh` uses the Hub commit snapshot.

If the pack is already in the Hub cache on a node, skip the download on that node.

## Build the image

On both nodes, from this repo:

```bash
docker pull vllm/vllm-openai:deepseekv41-flash-0909@sha256:d84a123255b822fc22508635218000187221794f59c0694c33b0650d1e377d58
docker build -f docker/Dockerfile -t dsv41-flash-exl3-sm121 docker
```

Stock `vllm/vllm-openai` wheels do not load `DeepseekV41ForCausalLM`. The image starts from the pinned `deepseekv41-flash-0909` digest and overlays Engram-on-disk plus `vllm-exl3`.

If the image `dsv41-flash-exl3-sm121` is already present, skip the pull and the build on that node.

## Run

If another `--gpus all` container is up, stop it first.

On the head node:

```bash
./run.sh
python3 smoke_chat.py
python3 bench_decode.py --phase prose --concurrency 1
python3 tools/measure_lail_prose.py
```

The API is `http://127.0.0.1:8000/v1`. The served model is `deepseek-ai/DeepSeek-V4.1-Flash`. Cap is `MAX_NUM_SEQS=2`. Do not send a third stream.

The `smoke_chat.py` default prompt is `What is 17*19? Return only the integer.` Thinking is off. Non-empty `content` is the pass. `323` is enough. `bench_decode.py` is streamed greedy, thinking off, 200 completion tokens, 3-run median.

If `ORCHESTRATE=auto` (the default) and SSH to `WORKER_HOST` fails, `run.sh` exits 1. It does not start a TP=2 head rank alone.

When you are done:

```bash
./stop.sh
```

## Defaults

<!-- BEGIN generated defaults from recipe.yaml — edit recipe.yaml and run kit/render.py -->
| Setting | Value |
|---|---|
| Image | `dsv41-flash-exl3-sm121` |
| Model | `sfxnz/DeepSeek-V4.1-Flash-EXL3` revision `2.0bpw-mcg` |
| `--tensor-parallel-size` / `--nnodes` | 2 / 2 |
| `--max-model-len` | 1048576 |
| `--max-num-seqs` | 2 |
| `--max-num-batched-tokens` | 2048 |
| `--kv-cache-dtype` | `fp8` |
| `--kv-cache-memory` | `4294967296` |
| `--quantization` | `exl3` |
| Engram | disk (`DSV41_ENGRAM_DISK=1`) |
| `--block-size` | 64 |
| Speculative | DSpark-5 (`SPEC=dspark`) |
| CUDA graphs | `FULL_AND_PIECEWISE` (`ENFORCE_EAGER=0`, `DSV41_ALLOW_CUDA_GRAPHS=1`) |
| Tokenizers / tools / reasoning | `deepseek_v41` |
| Default thinking | `thinking=false`, `reasoning_effort=low` |
| API | `http://<head>:8000/v1` |
| Container | `dsv41-flash-exl3` |
| Master port | 29524 |
<!-- END generated defaults -->

## Measured on 2× DGX Spark

`bench_decode.py` is streamed greedy, 200 completion tokens, 3-run median. `tools/measure_lail_prose.py` matches L.A.I.L streams prose (512 tokens, temperature 0.2). Default is DSpark-5 with CUDA graphs. Smoke is `python3 smoke_chat.py` with thinking off.

<!-- BEGIN generated measured from recipe.yaml — edit recipe.yaml and run kit/render.py -->
| Phase | Concurrency | Decode tok/s (median per stream) | Aggregate tok/s | TTFT p50 |
|---|---|---:|---:|---:|
| prose | 1 | 21.2 | 21.2 | 0.365 s |
| lail_prose | 1 | 15.3 | 15.3 | 0.474 s |
<!-- END generated measured -->

## Rebuild the pack

If you already downloaded `sfxnz/DeepSeek-V4.1-Flash-EXL3` at revision `2.0bpw-mcg`, skip this section. The published pack is the serve path.

Exclusive GPU. Stop any quant container before `./run.sh`.

1. Download the official snapshot.

```bash
python3 tools/download_official.py
```

The commit is `dba1be0a40aa45a94ad051997016db3960a90277`. The two Engram shards are about 95 GiB each and need `hf_xet`. Do not set `HF_HUB_DISABLE_XET`.

2. Hardlink the non-expert shards on the host. Quantize routed experts inside image `dsv41-flash-exl3-sm121` with exclusive GPU. Host Python does not have ExLlamaV3. Use `--greedy --beam 16`. Split expert shards 3-22 and 23-42 across the two Sparks.

```bash
python3 tools/quantize_experts_exl3.py --link-only

# spark1: expert shards 3-22, inside the recipe image
python3 tools/quantize_experts_exl3.py \
  --allow-partial --batch 8 --greedy --beam 16 --only-files $(python3 -c "print(' '.join(f'model-{i:05d}-of-00048.safetensors' for i in range(3,23)))")

# spark2: expert shards 23-42, inside the recipe image
python3 tools/quantize_experts_exl3.py \
  --allow-partial --batch 8 --greedy --beam 16 --only-files $(python3 -c "print(' '.join(f'model-{i:05d}-of-00048.safetensors' for i in range(23,43)))")
```

3. Merge the two node outputs.

```bash
bash tools/assemble_pack.sh
```

## License

MIT for the scripts. Model weights are MIT from DeepSeek. `vllm-exl3` is AGPL-3.0.
