# Serve DeepSeek-V4.1-Flash EXL3 on 2× DGX Spark

Serve an EXL3 pack of [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) across two NVIDIA DGX Spark (GB10) nodes at tensor-parallel 2.

Native MXFP4 and MXFP8 weights are about 511 GB and do not fit 2× Spark UMA. Routed experts in this pack are EXL3 2.0 bpw (MCG). Engram stays on NVMe (`DSV41_ENGRAM_DISK=1`). Official vLLM `cpu_offload` pins those tables in host RAM, which on a Spark is the GPU pool.

The Hugging Face card is [model-card.md](model-card.md).

Default thinking is off. If you omit `chat_template_kwargs`, V4.1 thinking is on at effort 50. A small `max_tokens` then returns empty `content`.

## Prerequisites

You need:

- Two DGX Spark nodes
- Passwordless SSH from the head to `WORKER_HOST` (default `spark2`)
- Exclusive GPUs on both nodes. Do not start this while another `--gpus all` serve is up.
- About 340 GB free disk per node
- Hugging Face `hf` (or `huggingface-cli`) and Docker on both nodes

Pin `NCCL_IB_HCA`. GB10 exposes four HCAs and two are DOWN. Read unified memory with `free -h`. Never read VRAM from `nvidia-smi`.

This lab's fabric defaults are `HEAD_IP=10.100.8.1`, `WORKER_HOST=spark2`, `IFACE=enp1s0f1np1`, and `HCA=rocep1s0f1`. If your fabric differs, export `HEAD_IP`, `WORKER_HOST`, `IFACE`, and `HCA` before `./run.sh`.

## Clone the recipe

The default branch is `recipe/dsv41-flash-exl3`.

```bash
git clone https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark.git
cd DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark
```

## Download the pack

Run this on both nodes. Keep Hugging Face xet enabled. Do not set `HF_HUB_DISABLE_XET`. The revision is about 334 GB.

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

Stock `vllm/vllm-openai` wheels do not load `DeepseekV41ForCausalLM`. This image starts from the pinned `deepseekv41-flash-0909` digest and overlays Engram-on-disk plus `vllm-exl3`.

If the image `dsv41-flash-exl3-sm121` is already present, skip the pull and the build on that node.

## Run

If another `--gpus all` container is up, stop it first.

On the head node:

```bash
./run.sh
python3 smoke_chat.py
```

If `ORCHESTRATE=auto` (the default) and SSH to `WORKER_HOST` fails, `run.sh` exits 1. It does not start a TP=2 head rank alone.

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
| Speculative | DSpark-5 (`SPEC=none`) |
| Tokenizers / tools / reasoning | `deepseek_v41` |
| Default thinking | `thinking=false`, `reasoning_effort=low` |
| API | `http://<head>:8000/v1` |
| Container | `dsv41-flash-exl3` |
| Master port | 29524 |
<!-- END generated defaults -->

## Measured on 2× DGX Spark

Not frozen yet. Smoke is `python3 smoke_chat.py` with thinking off.

<!-- BEGIN generated measured from recipe.yaml — edit recipe.yaml and run kit/render.py -->
| Phase | Concurrency | Decode tok/s (median per stream) | Aggregate tok/s | TTFT p50 |
|---|---|---:|---:|---:|
| prose | 1 | — | — | — s |
<!-- END generated measured -->

## Rebuild the pack

Skip this section if you downloaded `sfxnz/DeepSeek-V4.1-Flash-EXL3` at revision `2.0bpw-mcg`.

1. Download the official snapshot (`dba1be0a40aa45a94ad051997016db3960a90277`). The two Engram shards are about 95 GiB each and need `hf_xet`. Do not set `HF_HUB_DISABLE_XET`.
2. Assemble the EXL3 pack with `python3 tools/quantize_experts_exl3.py` inside the recipe image (GPU, exclusive). This recipe uses `--greedy --beam 16` (about 0.85 s/expert, about 1.15 w/s on GB10, about 5.5 h per 20 expert shards). Omit `--greedy` for tail-biting Viterbi (about 4.1 s/expert). Split across the two Sparks:

```bash
# non-expert shards (vision / DSpark / Engram): hardlinks, no GPU
python3 tools/quantize_experts_exl3.py --link-only

# spark1: expert shards 3-22
docker exec dsv41-quant python3 -u /recipe/tools/quantize_experts_exl3.py \
  --allow-partial --batch 8 --greedy --beam 16 --only-files $(python3 -c "print(' '.join(f'model-{i:05d}-of-00048.safetensors' for i in range(3,23)))")

# spark2: expert shards 23-42
docker exec dsv41-quant python3 -u /recipe/tools/quantize_experts_exl3.py \
  --allow-partial --batch 8 --greedy --beam 16 --only-files $(python3 -c "print(' '.join(f'model-{i:05d}-of-00048.safetensors' for i in range(23,43)))")
```

3. Merge the two node outputs and copy the pack to spark2:

```bash
bash tools/assemble_pack.sh
```

`assemble_pack.sh` writes `snapshots/2.0bpw-mcg` on both nodes. Stop the `dsv41-quant` containers before `./run.sh`.

## License

MIT for the scripts. Model weights are MIT from DeepSeek. `vllm-exl3` is AGPL-3.0.
