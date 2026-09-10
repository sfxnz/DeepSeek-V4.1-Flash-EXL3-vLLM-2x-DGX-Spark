# DeepSeek-V4.1-Flash EXL3 · vLLM · 2× DGX Spark

Serve an **EXL3** pack of [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) across two NVIDIA DGX Spark (GB10) nodes at tensor-parallel 2.

552B backbone, 8B active at prefill and 16B at decode, 384 routed experts (top-6), CSA2, CED, Engram at layers 1 and 14. Native MXFP4/MXFP8 is ~511 GB and does not fit 2× Spark UMA. This recipe trellis-quantizes routed experts to EXL3 2.0 bpw (MCG) and keeps Engram on NVMe (`DSV41_ENGRAM_DISK=1`). Official vLLM `cpu_offload` pins those tables in host RAM, which on a Spark is the GPU pool.

Stock `vllm/vllm-openai` wheels do not know `DeepseekV41ForCausalLM`. Build the local image from the pinned `deepseekv41-flash-0909` digest, then overlay Engram-on-disk and `vllm-exl3`.

Default thinking is **off**. V4.1 thinking is on at effort 50 if you send nothing, so a small `max_tokens` returns empty `content`.

## Measured on 2× DGX Spark

Not frozen yet. Smoke is `python3 smoke_chat.py` with thinking off.

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
| `--block-size` | 256 |
| Speculative | DSpark-5 (`SPEC=dspark`) |
| Tokenizers / tools / reasoning | `deepseek_v41` |
| Default thinking | `thinking=false`, `reasoning_effort=low` |
| API | `http://<head>:8000/v1` |
| Container | `dsv41-flash-exl3` |
| Master port | 29524 |
<!-- END generated defaults -->

<!-- BEGIN generated measured from recipe.yaml — edit recipe.yaml and run kit/render.py -->
| Phase | Concurrency | Decode tok/s (median per stream) | Aggregate tok/s | TTFT p50 |
|---|---|---:|---:|---:|
| prose | 1 | — | — | — s |
<!-- END generated measured -->

## Build the image

On both nodes, from this repo:

```bash
docker pull vllm/vllm-openai:deepseekv41-flash-0909@sha256:d84a123255b822fc22508635218000187221794f59c0694c33b0650d1e377d58
docker build -f docker/Dockerfile -t dsv41-flash-exl3-sm121 docker
```

## Weights

1. Download the official snapshot (`dba1be0a40aa45a94ad051997016db3960a90277`). The two Engram shards are ~95 GiB each and need `hf_xet` (`HF_HUB_DISABLE_XET` must be unset).
2. Assemble the EXL3 pack with `python3 tools/quantize_experts_exl3.py` inside the recipe image (GPU, exclusive). Uncalibrated 2.0 bpw MCG is about 5.3 s/expert on one GB10 (~33 h per 20 expert shards). Split across the two Sparks:

```bash
# non-expert shards (vision / DSpark / Engram): hardlinks, no GPU
python3 tools/quantize_experts_exl3.py --link-only

# spark1: expert shards 3-22
docker exec dsv41-quant python3 -u /recipe/tools/quantize_experts_exl3.py \
  --allow-partial --batch 8 --only-files $(python3 -c "print(' '.join(f'model-{i:05d}-of-00048.safetensors' for i in range(3,23)))")

# spark2: expert shards 23-42
docker exec dsv41-quant python3 -u /recipe/tools/quantize_experts_exl3.py \
  --allow-partial --batch 8 --only-files $(python3 -c "print(' '.join(f'model-{i:05d}-of-00048.safetensors' for i in range(23,43)))")
```

3. Merge the two node outputs and copy the pack to spark2:

```bash
bash tools/assemble_pack.sh
```

4. Place the pack at `models--sfxnz--DeepSeek-V4.1-Flash-EXL3/snapshots/2.0bpw-mcg` on both nodes, or `hf download sfxnz/DeepSeek-V4.1-Flash-EXL3 --revision 2.0bpw-mcg` once published. Stop the `dsv41-quant` containers before `./run.sh` — exclusive GPUs.

## Run

```bash
./run.sh
```

`ORCHESTRATE=auto` (default): if SSH to `WORKER_HOST` fails, `run.sh` exits 1. It does not start a TP=2 head rank alone.

```bash
python3 smoke_chat.py
./stop.sh
```

Exclusive GPUs. Do not start this while another `--gpus all` serve is up. Pin `NCCL_IB_HCA`. Read unified memory with `free -h`. Never `nvidia-smi` VRAM.

## License

MIT for the scripts. Model weights are MIT from DeepSeek. `vllm-exl3` is AGPL-3.0.
