---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
library_name: transformers
tags:
  - exl3
  - quantized
  - vllm
pipeline_tag: text-generation
---

# DeepSeek-V4.1-Flash EXL3 2.0 bpw MCG

EXL3 pack of [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash). Routed experts are 2.0 bpw MCG. Other tensors keep the source dtypes. Engram tables stay on NVMe at serve time.

This pack exists so 2× NVIDIA DGX Spark (GB10) can serve the model at tensor-parallel 2. Native MXFP4/MXFP8 is about 511 GB and does not fit 2× Spark UMA.

## Download

```bash
hf download sfxnz/DeepSeek-V4.1-Flash-EXL3 --revision 2.0bpw-mcg
```

The two Engram shards are about 95 GiB each. Keep Hugging Face xet enabled. Do not set `HF_HUB_DISABLE_XET`.

## Serve

The 2× DGX Spark cookbook is [sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark](https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark). Clone that repo, download this revision, build the image on both nodes, and run `./run.sh`.

## Pack

| Field | Value |
|---|---|
| Source | `deepseek-ai/DeepSeek-V4.1-Flash` commit `dba1be0a40aa45a94ad051997016db3960a90277` |
| Quant | EXL3 2.0 bpw, codebook MCG, `quant_method=exl3` |
| Shards | `model-00001-of-00048.safetensors` through `model-00048-of-00048.safetensors` |
| Size on disk | about 334 GB |

Model weights are MIT from DeepSeek. `vllm-exl3` is AGPL-3.0. The recipe image clones it at build.

## Rebuild

The GitHub recipe documents `tools/quantize_experts_exl3.py` and `tools/assemble_pack.sh` if you want to rebuild from the official snapshot.
