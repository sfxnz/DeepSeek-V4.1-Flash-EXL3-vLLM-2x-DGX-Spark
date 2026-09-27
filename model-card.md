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

EXL3 pack of [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash). Routed experts are 2.0 bpw MCG. In the default revision they are re-encoded with tail-biting Viterbi plus a scale refit, and the lm_head is MXFP8. Other tensors keep the source dtypes. Engram tables stay on NVMe at serve time.

This pack exists so 2× NVIDIA DGX Spark (GB10) can serve the model at tensor-parallel 2. Native MXFP4/MXFP8 is about 511 GB and does not fit 2× Spark UMA.

## Download

```bash
hf download sfxnz/DeepSeek-V4.1-Flash-EXL3 --revision 2.0bpw-mcg-viterbi-lmhead-mxfp8
```

The two Engram shards are about 95 GiB each. Keep Hugging Face xet enabled. Do not set `HF_HUB_DISABLE_XET`. The recipe measured the Viterbi revision against `2.0bpw-mcg-lmhead-mxfp8` in an interleaved 2+2 boot comparison (round 35 P4). Teacher-forced NLL on 40 fixed passages was 0.139 vs 0.243 nats/token (−42%, 39 of 40 passages lower). GSM8K-100 read 97/97 vs 95/94, MMLU 4×57 206/204 vs 196/198, and tool-call exact args 88/88 vs 84/88. Decode ms per step is unchanged; speed moves only through speculative acceptance (prose_long +7.0%, L.A.I.L prose −2.4% in that comparison).

## Serve

The 2× DGX Spark cookbook is [sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark](https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark). Clone that repo, download this revision, build the image on both nodes, and run `./run.sh`.

## Pack

| Field | Value |
|---|---|
| Source | `deepseek-ai/DeepSeek-V4.1-Flash` commit `dba1be0a40aa45a94ad051997016db3960a90277` |
| Quant | EXL3 2.0 bpw, codebook MCG, `quant_method=exl3` |
| Revision `2.0bpw-mcg-viterbi-lmhead-mxfp8` (serve pin since round 36) | All 46,080 routed-expert tensors re-encoded from the MXFP4 source: exllamav3 1.5.1 tail-biting Viterbi + `refit_scales`, same 2.0 bpw MCG K=2 format and bytes, mean relative error 0.262 vs 0.378. `model-00043` (lm_head MXFP8) is the same as in `2.0bpw-mcg-lmhead-mxfp8` |
| Revision `2.0bpw-mcg-lmhead-mxfp8` (serve pin in rounds 33-35) | `2.0bpw-mcg` with only `model-00043` re-encoded (lm_head MXFP8, +5.9% L.A.I.L in round 33) |
| Revision `2.0bpw-mcg` | The original MCG pack; the lm_head MXFP8 path turns itself off on it |
| Shards | `model-00001-of-00048.safetensors` through `model-00048-of-00048.safetensors` |
| Size on disk | about 334 GB |

Model weights are MIT from DeepSeek. `vllm-exl3` is AGPL-3.0. The recipe image clones it at build.

## Rebuild

The GitHub recipe documents `tools/quantize_experts_exl3.py --codebook mul1` and `tools/assemble_pack.sh`. That writes revision `2.0bpw-mul1`. Serve pins `2.0bpw-mcg-viterbi-lmhead-mxfp8`, built by the recipe's `tools/requant_full.py` (Viterbi re-encode of the routed experts on top of `2.0bpw-mcg-lmhead-mxfp8`). MUL1 + p2b `cb=2` lost prose decode (23.52 vs 27.98). A pack-only swap without `cb=2` also drops the fused path. Stay at K=2 and calibrate before raising bits.
