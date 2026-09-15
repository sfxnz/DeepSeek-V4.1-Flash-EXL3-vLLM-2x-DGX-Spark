# AGENTS.md — DeepSeek-V4.1-Flash EXL3 · 2× DGX Spark

Serve an EXL3 pack of `deepseek-ai/DeepSeek-V4.1-Flash` at TP=2. Local image `dsv41-flash-exl3-sm121` from the pinned `vllm/vllm-openai:deepseekv41-flash-0909` digest. Engram stays on NVMe. Stock ExLlamaV3 and stock vLLM wheels do not load `deepseek_v41` / CED / CSA2 / Engram.

Humans read [README.md](README.md).

The public path is clone, `hf download` of `sfxnz/DeepSeek-V4.1-Flash-EXL3` at `2.0bpw-mcg`, image build on both nodes, then `./run.sh`. `run.sh` resolves Hub `refs/<rev>` to `snapshots/<commit>/`. It also accepts an assembled pack at `snapshots/2.0bpw-mcg`. Rebuild defaults to MUL1 (`2.0bpw-mul1`) and needs p2b `cb=2`. Pack-only MUL1 is a decode regression.

## Working rules

- `recipe.yaml` is the source of truth for pins and generated blocks. Edit it, then `python3 kit/render.py`. Do not hand-edit `# BEGIN generated` or `<!-- BEGIN generated` blocks.
- Read unified memory with `free -h`. Never `nvidia-smi` VRAM.
- Exclusive GPUs. Do not start this while another `--gpus all` serve is up. Unload that serve first (`./stop.sh` in its recipe). Stop `dsv41-quant` too — it holds the GPU during pack conversion.
- Pin `NCCL_IB_HCA`. GB10 exposes four HCAs and two are DOWN. Defaults are `enp1s0f1np1` / `rocep1s0f1`.
- Default thinking is off. `chat_template_kwargs`: `thinking=false`, `reasoning_effort=low`. Tokenizer/tool/reasoning parsers are `deepseek_v41`.
- DeepJIT is a CUDA/Ascend kernel JIT library, not a serving stack. Official V4.1 kernels in the HF `inference/` tree are TileLang. Do not vendor DeepJIT into this image.
- Stay at EXL3 K=2. Calibrate before raising bits. Do not raise `MAX_NUM_SEQS` before a new occupancy row. DSpark block stays 5.
- Keep `--max-num-batched-tokens` at 2048. spark1+spark2 measured 8192 at 12,712 tokens as 755 vs 797 tok/s (−5%). Do not ship 8192 as a proven prefill upgrade. Override is `MAX_NUM_BATCHED_TOKENS=8192`.

`ORCHESTRATE=auto` (default): if SSH to `WORKER_HOST` fails, `run.sh` exits 1. Do not start a TP=2 head rank alone.

## Refuse-guards (`run.sh`)

Exits unless the matching `FORCE_UNSAFE_*=1`:

- `QUANTIZATION` not `exl3` (`FORCE_UNSAFE_QUANT`)
- `DSV41_ENGRAM_DISK` not `1` (`FORCE_UNSAFE_ENGRAM`)
- `--max-model-len` above 1048576 (`FORCE_UNSAFE_CTX`)
- `MAX_NUM_SEQS` above 2 (`FORCE_UNSAFE_CTX`)
- `KV_CACHE_MEMORY` above 8 GiB (`FORCE_UNSAFE_CTX`)
- `NUM_SPECULATIVE_TOKENS` not divisible by 5 when `SPEC=dspark` (`FORCE_UNSAFE_CTX`)

Native MXFP4 experts are ~130 GiB per TP=2 rank before Engram. That does not fit 121 GiB UMA. Engram pinned in host RAM is the same pool.

## Verify

```bash
python3 -m unittest discover -s tests -q
python3 kit/render.py --check
```

After `./run.sh` is up:

```bash
python3 smoke_chat.py
python3 smoke_vision.py
```

`smoke_chat.py` must have non-empty `choices[0].message.content`. `323` is enough for `17*19`. `smoke_vision.py` must not return HTTP 400 `is not a multimodal model`.

## Never touch

- Live HF tokens
- Floating `:latest` on the dsv41 base. Digest is pinned in `recipe.yaml` and the Dockerfile
- Hand-edited generated README / `run.sh` blocks
