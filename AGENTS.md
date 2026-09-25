# AGENTS.md — DeepSeek-V4.1-Flash EXL3 · 2× DGX Spark

Serve an EXL3 pack of `deepseek-ai/DeepSeek-V4.1-Flash` at TP=2. Local image `dsv41-flash-exl3-sm121:canonical-e13`, built by `docker/Dockerfile` from the pinned `vllm/vllm-openai:deepseekv41-flash-0909` digest. Engram stays on NVMe. Stock ExLlamaV3 and stock vLLM wheels do not load `deepseek_v41` / CED / CSA2 / Engram.

Humans read [README.md](README.md).

The public path is clone, `hf download` of `sfxnz/DeepSeek-V4.1-Flash-EXL3` at `2.0bpw-mcg-lmhead-mxfp8`, image build on both nodes, then `./run.sh`. That revision is the stock `2.0bpw-mcg` pack with only `model-00043` re-encoded (lm_head MXFP8). `run.sh` resolves Hub `refs/<rev>` to `snapshots/<commit>/`. It also accepts an assembled pack at `snapshots/<rev>`. `SNAPSHOT_SHA=2.0bpw-mcg` still serves the stock pack; lm_head MXFP8 then turns itself off. Rebuild tools can write `2.0bpw-mul1` and need p2b `cb=2`. Do not switch the serve pin: MUL1 + p2b `cb=2` lost prose decode (23.52 vs 27.98). Pack-only MUL1 is also a decode regression.

## Working rules

- `recipe.yaml` is the source of truth for pins and generated blocks. Edit it, then `python3 kit/render.py`. Do not hand-edit `# BEGIN generated` or `<!-- BEGIN generated` blocks.
- Read unified memory with `free -h`. Never `nvidia-smi` VRAM.
- Exclusive GPUs. Do not start this while another `--gpus all` serve is up. Unload that serve first (`./stop.sh` in its recipe). Stop `dsv41-quant` too — it holds the GPU during pack conversion.
- Pin `NCCL_IB_HCA`. GB10 exposes four HCAs and two are DOWN. Defaults are `enp1s0f1np1` / `rocep1s0f1`. A second rail, `roceP2p1s0f1` / `enP2p1s0f1np1`, is also ACTIVE. The recipe does not use it, and it is unmeasured.
- Default thinking is off. `chat_template_kwargs`: `thinking=false`, `reasoning_effort=low`. Tokenizer/tool/reasoning parsers are `deepseek_v41`.
- DeepJIT is a CUDA/Ascend kernel JIT library, not a serving stack. Official V4.1 kernels in the HF `inference/` tree are TileLang. Do not vendor DeepJIT into this image.
- Stay at EXL3 K=2. Calibrate before raising bits. Do not raise `MAX_NUM_SEQS` before a new occupancy row.
- DSpark k=3 (`NUM_SPECULATIVE_TOKENS=3`). The R15/R17 sweep measured L.A.I.L k5 26.32 / k4 25.87 / k3 28.76 / k2 26.89. `run.sh` derives cudagraph capture sizes from k and `MAX_NUM_SEQS` as {1} ∪ {s·k, s·(k+1)}, which gives `[1,3,4,6,8]`. Matched captures measured +5.5% in R16. k must stay 1..5 (the DSpark block is 5). E6 k=10 collapsed to 12.1.
- `--max-num-batched-tokens` defaults to 8192. E1 measured 2048 at pp@16k −28%. PR 6 measured 8192 at 12,712-token prompts as 755 vs 797 tok/s (−5%) against 2048. `MAX_NUM_BATCHED_TOKENS=2048` is the override for that workload.
- The serve pin is `SNAPSHOT_SHA=2.0bpw-mcg-lmhead-mxfp8`: the MCG codebook plus lm_head MXFP8 (R33 +5.9% L.A.I.L). Do not switch to MUL1. MUL1 + p2b `cb=2` measured 23.52 vs 27.98 prose decode. Prefill 792 vs 797 is flat and does not change the call.
- Round 34 defaults (2026-09-24 review campaign, `results/RESULTS.md`): `DSV41_ENGRAM_WILLNEED=1`, `DSV41_STREAM_FEED=1`, `DSV41_WOA_PREPACK=1` and `DSV41_DSPARK_SPARSE_MARKOV=1`. `DSV41_WOA_PREPACK=1` needs `canonical-e13`: on `canonical-e12` it logs `the lever is OFF` and `AUDIT=strict` fails. To turn a lever off, set it to `0`; an empty value keeps the default. `DSV41_MHC_DECODE_SPLITS` stays 0 (quick quality failed at 40). `DSV41_P2B_SRC_SORT`, `DSV41_DENSE_DG_SMALLM` and dual-rail NCCL stay off (flags.md).
- Never pull, merge or check out in the main checkout (`~/projects/ai-lab/recipes/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark`) while the serve is up. The head container bind-mounts its `docker/patch` and `sitecustomize.py` read-only, so a pull changes head patch code under a running serve and skews it from the worker's `~/.cache/dsv41-patch` copy. Work in a worktree.
- Every env var a `docker/patch/*.py` file reads goes in `FORWARD_ENVS` in `run.sh`. That one list feeds `docker run -e` on the head and the worker ssh line. `tests/test_recipe_ops.py` enforces it. `python3 tests/run_sh_harness.py` prints both `docker run` commands with docker and ssh stubbed.

`ORCHESTRATE=auto` (default): if SSH to `WORKER_HOST` fails, `run.sh` exits 1. Do not start a TP=2 head rank alone.

## Refuse-guards (`run.sh`)

Exits unless the matching `FORCE_UNSAFE_*=1`. All three default to 0. Campaign boots set one for that boot only and record it. Never commit one as a default.

- `QUANTIZATION` not `exl3` (`FORCE_UNSAFE_QUANT`)
- `DSV41_ENGRAM_DISK` not `1` (`FORCE_UNSAFE_ENGRAM`)
- `--max-model-len` above 1048576 (`FORCE_UNSAFE_CTX`)
- `MAX_NUM_SEQS` above 2 (`FORCE_UNSAFE_CTX`)
- `KV_CACHE_MEMORY` above 8 GiB (`FORCE_UNSAFE_CTX`)
- `NUM_SPECULATIVE_TOKENS` not an integer 1..5 when `SPEC=dspark` (`FORCE_UNSAFE_CTX`)

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

`smoke_chat.py` fails unless `choices[0].message.content` matches `323` for `17*19` (`--expect ''` accepts any non-empty content). `smoke_vision.py` must not return HTTP 400 `is not a multimodal model`, and its answer for the solid-red PNG must contain `red`.

## Never touch

- Live HF tokens
- Floating `:latest` on the dsv41 base. Digest is pinned in `recipe.yaml` and the Dockerfile
- Hand-edited generated README / `run.sh` blocks
