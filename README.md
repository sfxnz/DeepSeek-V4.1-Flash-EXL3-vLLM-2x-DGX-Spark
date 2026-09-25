# Serve DeepSeek-V4.1-Flash EXL3 on 2× DGX Spark

Serve an EXL3 pack of [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) across two NVIDIA DGX Spark (GB10) nodes at tensor-parallel 2.

The pack is [sfxnz/DeepSeek-V4.1-Flash-EXL3](https://huggingface.co/sfxnz/DeepSeek-V4.1-Flash-EXL3) at revision `2.0bpw-mcg` (served through the `2.0bpw-mcg-lmhead-mxfp8` derivative — same pack, only the lm_head tensor re-encoded to mxfp8, see Rebuild). Routed experts are EXL3 2.0 bpw (MCG). Engram stays on NVMe (`DSV41_ENGRAM_DISK=1`). Native MXFP4 and MXFP8 weights are about 511 GB and do not fit 2× Spark UMA.

**Decode campaign 2026-09-20/22 (+44% vs published)**: real-use prose 23 → **33.2 tok/s** (L.A.I.L cell, c=1 t=0.2, pooled n=10), greedy single-stream **39.6 tok/s** (9-run median, acc 2.61), 21.6/23.2 GiB free per Spark after a 32k prefill, zero OOMs in ~30 boots. Won levers: spec k=5→k=3 + matched cudagraph captures, Engram prefetch v3 (pf_hit 100%) + gather v2, NCCL AR-tail set, mem-hygiene bundle, lm_head mxfp8. Full evidence: `results/RESULTS.md` rounds 15–33.

**Review campaign 2026-09-24 (round 34)**: before is s2, a fresh boot of the round-33 config. After is pooled over s8 and s13, the two boots that ran the identical new defaults under the same protocol (medians of the pooled per-run values):
- L.A.I.L fresh: 32.44 → **33.68 tok/s** (+3.8%, n=20; 66.38 → 62.63 ms/step). Per boot 34.51 / 32.44.
- prose c=1: 37.07 → **39.52 tok/s** (+6.6%, n=18; 67.77 → 63.5 ms/step). Per boot 39.87 / 39.18.
- structured c=1: 58.57 → **62.52 tok/s** (+6.7%)
- c=2 aggregate: prose 52.93 → 56.77 (+7.3%), structured 84.04 → 89.44 (+6.4%)
- L.A.I.L after c=2 traffic: 31.91 → 33.75 tok/s (+5.8%, n=20)
- novel-text prefill (s13 only): 8k 184.2 → **838.0 tok/s**, 32k 231.1 → **811.5 tok/s** (Engram WILLNEED read-ahead)
- quick quality eval: pass
- SPARSE_MARKOV's effect on acceptance is open (round-2 ABAB pending).

New default levers: `DSV41_ENGRAM_WILLNEED`, `DSV41_STREAM_FEED`, `DSV41_WOA_PREPACK` and `DSV41_DSPARK_SPARSE_MARKOV`, on image `canonical-e13`. Evidence: `results/RESULTS.md` round 34.

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
hf download sfxnz/DeepSeek-V4.1-Flash-EXL3 --revision 2.0bpw-mcg-lmhead-mxfp8
```

`2.0bpw-mcg-lmhead-mxfp8` is the serve pin: the stock `2.0bpw-mcg` pack with only `model-00043` re-encoded (lm_head to MXFP8). `hf download` writes `refs/2.0bpw-mcg-lmhead-mxfp8` and `snapshots/<commit>/`. `./run.sh` resolves that layout. An assembled pack at `snapshots/2.0bpw-mcg-lmhead-mxfp8` still works. If both exist, `run.sh` uses the Hub commit snapshot. To serve the stock pack, download `--revision 2.0bpw-mcg` and run `SNAPSHOT_SHA=2.0bpw-mcg ./run.sh`; the lm_head MXFP8 path then turns itself off.

If the pack is already in the Hub cache on a node, skip the download on that node.

## Build the image

On both nodes, from this repo:

```bash
docker pull vllm/vllm-openai:deepseekv41-flash-0909@sha256:d84a123255b822fc22508635218000187221794f59c0694c33b0650d1e377d58
docker build -f docker/Dockerfile -t dsv41-flash-exl3-sm121:canonical-e13 docker
```

Stock `vllm/vllm-openai` wheels do not load `DeepseekV41ForCausalLM`. The image starts from the pinned `deepseekv41-flash-0909` digest and overlays Engram-on-disk plus `vllm-exl3`.

If the image `dsv41-flash-exl3-sm121:canonical-e13` is already present, skip the pull and the build on that node. A node that already has `canonical-e12` can layer the round-34 stages on it instead of a full build: `docker build -f docker/Dockerfile.e13 -t dsv41-flash-exl3-sm121:canonical-e13 docker`.

`docker/Dockerfile` builds the canonical serve image `dsv41-flash-exl3-sm121:canonical-e13`, which is the `IMAGE` default. It already applies the E10+E11 keeps (p2b mrow/cfg1/codebook/fshift, b12x smalls, `fix_o_proj_woa_fp8`) that the historical `docker/Dockerfile.e10` → `docker/Dockerfile.e11` chain added. `results/RESULTS.md` round 7 records the rebuild as content-equivalent. Round 34 adds the p2b srcsort build (`DSV41_P2B_SRC_SORT`, default off, byte-identical kernels when off) and the `fix_o_proj_woa_fp8` stage 2 that `DSV41_WOA_PREPACK=1` needs; on `canonical-e12` that lever logs `the lever is OFF` and the strict audit fails. `docker/Dockerfile.e13` layers the same two stages on `canonical-e12`; the Sparks' `canonical-e13` is that build (`sha256:c81762335a12`, `results/RESULTS.md` round 34). The image carries a `dsv41.recipe.patches` label. `run.sh` warns, but still boots, when `IMAGE` lacks the label: a stale local `:latest`, or a canonical-e12 built before the label existed. The experiment Dockerfiles (`docker/Dockerfile.e10`, `docker/Dockerfile.mma`) chain `FROM` the untagged base and are history.

## Run

If another `--gpus all` container is up, stop it first.

On the head node:

```bash
./run.sh
python3 smoke_chat.py
python3 smoke_vision.py
python3 bench_decode.py --phase prose --concurrency 1
python3 tools/measure_lail_prose.py
```

The API is `http://127.0.0.1:8000/v1`. The served model is `deepseek-ai/DeepSeek-V4.1-Flash`. Cap is `MAX_NUM_SEQS=2`. Do not send a third stream.

The promoted recipe additionally ships, all default-on and individually A/B'd (results/RESULTS.md rounds 15–33): `NUM_SPECULATIVE_TOKENS=3` with cudagraph capture sizes `[1,3,4,6,8]`, `MAX_NUM_BATCHED_TOKENS=8192`, the NCCL AR-tail set (`NCCL_BUFFSIZE=1048576`, `NCCL_LL128_BUFFSIZE=262144`, `NCCL_PROTO=^LL128`, `NCCL_MAX_NCHANNELS=8`), the mem-hygiene bundle (`DSV41_DROP_PAGE_CACHE=1`, `DSV41_INDEXER_PREFILL_FACTOR=1`, `DSV41_PREFILL_EMPTY_CACHE_TOKENS=8192`, `DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB=2.5`, `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256`), Engram prefetch v3 + gather v2 + census (`DSV41_ENGRAM_PREFETCH=1`, `DSV41_ENGRAM_GATHER_V2=1`, `DSV41_ENGRAM_CENSUS=1`), and lm_head mxfp8 (`DSV41_LMHEAD_MXFP8=1`, pack revision `2.0bpw-mcg-lmhead-mxfp8` — the stock pack with only `model-00043` re-encoded; `tools/quantize_lmhead_mxfp8.py` builds it from `2.0bpw-mcg` in ~10 min). Round 34 (2026-09-24 review campaign, `results/RESULTS.md`) adds four more default-on levers: `DSV41_ENGRAM_WILLNEED=1` (prefill read-ahead) and `DSV41_STREAM_FEED=1` (weight-load drain), both from the s4 prefill arm; `DSV41_WOA_PREPACK=1`, bit-exact and ABAB-tested (bundled with MHC split-K, which was rejected); and `DSV41_DSPARK_SPARSE_MARKOV=1`, from one boot. Set any of them to `0` to turn it off.

The `smoke_chat.py` default prompt is `What is 17*19? Return only the integer.` Thinking is off. The pass is `content` matching `323` (`--expect ''` accepts any non-empty content). `smoke_vision.py` sends a 64x64 solid-red PNG as OpenAI `image_url`. It must not return HTTP 400 `is not a multimodal model`, and the answer must contain `red`. `bench_decode.py` is streamed greedy, thinking off, 200 completion tokens, 3-run median by default (the headline cell uses `--runs 9`).

If `ORCHESTRATE=auto` (the default) and SSH to `WORKER_HOST` fails, `run.sh` exits 1. It does not start a TP=2 head rank alone.

When you are done:

```bash
./stop.sh
```

## Quality eval

`tests/quality_eval.py` measures output quality against a running serve. It is stdlib-only and uses HTTP only. `--quick` (~6-9 min) covers:

- teacher-forced NLL on 40 public-domain passages (`prompt_logprobs`, with BOS)
- a decode-vs-prefill logprob probe
- 30 tool-call items (JSON-valid, exact-args and no-call rates)
- needle recall at 8k/32k × 3 depths, on generated filler
- a 12×2 greedy self-consistency control
- a c=2 sanity check
- the red-PNG vision check

`--full` adds GSM8K-100 with thinking off, GSM8K-40 with thinking on, MMLU 4×57, and the needle at 128k. The run exits 1 when a gate fails.

```bash
python3 tests/quality_eval.py --quick --out /tmp/q.json \
  --baseline results/2026-09-24-review/quality-baseline/quick.json
```

With `--baseline`, it gates on the following:

| Metric | Gate |
|--------|------|
| NLL | ≤ baseline + max(0.01, 3× repeat noise) nats |
| Decode probe | median \|Δlogprob\| ≤ baseline + 0.05, and greedy-text NLL ≤ baseline + 0.15 |
| Rates | Wilson 95% upper bound with one item of slack ((k+1)/n) ≥ baseline rate |
| Needle | found ≥ baseline |
| Golden flip hazard | ≤ 2× max(baseline A/A hazard, 0.005); the run's own A/A hazard has the same limit |

Vision and c=2 must always pass. `--result saved.json --baseline other.json` re-gates two saved runs offline with no traffic, which is how an A/B compares two boots. Run it serialized: never next to a bench, and never with a third stream. The vendored data and licenses are in `tests/quality/README.md`. The baseline numbers are in `results/2026-09-24-review/quality-baseline/README.md`.

## Defaults

<!-- BEGIN generated defaults from recipe.yaml — edit recipe.yaml and run kit/render.py -->
| Setting | Value |
|---|---|
| Image | `dsv41-flash-exl3-sm121:canonical-e13` |
| Model | `sfxnz/DeepSeek-V4.1-Flash-EXL3` revision `2.0bpw-mcg-lmhead-mxfp8` |
| `--tensor-parallel-size` / `--nnodes` | 2 / 2 |
| `--max-model-len` | 1048576 |
| `--max-num-seqs` | 2 |
| `--max-num-batched-tokens` | 8192 |
| `--kv-cache-dtype` | `fp8` |
| `--kv-cache-memory` | 8589934592 |
| `--quantization` | `exl3` |
| Engram | disk (`DSV41_ENGRAM_DISK=1`) |
| `--block-size` | 64 |
| Speculative | DSpark-3 (`SPEC=dspark`) |
| CUDA graphs | `FULL_AND_PIECEWISE`, capture sizes {1} ∪ {s·k, s·(k+1)} for s ≤ `--max-num-seqs` (`[1,3,4,6,8]` at k=3) (`ENFORCE_EAGER=0`, `DSV41_ALLOW_CUDA_GRAPHS=1`) |
| Engram prefetch / census / gather | `DSV41_ENGRAM_PREFETCH=1` `DSV41_ENGRAM_CENSUS=1` `DSV41_ENGRAM_GATHER_V2=1` |
| Engram prefill read-ahead | `DSV41_ENGRAM_WILLNEED=1` (gv2 calls with >= 512 rows fadvise their pages before the preadv loop) |
| lm_head | MXFP8 (`DSV41_LMHEAD_MXFP8=1`), needs the `2.0bpw-mcg-lmhead-mxfp8` pack; self-disarms on stock `2.0bpw-mcg` |
| Weight-load stream feed | `DSV41_STREAM_FEED=1` |
| Decode levers | `DSV41_WOA_PREPACK=1` (needs the `dsv41-flash-exl3-sm121:canonical-e13` o_proj stage) `DSV41_DSPARK_SPARSE_MARKOV=1` (top-k 256) |
| NCCL AR-tail set | `NCCL_BUFFSIZE=1048576` `NCCL_LL128_BUFFSIZE=262144` `NCCL_PROTO=^LL128` `NCCL_MAX_NCHANNELS=8` |
| Memory hygiene | `DSV41_DROP_PAGE_CACHE=1` `DSV41_INDEXER_PREFILL_FACTOR=1` `DSV41_PREFILL_EMPTY_CACHE_TOKENS=8192` `DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB=2.5` `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256` |
| Tokenizers / tools / reasoning | `deepseek_v41` |
| Vision | on (`LANGUAGE_MODEL_ONLY=0`) |
| `--mm-encoder-tp-mode` | data |
| Default thinking | `thinking=false`, `reasoning_effort=low` |
| Post-ready | engagement audit of both ranks' logs (`AUDIT=warn`; `strict` fails the boot, `off` skips), then greedy, t=0.7 and ~3k-token nonce warmup requests (`WARMUP=1`) |
| API | `http://<head>:8000/v1` |
| Container | `dsv41-flash-exl3` |
| Master port | 29524 |
<!-- END generated defaults -->

## Measured on 2× DGX Spark

`bench_decode.py` is streamed greedy, 200 completion tokens, 3-run median; the round-34 table below pools 9 runs from each of two identical-config boots (39.52 tok/s prose c=1, n=18; the prompt stops naturally at ~78 tokens, so the cell forces `ignore_eos` and 59% of it is post-EOS text). `tools/measure_lail_prose.py` matches L.A.I.L streams prose (512 tokens, temperature 0.2) — this is the real-world-use cell: 33.68 tok/s (n=20), +46% vs the pre-campaign published recipe (23 tok/s). The table pools the round-34 boots `s8-S-sparse-markov` and `s13-promote-final`, which both ran exactly the current defaults (`results/2026-09-24-review/campaign/pooled-s8-s13.json`). Default is DSpark-3 with matched cudagraph captures, vision on. These cells are the MCG pack with the lm_head-mxfp8 head (`2.0bpw-mcg-lmhead-mxfp8`) on native p2b `cb=1`. A MUL1 pack measured and lost prose decode at every bit-width tested (see below). The KV pool is 8 GiB. Every accepted/rejected experiment lives in `results/RESULTS.md` (34 rounds); run `benches/micro.sh` and `benches/e2e.sh` to reproduce cells, and `tests/correctness.sh --full` for the quality gate.

`MAX_NUM_BATCHED_TOKENS` history: at 12.7k-token prompts 8192 measured −5% vs 2048 (`evidence/pr6-batched-8192/`), but on the campaign's prose/prefill cells 8192 was re-measured across rounds 15–33 as part of the promoted config — every kept lever was A/B'd on top of it. It ships as the default now; 2048 remains available for long-prompt-heavy workloads.

MUL1 + p2b `cb=2` (`2.0bpw-mul1` K=2 on `dsv41-flash-exl3-sm121:cb2`) lost prose decode: 23.52 vs 27.98 MCG. Runs 26.43 / 23.52 / 23.17 vs baseline 26.34 / 27.98 / 31.77. MUL1 median and two of three runs sit below the worst baseline run. 12,712-token prefill 792 vs 797 (flat). Rebuild tools stay. Serve stays on the MCG pack (`2.0bpw-mcg-lmhead-mxfp8`). See `evidence/pr6-mul1-cb2/`.

<!-- BEGIN generated measured from recipe.yaml — edit recipe.yaml and run kit/render.py -->
| Phase | Concurrency | Decode tok/s (median per stream) | Aggregate tok/s | TTFT p50 |
|---|---|---:|---:|---:|
| prose | 1 | 39.52 | 39.52 | 0.25 s |
| prose | 2 | 29.03 | 56.77 | 0.28 s |
| structured | 1 | 62.52 | 62.50 | 0.18 s |
| structured | 2 | 45.28 | 89.44 | 0.28 s |
| lail_prose | 1 | 33.68 | 33.68 | 0.34 s |
<!-- END generated measured -->

## Rebuild the pack

If you already downloaded `sfxnz/DeepSeek-V4.1-Flash-EXL3` at revision `2.0bpw-mcg`, skip the expert rebuild. The published pack is the serve path. Two rebuilds were measured end-to-end and **both lost decode**:

- **MUL1 + p2b `cb=2`** (`2.0bpw-mul1`, K=2): 23.52 vs 27.98 MCG on our kit. Do not serve it. Pack-only MUL1 without p2b `cb=2` drops native fused MoE onto generic `exl3_moe` and is a decode regression.
- **Her 2.9 bpw mul1 pack** (MiaAI-Lab kit, 4 boots, results/2026-09-20-mul1-lane/): stock k=3 prose 16.35, spec-off 23.64 vs our MCG 34+ — its MTP drafter is quantized to 4-bit EXL3 (`mtp_bits: 4`, acceptance 1.33 vs 2.77 source-precision). Her k=3 numbers are not reproducible on that pack; the deficit is a pack property, not a flag.

The lm_head-mxfp8 pack is the one derivative that **won** (+5.9% L.A.I.L, round 33). It re-encodes only `model-00043` of the stock pack and is published as the Hub branch `2.0bpw-mcg-lmhead-mxfp8`, which `./run.sh` downloads by default. To rebuild it, run the tool on a copy of the stock snapshot (it edits in place; needs torch + safetensors, e.g. inside the image):

```bash
S=~/.cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-EXL3/snapshots
cp -a "$S/<stock 2.0bpw-mcg snapshot>" "$S/2.0bpw-mcg-lmhead-mxfp8"
python3 tools/quantize_lmhead_mxfp8.py --snapshot "$S/2.0bpw-mcg-lmhead-mxfp8"
```

Stay at K=2 for expert re-encodes. Calibrate (activation Hessian / official convert) before raising bits. Do not pass `--hq` or `bits!=2` here. This recipe does not ship a calibration harness.

Exclusive GPU. Stop any quant container before `./run.sh`.

1. Download the official snapshot.

```bash
python3 tools/download_official.py
```

The commit is `dba1be0a40aa45a94ad051997016db3960a90277`. The two Engram shards are about 95 GiB each and need `hf_xet`. Do not set `HF_HUB_DISABLE_XET`.

2. Hardlink the non-expert shards on the host. Quantize routed experts inside image `dsv41-flash-exl3-sm121:canonical-e12` with exclusive GPU. Host Python does not have ExLlamaV3. Use `--codebook mul1 --greedy --beam 16`. Split expert shards 3-22 and 23-42 across the two Sparks. Destination is `snapshots/2.0bpw-mul1`.

```bash
python3 tools/quantize_experts_exl3.py --codebook mul1 --link-only

# spark1: expert shards 3-22, inside the recipe image
python3 tools/quantize_experts_exl3.py \
  --codebook mul1 --allow-partial --batch 8 --greedy --beam 16 --only-files $(python3 -c "print(' '.join(f'model-{i:05d}-of-00048.safetensors' for i in range(3,23)))")

# spark2: expert shards 23-42, inside the recipe image
python3 tools/quantize_experts_exl3.py \
  --codebook mul1 --allow-partial --batch 8 --greedy --beam 16 --only-files $(python3 -c "print(' '.join(f'model-{i:05d}-of-00048.safetensors' for i in range(23,43)))")
```

3. Merge the two node outputs.

```bash
CODEBOOK=mul1 bash tools/assemble_pack.sh
```

## License

MIT for the scripts. Model weights are MIT from DeepSeek. `vllm-exl3` is AGPL-3.0.
