# Serve DeepSeek-V4.1-Flash EXL3 on 2× DGX Spark

Serve an EXL3 pack of [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) across two NVIDIA DGX Spark (GB10) nodes at tensor-parallel 2.

The pack is [sfxnz/DeepSeek-V4.1-Flash-EXL3](https://huggingface.co/sfxnz/DeepSeek-V4.1-Flash-EXL3) at revision `2.0bpw-mcg-viterbi-lmhead-mxfp8`. It has the same format and bytes as the stock `2.0bpw-mcg` pack, but the routed experts are re-encoded with tail-biting Viterbi plus a scale refit (relative error 0.262 vs 0.378), and the lm_head tensor is in MXFP8. See Download the pack and Rebuild. Routed experts are EXL3 2.0 bpw (MCG). Engram stays on NVMe (`DSV41_ENGRAM_DISK=1`). Native MXFP4 and MXFP8 weights are about 511 GB and do not fit 2× Spark UMA.

**Decode campaign 2026-09-20/22 (+44% vs published)**: real-use prose 23 → **33.2 tok/s** (L.A.I.L cell, c=1 t=0.2, pooled n=10), greedy single-stream **39.6 tok/s** (9-run median, acc 2.61), 21.6/23.2 GiB free per Spark after a 32k prefill, zero OOMs in ~30 boots. Won levers: spec k=5→k=3 + matched cudagraph captures, Engram prefetch v3 (pf_hit 100%) + gather v2, NCCL AR-tail set, mem-hygiene bundle, lm_head mxfp8. Full evidence: `results/RESULTS.md` rounds 15–33.

**Review campaign 2026-09-24 (round 34)**: before is s2, a fresh boot of the round-33 config. After is pooled over s8 and s13, the two boots that ran the identical new defaults under the same protocol (medians of the pooled per-run values):
- L.A.I.L fresh: 32.44 → **33.68 tok/s** (+3.8%, n=20; 66.38 → 62.63 ms/step). Per boot 34.51 / 32.44.
- prose c=1: 37.07 → **39.52 tok/s** (+6.6%, n=18; 67.77 → 63.5 ms/step). Per boot 39.87 / 39.18.
- structured c=1: 58.57 → **62.52 tok/s** (+6.7%)
- c=2 aggregate: prose 52.93 → 56.77 (+7.3%), structured 84.04 → 89.44 (+6.4%)
- L.A.I.L after c=2 traffic: 31.91 → 33.75 tok/s (+5.8%, n=20)
- novel-text prefill (s13 only): 8k 184.2 → **838.0 tok/s**, 32k 231.1 → **811.5 tok/s** (Engram WILLNEED read-ahead)
- quick quality eval: pass
- SPARSE_MARKOV's effect on acceptance is open (round-2 ABAB pending; settled in round 35: it stays).

**Round-3 kernels 2026-09-26/27 (round 35)**: ten bit-exact (COOP: within 1 fp16 ulp) decode kernels promoted as one bundle, plus `DSV41_DSPARK_SPARSE_MARKOV_TOPK=1024`, on image `canonical-e14`. Before is arm A, the round-34 defaults on the same image (2 boots). After is the final config (3 boots: S1024-1, S1024-2 and the validation boot final-1). Values are medians of the per-boot medians; the per-boot ranges do not overlap on any decode row:
- L.A.I.L fresh: 33.95 → **43.17 tok/s** (+27.1%; 62.79 → 49.00 ms/step at unchanged acceptance, 2.138 → 2.133). Per boot 43.17 / 43.62 / 42.74 vs 33.69 / 34.21.
- prose_long c=1 (natural finish, no post-EOS text): 33.00 → **43.73 tok/s** (+32.5%; 63.34 → 48.98 ms/step)
- four_numbers prose c=1: 39.16 → **52.52 tok/s** (+34.1%); structured c=1 62.19 → **83.81** (+34.8%)
- c=2 aggregate: prose 56.70 → 81.69 (+44.1%), structured 89.04 → 138.34 (+55.4%), prose_long 47.40 → 65.62 (+38.4%)
- prefill unchanged within noise (pp_novel 8k/32k 821.4 / 806.3)
- quality: quick PASS on all three boots; full PASS on final-1 (GSM8K 94/100, thinking 39/40, MMLU 199/228, needle 9/9)
- Not promoted: the five KP add-ons (no tok/s gain beyond noise). Rejected as written: the Viterbi re-encoded pack (golden-hazard gate and L.A.I.L). Round 36 adopted it (below).

**Viterbi pack 2026-09-27 (round 36)**: the default pack is now `2.0bpw-mcg-viterbi-lmhead-mxfp8`. Format and kernels are the same; the routed experts are re-encoded (tail-biting Viterbi + scale refit). Round 35 rejected it as its rules were written. The protocol owner adopted it on its quality evidence, and the golden-hazard gate now applies only to same-pack levers (ARMS.md step 6). Before is the round-35 final config on the old pack (S1024-1, S1024-2, final-1). After is the new defaults (V-1, V-2). The two sets were not interleaved. Values are medians of the per-boot medians:
- quality: NLL 0.243 → **0.139** nats/token (−42%, 39/40 passages lower, paired), GSM8K 94 → 97 / 96, MMLU 199 → 205 / 202, tools exact args 21/22 → 22/22; V-2 passes quick and full against the new baseline
- L.A.I.L fresh: 43.17 → 42.58 tok/s (−1.4%, boot-to-boot noise 0.88); 49.00 → 48.78 ms/step; acceptance 2.133 → 2.115
- prose_long c=1: 43.73 → 43.80 tok/s (+0.2%, noise 2.75); four_numbers prose c=1 52.52 → 50.51 (−3.8%; acceptance 2.513 → 2.416)
- structured c=2 aggregate: 138.34 → 156.29 (+13.0%; both streams run at acceptance 4.0)
- c=1 ms/step and prefill: unchanged within noise

New default levers (round 35): `DSV41_P2B_COOP=2`, `DSV41_DENSE_GEMV=1`, `DSV41_MHC_DET_SPLITS=16`, `DSV41_ENGRAM_NATIVE_STAGE=1`, `DSV41_ENGRAM_EARLY_HASH=1`, `DSV41_ATTN_T2R_DEDUP=1`, `DSV41_SWA_META_FUSED=1`, `DSV41_MOE_PREP_FUSED=1`, `DSV41_CANDIDATE_MASK_BOUNDED=1`, `DSV41_INDEXER_WP_GEMV=1` and `DSV41_DSPARK_SPARSE_MARKOV_TOPK=1024`, on image `canonical-e14`. Evidence: `results/RESULTS.md` round 35 (per-lever attribution in `results/2026-09-26-serve-r3/P3/attribution.txt`). Round 34's (`DSV41_ENGRAM_WILLNEED`, `DSV41_STREAM_FEED`, `DSV41_WOA_PREPACK`, `DSV41_DSPARK_SPARSE_MARKOV`) stay on.

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
hf download sfxnz/DeepSeek-V4.1-Flash-EXL3 --revision 2.0bpw-mcg-viterbi-lmhead-mxfp8
```

`2.0bpw-mcg-viterbi-lmhead-mxfp8` is the serve pin (round 36). It is `2.0bpw-mcg-lmhead-mxfp8` with all 40 routed-expert shards re-encoded from the MXFP4 source: tail-biting Viterbi plus a closed-form scale refit, the same 2.0 bpw MCG K=2 format, and the lm_head MXFP8 `model-00043` unchanged. Against round 35's pin it has 42% lower teacher-forced NLL on the fixed passages (0.139 vs 0.243 nats/token, 39 of 40 passages lower), GSM8K 97/97 vs 95/94 and MMLU 206/204 vs 196/198 (`results/RESULTS.md` rounds 35 and 36). `hf download` writes `refs/2.0bpw-mcg-viterbi-lmhead-mxfp8` and `snapshots/<commit>/`. `./run.sh` resolves that layout. An assembled pack at `snapshots/2.0bpw-mcg-viterbi-lmhead-mxfp8` still works. If both exist, `run.sh` uses the Hub commit snapshot. Previous revisions stay on the Hub. `2.0bpw-mcg-lmhead-mxfp8` was the serve pin in rounds 33-35: the stock pack with only `model-00043` re-encoded (lm_head to MXFP8). To serve it, download `--revision 2.0bpw-mcg-lmhead-mxfp8` and run `SNAPSHOT_SHA=2.0bpw-mcg-lmhead-mxfp8 ./run.sh`. `2.0bpw-mcg` is the stock pack. To serve it, download `--revision 2.0bpw-mcg` and run `SNAPSHOT_SHA=2.0bpw-mcg ./run.sh`; the lm_head MXFP8 path then turns itself off.

If the pack is already in the Hub cache on a node, skip the download on that node.

## Build the image

On both nodes, from this repo:

```bash
docker pull vllm/vllm-openai:deepseekv41-flash-0909@sha256:d84a123255b822fc22508635218000187221794f59c0694c33b0650d1e377d58
docker build -f docker/Dockerfile -t dsv41-flash-exl3-sm121:canonical-e14 docker
```

Stock `vllm/vllm-openai` wheels do not load `DeepseekV41ForCausalLM`. The image starts from the pinned `deepseekv41-flash-0909` digest and overlays Engram-on-disk plus `vllm-exl3`.

If the image `dsv41-flash-exl3-sm121:canonical-e14` is already present, skip the pull and the build on that node. A node that already has `canonical-e13` can layer the round-35 stages on it instead of a full build: `docker build -f docker/Dockerfile.e14 -t dsv41-flash-exl3-sm121:canonical-e14 docker`. A node with only `canonical-e12` builds `docker/Dockerfile.e13` as `canonical-e13` first.

`docker/Dockerfile` builds the canonical serve image `dsv41-flash-exl3-sm121:canonical-e14`, which is the `IMAGE` default. It already applies the E10+E11 keeps (p2b mrow/cfg1/codebook/fshift, b12x smalls, `fix_o_proj_woa_fp8`) that the historical `docker/Dockerfile.e10` → `docker/Dockerfile.e11` chain added. `results/RESULTS.md` round 7 records the rebuild as content-equivalent. Round 34 adds the p2b srcsort build (`DSV41_P2B_SRC_SORT`, default off, byte-identical kernels when off) and the `fix_o_proj_woa_fp8` stage 2 that `DSV41_WOA_PREPACK=1` needs; on `canonical-e12` that lever logs `the lever is OFF` and the strict audit fails. `docker/Dockerfile.e13` layers the same two stages on `canonical-e12`; the Sparks' `canonical-e13` is that build (`sha256:c81762335a12`, `results/RESULTS.md` round 34). Round 35 (the k3 round-3 kernel work, `perf/kernels-r3`) adds the p2b coop and dataflow builds (`DSV41_P2B_COOP=1|2`; with the env unset or `0` the kernels are byte-identical to `canonical-e13`'s). `docker/Dockerfile.e14` layers them on `canonical-e13`; the Sparks' `canonical-e14` is that build, the round-3 serve arms' `review-e14` (`sha256:3a002b55c9bc`) tagged on both nodes (`results/RESULTS.md` round 35). `DSV41_P2B_COOP=2`, a default since round 35, needs it: on `canonical-e13` the boot logs `the lever is OFF. Rebuild docker/Dockerfile.e14` and `AUDIT=strict` fails. The other round-3 kernels compile at first use from the mounted `docker/patch` and need no build stage. The image carries a `dsv41.recipe.patches` label. `run.sh` warns, but still boots, when `IMAGE` lacks the label: a stale local `:latest`, or a canonical-e12 built before the label existed. The experiment Dockerfiles (`docker/Dockerfile.e10`, `docker/Dockerfile.mma`) chain `FROM` the untagged base and are history.

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

The promoted recipe additionally ships, all default-on and individually A/B'd (results/RESULTS.md rounds 15–33): `NUM_SPECULATIVE_TOKENS=3` with cudagraph capture sizes `[1,3,4,6,8]`, `MAX_NUM_BATCHED_TOKENS=8192`, the NCCL AR-tail set (`NCCL_BUFFSIZE=1048576`, `NCCL_LL128_BUFFSIZE=262144`, `NCCL_PROTO=^LL128`, `NCCL_MAX_NCHANNELS=8`), the mem-hygiene bundle (`DSV41_DROP_PAGE_CACHE=1`, `DSV41_INDEXER_PREFILL_FACTOR=1`, `DSV41_PREFILL_EMPTY_CACHE_TOKENS=8192`, `DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB=2.5`, `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256`), Engram prefetch v3 + gather v2 + census (`DSV41_ENGRAM_PREFETCH=1`, `DSV41_ENGRAM_GATHER_V2=1`, `DSV41_ENGRAM_CENSUS=1`), and lm_head mxfp8 (`DSV41_LMHEAD_MXFP8=1`, pack revision `2.0bpw-mcg-lmhead-mxfp8` — the stock pack with only `model-00043` re-encoded; `tools/quantize_lmhead_mxfp8.py` builds it from `2.0bpw-mcg` in ~10 min). Round 34 (2026-09-24 review campaign, `results/RESULTS.md`) adds four more default-on levers: `DSV41_ENGRAM_WILLNEED=1` (prefill read-ahead) and `DSV41_STREAM_FEED=1` (weight-load drain), both from the s4 prefill arm; `DSV41_WOA_PREPACK=1`, bit-exact and ABAB-tested (bundled with MHC split-K, which was rejected); and `DSV41_DSPARK_SPARSE_MARKOV=1`, from one boot. Round 35 (the k3 round-3 kernels, `results/RESULTS.md`) adds the kernel bundle `DSV41_P2B_COOP=2` (coop dataflow MoE kernel, needs `canonical-e14`), `DSV41_DENSE_GEMV=1`, `DSV41_MHC_DET_SPLITS=16`, `DSV41_ENGRAM_NATIVE_STAGE=1`, `DSV41_ENGRAM_EARLY_HASH=1`, `DSV41_ATTN_T2R_DEDUP=1`, `DSV41_SWA_META_FUSED=1`, `DSV41_MOE_PREP_FUSED=1`, `DSV41_CANDIDATE_MASK_BOUNDED=1` and `DSV41_INDEXER_WP_GEMV=1` (A, K, KC interleaved ABAB: −14.5 to −15.8 ms/step at c=1), and `DSV41_DSPARK_SPARSE_MARKOV_TOPK=1024` (its own ABAB). Set any of them to `0` to turn it off (`DSV41_DSPARK_SPARSE_MARKOV_TOPK=256` restores the round-34 width).

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
  --baseline results/2026-09-27-viterbi-adopt/quality-baseline/quick.json
```

With `--baseline`, it gates on the following:

| Metric | Gate |
|--------|------|
| NLL | ≤ baseline + max(0.01, 3× repeat noise) nats |
| Decode probe | median \|Δlogprob\| ≤ baseline + 0.05, and greedy-text NLL ≤ baseline + 0.15 |
| Rates | Wilson 95% upper bound with one item of slack ((k+1)/n) ≥ baseline rate |
| Needle | found ≥ baseline |
| Golden flip hazard | ≤ 2× max(baseline A/A hazard, 0.005); the run's own A/A hazard has the same limit |

The golden flip hazard compares greedy text with the baseline's own greedy run, so it applies only when the baseline ran the same pack (weights). It gates kernel and numerics levers. A pack change is judged on paired NLL over the fixed passages, paired task scores and A/A self-consistency, and then the baseline is re-recorded on the new pack (ARMS.md step 6). Vision and c=2 must always pass. `--result saved.json --baseline other.json` re-gates two saved runs offline with no traffic, which is how an A/B compares two boots. Run it serialized: never next to a bench, and never with a third stream. The vendored data and licenses are in `tests/quality/README.md`. The baseline for the default pack `2.0bpw-mcg-viterbi-lmhead-mxfp8` was recorded on boot V-1 of round 36; its numbers are in `results/2026-09-27-viterbi-adopt/quality-baseline/README.txt`. The baseline for `2.0bpw-mcg-lmhead-mxfp8` (rounds 33-35) is `results/2026-09-24-review/quality-baseline/`.

## Defaults

<!-- BEGIN generated defaults from recipe.yaml — edit recipe.yaml and run kit/render.py -->
| Setting | Value |
|---|---|
| Image | `dsv41-flash-exl3-sm121:canonical-e14` |
| Model | `sfxnz/DeepSeek-V4.1-Flash-EXL3` revision `2.0bpw-mcg-viterbi-lmhead-mxfp8` |
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
| lm_head | MXFP8 (`DSV41_LMHEAD_MXFP8=1`), needs an lm_head-MXFP8 pack (`2.0bpw-mcg-viterbi-lmhead-mxfp8` or `2.0bpw-mcg-lmhead-mxfp8`); self-disarms on stock `2.0bpw-mcg` |
| Weight-load stream feed | `DSV41_STREAM_FEED=1` |
| Decode levers | `DSV41_WOA_PREPACK=1` (needs the `dsv41-flash-exl3-sm121:canonical-e14` o_proj stage) `DSV41_DSPARK_SPARSE_MARKOV=1` `DSV41_DSPARK_SPARSE_MARKOV_TOPK=1024` |
| Round-3 kernel bundle | `DSV41_P2B_COOP=2` (coop dataflow MoE kernel, needs the `dsv41-flash-exl3-sm121:canonical-e14` build) `DSV41_DENSE_GEMV=1` `DSV41_MHC_DET_SPLITS=16` `DSV41_ENGRAM_NATIVE_STAGE=1` `DSV41_ENGRAM_EARLY_HASH=1` `DSV41_ATTN_T2R_DEDUP=1` `DSV41_SWA_META_FUSED=1` `DSV41_MOE_PREP_FUSED=1` `DSV41_CANDIDATE_MASK_BOUNDED=1` `DSV41_INDEXER_WP_GEMV=1` (set one to `0` to turn it off) |
| NCCL AR-tail set | `NCCL_BUFFSIZE=1048576` `NCCL_LL128_BUFFSIZE=262144` `NCCL_PROTO=^LL128` `NCCL_MAX_NCHANNELS=8` |
| Memory hygiene | `DSV41_DROP_PAGE_CACHE=1` `DSV41_INDEXER_PREFILL_FACTOR=1` `DSV41_PREFILL_EMPTY_CACHE_TOKENS=8192` `DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB=2.5` `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256` |
| Tokenizers / tools / reasoning | `deepseek_v41` |
| Vision | on (`LANGUAGE_MODEL_ONLY=0`) |
| `--mm-encoder-tp-mode` | data |
| Default thinking | `thinking=false`, `reasoning_effort=low` |
| Post-ready | engagement audit of both ranks' logs (`AUDIT=warn`; `strict` fails the boot, `off` skips), then greedy, t=0.7, ~3k/~1k/~300-token nonce prefill and small-image warmup requests (`WARMUP=1`) |
| API | `http://<head>:8000/v1` |
| Container | `dsv41-flash-exl3` |
| Master port | 29524 |
<!-- END generated defaults -->

## Measured on 2× DGX Spark

`bench_decode.py` is streamed greedy, 200 completion tokens, 3-run median; the round-36 table below pools 9 runs from each of two identical-config boots (50.41 tok/s prose c=1, n=18; the prompt stops naturally at ~78-86 tokens, so the cell forces `ignore_eos` and about 60% of it is post-EOS text). `tools/measure_lail_prose.py` matches L.A.I.L streams prose (512 tokens, temperature 0.2) — this is the real-world-use cell: 42.58 tok/s (n=20), +85% vs the pre-campaign published recipe (23 tok/s). The table pools the round-36 boots `V-1` and `V-2`, which both ran exactly the current defaults (`results/2026-09-27-viterbi-adopt/round36-headline.json`; round 35's table on the previous pack was 52.97 / 42.98 on `results/2026-09-26-serve-r3/round35-headline.json`). Default is DSpark-3 with matched cudagraph captures, vision on. These cells are the Viterbi-re-encoded MCG pack with the lm_head-mxfp8 head (`2.0bpw-mcg-viterbi-lmhead-mxfp8`) on native p2b `cb=1`. A MUL1 pack measured and lost prose decode at every bit-width tested (see below). The KV pool is 8 GiB. Every accepted/rejected experiment lives in `results/RESULTS.md` (36 rounds); run `benches/micro.sh` and `benches/e2e.sh` to reproduce cells, and `tests/correctness.sh --full` for the quality gate.

`MAX_NUM_BATCHED_TOKENS` history: at 12.7k-token prompts 8192 measured −5% vs 2048 (`evidence/pr6-batched-8192/`), but on the campaign's prose/prefill cells 8192 was re-measured across rounds 15–33 as part of the promoted config — every kept lever was A/B'd on top of it. It ships as the default now; 2048 remains available for long-prompt-heavy workloads.

MUL1 + p2b `cb=2` (`2.0bpw-mul1` K=2 on `dsv41-flash-exl3-sm121:cb2`) lost prose decode: 23.52 vs 27.98 MCG. Runs 26.43 / 23.52 / 23.17 vs baseline 26.34 / 27.98 / 31.77. MUL1 median and two of three runs sit below the worst baseline run. 12,712-token prefill 792 vs 797 (flat). Rebuild tools stay. Serve stays on the MCG codebook (`2.0bpw-mcg-viterbi-lmhead-mxfp8`). See `evidence/pr6-mul1-cb2/`.

<!-- BEGIN generated measured from recipe.yaml — edit recipe.yaml and run kit/render.py -->
| Phase | Concurrency | Decode tok/s (median per stream) | Aggregate tok/s | TTFT p50 |
|---|---|---:|---:|---:|
| prose | 1 | 50.41 | 50.39 | 0.23 s |
| prose | 2 | 41.57 | 81.02 | 0.25 s |
| structured | 1 | 84.20 | 84.18 | 0.21 s |
| structured | 2 | 78.17 | 156.29 | 0.23 s |
| lail_prose | 1 | 42.58 | 42.58 | 0.32 s |
<!-- END generated measured -->

## Rebuild the pack

If you already downloaded `sfxnz/DeepSeek-V4.1-Flash-EXL3` at revision `2.0bpw-mcg`, skip the expert rebuild. The published pack is the serve path. Two rebuilds were measured end-to-end and **both lost decode**:

- **MUL1 + p2b `cb=2`** (`2.0bpw-mul1`, K=2): 23.52 vs 27.98 MCG on our kit. Do not serve it. Pack-only MUL1 without p2b `cb=2` drops native fused MoE onto generic `exl3_moe` and is a decode regression.
- **Her 2.9 bpw mul1 pack** (MiaAI-Lab kit, 4 boots, results/2026-09-20-mul1-lane/): stock k=3 prose 16.35, spec-off 23.64 vs our MCG 34+ — its MTP drafter is quantized to 4-bit EXL3 (`mtp_bits: 4`, acceptance 1.33 vs 2.77 source-precision). Her k=3 numbers are not reproducible on that pack; the deficit is a pack property, not a flag.

Two derivatives were adopted. The lm_head-mxfp8 pack **won** on speed (+5.9% L.A.I.L, round 33); it re-encodes only `model-00043` of the stock pack and is published as the Hub branch `2.0bpw-mcg-lmhead-mxfp8`. The Viterbi pack (round 36, the default `2.0bpw-mcg-viterbi-lmhead-mxfp8`) builds on it and was adopted on quality: NLL −42%, with L.A.I.L within noise and short-prose c=1 −3.8%. `tools/requant_full.py` re-encodes the 46,080 routed-expert tensors from the MXFP4 source across both Sparks with exllamav3's tail-biting Viterbi and `refit_scales`, and keeps that `model-00043` (round 35 in `results/RESULTS.md`). To rebuild the lm_head pack, run the tool on a copy of the stock snapshot (it edits in place; needs torch + safetensors, e.g. inside the image):

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
