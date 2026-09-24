# draft-subvocab: CPU feasibility (2026-09-24)

**Verdict: NO-GO.** At N=32k, the sub-vocab covers 87-94% of tokens in English prose, chat and code. The corrections set a bar of about 99% coverage to net a gain, and at minimum 97% for prose and code. On the measured coverage, the modelled tokens/step loss is larger than the draft-side saving on every English and code corpus, at every N from 16k to 64k. The only corpus that comes out slightly positive is Chinese Wikipedia, and only at 48k-64k (+0.7 to +1.3%), which is within noise. No GPU run is proposed.

## What would change (per corrections)

- The live path drafts greedily. `DSV41_DSPARK_SOFTMAX_VERIFY=0` means `draft_logits` is None, so the draft token comes from `map_draft_to_target(argmax)` (speculator.py:127-128). The d2t scatter is not on the live path.
- The draft head is the aliased MXFP8 **target** head (lmhead_mxfp8.py). Setting `draft_vocab_size` < vocab turns off the alias (spec_decode/dspark/utils.py:104-117), so the patch would have to supply its own MXFP8 head rows.
- A target token outside the sub-vocab can never be proposed, so it truncates the accepted chain.

## Method

`tools/draft_subvocab_coverage.py` runs CPU-only inside the recipe image (tokenizers 0.23.2, `--network none`, no `--gpus`). It uses the model's `tokenizer.json` (vocab 129280) from the local source snapshot.

- **Ranking corpus** (5.09M tokens): wikitext-103 train (7.4k rows), zh Wikipedia (300 articles), OpenAssistant oasst1 train (~6k messages), and the vLLM `v1/` Python sources from the image. The public corpora came from HF datasets-server and sit in the scratchpad only.
- **Held-out eval corpora:**
  - `essay_prose`: `docker/patch/essay_corpus.json`. These are 8 × 511 target-generated tokens for the L.A.I.L prose prompt, so it is the closest proxy for live target output.
  - `chat_greedy`: 12 greedy completions in `results/2026-09-22-lmhead/greedy_stock.json`.
  - `oasst_val`.
  - `wikitext` test.
  - HumanEval (prompt + solution).
  - MBPP. Its code has CRLF line endings, so it is pessimistic.
  - `repo_code`: `tools/*.py`.
  - zh Wikipedia (held-out articles).
- **Layouts:**
  - `global`: one replicated top-N head, no draft AllGather.
  - `tp2`: each rank keeps its top N/2 ids inside its own half of the vocab (correction 3), so each draft shard is a row subset of the local target shard.
- **Acceptance model:** τ = 1 + Σᵢ₌₁³ (α·c)ⁱ with τ = 2.3 at c = 1, so α = 0.637. This treats positions as independent, which overstates the loss somewhat because out-of-vocab tokens are also the rare ones the drafter misses anyway. The rejected `DSV41_DSPARK_DRAFT_TOPK=32` mask measured 4-6% acceptance loss (flags.md:117), which is the same order.

## Coverage (fraction of eval tokens inside the sub-vocab)

| eval | tokens | 16k glob / tp2 | 32k glob / tp2 | 48k glob / tp2 | 64k glob / tp2 |
|---|---:|---|---|---|---|
| essay_prose (target outputs) | 4,088 | 0.862 / 0.804 | **0.930** / 0.891 | 0.959 / 0.934 | 0.981 / 0.963 |
| chat_greedy (target outputs) | 1,533 | 0.789 / 0.718 | **0.875** / 0.824 | 0.912 / 0.887 | 0.930 / 0.914 |
| oasst_val | 187,351 | 0.817 / 0.759 | **0.905** / 0.864 | 0.944 / 0.918 | 0.968 / 0.951 |
| wikitext | 103,266 | 0.819 / 0.754 | 0.902 / 0.857 | 0.945 / 0.913 | 0.970 / 0.952 |
| humaneval | 30,816 | 0.899 / 0.858 | **0.942** / 0.917 | 0.962 / 0.948 | 0.977 / 0.963 |
| mbpp (CRLF) | 43,047 | 0.839 / 0.798 | 0.876 / 0.859 | 0.895 / 0.881 | 0.911 / 0.895 |
| repo_code | 22,921 | 0.869 / 0.826 | **0.933** / 0.900 | 0.963 / 0.940 | 0.977 / 0.965 |
| zh_wiki | 38,556 | 0.916 / 0.885 | 0.976 / 0.962 | 0.989 / 0.986 | 0.994 / 0.993 |

Raw numbers are in `draft-subvocab-coverage.json`. As a baseline, ranking by token id alone (BPE merge order, `--id-order`) is worse on everything except MBPP and chat_greedy; see `draft-subvocab-coverage-idorder.json`.

**Sensitivity to ranking-corpus size.** Coverage at 32k over 1/8 → 1/4 → 1/2 → full corpus:

| eval | 1/8 | 1/4 | 1/2 | full |
|---|---:|---:|---:|---:|
| oasst_val | 0.888 | 0.894 | 0.900 | 0.905 |
| essay_prose | 0.913 | 0.923 | 0.925 | 0.930 |
| repo_code | 0.921 | 0.926 | 0.932 | 0.933 |

That is about +0.5 pt per doubling. Extrapolated to a 1B-token ranking corpus (~7.6 more doublings), that gives about 94-96%, still under the 97% floor.

## Saving vs loss (step ≈ 66 ms, DSpark-3 verify; correction: not 25 ms)

Gross saving uses the corrected per-step costs: draft MXFP8 head ~1.35 ms, draft AllGathers 4 × 112 µs ≈ 0.45 ms, Markov 3 × 279 µs ≈ 0.84 ms. Head and Markov scale with N; AllGather goes to 0 when replicated and scales by N/V under tp2.

| N | gross, replicated | gross, tp2 | net essay | net chat | net oasst | net humaneval | net repo_code | net zh |
|---|---|---|---|---|---|---|---|---|
| 16k | 2.19 ms (3.3%) | 2.31 ms (3.5%) | −9.5 / −14.1 | −15.5 / −20.7 | −13.2 / −17.7 | −6.2 / −9.6 | −8.9 / −12.3 | −4.7 / −7.4 |
| 32k | 1.74 ms (2.6%) | 1.97 ms (3.0%) | −4.1 / −7.2 | −9.0 / −13.0 | −6.4 / −9.6 | −3.0 / −4.9 | −3.8 / −6.4 | +0.3 / −0.7 |
| 48k | 1.29 ms (2.0%) | 1.64 ms (2.5%) | −2.0 / −3.9 | −6.3 / −8.0 | −3.4 / −5.3 | −1.7 / −2.5 | −1.6 / −3.3 | +0.9 / +1.1 |
| 64k | 0.85 ms (1.3%) | 1.30 ms (2.0%) | −0.5 / −1.6 | −5.4 / −6.1 | −1.8 / −2.8 | −0.9 / −1.6 | −0.9 / −1.4 | +0.7 / +1.3 |

Net values are % tokens/s, replicated / tp2. The Markov term (~0.6 ms of the gross) overlaps with sparse-markov. If that lands first, every net number above drops by roughly another 1 pt.

## Gate outcome

- History-lens abort rule: "abort if 32k coverage < ~97% on prose or code". **Tripped** (93.0% on target prose, 93.3% on repo code).
- Mechanism-lens requirement: ≥ ~99% coverage for a +2% budget. **Not met** at any N ≤ 64k.
- Impact-lens rule: ship only if English, code and Chinese all lose < 1% acceptance. **Not met**; English and code lose 3-11% at 32k.

## Reproduce (CPU, serve untouched)

```bash
S=~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277
docker run --rm --network none -e NVIDIA_VISIBLE_DEVICES=void \
  -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages \
  -v $PWD:/repo:ro -v $CORPUS:/work -v $S/tokenizer.json:/tok/tokenizer.json:ro \
  --entrypoint python3 dsv41-flash-exl3-sm121:canonical-e12 -S /repo/tools/draft_subvocab_coverage.py \
  --tokenizer /tok/tokenizer.json \
  --freq rows:/work/wt_train.jsonl:text --freq rows:/work/zh_train.jsonl:text \
  --freq 'files:/work/vllm/v1/**/*.py' --freq rows:/work/oa_train.jsonl:text \
  --eval essay_prose=ids:/repo/docker/patch/essay_corpus.json \
  --eval chat_greedy=completions:/repo/results/2026-09-22-lmhead/greedy_stock.json \
  --eval repo_code='files:/repo/tools/*.py' ...   # see JSON for the full eval list
```

`$CORPUS` holds datasets-server rows: `Salesforce/wikitext` (wikitext-103-raw-v1), `wikimedia/wikipedia` (20231101.zh), `OpenAssistant/oasst1`, `openai/openai_humaneval` and `google-research-datasets/mbpp`.
