# VERDICT — 2026-09-21 draftk2: draft-sampling-method A/B + k=2 matched captures

Baseline: k3 + captures [1,3,4,6,8] greedy drafter = **28.76** median L.A.I.L
(results/2026-09-21-capture-pf, jobs ba2b5d9f2654/094bb42c84ac/d04d844d5b97).
Target: 35+ tok/s. KEEP gate both arms: ≥+3% → ≥29.62.

| arm | L.A.I.L median (c=1 decode, t=0.2, 256tok) | acc_len (CLI t=0, 512tok) | MemAvail s1/s2 | call |
|-----|----------------------------------------------|---------------------------|----------------|------|
| A: k3c + `draft_sample_method=probabilistic` | 28.50 (runs 28.80/28.50/26.92; jobs 3d35917a6a92/ce1383291aab/967aafdeed2c) | 3.28 (draft_accept 0.76) | 25/27 GiB | **REVERT** (−0.9% vs 28.76; gate ≥29.62 not met) |
| B: k=2 + captures [1,2,3,4,6] (greedy) | 26.89 (runs 26.08/27.57/26.89; jobs f8c7df9748ce/e59af28ed8ac/d6bb8070f4ae) | 2.57 (draft_accept 0.79) | 25/27 GiB | **REVERT** (−6.5% vs 28.76) |
| C | — skipped: both A and B reverted → abort-early rule; best stays 28.76 | | | skip |

## ARM A — sampled drafter (REVERT)

Plan deviation, documented: the session plan said "REMOVE draft_sample_method
from the speculative config (engine default drafter sampling)". That is a
provable no-op on this vLLM: `SpeculativeConfig.draft_sample_method` defaults
to `"greedy"` (vllm-snip config/speculative.py:588) and run.sh pins exactly
that value — removing the field reproduces the pinned config byte-for-byte.
The arm's stated theory ("a sampled drafter can match the sampled trajectory
better than greedy at t=0.2") is implemented by `"probabilistic"`: it allocates
`draft_logits`, switches DSpark's `_sample_logits` from argmax to
temperature-sampled gumbel, and uses full draft logits in the rejection ratio
test. So ARM A was run as `probabilistic` (SPEC_CONFIG override in
boot-k3c-dsampled.sh; everything else identical to boot-k3c.sh).

- Verified in engine log (APIServer non-default args):
  `speculative_config: {'method':'dspark','num_speculative_tokens':3,
  'draft_sample_method':'probabilistic'}`.
- L.A.I.L n=3 median 28.50 = −0.9% vs 28.76. Below gate → REVERT.
- acc observation: at CLI t=0 (512 tok) the sampled drafter shows acc_len
  3.28 / draft_acceptance 0.76 vs the greedy k3c lane's prose-bench 2.62
  (256 tok) — different harnesses, but no collapse; the sampled drafter is
  *functionally healthy*, it just doesn't convert to L.A.I.L speed at t=0.2.
  mul1's A3-null does not transfer, as predicted, but the win doesn't either.
- Warmup discarded: a224e23dfa42 (27.28).

## ARM B — k=2 with matched captures (REVERT)

- Lever: `NUM_SPECULATIVE_TOKENS=2` + capture sizes resized to the k2 formula
  [1,2,3,4,6] (k2 c=1 verify batch = 3). Single lever (the k change), captures
  matched to it as part of the same arm, per plan.
- Verified in engine log: `'num_speculative_tokens': 2`,
  `cudagraph_capture_sizes: [1,2,3,4,6]`.
- L.A.I.L n=3 median 26.89 = −6.5% vs 28.76. Below gate → REVERT.
- Theory refuted: acc 2.57 (CLI t=0) means k2's max 3.0 nearly saturates, but
  the cheaper verify batch (3 vs 4) and shallower draft do NOT pay for the
  lost ~0.5-0.7 acc/step at c=1 — per-step accepted tokens dominate. k=3
  remains the optimum of the sweep (k5 26.32 / k4 25.87 / **k3 28.76** /
  k2 26.89).
- Warmup discarded: 5396b3ad7600 (25.29).

## Abort-early rule applied

Both arms reverted and the best remains 28.76 → per plan, ARM C skipped and
the serve restored to the exact campaign-best config
(`results/2026-09-21-capture-pf/boot-k3c.sh`, byte-identical restore boot;
see 40-boot-k3c-restore.log).

## Serve state left UP

k3 + captures [1,3,4,6,8], greedy drafter — via
`results/2026-09-21-capture-pf/boot-k3c.sh` on :8000. Restore-boot smoke
17×19=323 OK (01→41-smoke logs), MemAvail verified after every smoke and
every L.A.I.L job (25-27 GiB throughout, ≫8 GiB floor).

## OOM floor log (all readings this session)

| boot | pre-boot MemAvail s1/s2 | post-smoke s1/s2 | post-L.A.I.L s1/s2 | abort lines |
|------|-------------------------|-------------------|---------------------|-------------|
| pre-session k3c restore (inherited) | 117/117 GiB | 25/26 GiB | 25/27 GiB | none |
| ARM A k3c-dsampled | 117/117 GiB | 25/27 GiB | 25/27 GiB | none |
| ARM B k2c | 117/117 GiB | 25/27 GiB | 25/27 GiB | none |
| final restore k3c | 117/117 GiB | see 41-smoke | — | none |

All boots: `./stop.sh` + `docker ps` verified empty (only dsv41-exl3-nfs +
conduit on s1, empty on s2) on BOTH nodes before every boot; no host CUDA
JIT / no big host allocations during serve; no trace files parsed.

## Gap to target

Best stays **28.76** vs 35 → **6.24 tok/s short** (82.2% of target).
k-lever now swept end-to-end (k2-k5); draft sampling swept (greedy/probabilistic).
Remaining lever families: acceptance shaping at t=0.2
(DSV41_DSPARK_SOFTMAX_VERIFY / CONF_GATE untested on k3c), scheduler/prefill
interactions in the L.A.I.L cell, or pack-level changes (source-precision
drafter re-quant).
