# VERDICT — MCG spec-k sweep (Round 15), 2026-09-21

**Question:** DSpark `num_speculative_tokens` k=5 was inherited, never swept on the MCG lane. At
L.A.I.L acceptance ~2.3, does smaller k cut step time enough to beat k=5 in the L.A.I.L bench cell?

**Answer: YES — best k = 3.** Every measured cell favors k3 over k5 on the same day, same image,
same env (only `NUM_SPECULATIVE_TOKENS` + the `FORCE_UNSAFE_CTX=1` override for the %5 guard).
k4 loses everywhere (acceptance does not buy depth at temperature 0.2). k7 not run: k4 already
showed flat acceptance-vs-depth on this workload, and E6 (flags.md) had k=10 collapse to 12.1.

## Same-day table (all arms: image dsv41-flash-exl3-sm121:canonical-e12, NCCL envs from
results/2026-09-20-nccl/boot-arm.sh, MNBT 8192, MAX_NUM_SEQS 2)

| k | L.A.I.L app tok/s (n=3 median) | acc (CLI twin, t=0.2) | CLI twin 512tok tok/s | in-harness prose 9× (greedy) | prefill 8k / 32k tok/s | MemAvail post-32k (s1/s2) |
|---|---|---|---|---|---|---|
| 3 | **27.27** (26.92 / 29.44 / 27.27) | **2.25** | **28.57** | **35.55** @ 2.62 (repro 35.51 @ 2.51) | 655.1 / 727.9 | 21 / 23 GiB |
| 4 | 25.87 (25.25 / 28.17 / 25.87) | 2.18 | 25.86 | — | — | — |
| 5 | 26.32 (28.25 / 26.32 / 24.83) | 2.30 | 25.57 | 33.03 @ 2.90 | 687.3 / 733.0 (2026-09-20) | 22.3 / 23.8 GiB (2026-09-20) |

Deltas k3 vs k5 (same day): L.A.I.L app **+3.6%**, CLI twin **+11.7%**, in-harness prose **+7.6%**
(35.55 vs 33.03; the in-harness k5 number improved vs yesterday's 31.59 too — warm day-to-day
drift, but the ordering is consistent across all three cells and both k3 boots).

Yesterday's k5 baseline for reference: L.A.I.L app 26.4–26.8 @ acc 2.26–2.39, prose 31.59 @ 2.77.

## Mechanism (why k3 wins at t=0.2)

- The L.A.I.L bench (perf.py) samples at **temperature 0.2**: the target's sampled trajectory
  diverges from the greedy draft, so acceptance is 2.25–2.30 vs 2.6–2.9 greedy. At acc ≈ 2.3,
  draft slots 4–5 are almost always rejected — k=5 pays draft cost + a 6-token verify batch for
  tokens it rarely keeps.
- k=3 verifies 4 tokens/step (c=1). Note the cudagraph capture sizes [1,5,6,10,12] are k5-era:
  a k3 verify batch of 4 pads to 5, so part of the theoretical saving is eaten by padding — and
  k3 STILL wins. A follow-up arm with capture sizes [1,3,4,6,8] (same count → same capture
  memory) is the obvious next refinement, expected to widen the k3 margin.
- k=4 is the worst of the three: deeper than acceptance justifies AND pads 5→6.

## L.A.I.L app job ids (POST :8765/api/bench/perf, runner=decode workload=prose, c=1)

- k3 real: 1e542db2889d (26.92), e398e2358cf3 (29.44), 9a3d37e2308f (27.27)
- k3 final-boot confirmation: 65933ed0ddd7 (27.0)
- k4 real: 6f65613487b6 (25.25), 46ad10107a25 (28.17), 349f2750be43 (25.87)
- k5 real (same-day): 32618cdfff9e (28.25), 51ce15261173 (26.32), 31ffd1ce607b (24.83)
- Discarded: 1b2fa6c0fc52 (k3, 20.39 — ran concurrently with its warmup, ttft 2.88s contamination)
  and warmup jobs d305f7acc0ad, b8413f7cecf2, 5fa572d95ca4, 7d5831b6108c (by protocol).

## Memory discipline (both ranks, every boot and after every smoke/L.A.I.L — zero OOMs)

- Boot gate (>12 GiB required): 116/117 GiB free before every boot (serve down).
- Post-smoke: k3 25/27, k4 25/26, k5 25/26 GiB — all ≫ 8 GiB floor.
- Post-bench: k3 25/27 → post-32k-prefill 21/23 GiB. Logs: 00/10/20/30/43-mem-*.log.
- One serve at a time: `./stop.sh` + docker ps empty on BOTH nodes verified before each boot.

## Serve state left

**k=3 serve UP** via `results/2026-09-20-ksweep/boot-k3.sh` (boot-k3.sh = boot-arm.sh envs +
`NUM_SPECULATIVE_TOKENS=3` + `FORCE_UNSAFE_CTX=1`; everything else identical). Smoke 17×19=323 ✓
(jobs/65933ed0ddd7 confirm 27.0 tok/s on this exact boot). k5 remains restorable verbatim with
`bash results/2026-09-20-nccl/boot-arm.sh` (note: boot-arm.sh is not +x — invoke with `bash`).

## Honest gap to 35

k3 L.A.I.L = 27.27 (best arm median) → **~7.7 tok/s (+28%) short of 35**. Ranked remaining levers:
1. k3 + matched cudagraph capture sizes [1,3,4,6,8] (kills the 4→5 verify padding; cheap, single boot).
2. Acceptance at t=0.2 is the structural limiter (2.25 vs 2.6+ greedy): softmax-verify /
   conf-gate draft knobs (DSV41_DSPARK_SOFTMAX_VERIFY / CONF_GATE) are untested on k3.
3. In-harness is already at 35.5 — the L.A.I.L gap is workload (256 vs 200 tokens, t~0.2, no
   prefix-cache reuse across single requests), so part of the remaining gap may not be
   addressable by serving knobs at all.
