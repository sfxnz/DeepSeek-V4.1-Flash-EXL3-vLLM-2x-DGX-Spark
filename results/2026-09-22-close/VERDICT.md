# CAMPAIGN CLOSE — VERDICT (Round 33, 2026-09-22, close session)

**Final best: stock k3c-pf-gv2 base + lm_head MXFP8 (DSV41_LMHEAD_MXFP8=1,
pack 2.0bpw-mcg-lmhead-mxfp8) — fresh L.A.I.L pooled n=10 median 33.23
(+5.9% vs 31.37) CLEARS the +3% gate → KEEP; this serve is LEFT UP
(script results/2026-09-22-endgame2/boot-lm.sh). 35 tok/s NOT crossed:
honest ceiling statement below. G8 lane stays closed (post-load site).**

## Part 1 — lm_head high-n gate (the last live lever)

Boot (boot-lm.sh, canonical-e12, pack 2.0bpw-mcg-lmhead-mxfp8, all four
round-23 keeps unchanged): healthy ~11 min, BOTH ranks logged
`dsv41: lm_head mxfp8 enabled (b12x, (64640, 5120))`, smoke "17*19" →
**323 PASS**. (Boots used this session: 2 of cap 4 — stop/verify only, no
wasted boots.)

L.A.I.L decode/prose (warmup 387735121015 = 32.82 DISCARDED, then two
independent n=5 batches — 10 fresh runs):

| batch | jobs | decode medians (tok/s) | median |
|---|---|---|---|
| real1 (n=5) | 8baac3e7745d 09b0737fa98a 8d10508f7132 + 2 in json | 31.78, 32.22, 33.78, 34.60, 36.91 | **33.78** |
| real2 (n=5) | 53621fcaafe9 e98e6b63b5fd + 3 in json | 30.29, 32.05, 32.67, 35.41, 35.45 | **32.67** |
| pooled n=10 | sorted | 30.29 31.78 32.05 32.22 **32.67 33.78** 34.60 35.41 35.45 36.91 | **33.23** |

GATE: 33.23 ≥ 32.31 (31.37 × 1.03) → **KEEP the lever, leave THIS serve
UP.** Both batch medians (33.78, 32.67) > 31.9. Round-30's miss (32.17,
short by 0.14) is resolved: with fresh n the lever's true effect is
+5.9%, not +2.5% — the round-30 sample sat in the acceptance-lottery
low band.

CLI twin (tools/measure_lail_prose.py, n=9, 512 tok): median
**33.56** lail-math (recipe 33.49), TTFT 0.36 s, acceptance_len 2.21,
draft acceptance 0.40, no collapse (repeat_ratio ~0.002).

35 claim: NOT made — requires two independent batch medians ≥35 AND CLI
twin ≥35; today's best single runs 36.91/35.45/35.41 are high-lottery
draws, batch medians 33.78/32.67. The rule stands.

## Part 2 — G8 post-load site (closure; no boot spent)

Cheap probes done, serve untouched:
1. Full re-read of the saved boot-g8-6 log (142 lines): the only captured
   traceback is the HEAD-side generic chain ("WorkerProc initialization
   failed ... See stack trace for root cause", EngineCore exit); the
   worker's own death-window stack died with the container (docker logs
   destroyed on rm). No new information recoverable from saved logs.
2. Watcher v3 logs re-read (spark2 /tmp/g8watch3.log): anon=0kB every
   sample — smaps_rollup unreadable on root-owned container procs, as
   documented; no anon time series exists for the death window.
3. Source-read of every DSV41-reachable post-load allocation in the g8
   image (vllm_exl3/exl3.py process_weights_after_loading:
   make_linear_exl3 ×3/expert = .contiguous() no-op on contiguous G8
   shards; ngram pwaf host work = two small .cpu().tolist() lists;
   no scale/pointer table build, no from_numpy, no large host buffer in
   the DSV41-reachable path). Stock vLLM post-load init (cudagraph
   capture pool, NCCL buffers, Engram tables, indexer workspace) is
   vLLM-build territory with no env/flag mitigation: the indexer
   workspace is already factor=1 (round-12 keep); capture set is already
   minimal [1,3,4,6,8]; NCCL buffer set is already the tuned minimum.

**No single named allocation ≥30 GiB with a config-level mitigation
emerged → closure paragraph (no fix boot attempted, per one-boot rule):**
the site is NAMED — post-load engine init on the G8 pack (the phase
between "48/48 shards loaded" and first health, under real
NCCL/EngineCore/cudagraph) — the blocker is host memory physics: spark1
MemAvail collapsed to 0.6 GiB and the worker died with no traceback
~7-8 min after a complete 48/48 load (boot-g8-6, fix live, load-phase
pin balloon already eliminated 39.3→1.9 GiB by g8_stream_feed). The fix
is a vLLM-build-level change (host-side allocation of the post-load init
path under the real engine) = out of scope per the stop rule. G8 lane
remains CLOSED; the G8→decode contribution (−1.34 ms/step projected)
stays unmeasured and is part of the named path to 35.

## Part 3 — campaign final table (rounds 15–32, L.A.I.L prose c=1 medians)

| Round | Config / lever (single change vs prior best) | L.A.I.L | verdict |
|---|---|---|---|
| 15 | MCG spec-k sweep → k=3 (vs k=5) | 27.27 | KEEP (k3 wins all cells) |
| 16a | capture sizes [1,3,4,6,8] matched to k3 | 28.76 | KEEP (+5.5%) |
| 16b | engram prefetch v3 (first attempt) | 28.69 | NO-GO (stale-id race) |
| 17a | draft_sample_method=probabilistic | 28.50 | REVERT (−0.9%) |
| 17b | k=2 + captures [1,2,3,4,6] | 26.89 | REVERT (−6.5%); k swept: k2 26.89 / k3 / k4 25.87 / k5 26.32 |
| 18 | SOFTMAX_VERIFY=1 | 27.14 | REVERT (−5.6%) |
| 19 | CPU-hash v1 (predict rule) | 28.87 (disarmed parity) | REVERT (stage-B predict failed) |
| 20 | CPU-hash v2 (stream-ordering fixed) | 27.46 | REVERT env (sync removal ≠ wall time) |
| 21 | prefetch v3 re-arm (ordering fixed) | 30.10 | KEEP (+4.6%) |
| 22 | (trace only, no A/B) | — | attribution: idle moved inside stage |
| 23 | engram gather v2 (preadv batching) | 31.37 | KEEP (+4.2%) — baseline for this session |
| 24 | engram stage defer (off-thread gather) | 30.28 | REVERT env (flat ⇒ gather not the wall) |
| 25 | (trace only, no A/B) | — | DEVICE-BOUND verdict: 35 needs kernels |
| 26 | PF-G8 gate repair (bit-exact 20/20) | — | PASS; fold+loader staged dormant |
| 28 | G8 first boot lane | — | ABORT at gate (harness artifact, later found) |
| 29 | G8 gate v3 24/24 ×2; boot ×3 | — | kernels PROVEN; boot OOM'd in load |
| 30 | lm_head MXFP8 v1 | 32.17 (n=10) | REVERT (+2.5% < +3% gate, missed by 0.14) |
| 31 | G8 load fix (stream feed, 39.3→1.9 GiB) | — | load OOM eliminated (diag, no serve) |
| 32 | G8 boot 6 (fix live) + stock confirm 31.89 | — | G8 LANE CLOSED (post-load site); stock 31.37 stands |
| **33** | **lm_head MXFP8 high-n re-measure (this session)** | **33.23 (n=10)** | **KEEP (+5.9%) — FINAL BEST, serve UP** |

Scoreboard: 26.4 → 27.27 → 28.76 → 30.10 → 31.37 → **33.23**.

## Final best config (serve UP now)

Script: `results/2026-09-22-endgame2/boot-lm.sh` = round-23 stock best
(boot-k3c-pf-gv2.sh) + `SNAPSHOT_SHA=2.0bpw-mcg-lmhead-mxfp8` +
`DSV41_LMHEAD_MXFP8=1`. Image canonical-e12. Full keep set:
MAX_NUM_BATCHED_TOKENS=8192, DROP_PAGE_CACHE=1, INDEXER_PREFILL_FACTOR=1,
PREFILL_EMPTY_CACHE_TOKENS=8192 / MEMAVAIL_GIB=2.5, LOGITS_MB=256,
NCCL {BUFFSIZE=1MiB, LL128=256KiB, PROTO=^LL128, MAX_NCHANNELS=8},
NUM_SPECULATIVE_TOKENS=3, FORCE_UNSAFE_CTX=1, COMPILATION_CONFIG
cudagraph [1,3,4,6,8] FULL_AND_PIECEWISE, ENGRAM_PREFETCH=1,
ENGRAM_CENSUS=1, ENGRAM_GATHER_V2=1, LMHEAD_MXFP8=1.

## Four numbers (fresh, this serve, 2026-09-22, fournumbers/ dir)

- **Prose 9-run median (bench_decode.py, c=1, 200 tok): 39.60 tok/s**
  (runs 37.87–40.82), acceptance 2.61.
- **Cold prefill: 8k = 240.1 tok/s, 32k = 589.6 tok/s** (3-run medians,
  fresh docs; 32k TTFT median 39.5 s @30.7k doc tokens, acc 2.46).
- L.A.I.L CLI twin median 33.56 (n=9).
- **MemAvail after 32k prefill: spark1 21.6 GiB / spark2 23.2 GiB**
  (floor 8; historic band 21–25).
- (MoE/attention ms per layer: no live probe exists — documented
  fallback in four_numbers.json notes; last measured composition
  (trace3): p2b MoE 22.4 ms + dense b12x 17.6 ms + AR 4.6 + draft/aux 13.)

## 35 status — honest ceiling statement

**35 NOT crossed.** Measured ceiling today: L.A.I.L pooled 33.23 / best
batch median 33.78 / best single runs 36.91·35.45 (acceptance lottery).
Named path to 35 (Round-25 arbitration + Round-27 endgame math): at acc
2.3, 35 tok/s needs step ≤65.7 ms; device-busy alone is 65.25 ms — the
box is DEVICE-BOUND. The path is: p2b MoE 22.4 ms/step (G8 fold −1.34 ms
projected — BLOCKED on the G8 post-load host-memory site, this verdict
Part 2) + dense b12x GEMM 17.6 ms/step (CUDA kernel work, e.g. coop cb=1
port ~1.5–3 wks) + lm_head pair −2.6 ms (NOW BANKED, this round) + idle
7.15 ms (fadvise cap ~3.1 ms staged dormant). Ceiling estimate with
lm_head banked and no kernel work: ~33–34 at acc 2.26. 35 requires the
CUDA work (p2b/dense) or the G8 stack — both outside env scope, per the
stop rule.

## Zero-OOM record + OOM log (MemAvail GiB, floor 12 pre-boot / 8 post-smoke)

| point | spark1 | spark2 |
|---|---|---|
| pre-stop (stock up) | 25 | 27 |
| post-stop (both empty) | 116 | 117 |
| pre-boot lm | 116 | 117 |
| post-health (lm) | 32.7 | 33.0 |
| post-smoke (lm) | 25.5 | 26.9 |
| post-bench/four-numbers (final) | 21.6 | 23.2 |

No kernel OOM, no worker kills, no abort lines, no host CUDA JIT, no
trace parsing. Full log: oom-log.md. Boot budget: 2 of 4 used.

## Artifacts index (results/2026-09-22-close/)

boot-lm-close.log (boot, both ranks armed, health green), smoke-lm.json
(323 PASS), lail_bench.sh, lail_warmup.json, lail_real{1,2}.{json,log}
(10 fresh runs + job ids), cli_prose9.log (twin 33.56),
fournumbers/four_numbers.json + 00–05 logs (four numbers, provenance:
image canonical-e12, cmd digest 107dc938…), oom-log.md, VERDICT.md (this
file). G8 evidence: results/2026-09-22-g8final/ (diags v2–v6, boots
1–6, watchers, VERDICT). lm_head lineage: results/2026-09-22-lmhead/
(round-30 verdict, integration test, numerics).

## Commits

(this verdict + flags Round 33 + RESULTS.md campaign summary follow;
see git log on cursor/mul1-p2b-prefill-95c3)
