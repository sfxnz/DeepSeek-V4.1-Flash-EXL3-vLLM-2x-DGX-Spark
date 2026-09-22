# VERDICT — NCCL AR-tail set: **KEEP**

Arm: nccl-set (ONE room lever: NCCL buffer/proto/channel set from validated
native-recipe). Image `dsv41-flash-exl3-sm121:canonical-e12`,
`MAX_NUM_BATCHED_TOKENS=8192`, on top of the KEEP'd mem-hygiene five-env
baseline (unchanged). Wiring commit `1a43d15` (adds NCCL_LL128_BUFFSIZE +
NCCL_PROTO to the run.sh loop list and worker ssh line).

Envs (exact boot, via `results/2026-09-20-nccl/boot-arm.sh`):

```
IMAGE=dsv41-flash-exl3-sm121:canonical-e12 MAX_NUM_BATCHED_TOKENS=8192 \
DSV41_DROP_PAGE_CACHE=1 DSV41_INDEXER_PREFILL_FACTOR=1 \
DSV41_PREFILL_EMPTY_CACHE_TOKENS=8192 DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB=2.5 \
VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 \
NCCL_BUFFSIZE=1048576 NCCL_LL128_BUFFSIZE=262144 NCCL_PROTO='^LL128' \
NCCL_MAX_NCHANNELS=8 ./serve.sh
```

Rationale: ~92.7 AllReduces/step; in-graph AR tail 124us vs 43us isolated
floor (~7ms/step). The set shrinks NCCL pinned buffers 4.7→0.14 GiB and cuts
per-op protocol overhead for small messages. NCCL_IB_GID_INDEX intentionally
NOT set (plumbing not a tunable); NCCL_MIN_NCHANNELS intentionally NOT set
(prior NULL was a different knob; no stacking).

## Numbers — baseline (2026-09-20-memhygiene) vs arm (2026-09-20-nccl)

| Metric | mem-hygiene baseline | nccl-set | Delta |
|---|---:|---:|---|
| Prose decode c=1 median tok/s (9x) | 34.67 | 31.59 first pass / **34.69 re-run** | flat (+0.06%; first pass = cold-cache artifact, same pattern as mem-hygiene's 30.04) |
| Cold prefill 8k tok/s | 690.4 | 687.3 | −0.4% (noise) |
| Cold prefill 32k tok/s | 718.2 | **733.0** | **+2.1%** |
| L.A.I.L prose median tok/s (3x) | 25.34 | **26.55** / 26.12 re-check | **+4.8% / +3.1%** — both passes above baseline and above the 25.2–26.3 band top |
| DSpark acceptance len | 2.743 | 2.77 / 2.871 (re-run) | noise |
| MemAvail after 32k prefill, spark1 GiB | 18.80 | **22.29** | **+3.49** |
| MemAvail after 32k prefill, spark2 GiB | 21.05 | **23.84** | **+2.79** |

## Verdict rationale (SPEED arm; score cell decides)

Success rule applied: prose median improvement ≥2% OR AR-tail evidence +
flat-but-not-worse with memory savings. Prose re-run 34.69 vs 34.67 = flat,
not worse (first-pass 31.59 is the established cold-cache artifact; kept both
logs per mem-hygiene precedent). The lever delivers on the memory side of its
rationale — pinned-buffer shrink shows up as +3.5/+2.8 GiB MemAvail after
32k prefill on top of the mem-hygiene baseline — with zero NCCL WARN/error
lines (NCCL_DEBUG=WARN on both nodes) and clean smokes (323 + vision ok).
Prefill 32k +2.1% and L.A.I.L 26.55/26.12 (both passes above the 25.2–26.3
band) point the right way but are not claimed as beyond-noise wins on their
own. No regression beyond noise anywhere → **KEEP**.

## Evidence

- `results/2026-09-20-nccl/four_numbers.json` — four numbers (first pass)
- `results/2026-09-20-nccl/01-prose.log` + `01-prose-rerun.log` — both prose passes
- `results/2026-09-20-nccl/04-lail.log` + `04-lail-rerun.log` — both L.A.I.L passes
- `results/2026-09-20-nccl/00-boot.log` — boot, armed lines, ready at 480s
- NCCL WARN grep on both nodes' docker logs: 0 matches (head and worker)
- L.A.I.L runner record: job_id `dcc67f0b824e` (completed,
  `decode_prose` c=1): decode median **26.39 tok/s** (+4.1% vs 25.34
  baseline) — third independent confirmation (26.55 in-harness, 26.12
  re-check, 26.39 via L.A.I.L's own bench).

## Restore / rollback

KEEP: serve left up with the full env set (boot-arm.sh above is the exact
invocation). To roll back: `./stop.sh`, unset the four NCCL envs, re-run
`boot-arm.sh` minus the NCCL lines (= mem-hygiene baseline boot).
