# VERDICT — mul1-lane Boot A2 (SPEC sweep: spec OFF) — decisive diagnostic

Date: 2026-09-20 · Run dir: `../` · Artifacts: this dir · Raw boot log: `04-boot-full.log`
Single lever vs Boot A: `SPEC_METHOD=none` in `.env`. Serve-command diff (verified token-by-token): only `--speculative-config {"method":"dspark","num_speculative_tokens":3}` removed. Engine log confirms `speculative_config=None` (Boot A: dspark k=3).

## Spec-sweep table (greedy prose, median, port 8888)

| Metric | A2: spec OFF | A: k=3 dspark | MCG lane (K4) | Δ A2 vs A |
|---|---|---|---|---|
| Prose decode c=1 tok/s (9 runs) | **23.64** | 16.35 | 34.69 | **+45%** |
| Cold prefill 8k tok/s | **587.9** | 593.7 | 690 | −1% |
| Cold prefill 32k tok/s | **845.6** | 711.7 | 733 | **+19%** |
| LAIL prose tok/s (t=0.2) | **23.46** | 25.63 | 26.39 | −8.5% |
| Micro tg 8k / 32k tok/s | 23.1 / 23.3 | 33.3 / 33.3 | — | −31% (spec loss) |
| acceptance_len (prose / tg32k) | n/a (spec off) | 1.32 / 2.83 | — | — |
| MemAvail after (s1/s2 GiB) | **8.8 / 10.0** | 5.3 / 6.7 | 22.3 / 23.8 | +3.5/+3.3 |

## Classification — spec-off ≈ 23+ → **greedy drafting pathology on the mul1 pack**

The decisive rule: spec-off greedy prose = **23.64 tok/s ≈ her spec-off ~23**. The pack + kernels are FINE; the k=3 dspark draft/verify path is the anomaly:

- With k=3, greedy decode runs at 16.35 (−31% vs its own spec-off) while acceptance_len is only 1.32 and draft acceptance 0.108 — the draft rarely wins but the verify overhead is paid every step. Spec is actively harmful at c=1 greedy on this pack.
- Prefill is healthy spec-off: 32k = **845.6 tok/s**, +19% vs k=3 boot and **+15% above the MCG lane's 733** — the mul1 pack's prefill kernels are not the problem; they beat MCG at 32k.
- LAIL temp-0.2 stays at parity (23.46 vs MCG 26.39, −11%; within this lane's temp-sampled acceptance behavior where k=3 LAIL acceptance was 2.08).
- Memory floors IMPROVE with spec off (8.8/10.0 vs 5.3/6.7) — the draft model's resident pools were a structural cost of the k=3 arm, not a leak.

**mul1 lane stays OPEN for a future draft fix** (e.g., draft-model quality at 2.9bpw, k tuning, or dspark drafter config); until then `SPEC_METHOD=none` is the mul1 lane's best greedy config on record — do not ship k=3 on this pack.

## Boot / smoke

- Boot clean, health OK, no NVRM, no abort criteria hit. Smoke: `17 × 19 = 323` ✓ (finish stop).
- Floors recorded at 8.8/10.0 GiB after the full 32k micro sweep — above the 8 GiB card; no downward drift (structural 2.9bpw footprint minus draft pools).

## Files

`00-launch-line.txt` · `01-prose-decode.log` · `02-micro.log` · `03-lail-prose.log` · `04-boot-full.log` · `05-mem-after.log` · `06-docker-inspect-head.txt` · `summary.json` · this `VERDICT.md`

## Restore (post-run)

MCG serve (dsv41-flash-exl3-sm121:canonical-e12) restored via saved `boot-arm.sh` — see restore section in the parent summary and `results-bootA/restore-mcg.log` for the reference procedure.
