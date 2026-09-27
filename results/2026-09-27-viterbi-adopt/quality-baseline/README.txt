Quality baseline for pack 2.0bpw-mcg-viterbi-lmhead-mxfp8 (round 36, recorded on boot V-1, 2026-09-27)
=====================================================================================================

This is the default quality baseline for tests/quality_eval.py since round 36. It is also where the goldens live:
the harness stores them in the baseline JSON itself, and the golden hazard of a later run is its divergence from
components.selfcons.runs[0] of this file. ARMS.md step 6 amends the golden gate. It gates kernel/numerics levers on
the pack the baseline ran, so a pack change re-records this baseline (the round-36 amendment, decision in
../decision-viterbi-addendum.json).

The previous baseline, results/2026-09-24-review/quality-baseline/ (09-24, canonical-e12,
2.0bpw-mcg-lmhead-mxfp8), stays the reference for 2.0bpw-mcg-lmhead-mxfp8 (the serve pin in rounds 33-35).

Provenance (both JSONs)
- Boot V-1 (../V-1/): the round-36 defaults at commit dcca87e, launched as `env AUDIT=strict ./run.sh` with no
  lever, pack or image env (../V-1/command.txt). Pack snapshots/2.0bpw-mcg-viterbi-lmhead-mxfp8 (assembled on both
  nodes, not yet on the Hub). Image dsv41-flash-exl3-sm121:canonical-e14 = sha256:3a002b55c9bc on both ranks.
  flags_sha256 74c6b7e5ce7cdb57 (the container DSV41_/VLLM_/NCCL_ env plus the cmd, model path included). DSpark k=3
  greedy verify, SPARSE_MARKOV_TOPK=1024, the round-35 kernel bundle, MAX_NUM_SEQS=2.
- Both runs had no --baseline ("baseline": null), so only vision and c2 were gated. Both passed, rc 0.
- They ran after the boot's perf capture (smokes, correctness, L.A.I.L x10, bench_decode x9, four_numbers, e2e),
  serialized, with nothing else sending traffic. gpu_guard was clean over the whole boot (../V-1/gpu_guard.txt).

  file                  mode     started (UTC)          wall     exit
  quick.json/quick.txt  --quick  2026-09-27T06:20:50Z   315.3 s  0
  full.json/full.txt    --full   2026-09-27T06:26:06Z   1294.3 s 0

Numbers
  component                                          quick                  full
  NLL, nats/token (40 passages, 19,828 tokens)       0.138564               0.139323 (passes 0.139323 / 0.138198,
                                                                            repeat delta 0.001125)
  decode probe median / p99 |dlogprob|               0.00734 / 0.95233      0.00139 / 0.65392
  decode probe, prefill NLL of greedy text           0.3336                 0.219
  tools JSON-valid / exact-args / no-call            22/22, 22/22, 8/8      22/22, 22/22, 8/8
  needle found (8k, 32k; full adds 128k) x 3 depths  6/6                    9/9
  self-consistency identical / A/A hazard            5/12, 0.0122           6/12, 0.00968
  c=2, vision                                        pass, "Red"            pass, "Red"
  GSM8K-100, thinking off                            -                      97/100 (misses 403, 962, 1001)
  GSM8K-40, thinking on                              -                      39/40 (miss 403)
  MMLU 4x57                                          -                      205/228 = 0.8991

Gate limits these files set (tests/quality_eval.py GATE_RULES)
  NLL <= 0.148564 quick / 0.149323 full (base + max(0.01, 3 x repeat delta));
  golden hazard and A/A hazard <= 0.0244 quick / 0.01936 full (2 x max(base A/A, 0.005));
  decode median <= base + 0.05, greedy-text NLL <= base + 0.15; tools exact-args needs >= 21/22 (base 22/22, one
  item of slack); needle >= 6/6 quick, 9/9 full; GSM8K >= base 0.97, thinking >= 0.975 and MMLU >= 0.8991 under the
  one-item Wilson slack.
  The full A/A band is tight (0.00968 here vs 0.01698 on round 35's final-1). A later boot of this pack must hold
  its greedy text within 0.01936.

Cross-checks, offline, no traffic (crosscheck/)
  The round-35 P4 readings of this same pack (VIT-1, VIT-2; review-e14 = the same image ID, TOPK 256, so drafting
  differed but greedy verify did not) re-gated against these files with `--result <json> --baseline <this>`:
    VIT-1 quick PASS (golden 0.01128, A/A 0.01609, limit 0.0244)   VIT-1 full PASS (golden 0.01034, A/A 0.00843, limit 0.01936)
    VIT-2 quick PASS (golden 0.01122, A/A 0.01166, limit 0.0244)   VIT-2 full PASS (golden 0.01006, A/A 0.01207, limit 0.01936)
  The old pack's P4 readings (CUR-1, CUR-2) against the same files fail, as they should: NLL 0.242-0.244 against
  0.1486/0.1493 and golden hazard 0.10204. CUR-2 quick also fails A/A 0.0274 and exact-args 20/22.

Use
  python3 tests/quality_eval.py --quick --baseline results/2026-09-27-viterbi-adopt/quality-baseline/quick.json --out <boot>/quality_quick.json
  python3 tests/quality_eval.py --full  --baseline results/2026-09-27-viterbi-adopt/quality-baseline/full.json  --out <boot>/quality_full.json
