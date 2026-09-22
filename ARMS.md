# ARMS.md — campaign arm discipline

One boot = one lever. Every arm produces the four numbers via
`tools/four_numbers.sh --arm NAME` and a verdict appended to `flags.md`,
`results/RESULTS.md`, and the `recipes` skill ledger.

## Arm loop

1. **Pre-flight (both nodes)** — `docker ps --format '{{.Names}} {{.Image}}'`
   must show only `dsv41-flash-exl3`; `ssh spark2 docker ps` same.
2. **Stop both nodes**: `./stop.sh` (reads `.run-state/worker_host`, ssh's
   to spark2).
3. **Boot with ONE env lever** (one knob changed vs baseline; full argv/env
   recorded — paste the exact `FOO=1 ./serve.sh` line into the arm notes):
   ```bash
   DSV41_ENGRAM_PREFETCH=1 ./serve.sh     # example lever
   ```
   run.sh refuses unsafe configs (`FORCE_UNSAFE_QUANT/ENGRAF/CTX` guards);
   if an A/B legitimately needs an override, set the guard for that boot
   only and record it — never commit it as a default.
4. **Boot floors** (read `free -h`, NEVER nvidia-smi):
   - abort if MemAvailable **< 12 GiB at boot** (before smoke);
   - abort if MemAvailable **< 8 GiB after smoke** (`smoke_chat.py` +
     `smoke_vision.py`).
5. **Four numbers** (serialized, ~10-15 min):
   ```bash
   tools/four_numbers.sh --arm <name>          # prose 9x median, pp 8k/32k,
                                               # MemAvailable both nodes, L.A.I.L 3x
   ```
   Number 3 (MoE/attention ms/layer at shipped chunk) has no live probe —
   fallback is boot knobs `DSV41_STEP_CENSUS=1` / `DSV41_ENGRAM_CENSUS=1`
   (decode-side only; E0 prefill-flush caveat); real profiling is a
   separate gated step.
6. **Promote rule**: 9-run prose median beats baseline (or 3-run min AND
   max both beat it). No promote on a single lucky run.
7. **Free the box between arms**: `./stop.sh`, confirm no GPU containers on
   either node, then boot the next arm. Never stack levers on a promoted
   arm's boot without re-running the full four numbers.
8. **Verdicts**: append to `flags.md` (per-round note), `results/RESULTS.md`
   (row with evidence path), and the `recipes` skill ledger. Reference band
   for prose c=1: 31-34.3 tok/s; L.A.I.L prose: 25.2-26.3 tok/s.

## Exact restore sequence

Current serve = canonical image `dsv41-flash-exl3-sm121:canonical-e12` on
both nodes, launched via `serve.sh` from this repo on spark1 (head). To
restore after any arm:

```bash
cd /home/sfxnz/projects/ai-lab/recipes/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark
./stop.sh                                  # both nodes (stops spark2 via ssh)
./serve.sh                                 # defaults = canonical-e12 + stock flags
# readiness: python3 smoke_chat.py && python3 smoke_vision.py
# then: tools/four_numbers.sh --arm <label> to re-confirm baseline cells
```

- `serve.sh` → `exec ./run.sh`; on the head with `ORCHESTRATE=auto` (default)
  and `NNODES=2`, run.sh starts the **worker on spark2 first** (scps itself +
  `docker/patch` to /tmp there, launches worker container with the full env
  forwarded), sleeps 25s for NCCL, then starts head rank 0 and `wait_ready`
  (`/health` + `/v1/models`; allow up to 60 min, plan ≥20).
- Kernel arms rollback: `IMAGE=dsv41-flash-exl3-sm121:canonical-e12 ./serve.sh`.
- Patch-script changes (`docker/patch/engram_*`) take effect on restart
  without an image rebuild (patch dir is volume-mounted read-only).
