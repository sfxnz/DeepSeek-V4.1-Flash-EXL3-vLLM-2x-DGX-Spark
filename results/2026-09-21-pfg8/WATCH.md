# PF-G8 pack rebuild — WATCH RECIPE (launched 2026-09-21 22:06-22:07 BST)

## What is running

- **spark1**: container `dsv41-quant-g8-r1`, expert shards **3-22** (20 shards),
  log `results/2026-09-21-pfg8/rebuild-spark1.log`, marker `rebuild-spark1.started`
  (2026-09-21T22:06:43+01:00).
- **spark2**: container `dsv41-quant-g8-r2`, expert shards **23-42** (20 shards),
  log `results/2026-09-21-pfg8/rebuild-spark2.log` (on spark2), marker
  `rebuild-spark2.started` (2026-09-21T22:06:44+01:00).
- Image `dsv41-flash-exl3-sm121:canonical-e12`, env `DSV41_PACK_PF_G8=1` (the
  quantize-time G8 fold `.view(kt, nt/8, 8*w).permute(1,0,2).contiguous()`),
  `--codebook mcg --allow-partial --batch 8 --greedy --beam 16`,
  dst `snapshots/2.0bpw-mcg-g8`. Detached `docker run -d` — daemon-owned,
  survives session exit. Relaunch scripts (idempotent, `docker rm -f` first):
  `launch-rebuild-spark1.sh` / `launch-rebuild-spark2.sh` (run spark2's ON spark2).
- Spark2's repo checkout is an older commit, so its `tools/{quantize_experts_exl3,
  pack_meta,mxfp4}` were rsync'd from spark1 @3242152 (md5-verified) — do not
  rebuild spark2's container from its own `tools/` until the repo is synced.

## How to read progress

Per rank (run the spark1 line on spark1, the spark2 line over ssh):

```bash
R=~/projects/ai-lab/recipes/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark
DST=~/.cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-EXL3/snapshots/2.0bpw-mcg-g8
# completed expert shards on this rank (of 20):
ls $DST/model-000{03..22}-of-00048.safetensors 2>/dev/null | wc -l   # spark1
ls $DST/model-000{23..42}-of-00048.safetensors 2>/dev/null | wc -l   # spark2
# live tail (each line = 8 experts; tot counts experts finished in the shard):
tail -f $R/results/2026-09-21-pfg8/rebuild-spark1.log
# container state (must stay Up; Exited means done or failed — check the log tail):
docker ps -a --filter name=dsv41-quant-g8-r1 --format '{{.Status}}'
```

Converter log cadence: one `exl3 <file> w1 a-b n=8 7.5s` line per 8-expert
batch, ~7.5 s/batch → ~1.05 expert/s (960 experts/shard → ~15.3 min/shard,
consistent with the plan's measured 16.4 min/shard).

## ETA formula (from REBUILD-PLAN.md measured rates)

- 20 shards x 16.4 min/shard ≈ **5.5 h** per rank (parallel); observed live
  rate 22:06 start → ~15.3-16.4 min/shard. Both ranks finish
  **~03:40-04:15 BST 2026-09-22** (container exited, 20 shard files present).
- ETA = start_marker + (20 − shards_done) × 16.4 min.
- After both ranks finish: `CODEBOOK=mcg REV=2.0bpw-mcg-g8 DST=.../2.0bpw-mcg-g8
  bash tools/assemble_pack.sh` (rsyncs 23-42 from spark2, rebuilds index) then
  `python3 tools/permute_pack_group_major.py <new-pack>` verification.

## Verify both ranks done

```bash
ssh spark2 'ls ~/.cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-EXL3/snapshots/2.0bpw-mcg-g8/model-000{23..42}-of-00048.safetensors | wc -l'  # expect 20
ls ~/.cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-EXL3/snapshots/2.0bpw-mcg-g8/model-000{03..22}-of-00048.safetensors | wc -l  # expect 20
```

## Abort / restore-the-serve

1. Stop rebuilds: `docker rm -f dsv41-quant-g8-r1` (spark1) and
   `ssh spark2 'docker rm -f dsv41-quant-g8-r2'`.
2. Restore serve on the OLD pack (untouched by this build — verified
   48/48 shards, mtime 2026-09-10):
   `bash results/2026-09-21-gatherv2/boot-k3c-pf-gv2.sh` (SNAPSHOT_SHA
   defaults to `2.0bpw-mcg`; do NOT point it at 2.0bpw-mcg-g8 — stock kernels
   would misread the G8 layout and produce garbage).
3. G8 partial packs are resumable per shard (atomic rename) — relaunch script
   re-skips finished shards.

## Phase logs

- `baseline-phase.log` — pre-stop free -h + docker ps both ranks.
- `stop-phase.log` — post-stop verify (docker ps empty of GPU containers,
  MemAvail 116/117 GiB).
- `shard3-verify.md` — first-shard verification verdict.
