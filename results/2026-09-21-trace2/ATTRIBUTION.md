# ATTRIBUTION — residual ~10.7 ms/step idle on the ENGAGED 30.10 serve (Round 22)

Re-trace of Round 18's method on the current best config (k3c +
`DSV41_ENGRAM_PREFETCH=1` + `DSV41_ENGRAM_CENSUS=1`, i.e. `boot-k3c-pf.sh` +
torch profiler). Boot: `boot-k3c-pf-trace.sh` (only addition vs the live-best
boot = `--profiler-config {"profiler":"torch","torch_profiler_dir":...}`).
One L.A.I.L prose request (61 prompt + 512 completion, streaming) profiled via
`profile_window.sh` (copied from Round 18); >10 s flush; trace docker-cp'd
out BEFORE `./stop.sh` (Round-18 lesson — see Incident note below); parsed on
the idle host with the ijson + 6 GB RLIMIT_AS parsers only.

Trace: `traces/dp0_pp0_tp0_dcp0_ep0_rank0.1790009443854305615.pt.trace.json.gz`
(129 MB gz, 5.78 M events, 608 756 kernels; NOT committed, gitignore
convention). Parsers: `kernel_study/gemv_bench/parse_gaps{,_stacks,_memcpy}.py`
+ new `parse_gap_split.py` (blocked-in-sync vs post-sync split). Raw outputs:
`parse-kernels.txt`, `parse-gaps.txt`, `parse-gaps-stacks.txt`,
`parse-gaps-memcpy.txt`, `parse-gap-split.txt`.

Census during the window (both ranks): pf_hit 100 % the entire run,
read_w 0.09–0.12 ms, read_s 0.05 ms, dequant 0.02 ms, rows/call 51 —
prefetch overlap fully engaged, pread waits eliminated (Round-18's 4.0 ms
warmup → 0.1 ms steady).

## Numbers (window 17.89 s ≈ 238 steps @ acc ~2.15, 512-token L.A.I.L prose)

| quantity | total | per step |
|---|---|---|
| window | 17893 ms | ~75 ms (profiler overhead inflates vs ~72-75 ms stock wall) |
| device busy (kernels) | 15523 ms | ~65.2 ms |
| device busy (kernels + memcpy) | 15542 ms | ~65.3 ms |
| **pure GPU idle** | **2351 ms** | **~9.9 ms** |
| idle gaps ≥ 1 ms | 242 gaps, 1804 ms | ~1 gap / step, median ~7.5 ms |
| total memcpy | 20.2 ms (13.9 DtoD / 3.7 DtoH / 2.6 HtoD) | ~0.08 ms — noise |

## Idle then vs now (per step, gaps ≥1 ms pool)

| owner | Round 18 (k3c, no prefetch) | Round 22 (engaged 30.10) |
|---|---|---|
| `engram.py stage` — `hashes_ready.synchronize()` wait-behind-work | ~13.0 ms | ~0.0 ms (see split) |
| `engram.py stage` — post-sync CPU-side gather (rows loop: pread syscalls → dequant → pinned H2D staging) | (inside the 13.0) | **~7.5 ms** |
| tail gaps <1 ms / sub-bucket STREAM_SYNC fragments | ~1.2 ms | ~2.4 ms |
| **total pure idle** | **~14.2 ms** | **~9.9 ms** |

Attribution evidence: `parse-gaps-memcpy.txt` — 1803.0 of 1804.4 ms (99.9 %)
of pure idle sits under the SAME innermost stack as Round 18
(tid 617 = rank-0 worker main thread):

```
cudaEventSynchronize
<built-in method synchronize of Event object at 0xf04cc80d6f20>
torch/cuda/streams.py(245): synchronize
vllm/models/deepseek_v4_1/common/engram.py(2395): stage
torch/utils/_contextlib.py(120): decorate_context
vllm/models/deepseek_v4_1/nvidia/model_state.py(85): prepare_inputs
```

(2395 = prefetch-v3-wrapped `EngramDiskStager.stage`; repo line 1708 in
Round 18. 241/241 gaps ≥2 ms under this stack.)

## New discriminator: the sync is NOT the wait anymore

`parse_gap_split.py`: of 1803 ms of ≥2 ms idle, the CPU thread is blocked
INSIDE the covering `cudaEventSynchronize` for **0.7 ms total** — the D2H
hash event now fires promptly (prefetch v3 emptied the queue behind it).
**1802.4 ms (7.48 ms/gap) elapses AFTER the sync returns**, inside
`engram.stage`'s per-row gather loop before the next graph replay launches.

Mechanism rewrite: Round 18's idle = wait-behind-work (event queued behind
previous step) + CPU gather. Prefetch v3 killed the first term and the pread
waits (census read_w 0.1 ms); what remains is raw host EXECUTION time of the
stage path itself — 51 rows/call × (pread syscall + dequant + pinned-copy/H2D
enqueue) ≈ 7.5 ms/step of serialized Python/host work per decode step, plus
~2.4 ms of sub-1-ms scheduling fragments.

## Why no A/B boot was run (gate not met)

The single dominant owner ≥5 ms exists (post-sync gather, ~7.5 ms/step) but
has **no clearly env-reachable fix**:

- The only env lever that deletes this exact code path
  (`DSV41_ENGRAM_CPU_HASH=1`, skips hash+D2H+sync AND the on-thread gather)
  was measured END-TO-END in Round 20 → 27.46 pooled median vs 28.76 —
  REVERT. Its failure mode (off-thread worker pread/dequant + per-step canary
  cost ≈ the sync saved) attacks the SAME host path we'd be re-arming.
- `DSV41_ENGRAM_STAGE_THREADS=16` is already set; the residual is loop
  execution, not IO wait (read_w 0.1 ms proves the data is in page cache).
- Shrinking rows/call or vectorizing/de-off-threading the gather is a code
  patch (next round's candidate), not an env flip.

Per the protocol ("diffuse or not env-reachable → do NOT boot"), and with the
boot budget at 2/3 (boot 3 = restore), no A/B was run.

## Next-lever pointer (for a future round, code-level)

The 7.5 ms/step is host execution of 51 rows: batch the per-row preads into
one `preadv`/single-buffer read, vectorize dequant, single pinned H2D per
layer — or move the whole gather for step N+1 to the (now idle-during-replay)
side thread and only event-wait at replay time (Round 18 option 2, now
testable since prefetch v3's ordering is fixed). Expected ceiling if fully
recovered: step ~72 → ~64-65 ms ⇒ ~33-34 tok/s at acc 2.15; 35 still needs
acceptance or device-side gains too.

## Incident note (boot 1 of this round)

Boot 1 ran the identical protocol but `docker cp` was attempted AFTER
`./stop.sh`, which `docker rm -f`s the container — /tmp/dsv41-traces is
in-container tmpfs (not a bind mount), so the first trace (127 MB) was lost.
Boot 2 repeated the window with cp-before-stop. Cost: one boot of the
3-boot budget; no host-memory incident (both boots' serve periods stayed
≥25 GiB except a transient 1.9 GiB MemAvail dip on spark1 at docker-cp time
— below the 8 GiB post-smoke floor, logged in the OOM table; both ranks
returned to 117 GiB immediately after stop and parsing then ran on the idle
host per protocol).
