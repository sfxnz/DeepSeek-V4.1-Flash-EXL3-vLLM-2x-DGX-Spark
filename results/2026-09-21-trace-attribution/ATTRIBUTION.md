# ATTRIBUTION — the 11-13 ms/step gap on k3c (2026-09-21, round 18)

Method (round-5/7/11 precedent): k3c serve booted byte-identical to
`results/2026-09-21-capture-pf/boot-k3c.sh` plus ONE addition —
`--profiler-config {"profiler":"torch","torch_profiler_dir":"/tmp/dsv41-traces"}`
which mounts the `/start_profile` / `/stop_profile` HTTP routes in this vLLM
build (they are NOT mounted by default; openapi.json confirms). Boot script:
`boot-k3c-trace.sh`. Profiler driven over exactly one L.A.I.L prose request
(frozen prompt, 512 tokens, t=0.2, c=1, streaming; warmup run first, profiler
off) via `profile_window.sh`. Trace docker-cp'd out of the container
immediately after >10 s flush; serve then STOPPED before parsing (the
in-container trace export dropped host MemAvail to 3 GiB — below the 8 GiB
floor — so parsing ran on the idle host, both ranks restored to 117 GiB).

Trace: `traces/dp0_pp0_tp0_dcp0_ep0_rank0.1789990966929221167.pt.trace.json.gz`
(119 MB gz, 5,023,052 events, 638,297 kernels, NOT committed — gitignore
convention). Parsers (all ijson-streaming + 6 GB RLIMIT_AS, per the round-11
OOM incident): `kernel_study/gemv_bench/parse_trace_safe.py` (kernel table),
`parse_gaps.py` + `parse_gaps_stacks.py` + `parse_gaps_memcpy.py` (new:
GPU-idle bucketing by co-temporal CPU frame). Raw outputs:
`parse-kernels.txt`, `parse-gaps.txt`, `parse-gaps-stacks.txt`,
`parse-gaps-memcpy.txt`.

## Numbers (window 20.57 s ≈ 234 decode steps @ acc ~2.2, 512-token L.A.I.L prose)

| quantity | total | per step |
|---|---|---|
| window | 20570 ms | ~88 ms (profiler overhead inflates vs 78.5 ms stock wall) |
| device busy (kernels) | 16699.5 ms | ~71.4 ms |
| device busy (kernels + memcpy) | 16719.2 ms | ~71.5 ms |
| **pure GPU idle** | **3850.8 ms** | **~14.2 ms** |
| idle gaps ≥ 1 ms | 256 gaps, 3334.5 ms | ~1 gap / step, ~13.0 ms median |
| total memcpy (whole window) | 20.4 ms (15.1 DtoD / 2.8 H2D / 2.5 D2H) | ~0.09 ms — noise |

Device-busy composition confirms the round-11 model (per window / 234 steps):
p2b_moe_batched 5596 ms (~24 ms/step), b12x dense blockscaled 4467 ms
(~19 ms/step), NCCL AR 1542 ms (~6.6 ms/step), wo_a deep_gemm fp8 814 ms,
draft sm80 wmma ~367+186 ms (~2.4 ms/step) + hc_prenorm 478 ms. The gap pool
is ON TOP of all of that and is ~99.9% ONE thing:

## Attribution: one owner — EngramDiskStager hard sync in prepare_inputs

`parse-gaps-stacks.txt` — 253 gaps ≥ 4 ms, 3370.9 ms total, **every single
one** under the same innermost stack (tid 690 = rank-0 worker main thread):

```
cudaEventSynchronize
<built-in method synchronize of Event object at 0xf6600c79c360>
torch/cuda/streams.py(245): synchronize
vllm/models/deepseek_v4_1/common/engram.py(1708): stage        [image line #]
torch/utils/_contextlib.py(120): decorate_context
vllm/models/deepseek_v4_1/nvidia/model_state.py(85): prepare_inputs
```

`parse-gaps-memcpy.txt` — pure-idle ownership: 3331.0 of 3334.5 ms (99.9%)
under the same `Event.synchronize`; remainder 2.2 ms cudaGraphLaunch +
1.4 ms getattr (rounding error).

Source (image `canonical-e12`, `engram.py` `EngramDiskStager.stage`,
image lines 1302-1339; repo trace names it :1708 because sitecustomize's
engram_stage_fast wrap sits in the stack):

```python
hashes = self.hash_state(ids, ...)          # GPU hash of input ids
host.copy_(hashes[...], non_blocking=True)  # D2H of tiny hash tensor
self.hashes_ready.record()
self.hashes_ready.synchronize()             # ← HARD SYNC, owns ~14 ms/step
for engram, buf in zip(...):                # CPU: pread NVMe + dequant
    ...                                     # H2D staged rows (on-thread)
```

Mechanism: the sync waits for the D2H hash copy which is queued BEHIND the
still-running previous step's graph work; the GPU then stays idle through
the entire CPU-side NVMe gather + H2D staging before the next replay
launches. k-independence (F1) is explained: one sync per step regardless of
draft depth. Class #2 (eager sampler region) and class #4 (sampler softmax)
are **closed by measurement**: they contribute <0.15% of idle (the eager
kernels run while the GPU would be busy anyway or inside tiny gaps < 1 ms).
Class #3 (cross-stream events) is the mechanism's tail, not an owner.

## Winning patch class + expected recovery

**Patch class #1 (per-step host critical path in prepare_inputs), anchor =
the engram stage sync.** Concretely, ranked:

1. **CPU-side hash** (kills the sync AND the wait-behind-work): input_ids
   for the next step are deterministic on host before prepare_inputs; a
   verified CPU port of `_hash_ids_kernel` already exists from the prefetch
   v3 work (`docker/patch/engram_prefetch_v3.py` hash port, verified
   against live stage dumps in round 16). Compute hashes on CPU → no D2H,
   no `hashes_ready.synchronize()` at all → recover up to the full
   ~13-14 ms/step idle pool.
2. Overlap/deferral: move the gather for step N+1 off the critical path
   (side thread + event waited only at replay time) — recovers the
   CPU-gather + H2D portion but keeps a (smaller) sync.

Expected: gap pool 11-14 ms/step → near zero ⇒ step ~78.5 → ~64-67 ms ⇒
**~34.5-37 tok/s at acc 2.26 — the 35 target is reachable from this one fix
alone**, consistent with THEORY.md's arithmetic (device-busy 61-64 ms vs
64.6 ms needed).

Sizing caveat: measured under profiler (CUPTI adds per-launch host cost, so
some of the 14.2 ms/step is profiler-inflated); stock-wall estimate from
RESULTS (78.5 ms step, 61-64 device) puts the true pool at 14-17 ms — same
conclusion, same owner.
