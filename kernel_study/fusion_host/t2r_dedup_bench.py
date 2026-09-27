#!/usr/bin/env python3
"""DSV41_ATTN_T2R_DEDUP: token -> request map per KV-cache group, stock vs dedup.

Runs in the serving image with a GPU. One decode step = two metadata preps as
build_attn_metadata makes them in the serve (r3 trace): the draft prep (3
KV-cache groups) and the target prep (18 groups). Each prep passes its own
query_start_loc slice objects to every group's CommonAttentionMetadata, and
every group's builder calls token_to_req_indices(its own persistent buffer).

GPU-bound timing: each iteration first queues a calibrated torch.cuda._sleep
of --busy-us, then enqueues the step's 21 group builds between two CUDA
events. The host must finish enqueuing before the GPU reaches the first
event (as in the serve, where the host runs ~40 ms ahead); the host enqueue
time of every timed iteration is recorded and the run fails if any reaches
the busy time (then the events would time host launch speed, not the GPU).
Arms alternate per iteration; --l2 cold flushes L2 (a 128 MiB write, GB10 L2
is 24 MiB) before the sleep, warm does not.

Correctness: after every iteration, every group buffer's [:n] must equal
repeat_interleave of the query lengths (plus the zero tail) bit for bit, and
the dedup arm must have engaged (verified reuses) and stayed armed. Then a
sabotage run: the first group's map is overwritten before the next group
copies it; the verify must restore the stock map there and disarm, and the
later groups must compute the stock map.
Prints one JSON line; exit 1 on a mismatch, a host-bound sample, a disarm in
the timed runs or a failed sabotage check.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "docker" / "patch"))

PREPS = (3, 18)  # groups per prep: draft, target (r3 trace, 21 DeviceScan per step)
CASES = (  # (label, qsl host values incl. padded requests, num_tokens)
    ("c1_m4", [0, 4], 4),
    ("c2_m8", [0, 4, 8], 8),
    ("c2_pad_m8_tail", [0, 3, 6, 6], 8),  # padded request + unmapped tail
    ("c1_m1", [0, 1], 1),
)


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def summarize(v):
    return dict(n=len(v), median=round(statistics.median(v), 2), p10=round(pct(v, 0.10), 2),
                p90=round(pct(v, 0.90), 2), mean=round(statistics.fmean(v), 2), max=round(max(v), 2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--busy-us", type=float, default=6000.0)
    ap.add_argument("--l2", choices=("warm", "cold", "both"), default="both")
    args = ap.parse_args()

    import attn_t2r_dedup as t2r
    import decode_levers
    from vllm.v1.attention.backend import CommonAttentionMetadata as CAM

    stock_fn = CAM.token_to_req_indices
    decode_levers._install_t2r_dedup({"DSV41_ATTN_T2R_DEDUP": "1"})
    dedup_fn = CAM.token_to_req_indices
    assert getattr(dedup_fn, "_dsv41_t2r_dedup", False)

    dev = torch.device("cuda")
    groups = sum(PREPS)
    bufs = [torch.zeros(8192, dtype=torch.int32, device=dev) for _ in range(groups)]
    qsl_buf = torch.zeros(9, dtype=torch.int32, device=dev)
    seq_lens = torch.full((8,), 1000, dtype=torch.int32, device=dev)
    slot = torch.zeros(8192, dtype=torch.int64, device=dev)
    table = torch.zeros((8, 64), dtype=torch.int32, device=dev)
    flush = torch.empty(128 << 20, dtype=torch.uint8, device=dev)

    torch.cuda._sleep(1000)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    torch.cuda._sleep(20_000_000)
    e1.record()
    torch.cuda.synchronize()
    cycles_per_us = 20_000_000 / (e0.elapsed_time(e1) * 1e3)
    busy_cycles = int(args.busy_us * cycles_per_us)

    def cam(qsl_gpu, qsl_cpu, num_reqs, num_tokens):
        return CAM(
            query_start_loc=qsl_gpu, query_start_loc_cpu=qsl_cpu, seq_lens=seq_lens[:num_reqs],
            max_seq_len=1000, num_reqs=num_reqs, num_actual_tokens=num_tokens, max_query_len=4,
            block_table_tensor=table[:num_reqs], slot_mapping=slot[:num_tokens],
        )

    def step(qsl_np, num_reqs, num_tokens):
        """One decode step's builds: each prep with its own query_start_loc objects."""
        g = 0
        for ngroups in PREPS:
            qsl_gpu, qsl_cpu = qsl_buf[: num_reqs + 1], torch.from_numpy(qsl_np[: num_reqs + 1])
            for _ in range(ngroups):
                cam(qsl_gpu, qsl_cpu, num_reqs, num_tokens).token_to_req_indices(bufs[g])
                g += 1

    modes = ("warm", "cold") if args.l2 == "both" else (args.l2,)
    out = {"args": vars(args), "torch": torch.__version__, "preps": PREPS, "cycles_per_us": round(cycles_per_us, 1),
           "results": {}}
    bad = host_bound = 0
    ev = {k: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for k in ("stock", "dedup")}
    fns = {"stock": stock_fn, "dedup": dedup_fn}
    for mode in modes:
        for label, qvals, num_tokens in CASES:
            num_reqs = len(qvals) - 1
            n = max(qvals[-1], num_tokens)
            qsl_np = np.array(qvals + [num_tokens] * (9 - len(qvals)), np.int32)
            qsl_buf.copy_(torch.from_numpy(qsl_np))
            ref = torch.repeat_interleave(
                torch.arange(num_reqs, dtype=torch.int32), torch.from_numpy(np.diff(qsl_np[: num_reqs + 1]))
            )
            ref = torch.cat([ref, torch.zeros(n - ref.numel(), dtype=torch.int32)])
            gpu = {"stock": [], "dedup": []}
            host = {"stock": [], "dedup": []}
            for it in range(args.warmup + args.iters):
                for name in (("stock", "dedup") if it % 2 == 0 else ("dedup", "stock")):
                    CAM.token_to_req_indices = fns[name]
                    for b in bufs:
                        b[:16].fill_(-7)  # poison: values must be rewritten
                    if mode == "cold":
                        flush.fill_(it & 0xFF)
                    torch.cuda.synchronize()
                    torch.cuda._sleep(busy_cycles)
                    h0 = time.perf_counter()
                    ev[name][0].record()
                    step(qsl_np, num_reqs, num_tokens)
                    ev[name][1].record()
                    h1 = time.perf_counter()
                    torch.cuda.synchronize()
                    for b in bufs:
                        if not torch.equal(b[:n].cpu(), ref):
                            bad += 1
                    if it >= args.warmup:
                        gpu[name].append(ev[name][0].elapsed_time(ev[name][1]) * 1e3)
                        host[name].append((h1 - h0) * 1e6)
                        host_bound += (h1 - h0) * 1e6 >= args.busy_us
            res = {"gpu_us": {k: summarize(v) for k, v in gpu.items()},
                   "host_enqueue_us": {k: summarize(v) for k, v in host.items()}}
            res["delta_us_median"] = round(res["gpu_us"]["stock"]["median"] - res["gpu_us"]["dedup"]["median"], 2)
            out["results"][f"{mode}_{label}"] = res
    CAM.token_to_req_indices = stock_fn

    # GPU ops per decode step (3 + 18 groups), c=1 m=4
    counts = {}
    for name in ("stock", "dedup"):
        CAM.token_to_req_indices = fns[name]
        qsl_np = np.array([0, 4] + [4] * 7, np.int32)
        qsl_buf.copy_(torch.from_numpy(qsl_np))
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            step(qsl_np, 1, 4)
            torch.cuda.synchronize()
        counts[name] = sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)
    CAM.token_to_req_indices = stock_fn
    out["gpu_ops_per_step"] = counts
    for res in out["results"].values():
        res["us_per_op"] = {k: round(res["gpu_us"][k]["median"] / counts[k], 3) for k in counts}
    out["mismatches"] = bad
    out["host_bound_samples"] = host_bound
    out["lever_state"] = {k: t2r._STATE[k] for k in ("armed", "engaged", "verify_left")}
    ok = not bad and not host_bound and t2r._STATE["armed"] and t2r._STATE["engaged"]

    # Sabotage, on the real CommonAttentionMetadata: the first group's map is
    # overwritten after its compute, the next group's reuse copies it, the
    # verify recomputes the stock map into that group's buffer and disarms;
    # every later group then computes the stock map itself.
    CAM.token_to_req_indices = dedup_fn
    t2r._STATE.update(armed=True, engaged=False, verify_left=2)
    qsl_np = np.array([0, 3, 4] + [4] * 6, np.int32)
    qsl_buf.copy_(torch.from_numpy(qsl_np))
    for b in bufs:
        b.fill_(-7)
    qsl_gpu, qsl_cpu = qsl_buf[:3], torch.from_numpy(qsl_np[:3])
    cam(qsl_gpu, qsl_cpu, 2, 4).token_to_req_indices(bufs[0])
    bufs[0][1] = 99
    for g in range(1, 6):
        cam(qsl_gpu, qsl_cpu, 2, 4).token_to_req_indices(bufs[g])
    torch.cuda.synchronize()
    CAM.token_to_req_indices = stock_fn
    out["sabotage"] = {
        "disarmed": not t2r._STATE["armed"],
        "engaged": t2r._STATE["engaged"],
        "groups_1_to_5_stock_map": all(bufs[g][:4].tolist() == [0, 0, 0, 1] for g in range(1, 6)),
    }
    sab = out["sabotage"]
    ok = ok and sab["disarmed"] and not sab["engaged"] and sab["groups_1_to_5_stock_map"]
    print(json.dumps(out))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
