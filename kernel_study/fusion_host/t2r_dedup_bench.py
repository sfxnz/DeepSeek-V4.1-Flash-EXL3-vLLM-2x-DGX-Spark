#!/usr/bin/env python3
"""DSV41_ATTN_T2R_DEDUP: token -> request map per KV-cache group, stock vs dedup.

Runs in the serving image with a GPU. One "prep" = what build_attn_metadata
does for the token map: one CommonAttentionMetadata per KV-cache group, all
sharing the step's query_start_loc tensors, each group's builder calling
token_to_req_indices(its own persistent buffer). The r3 trace shows 21 of
these per decode step (DeviceScan 21.08/step). The GPU work is queued behind
~--busy-us of work so the device runs it back to back, as in the serve (the
host is ~40 ms ahead); CUDA events around the prep give its GPU time. Arms
alternate per iteration. Correctness: after every prep, every group buffer's
[:n] must equal the stock values bit for bit, and the stock arm's buffers are
compared with a fresh stock computation too.
Prints one JSON line.
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


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def summarize(v):
    return dict(n=len(v), median=statistics.median(v), p10=pct(v, 0.10), p90=pct(v, 0.90),
                mean=statistics.fmean(v), min=min(v), max=max(v))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=21)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--busy-us", type=float, default=2000.0)
    args = ap.parse_args()

    import decode_levers
    from vllm.v1.attention.backend import CommonAttentionMetadata as CAM

    stock_fn = CAM.token_to_req_indices
    decode_levers._install_t2r_dedup({"DSV41_ATTN_T2R_DEDUP": "1"})
    dedup_fn = CAM.token_to_req_indices
    assert getattr(dedup_fn, "_dsv41_t2r_dedup", False)

    dev = torch.device("cuda")
    max_tokens = 8192
    bufs = [torch.zeros(max_tokens, dtype=torch.int32, device=dev) for _ in range(args.groups)]
    qsl_buf = torch.zeros(9, dtype=torch.int32, device=dev)
    seq_lens = torch.full((8,), 1000, dtype=torch.int32, device=dev)
    slot = torch.zeros(max_tokens, dtype=torch.int64, device=dev)
    table = torch.zeros((8, 64), dtype=torch.int32, device=dev)
    busy = torch.randn(1024, 1024, device=dev)

    def cam(qsl_gpu, qsl_cpu, num_reqs, num_tokens):
        return CAM(
            query_start_loc=qsl_gpu, query_start_loc_cpu=qsl_cpu, seq_lens=seq_lens[:num_reqs],
            max_seq_len=1000, num_reqs=num_reqs, num_actual_tokens=num_tokens, max_query_len=4,
            block_table_tensor=table[:num_reqs], slot_mapping=slot[:num_tokens],
        )

    # (label, qsl host values incl. padded requests, num_tokens)
    cases = [
        ("c1_m4", [0, 4], 4),
        ("c2_m8", [0, 4, 8], 8),
        ("c2_pad_m8_tail", [0, 3, 6, 6], 8),  # padded request + unmapped tail
        ("c1_m1", [0, 1], 1),
    ]
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(10):
        busy @ busy
    torch.cuda.synchronize()
    n_mm = max(1, int(args.busy_us / ((time.perf_counter() - t) / 10 * 1e6)))

    out = {"args": vars(args), "torch": torch.__version__, "n_mm": n_mm, "results": {}}
    bad = 0
    ev = [[torch.cuda.Event(enable_timing=True) for _ in range(2)] for _ in range(2)]
    for label, qvals, num_tokens in cases:
        num_reqs = len(qvals) - 1
        n = max(qvals[-1], num_tokens)
        times = {"stock": [], "dedup": []}
        for it in range(args.warmup + args.iters):
            order = [("stock", stock_fn), ("dedup", dedup_fn)]
            if it % 2:
                order.reverse()
            for k, (name, fn) in enumerate(order):
                CAM.token_to_req_indices = fn
                for b in bufs:
                    b[:16].fill_(-7)  # poison: values must be rewritten
                qsl_np = np.array(qvals + [num_tokens] * (9 - len(qvals)), np.int32)
                qsl_buf.copy_(torch.from_numpy(qsl_np))
                qsl_gpu = qsl_buf[: num_reqs + 1]
                qsl_cpu = torch.from_numpy(qsl_np[: num_reqs + 1])
                for _ in range(n_mm):
                    busy @ busy
                ev[k][0].record()
                views = []
                for g in range(args.groups):
                    views.append(cam(qsl_gpu, qsl_cpu, num_reqs, num_tokens).token_to_req_indices(bufs[g]))
                ev[k][1].record()
                torch.cuda.synchronize()
                ref = torch.repeat_interleave(
                    torch.arange(num_reqs, dtype=torch.int32), torch.from_numpy(np.diff(qsl_np[: num_reqs + 1]))
                )
                ref = torch.cat([ref, torch.zeros(n - ref.numel(), dtype=torch.int32)])
                for g in range(args.groups):
                    if not torch.equal(bufs[g][:n].cpu(), ref) or views[g].shape[0] != num_tokens:
                        bad += 1
                if it >= args.warmup:
                    times[name].append(ev[k][0].elapsed_time(ev[k][1]) * 1e3)
        out["results"][label] = {k: summarize(v) for k, v in times.items()}
    CAM.token_to_req_indices = stock_fn
    # kernel counts for one prep of the c=1 case
    counts = {}
    for name, fn in (("stock", stock_fn), ("dedup", dedup_fn)):
        CAM.token_to_req_indices = fn
        qsl_np = np.array([0, 4] + [4] * 7, np.int32)
        qsl_buf.copy_(torch.from_numpy(qsl_np))
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            qsl_gpu, qsl_cpu = qsl_buf[:2], torch.from_numpy(qsl_np[:2])
            for g in range(args.groups):
                cam(qsl_gpu, qsl_cpu, 1, 4).token_to_req_indices(bufs[g])
            torch.cuda.synchronize()
        counts[name] = sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)
    CAM.token_to_req_indices = stock_fn
    out["gpu_ops_per_prep_c1"] = counts
    out["mismatches"] = bad
    print(json.dumps(out))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
