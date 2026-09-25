#!/usr/bin/env python3
"""All-reduce latency under one NCCL env, two ranks on one GB10 (two_rank_mps.sh).

k3 comm fix pass (review: AR overhead above the wire floor). With mixing off the m=4 AR is
22.5 us on two nodes against a 13.1 us RoCE floor; GDR is off today ("cuMemGdrSupport 0",
NCCL_CUMEM_ENABLE=0 in run.sh). DMA-BUF GDR would need NCCL_CUMEM_ENABLE=1. This measures the
decode-size AR (bf16, m x 5120) eagerly, back to back in a CUDA graph, and gapped (a 2x-L2
streaming read before each AR inside the graph), so two runs that differ only in the env
(e.g. EXTRA_ENV="NCCL_CUMEM_ENABLE=1" with NCCL_DEBUG=INFO) show whether NCCL takes a GDR
path here and what it changes. Outputs are checked (sum of both ranks' inputs, exact).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
HIDDEN = 5120


def stats(xs):
    s = sorted(xs)
    q = lambda f: s[min(len(s) - 1, int(round(f * (len(s) - 1))))]  # noqa: E731
    return {"median": round(statistics.median(s), 2), "p10": round(q(0.1), 2), "p90": round(q(0.9), 2),
            "n": len(s)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--master", default="127.0.0.1:29761")
    ap.add_argument("--m", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--build-dir", default="/repo/kernel_study/comm/.l2pf_var_build")
    ap.add_argument("--json")
    args = ap.parse_args()
    sys.path.insert(0, str(HERE))
    from datetime import timedelta

    import torch
    import torch.distributed as dist
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    import l2pf_variants

    ext = l2pf_variants.build(args.build_dir)
    torch.cuda.set_device(0)
    dev = torch.device("cuda:0")
    dist.init_process_group("gloo", init_method=f"tcp://{args.master}", rank=args.rank, world_size=2,
                            timeout=timedelta(seconds=120))
    comm = PyNcclCommunicator(group=dist.group.WORLD, device=dev)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    l2 = int(torch.cuda.get_device_properties(0).L2_cache_size)
    flush = torch.empty(2 * l2, dtype=torch.uint8, device=dev)
    sink = torch.zeros(1, dtype=torch.int32, device=dev)
    res = {}
    ok = True
    for m in args.m:
        x = torch.full((m, HIDDEN), float(args.rank + 1), device=dev, dtype=torch.bfloat16)
        y = torch.empty_like(x)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        eager = []
        for i in range(args.iters + 50):
            dist.barrier()
            e0.record()
            comm.all_reduce(x, y)
            e1.record()
            e1.synchronize()
            if i >= 50:
                eager.append(e0.elapsed_time(e1) * 1e3)
        ok &= bool((y == 3.0).all())
        g_ev = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(2)]
        gb = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gb, stream=stream):
            g_ev[0].record()
            for _ in range(20):
                comm.all_reduce(x, y)
            g_ev[1].record()
        gap_ev = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(2)]
        gg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gg, stream=stream):
            ext.k_read(flush, flush.numel(), 48, 0, sink)
            gap_ev[0].record()
            comm.all_reduce(x, y)
            gap_ev[1].record()
        b2b, gapped = [], []
        for i in range(args.iters // 3 + 10):
            dist.barrier()
            gb.replay()
            torch.cuda.synchronize()
            dist.barrier()
            gg.replay()
            torch.cuda.synchronize()
            if i >= 10:
                b2b.append(g_ev[0].elapsed_time(g_ev[1]) * 1e3 / 20)
                gapped.append(gap_ev[0].elapsed_time(gap_ev[1]) * 1e3)
        ok &= bool((y == 3.0).all())
        res[str(m)] = {"eager_us": stats(eager), "graph_b2b_us": stats(b2b), "graph_gapped_us": stats(gapped)}
        print(f"rank {args.rank} m={m} {json.dumps(res[str(m)])}", flush=True)
        del gb, gg
    out = {"rank": args.rank, "nccl": comm.nccl.ncclGetVersion(), "outputs_ok": ok,
           "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("NCCL_")}, "results": res,
           "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    dist.barrier()
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
