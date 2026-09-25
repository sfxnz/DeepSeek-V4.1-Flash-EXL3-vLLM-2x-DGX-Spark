#!/usr/bin/env python3
"""Which L2 'flush' actually makes the qkv_a b12x GEMM cold on GB10?

The serve's qkv_a runs at 47.6 us median (r3 profile, 199 GB/s). A microbench only
predicts a prefetch gain if its 'cold' GEMM matches that. For each scheme, a CUDA
graph holds R x [flush ; e_a ; qkv_a GEMM ; e_b] over R weight copies (rotation), and
the GEMM time is read from the events. Schemes:
  warm        R=1, no flush (weights stay in L2)
  memset1x/2x/4x   R=1, memset of 1x / 2x / 4x L2 before each GEMM
  read2x/4x        R=1, streaming read of 2x / 4x L2 before each GEMM
  rot8        R=8 weight copies, no flush (72 MiB of weights between reuses)
  rot8_memset2x    R=8 and a 2x memset
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from l2_prefetch_window import QKV_A, build_ext, stats  # noqa: E402

SCHEMES = {  # name: (copies, flush kind, flush multiple of L2)
    "warm": (1, None, 0),
    "memset1x": (1, "memset", 1),
    "memset2x": (1, "memset", 2),
    "memset4x": (1, "memset", 4),
    "read2x": (1, "read", 2),
    "read4x": (1, "read", 4),
    "rot8": (8, None, 0),
    "rot8_memset2x": (8, "memset", 2),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="/repo/kernel_study/comm/.l2pf_build2")
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--replays", type=int, default=100)
    ap.add_argument("--json")
    args = ap.parse_args()
    import torch
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
        swizzle_mxfp8_scale,
    )
    from vllm.utils import flashinfer as vfi

    ext = build_ext(args.build_dir)
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    l2 = int(torch.cuda.get_device_properties(0).L2_cache_size)
    k, n = QKV_A
    w = torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02
    w8, sc = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=False)
    sw = swizzle_mxfp8_scale(sc, M=n, K=k).contiguous()
    copies = [(w8.clone(), sw.clone()) for _ in range(8)]
    x = torch.randn(args.m, k, device=dev, dtype=torch.bfloat16)
    q, s = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
    big = torch.empty(4 * l2, dtype=torch.uint8, device=dev)
    sink = torch.zeros(1, dtype=torch.int32, device=dev)
    wbytes = w8.numel() + sw.numel()
    out = {}
    for name, (r, kind, mult) in SCHEMES.items():
        evs = [(torch.cuda.Event(enable_timing=True, external=True),
                torch.cuda.Event(enable_timing=True, external=True)) for _ in range(r)]

        def body():
            for i in range(r):
                if kind == "memset":
                    big[: mult * l2].zero_()
                elif kind == "read":
                    ext.read_l2(big, mult * l2, 48, False, sink)
                evs[i][0].record()
                vfi.mm_mxfp8(q, copies[i][0].t(), s, copies[i][1], out_dtype=torch.bfloat16, backend="auto")
                evs[i][1].record()

        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            body()
        torch.cuda.current_stream().wait_stream(st)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            body()
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
        us = []
        for _ in range(args.replays):
            g.replay()
            torch.cuda.synchronize()
            us += [a.elapsed_time(b) * 1e3 for a, b in evs]
        st_ = stats(us)
        st_["gbps"] = round(wbytes / (st_["median"] * 1e-6) / 1e9, 1)
        out[name] = st_
        print(name, json.dumps(st_), flush=True)
        del g
    res = {"what": "qkv_a b12x m=%d GEMM time under L2 eviction schemes" % args.m, "l2_bytes": l2,
           "qkv_a_bytes": wbytes, "serve_r3_median_us": 47.6, "schemes": out,
           "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
