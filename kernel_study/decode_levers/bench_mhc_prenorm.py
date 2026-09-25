#!/usr/bin/env python3
"""mHC prenorm split-K microbench (GPU, serve DOWN). Lever: DSV41_MHC_DECODE_SPLITS=N.

For each token count T and split count S, times inside a CUDA graph:
  gemm : deep_gemm tf32_hc_prenorm_gemm(x[T,20480], fn[24,20480]) -> [S,T,24]
  pre  : the whole mhc_pre_delayed_tilelang (prenorm GEMM + MHC_PRE_NORM_KERNEL,
         which reduces the S partials serially) with the split count forced
It also reports max |diff| of the pre outputs against S=16 (the stock decode
value). Split-K changes the fp32 summation order, so the outputs are not
bit-exact.

Shapes come from the V4.1-Flash config: hidden 5120, hc_mult 4 (K=20480),
hc_sinkhorn_iters 20, hc_eps 1e-6. Run inside the serve image:

  docker run --rm --gpus all --ipc host --network none \
    -v "$PWD":/w -w /w --entrypoint python3 dsv41-flash-exl3-sm121:canonical-e12 \
    kernel_study/decode_levers/bench_mhc_prenorm.py \
    --json results/2026-09-24-review/decode-levers/mhc_prenorm.json
"""

from __future__ import annotations

import argparse
import json
import sys


def graph_us(fn, iters: int, reps: int) -> float:
    """Median us per call of fn() replayed from one CUDA graph of `iters` calls."""
    import torch

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    g.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1000.0 / iters)
    times.sort()
    return times[len(times) // 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--splits", type=int, nargs="+", default=[16, 24, 32, 40, 48])
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--hc-mult", type=int, default=4)
    ap.add_argument("--sinkhorn", type=int, default=20)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--no-pdl", action="store_true", help="turn DeepGEMM PDL off (the serve has it on)")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    import torch
    from vllm.model_executor.kernels.mhc import tilelang as mhc_tl
    from vllm.model_executor.kernels.mhc import warmup as mhc_wu
    from vllm.utils import deep_gemm as vdg

    vdg._lazy_init()
    if args.no_pdl:
        vdg._apply_pdl(vdg._import_deep_gemm(), False)
    torch.manual_seed(0)
    dev = "cuda"
    hc, hid = args.hc_mult, args.hidden
    k = hc * hid
    mix = hc * (hc + 2)
    fn = torch.randn(mix, k, device=dev, dtype=torch.float32) * 0.02
    hc_scale = torch.rand(3, device=dev, dtype=torch.float32) + 0.5
    hc_base = torch.randn(mix, device=dev, dtype=torch.float32) * 0.1
    norm_w = (torch.rand(hid, device=dev) + 0.5).to(torch.bfloat16)
    rows = []
    stock_splits = mhc_wu.compute_mhc_pre_num_splits
    try:
        for t in args.tokens:
            residual = torch.randn(t, hc, hid, device=dev, dtype=torch.bfloat16)
            pre_mix = torch.softmax(torch.randn(t, hc, device=dev), -1).float().contiguous()
            x = residual.view(t, k)
            ref = None
            for s in [16] + [s for s in args.splits if s != 16]:
                mixes = torch.empty(s, t, mix, device=dev, dtype=torch.float32)
                sqr = torch.empty(s, t, device=dev, dtype=torch.float32)
                gemm_us = graph_us(
                    lambda: vdg.tf32_hc_prenorm_gemm(x, fn, mixes, sqr, s), args.iters, args.reps
                )
                mhc_wu.compute_mhc_pre_num_splits = lambda _k, _t, s=s: s

                def pre():
                    return mhc_tl.mhc_pre_delayed_tilelang(
                        residual, fn, hc_scale, hc_base, args.eps, args.eps, args.eps, 2.0,
                        args.sinkhorn, pre_mix=pre_mix, norm_weight=norm_w, norm_eps=args.eps,
                    )

                out = [o.float().clone() for o in pre()]
                pre_us = graph_us(pre, args.iters, args.reps)
                if s == 16:
                    ref = out
                diff = max((a - b).abs().max().item() for a, b in zip(out, ref))
                row = {"tokens": t, "splits": s, "gemm_us": round(gemm_us, 2),
                       "pre_us": round(pre_us, 2), "maxabs_vs_16": diff}
                rows.append(row)
                print(json.dumps(row), flush=True)
    finally:
        mhc_wu.compute_mhc_pre_num_splits = stock_splits

    by_t = {}
    for r in rows:
        by_t.setdefault(r["tokens"], []).append(r)
    summary = {}
    for t, rs in by_t.items():
        base = next(r for r in rs if r["splits"] == 16)
        best = min(rs, key=lambda r: r["pre_us"])
        summary[t] = {"best_splits": best["splits"], "pre_us_16": base["pre_us"],
                      "pre_us_best": best["pre_us"],
                      "saving_us_per_call": round(base["pre_us"] - best["pre_us"], 2)}
    print("SUMMARY " + json.dumps(summary), flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"pdl": not args.no_pdl, "rows": rows, "summary": summary}, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
