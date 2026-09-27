#!/usr/bin/env python3
"""Serve-like chain: a layer's dense decode GEMMs back to back inside one CUDA graph.

Per layer (real rank-0 weights of L distinct layers, so every weight is read once
per replay and L*87 MB >> caches: cold by construction, no flush):
  qkv_a(x) -> [q slice, scale] -> quant (stand-in for the fused q/kv norm-quant)
  -> wq_b(pre-quantized) -> [slice to 4096, scale] -> wo_b -> [scale] -> gate_up
  -> [silu*mul stand-in -> 1152] -> down -> [residual add] -> next layer
Arms: b12x (FlashInfer mxfp8_quantize + mm_mxfp8 auto for each GEMM, as the serve)
vs the dsv41 GEMV (fused quant; wq_b takes the same quantized activation), with and
without PDL. The glue kernels are identical in both arms. Timed per graph replay
with CUDA events, interleaved arms, >= 200 replays; per-layer = replay / L.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weights  # noqa: E402
from benchutil import load_ext, stats  # noqa: E402

SHAPES = ["qkv_a", "wq_b", "wo_b", "shared_gate_up", "shared_down"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--ms", default="3,4,6,8")
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    import dense_gemv  # /opt/dsv41-patch (SERVE_PATCH=1) - the serve's module and kernel
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize, swizzle_mxfp8_scale
    from vllm.utils import flashinfer as vfi

    ext = dense_gemv._ext()
    L = args.layers
    layers = []
    for li in range(L):
        per = {}
        for name in SHAPES:
            _, N, K, _ = weights.SHAPES[name]
            w, s2d = weights.load(name, li)
            wsw = swizzle_mxfp8_scale(s2d, M=N, K=K).contiguous()
            key = (K, N)
            nm, kc, smode, buckets = dense_gemv.CONFIGS[key]
            sc = dense_gemv.build_scales(s2d, N, K, kc, smode)
            per[name] = (w, wsw, sc, kc, smode, buckets)
        layers.append(per)

    def gemv(name, per, M, x=None, q=None, s=None, pdl=True):
        w, _, sc, kc, smode, buckets = per[name]
        W, S, MR = dense_gemv.pick(buckets, M)
        N, K = w.shape
        grid = int(ext.plan_grid(W, S, kc, MR, smode, 0, N, K))
        y = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        ext.gemv(x, None if q is None else q.view(torch.uint8), None if s is None else s.view(torch.uint8),
                 w.view(torch.uint8), sc, smode, y, W, S, kc, MR, grid, pdl)
        return y

    def b12x(name, per, x=None, q=None, s=None):
        w, wsw, *_ = per[name]
        if x is not None:
            q, s = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
        return vfi.mm_mxfp8(q, w.t(), s, wsw, out_dtype=torch.bfloat16, backend="auto")

    def chain(arm, x0, M):
        h = x0
        for per in layers:
            if arm == "b12x":
                g = lambda n, **kw: b12x(n, per, **kw)  # noqa: E731
            else:
                g = lambda n, **kw: gemv(n, per, M, pdl=(arm == "gemv_pdl"), **kw)  # noqa: E731
            y1 = g("qkv_a", x=h)
            qin = (y1[:, :1280] * 0.05).contiguous()
            q, s = mxfp8_e4m3_quantize(qin, is_sf_swizzled_layout=True)
            y2 = g("wq_b", q=q, s=s)
            y3 = g("wo_b", x=(y2[:, :4096] * 0.02).contiguous())
            y4 = g("shared_gate_up", x=(y3 * 0.05))
            y5 = g("shared_down", x=(torch.nn.functional.silu(y4[:, :1152]) * y4[:, 1152:] * 0.1).contiguous())
            h = (h + y5 * 0.01).contiguous()
        return h

    res = {"layers": L, "arms": {}}
    for M in [int(m) for m in args.ms.split(",")]:
        x0 = torch.randn(M, 5120, device="cuda").to(torch.bfloat16)
        graphs = {}
        outs = {}
        for arm in ("b12x", "gemv_pdl", "gemv_nopdl"):
            for _ in range(2):
                chain(arm, x0, M)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                outs[arm] = chain(arm, x0, M)
            graphs[arm] = g
        # same math in every arm: outputs must be bitwise equal after a replay
        for g in graphs.values():
            g.replay()
        torch.cuda.synchronize()
        eq = {a: bool(torch.equal(outs[a].view(torch.int16), outs["b12x"].view(torch.int16))) for a in outs}
        evs = {a: [] for a in graphs}
        names = list(graphs)
        for i in range(args.iters + 10):
            order = names if i % 2 == 0 else names[::-1]
            for a in order:
                st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                st.record()
                graphs[a].replay()
                en.record()
                if i >= 10:
                    evs[a].append((st, en))
        torch.cuda.synchronize()
        row = {}
        for a in names:
            per_layer = [s.elapsed_time(e) * 1000.0 / L for s, e in evs[a]]
            row[a] = {**stats(per_layer), "bitwise_eq_b12x": eq[a]}
        res["arms"][M] = row
        b = row["b12x"]["median"]
        print(f"M={M} per-layer dense chain us (median p10 p90): "
              + " | ".join(f"{a} {row[a]['median']:.1f} {row[a]['p10']:.1f} {row[a]['p90']:.1f} "
                           f"({100 * (b / row[a]['median'] - 1):+.1f}%, eq={eq[a]})" for a in names), flush=True)
        del graphs
    json.dump(res, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
