#!/usr/bin/env python3
# Adopted from the k3 dense-gemv review (the reviewer's independent harness); only the output
# path and comments that described the since-fixed smem-attribute defect were changed.
"""Reviewer check (k3 dense-gemv): does the persistent-grid GEMV keep its gain when the
serve's side-stream router GEMM runs concurrently with the shared expert?

Per layer (R real layers, cold by construction), captured in one CUDA graph per arm:
  fork -> side stream: router logits x @ Wgate^T (bf16 [M,5120] x [384,5120]^T, cuBLAS)
          main stream: shared gate_up -> silu_and_mul -> shared down
  join (main waits on side)
Arms: stock (serve's apply path: mxfp8 quant + b12x for gate_up and down) vs gemv (production
configs), each with and without the concurrent router. Timed per replay (CUDA events, ABBA,
20 warmup + ITERS), per-layer = replay / R. Outputs of every layer compared bitwise between arms.
"""
from __future__ import annotations

import json
import math
import statistics
import sys

import torch

sys.path.insert(0, "/repo/kernel_study/dense_gemv")
import weights  # noqa: E402

import dense_gemv  # noqa: E402


def pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(math.ceil(q * len(s))) - 1))]


def main() -> int:
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize, swizzle_mxfp8_scale
    from vllm.utils import flashinfer as vfi

    iters = 200
    R = 32
    ext = dense_gemv._ext()
    L = []
    for li in range(R):
        per = {}
        for name in ("shared_gate_up", "shared_down"):
            _, N, K, _ = weights.SHAPES[name]
            w, s2d = weights.load(name, li)
            nm, kc, smode, buckets = dense_gemv.CONFIGS[(K, N)]
            per[name] = (w, swizzle_mxfp8_scale(s2d, M=N, K=K).contiguous(), dense_gemv.build_scales(s2d, N, K, kc, smode),
                         kc, smode, buckets, N, K)
        per["router"] = weights.load("router_gate", li)[0]
        L.append(per)
    try:
        silu_and_mul = torch.ops._C.silu_and_mul

        def act(h):
            out = torch.empty(h.shape[0], h.shape[1] // 2, dtype=h.dtype, device=h.device)
            silu_and_mul(out, h)
            return out
    except Exception:  # noqa: BLE001
        def act(h):
            d = h.shape[1] // 2
            return torch.nn.functional.silu(h[:, :d]) * h[:, d:]

    side = torch.cuda.Stream()
    res = {}
    for M in (1, 4, 8):
        x = torch.randn(M, 5120, device="cuda").to(torch.bfloat16)
        plans = {}
        for name in ("shared_gate_up", "shared_down"):
            w, wsw, sc, kc, smode, buckets, N, K = L[0][name]
            W, S, MR = dense_gemv.pick(buckets, M)
            plans[name] = (W, S, MR, int(ext.plan_grid(W, S, kc, MR, smode, 0, N, K)))

        def lin(arm, per, name, inp):
            w, wsw, sc, kc, smode, buckets, N, K = per[name]
            if arm == "stock":
                q, s = mxfp8_e4m3_quantize(inp, is_sf_swizzled_layout=True)
                return vfi.mm_mxfp8(q, w.t(), s, wsw, out_dtype=torch.bfloat16, backend="auto")
            W, S, MR, grid = plans[name]
            y = torch.empty(inp.shape[0], N, dtype=torch.bfloat16, device="cuda")
            ext.gemv(inp, None, None, w.view(torch.uint8), sc, smode, y, W, S, kc, MR, grid, False)
            return y

        def layer_chain(arm, router):
            outs = []
            for per in L:
                if router:
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        logits = x @ per["router"].t()
                h = lin(arm, per, "shared_gate_up", x)
                y = lin(arm, per, "shared_down", act(h))
                if router:
                    torch.cuda.current_stream().wait_stream(side)
                    outs.append((y, logits))
                else:
                    outs.append((y, None))
            return outs

        graphs, outs = {}, {}
        for arm in ("stock", "gemv"):
            for router in (True, False):
                key = f"{arm}{'+router' if router else ''}"
                layer_chain(arm, router)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    outs[key] = layer_chain(arm, router)
                graphs[key] = g
        for g in graphs.values():
            g.replay()
        torch.cuda.synchronize()
        eq = all(torch.equal(a[0].view(torch.int16), b[0].view(torch.int16))
                 for a, b in zip(outs["gemv+router"], outs["stock+router"]))
        names = list(graphs)
        for _ in range(20):
            for n in names:
                graphs[n].replay()
        torch.cuda.synchronize()
        ev = {n: [] for n in names}
        for i in range(iters):
            for n in (names if i % 2 == 0 else names[::-1]):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                graphs[n].replay()
                b.record()
                ev[n].append((a, b))
        torch.cuda.synchronize()
        row = {}
        for n in names:
            t = [a.elapsed_time(b) * 1000.0 / R for a, b in ev[n]]
            row[n] = {"median": statistics.median(t), "p10": pct(t, 0.1), "p90": pct(t, 0.9), "mean": statistics.fmean(t)}
        row["gain_no_router_pct"] = 100 * (row["stock"]["median"] / row["gemv"]["median"] - 1)
        row["gain_with_router_pct"] = 100 * (row["stock+router"]["median"] / row["gemv+router"]["median"] - 1)
        row["saving_us_no_router"] = row["stock"]["median"] - row["gemv"]["median"]
        row["saving_us_with_router"] = row["stock+router"]["median"] - row["gemv+router"]["median"]
        row["bitwise_down_out"] = bool(eq)
        res[M] = row
        print(f"M={M} per-layer us (median [p10,p90]): " + " | ".join(
            f"{n} {row[n]['median']:.2f} [{row[n]['p10']:.2f},{row[n]['p90']:.2f}]" for n in names)
            + f" || saving no-router {row['saving_us_no_router']:.2f} us ({row['gain_no_router_pct']:+.1f}%), "
              f"with router {row['saving_us_with_router']:.2f} us ({row['gain_with_router_pct']:+.1f}%), bitwise {eq}",
            flush=True)
        del graphs, outs
    json.dump(res, open("/repo/results/2026-09-25-kernels/dense-gemv/concur.json", "w"), indent=1)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
