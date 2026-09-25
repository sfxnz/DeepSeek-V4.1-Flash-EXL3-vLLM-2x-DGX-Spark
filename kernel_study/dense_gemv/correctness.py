#!/usr/bin/env python3
"""Bitwise check of the dense GEMV against the production b12x MXFP8 path.

Gate (fixed before any result): for every shape, M in 1..8, activation
distribution and real layer tested, the GEMV output must equal the b12x output
bit for bit (bf16 viewed as int16), both from bf16 input (fused quant vs
FlashInfer mxfp8_quantize + b12x) and from pre-quantized input (b12x GEMM
only). The fused quant must also equal FlashInfer mxfp8_quantize byte for
byte (values and scales). CUDA-graph capture + replay must equal eager bit for
bit, including after the captured input buffer is rewritten.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weights  # noqa: E402
from benchutil import load_ext  # noqa: E402

MXFP8_SHAPES = ["qkv_a", "wq_b", "wo_b", "shared_gate_up", "shared_down", "indexer_wq_b", "engram_wkv", "lm_head"]
DISTS = ["normal", "lognormal", "outlier", "edge"]


def make_x(M: int, K: int, dist: str, gen: torch.Generator) -> torch.Tensor:
    dev = "cuda"
    if dist == "normal":
        x = torch.randn(M, K, generator=gen, device=dev)
    elif dist == "lognormal":  # heavy-tailed magnitudes, random signs
        x = torch.randn(M, K, generator=gen, device=dev) * torch.exp(1.5 * torch.randn(M, K, generator=gen, device=dev))
    elif dist == "outlier":  # a few massive channels, like LLM residual streams
        x = torch.randn(M, K, generator=gen, device=dev)
        idx = torch.randint(0, K, (8,), generator=gen, device=dev)
        x[:, idx] *= 300.0
    elif dist == "edge":  # powers of two, the 448*2^k boundary, zeros, -0, tiny, bf16 subnormals
        x = torch.randn(M, K, generator=gen, device=dev)
        pick = torch.randint(0, 8, (M, K), generator=gen, device=dev)
        e = torch.randint(-20, 20, (M, K), generator=gen, device=dev).float()
        x = torch.where(pick == 0, torch.exp2(e), x)
        x = torch.where(pick == 1, 448.0 * torch.exp2(e), x)
        x = torch.where(pick == 2, torch.zeros_like(x), x)
        x = torch.where(pick == 3, -torch.zeros_like(x), x)
        x = torch.where(pick == 4, x * 1e-30, x)
        x = torch.where(pick == 5, x * 9.2e-41, x)  # bf16 subnormal range
        x[:, :32] = 0.0  # one all-zero block per row
    else:
        raise KeyError(dist)
    return x.to(torch.bfloat16)


def b12x_ref(x, w, wsw):
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
    from vllm.utils import flashinfer as vfi

    q, s = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
    y = vfi.mm_mxfp8(q, w.t(), s, wsw, out_dtype=torch.bfloat16, backend="auto")
    return q, s, y


def unswizzle(s_swz: torch.Tensor, M: int, KB: int) -> torch.Tensor:
    mt, kt = (M + 127) // 128, (KB + 3) // 4
    v = s_swz.view(mt, kt, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(mt * 128, kt * 4)
    return v[:M, :KB]


def cmp(a: torch.Tensor, b: torch.Tensor) -> dict:
    ai, bi = a.view(torch.int16), b.view(torch.int16)
    ne = ai != bi
    d = (a.float() - b.float()).abs()
    fin = torch.isfinite(d)
    return {
        "bitwise": bool(not ne.any()),
        "n_diff": int(ne.sum()),
        "numel": a.numel(),
        "max_abs": float(d[fin].max()) if fin.any() else 0.0,
        "max_ulp": int((ai.int() - bi.int()).abs().max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--cfg", default="4,4,128", help="W,STAGES,KSPAN")
    ap.add_argument("--impl", default="v1", choices=["v1", "v3"])
    args = ap.parse_args()
    W, STG, KSP = (int(v) for v in args.cfg.split(","))
    ext = load_ext("dgemv_v1", ["gemv_ext.cu"])
    ext3 = load_ext("dgemv_v3", ["gemv3_ext.cu"]) if args.impl == "v3" else None
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import swizzle_mxfp8_scale

    only = {s for s in args.only.split(",") if s}
    gen = torch.Generator(device="cuda").manual_seed(20260925)
    report = {"cfg": args.cfg, "shapes": {}}
    all_ok = True
    for name in MXFP8_SHAPES:
        if only and name not in only:
            continue
        kind, N, K, layers = weights.SHAPES[name]
        rows = []
        for layer in layers[: args.layers]:
            w, s2d = weights.load(name, layer)
            assert tuple(w.shape) == (N, K) and w.dtype == torch.float8_e4m3fn, (w.shape, w.dtype)
            wsw = swizzle_mxfp8_scale(s2d, M=N, K=K).contiguous()
            sc = weights.compact_scale(s2d)
            modes = [(1, wsw)] + ([(0, sc)] if sc is not None else [])
            if ext3 is not None:
                if K % KSP:
                    print(f"{name}: K={K} not a multiple of KSPAN={KSP}; skipped", flush=True)
                    break
                m3, t3 = weights.v3_scales(s2d, N, K, KSP)
                modes = [(m3, t3)]
                tiles = N // 16
                g3 = min((tiles + W - 1) // W, 48 * max(1, 102400 // (ext3.smem3(W, STG, KSP, K) + 1024)))

                class _E:
                    @staticmethod
                    def gemv(x_, q_, s_, w_, sc_, sm_, y_, W_, S_, K_, grid_, pdl_):
                        ext3.gemv3(x_, q_, s_, w_, sc_, sm_, y_, W_, S_, K_, g3, pdl_)
                    quant_check = ext.quant_check
                gext = _E
            else:
                gext = ext
            for M in range(1, 9):
                for dist in DISTS:
                    x = make_x(M, K, dist, gen)
                    q, s, yref = b12x_ref(x, w, wsw)
                    # fused quant vs FlashInfer quant
                    xq_m = torch.empty(M, K, dtype=torch.uint8, device="cuda")
                    xs_m = torch.empty(M, K // 32, dtype=torch.uint8, device="cuda")
                    gext.quant_check(x, xq_m, xs_m)
                    q_ok = torch.equal(xq_m, q.view(torch.uint8)) and torch.equal(
                        xs_m, unswizzle(s.view(torch.uint8), M, K // 32))
                    for sm, wsc in modes:
                        grid = (N // 16 + W - 1) // W
                        y1 = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
                        gext.gemv(x, None, None, w.view(torch.uint8), wsc, sm, y1, W, STG, KSP, grid, False)
                        y2 = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
                        gext.gemv(None, q.view(torch.uint8), s.view(torch.uint8), w.view(torch.uint8), wsc, sm, y2,
                                  W, STG, KSP, grid, False)
                        torch.cuda.synchronize()
                        c1, c2 = cmp(y1, yref), cmp(y2, yref)
                        ok = c1["bitwise"] and c2["bitwise"] and q_ok
                        all_ok &= ok
                        rows.append({"layer": layer, "M": M, "dist": dist, "scale_mode": sm, "quant_equal": q_ok,
                                     "fused_vs_b12x": c1, "preq_vs_b12x": c2})
                        if not ok:
                            print(f"MISMATCH {name} L{layer} M={M} {dist} sm={sm} quant_eq={q_ok} "
                                  f"fused={c1} preq={c2}", flush=True)
            # graph capture + replay (fused path, M = 4 and 8, each scale mode)
            for sm, wsc in modes:
                for M in (1, 4, 8):
                    xs_ = make_x(M, K, "normal", gen)
                    y = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
                    grid = (N // 16 + W - 1) // W
                    gext.gemv(xs_, None, None, w.view(torch.uint8), wsc, sm, y, W, STG, KSP, grid, True)
                    torch.cuda.synchronize()
                    eager = y.clone()
                    g = torch.cuda.CUDAGraph()
                    s_ = torch.cuda.Stream()
                    with torch.cuda.stream(s_):
                        with torch.cuda.graph(g):
                            gext.gemv(xs_, None, None, w.view(torch.uint8), wsc, sm, y, W, STG, KSP, grid, True)
                    y.zero_()
                    g.replay()
                    torch.cuda.synchronize()
                    rep1 = torch.equal(y.view(torch.int16), eager.view(torch.int16))
                    xs_.copy_(make_x(M, K, "lognormal", gen))
                    g.replay()
                    ye = torch.empty_like(y)
                    gext.gemv(xs_, None, None, w.view(torch.uint8), wsc, sm, ye, W, STG, KSP, grid, False)
                    torch.cuda.synchronize()
                    rep2 = torch.equal(y.view(torch.int16), ye.view(torch.int16))
                    all_ok &= rep1 and rep2
                    rows.append({"layer": layer, "M": M, "graph_scale_mode": sm, "replay_eq_eager": rep1,
                                 "replay_after_input_rewrite_eq_eager": rep2})
                    del g
            del w, s2d, wsw, sc
        n = len([r for r in rows if "dist" in r])
        nbit = sum(1 for r in rows if "dist" in r and r["fused_vs_b12x"]["bitwise"] and r["preq_vs_b12x"]["bitwise"]
                   and r["quant_equal"])
        ng = [r for r in rows if "graph_scale_mode" in r]
        gok = all(r["replay_eq_eager"] and r["replay_after_input_rewrite_eq_eager"] for r in ng)
        report["shapes"][name] = {"N": N, "K": K, "cases": n, "bitwise_cases": nbit, "graph_ok": gok,
                                  "graph_cases": len(ng), "rows": rows}
        print(f"{name:15s} N={N:6d} K={K:6d}: bitwise {nbit}/{n} cases, graph replay {'ok' if gok else 'FAIL'} "
              f"({len(ng)} captures)", flush=True)
        torch.cuda.empty_cache()
    report["all_ok"] = all_ok
    json.dump(report, open(args.out, "w"), indent=1)
    print("ALL_OK" if all_ok else "FAILURES", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
