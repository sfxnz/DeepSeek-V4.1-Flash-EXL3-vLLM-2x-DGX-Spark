#!/usr/bin/env python3
"""o_proj wo_a: grouped GEMV (gemv_grouped, production kernel file) vs the serve's
DeepGEMM fp8_einsum on the serve's own inputs: the activation and its INT32-packed
MN-major ue8m0 scales come from fused_inv_rope_fp8_quant (SM12x layout), the weight
scale is the prepacked UE8M0 (DSV41_WOA_PREPACK=1). Bitwise check (M 1..8, 3 input
distributions, rotating real layers, graph replay), then cold timing (benchutil
protocol; producer outputs re-touched after the flush)."""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../docker/patch"))
import dense_gemv  # noqa: E402
import weights  # noqa: E402
from benchutil import Flusher, Rotation, load_ext, stats, time_arms  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--layers", type=int, default=40)
    args = ap.parse_args()
    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import fused_inv_rope_fp8_quant
    from vllm.utils.deep_gemm import fp8_einsum, transform_sf_into_required_layout

    ext = load_ext("dgemv_v1", ["gemv_ext.cu"])
    prod = load_ext("dsv41_dense_gemv_study", ["../../docker/patch/dense_gemv_kernel.cu"])
    fl = Flusher(ext)
    o = torch.zeros(4, dtype=torch.int32, device="cuda")
    G, D, K, recipe = 4, 1024, 4096, (1, 1, 32)
    name, kc, smode, buckets = dense_gemv.WOA_CONFIG
    copies = []
    for li in range(args.layers):
        w, s2d = weights.load("wo_a", li)
        s_f32 = torch.exp2(s2d.view(G, D, K // 32).float() - 127.0)
        sp = transform_sf_into_required_layout(s_f32, D, K, recipe, G, False)
        sc = dense_gemv.build_scales(s2d, G * D, K, kc, smode)
        copies.append((w, sp, sc))
    cos_sin = torch.randn(8192, 64, device="cuda", dtype=torch.float32)
    gen = torch.Generator(device="cuda").manual_seed(99)

    def produce(M, dist):
        oa = torch.randn(M, 32, 512, generator=gen, device="cuda")
        if dist == "lognormal":
            oa = oa * torch.exp(1.5 * torch.randn(M, 32, 512, generator=gen, device="cuda"))
        elif dist == "outlier":
            oa[:, :, torch.randint(0, 448, (4,), generator=gen, device="cuda")] *= 200.0
        pos = torch.randint(0, 8000, (M,), generator=gen, device="cuda")
        return fused_inv_rope_fp8_quant(oa.to(torch.bfloat16), pos, cos_sin, n_groups=G, heads_per_group=8,
                                        nope_dim=448, rope_dim=64, quant_group_size=32, tma_aligned_scales=True)

    report = {"correct": {}, "M": {}}
    all_ok = True
    for M in range(1, 9):
        W, S, MR = dense_gemv.pick(buckets, M)
        grid = int(prod.plan_grid(W, S, kc, MR, smode, 1, G * D, K, G))
        bad = n = 0
        for r in range(24):
            c = copies[r % len(copies)]
            x8, xs = produce(M, ("normal", "lognormal", "outlier")[r % 3])
            z = torch.empty(M, G, D, device="cuda", dtype=torch.bfloat16)
            fp8_einsum("bhr,hdr->bhd", (x8, xs), (c[0].view(G, D, K), c[1]), z, recipe=recipe)
            y = torch.empty(M, G, D, device="cuda", dtype=torch.bfloat16)
            prod.gemv_grouped(x8.view(torch.uint8), xs, c[0].view(torch.uint8), c[2], smode, y.view(M, G * D),
                              W, S, kc, MR, grid, False, G)
            torch.cuda.synchronize()
            n += 1
            bad += not torch.equal(y.view(torch.int16), z.view(torch.int16))
        # graph replay: capture once, new producer outputs copied into the captured buffers
        c = copies[0]
        x8, xs = produce(M, "normal")
        y = torch.empty(M, G, D, device="cuda", dtype=torch.bfloat16)
        prod.gemv_grouped(x8.view(torch.uint8), xs, c[0].view(torch.uint8), c[2], smode, y.view(M, G * D),
                          W, S, kc, MR, grid, False, G)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            prod.gemv_grouped(x8.view(torch.uint8), xs, c[0].view(torch.uint8), c[2], smode, y.view(M, G * D),
                              W, S, kc, MR, grid, False, G)
        gbad = 0
        for r in range(16):
            x8n, xsn = produce(M, ("lognormal", "outlier")[r % 2])
            x8.copy_(x8n)
            xs.as_strided((xs.untyped_storage().nbytes() // 4,), (1,)).copy_(
                xsn.as_strided((xsn.untyped_storage().nbytes() // 4,), (1,)))
            g.replay()
            z = torch.empty(M, G, D, device="cuda", dtype=torch.bfloat16)
            fp8_einsum("bhr,hdr->bhd", (x8, xs), (c[0].view(G, D, K), c[1]), z, recipe=recipe)
            torch.cuda.synchronize()
            gbad += not torch.equal(y.view(torch.int16), z.view(torch.int16))
        del g
        report["correct"][M] = {"cfg": [W, S, kc, MR, grid], "eager_bad": bad, "eager": n, "graph_bad": gbad,
                                "graph": 16}
        all_ok &= bad == 0 and gbad == 0
        print(f"wo_a M={M} W{W} S{S} KC{kc} MR{MR} grid{grid}: eager bad {bad}/{n}, graph bad {gbad}/16", flush=True)
    for M in (1, 3, 4, 6, 8):
        W, S, MR = dense_gemv.pick(buckets, M)
        grid = int(prod.plan_grid(W, S, kc, MR, smode, 1, G * D, K, G))
        x8, xs = produce(M, "normal")
        z = torch.empty(M, G, D, device="cuda", dtype=torch.bfloat16)
        y = torch.empty(M, G * D, device="cuda", dtype=torch.bfloat16)
        x8u = x8.view(torch.uint8)
        fns = {
            "dg_einsum": lambda c: fp8_einsum("bhr,hdr->bhd", (x8, xs), (c[0].view(G, D, K), c[1]), z, recipe=recipe),
            "gemv_grouped": lambda c: prod.gemv_grouped(x8u, xs, c[0].view(torch.uint8), c[2], smode, y, W, S, kc,
                                                        MR, grid, False, G),
            "read": lambda c: ext.read_flat(c[0].view(torch.uint8), o, 48, 2),
        }
        names = list(fns)
        rot = Rotation(copies, narms=len(names))
        arms = {nm: (lambda nm=nm, k=k: fns[nm](rot.get(k))) for k, nm in enumerate(names)}
        x8_flat = x8.transpose(0, 1).contiguous().view(-1).view(torch.uint8)
        xs_flat = xs.as_strided((xs.untyped_storage().nbytes() // 4,), (1,)).view(torch.uint8)

        def pre():
            fl()
            ext.read_flat(x8_flat, o, 8, 2)
            ext.read_flat(xs_flat, o, 8, 2)

        t = time_arms(arms, iters=args.iters, pre=pre)
        report["M"][M] = {nm: stats(v) for nm, v in t.items()}
        med = {nm: report["M"][M][nm]["median"] for nm in names}
        print(f"wo_a M={M} cold us: dg_einsum {med['dg_einsum']:.2f} | gemv_grouped {med['gemv_grouped']:.2f} "
              f"({100 * (med['dg_einsum'] / med['gemv_grouped'] - 1):+.1f}%) | read {med['read']:.2f}", flush=True)
    report["all_ok"] = all_ok
    json.dump(report, open(args.out, "w"), indent=1)
    print("ALL_OK" if all_ok else "FAILURES", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
