#!/usr/bin/env python3
"""o_proj wo_a: every compiled KC512 COMPACT32 kernel config (W x MR) of the grouped
GEMV against the production pick (dense_gemv.WOA_CONFIG), same inputs and protocol as
woa_bench.py (serve producer, prepacked scale, rotating real layers, cold, arms
interleaved). Each config is checked bitwise vs fp8_einsum before it is timed."""
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

CFGS = [(4, 2, 4), (4, 2, 8), (3, 2, 8), (2, 2, 4), (2, 2, 8)]  # (W, S, MR) compiled at KC512 COMPACT32


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
        copies.append((w, sp, dense_gemv.build_scales(s2d, G * D, K, kc, smode)))
    cos_sin = torch.randn(8192, 64, device="cuda", dtype=torch.float32)
    gen = torch.Generator(device="cuda").manual_seed(7)

    def produce(M, dist="normal"):
        oa = torch.randn(M, 32, 512, generator=gen, device="cuda")
        if dist == "lognormal":
            oa = oa * torch.exp(1.5 * torch.randn(M, 32, 512, generator=gen, device="cuda"))
        pos = torch.randint(0, 8000, (M,), generator=gen, device="cuda")
        return fused_inv_rope_fp8_quant(oa.to(torch.bfloat16), pos, cos_sin, n_groups=G, heads_per_group=8,
                                        nope_dim=448, rope_dim=64, quant_group_size=32, tma_aligned_scales=True)

    report = {"prod": {}, "M": {}}
    all_ok = True
    for M in (1, 3, 4, 6, 8):
        pw, ps, pmr = dense_gemv.pick(buckets, M)
        report["prod"][M] = f"W{pw}S{ps}MR{pmr}"
        cfgs = []
        for W, S, MR in CFGS:
            if MR < M:
                continue
            grid = int(prod.plan_grid(W, S, kc, MR, smode, 1, G * D, K, G))
            if grid < G:
                continue
            bad = 0
            for r in range(6):
                c = copies[r % len(copies)]
                x8, xs = produce(M, ("normal", "lognormal")[r % 2])
                z = torch.empty(M, G, D, device="cuda", dtype=torch.bfloat16)
                fp8_einsum("bhr,hdr->bhd", (x8, xs), (c[0].view(G, D, K), c[1]), z, recipe=recipe)
                y = torch.empty(M, G, D, device="cuda", dtype=torch.bfloat16)
                prod.gemv_grouped(x8.view(torch.uint8), xs, c[0].view(torch.uint8), c[2], smode,
                                  y.view(M, G * D), W, S, kc, MR, grid, False, G)
                torch.cuda.synchronize()
                bad += not torch.equal(y.view(torch.int16), z.view(torch.int16))
            all_ok &= bad == 0
            if bad == 0:
                cfgs.append((f"W{W}S{S}MR{MR}g{grid}", W, S, MR, grid))
            else:
                print(f"M={M} W{W} S{S} MR{MR}: {bad}/6 not bitwise", flush=True)
        x8, xs = produce(M)
        y = torch.empty(M, G * D, device="cuda", dtype=torch.bfloat16)
        x8u = x8.view(torch.uint8)
        fns = {tag: (lambda c, W=W, S=S, MR=MR, grid=grid: prod.gemv_grouped(
            x8u, xs, c[0].view(torch.uint8), c[2], smode, y, W, S, kc, MR, grid, False, G))
            for tag, W, S, MR, grid in cfgs}
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
        line = " | ".join(f"{nm} {report['M'][M][nm]['median']:.2f}" for nm in names)
        print(f"wo_a M={M} (prod W{pw}S{ps}MR{pmr}) cold median us: {line}", flush=True)
    report["all_ok"] = all_ok
    json.dump(report, open(args.out, "w"), indent=1)
    print("ALL_OK" if all_ok else "FAILURES", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
