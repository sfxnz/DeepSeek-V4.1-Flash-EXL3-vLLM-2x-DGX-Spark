#!/usr/bin/env python3
"""wo_a opportunity: cold timing of the serve's deep_gemm fp8_einsum (4 groups x
[1024, 4096], packed ue8m0 weight scale as with DSV41_WOA_PREPACK=1) vs the GEMV on
the same bytes stacked as one [4096, 4096] matrix (upper bound for a grouped GEMV:
same weight stream, one activation). Real layers rotated, benchutil cold protocol."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../docker/patch"))
import dense_gemv  # noqa: E402
import weights  # noqa: E402
from benchutil import Flusher, Rotation, load_ext, stats, time_arms  # noqa: E402

from vllm.utils.deep_gemm import fp8_einsum, transform_sf_into_required_layout  # noqa: E402

ext = load_ext("dgemv_v1", ["gemv_ext.cu"])
prod = load_ext("dsv41_dense_gemv_study", ["../../docker/patch/dense_gemv_kernel.cu"])
fl = Flusher(ext)
o = torch.zeros(4, dtype=torch.int32, device="cuda")
G, D, K = 4, 1024, 4096
recipe = (1, 1, 32)
copies = []
for li in range(40):
    w, s2d = weights.load("wo_a", li)
    s_f32 = torch.exp2(s2d.view(G, D, K // 32).float() - 127.0)
    sp = transform_sf_into_required_layout(s_f32, D, K, recipe, G, False)
    sc = dense_gemv.build_scales(s2d, G * D, K, 512, 0)
    copies.append((w, sp, sc))
from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import fused_inv_rope_fp8_quant  # noqa: E402

res = {}
cos_sin = torch.randn(4096, 64, device="cuda", dtype=torch.float32)
for M in (1, 3, 4, 6, 8):
    # the serve's producer: inverse RoPE + per-32 ue8m0 quant, INT32-packed MN-major scales (SM12x)
    o_att = torch.randn(M, 32, 512, device="cuda").to(torch.bfloat16)
    pos = torch.arange(M, device="cuda", dtype=torch.int64) + 100
    x8, xs = fused_inv_rope_fp8_quant(o_att, pos, cos_sin, n_groups=G, heads_per_group=8, nope_dim=448,
                                      rope_dim=64, quant_group_size=32, tma_aligned_scales=True)
    if M == 4:
        print("producer:", x8.shape, x8.stride(), x8.dtype, xs.shape, xs.stride(), xs.dtype, flush=True)
    z = torch.empty(M, G, D, device="cuda", dtype=torch.bfloat16)
    xb = torch.randn(M, K, device="cuda").to(torch.bfloat16)
    y = torch.empty(M, G * D, device="cuda", dtype=torch.bfloat16)
    W, S, MR = dense_gemv.pick(dense_gemv.CONFIGS[(4096, 5120)][3], M)
    grid = int(prod.plan_grid(W, S, 512, MR, 0, 0, G * D, K))
    fns = {
        "dg_einsum": lambda c: fp8_einsum("bhr,hdr->bhd", (x8, xs), (c[0].view(G, D, K), c[1]), z, recipe=recipe),
        "gemv_stacked": lambda c: prod.gemv(xb, None, None, c[0].view(torch.uint8), c[2], 0, y, W, S, 512, MR,
                                            grid, False),
        "read": lambda c: ext.read_flat(c[0].view(torch.uint8), o, 48, 2),
    }
    names = list(fns)
    rot = Rotation(copies, narms=len(names))
    arms = {n: (lambda n=n, k=k: fns[n](rot.get(k))) for k, n in enumerate(names)}

    x8_flat = x8.transpose(0, 1).contiguous().view(-1).view(torch.uint8)  # same bytes as out_buf
    xs_flat = xs.as_strided((xs.untyped_storage().nbytes() // 4,), (1,)).view(torch.uint8)

    def pre():
        fl()
        for t in (x8_flat, xs_flat, xb.view(torch.uint8)):
            ext.read_flat(t, o, 8, 2)

    t = time_arms(arms, iters=200, pre=pre)
    res[M] = {n: stats(v) for n, v in t.items()}
    print(f"M={M} cold med (p10 p90) us: " + " | ".join(
        f"{n} {res[M][n]['median']:.2f} ({res[M][n]['p10']:.2f} {res[M][n]['p90']:.2f})" for n in names), flush=True)
json.dump(res, open("/repo/results/2026-09-25-kernels/dense-gemv/woa-time.json", "w"), indent=1)
