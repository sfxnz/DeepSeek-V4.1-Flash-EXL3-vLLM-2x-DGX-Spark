#!/usr/bin/env python3
"""wo_a probe: which deep_gemm sm120 config (split-K?) does the serve's fp8_einsum pick
for 4 groups x [1024, 4096] at M = 1..8, and is it bit-identical to 4 independent
per-group GEMVs of the same MMA chain (our kernel, pre-quantized activation)?"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../docker/patch"))
import dense_gemv  # noqa: E402
import weights  # noqa: E402
from benchutil import load_ext  # noqa: E402

os.environ.setdefault("DG_PRINT_CONFIGS", "1")
from vllm.utils.deep_gemm import fp8_einsum, transform_sf_into_required_layout  # noqa: E402

ext = load_ext("dsv41_dense_gemv_study", ["../../docker/patch/dense_gemv_kernel.cu"])
G, D, K = 4, 1024, 4096
w, s2d = weights.load("wo_a", 0)  # [4096, 4096] e4m3, per-row [4096, 128] (32x32 blocks)
w3 = w.view(G, D, K)
s_f32 = torch.exp2(s2d.view(G, D, K // 32).float() - 127.0)  # the serve's fp32 scale
recipe = (1, 1, 32)
sp = transform_sf_into_required_layout(s_f32, D, K, recipe, G, False)
res = {}
gen = torch.Generator(device="cuda").manual_seed(3)
for M in (1, 3, 4, 6, 8):
    x8 = (torch.randn(M, G, K, generator=gen, device="cuda") * 3).to(torch.float8_e4m3fn)
    xs = torch.exp2(torch.randint(-6, 6, (M, G, K // 32), generator=gen, device="cuda").float())
    z = torch.empty(M, G, D, device="cuda", dtype=torch.bfloat16)
    fp8_einsum("bhr,hdr->bhd", (x8, xs), (w3, sp), z, recipe=recipe)
    torch.cuda.synchronize()
    # per group, through our GEMV (pre-quantized path needs the 128x4 swizzled ue8m0 scale)
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import swizzle_mxfp8_scale
    ours = torch.empty_like(z)
    _, kc, smode, buckets = dense_gemv.CONFIGS[(4096, 5120)]  # wo_b's K=4096 configs
    W, S, MR = dense_gemv.pick(buckets, M)
    for g in range(G):
        wg = w3[g].contiguous()
        sc = dense_gemv.build_scales(s2d.view(G, D, K // 32)[g].contiguous(), D, K, kc, 0)
        xq = x8[:, g, :].contiguous().view(torch.uint8)
        xs_u8 = (torch.log2(xs[:, g, :]) + 127).round().to(torch.uint8)
        xs_sw = swizzle_mxfp8_scale(xs_u8, M=M, K=K).contiguous()
        grid = int(ext.plan_grid(W, S, kc, MR, 0, 0, D, K))
        y = torch.empty(M, D, dtype=torch.bfloat16, device="cuda")
        ext.gemv(None, xq, xs_sw, wg.view(torch.uint8), sc, 0, y, W, S, kc, MR, grid, False)
        ours[:, g, :] = y
    torch.cuda.synchronize()
    eq = bool(torch.equal(ours.view(torch.int16), z.view(torch.int16)))
    d = (ours.float() - z.float()).abs()
    res[M] = {"bitwise": eq, "n_diff": int((ours.view(torch.int16) != z.view(torch.int16)).sum()),
              "max_abs": float(d.max()), "rel": float(d.max() / z.float().abs().max())}
    print(f"M={M}: einsum vs per-group GEMV bitwise={eq} {res[M]}", flush=True)
json.dump(res, open("/repo/results/2026-09-25-kernels/dense-gemv/woa-probe.json", "w"), indent=1)
