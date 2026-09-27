#!/usr/bin/env python3
"""Driver for compute-sanitizer over every det mHC kernel (serve image, spark2; see san.sh).

From the review of 4f661a5 (reviewer's rv_sanitize.py / rv_init.py), plus the split kernels of
DSV41_MHC_DET_OVERLAP. Every T in 1..16 (--init: T 1, 5, 8, 9, 16), K 20480 and 5120, carried and
one-hot pre-mix, PDL launches as shipped: det post; fused pre (det GEMM + mhc_det_norm); split
pre (mhc_det_norm_li, then det GEMM + mhc_det_norm_coef forked to a side stream and joined).
Random data: these checks are about memory, barriers and initialization, not numerics.
--init: initcheck mode (run with PYTORCH_NO_CUDA_MEMORY_CACHING=1 and no kernel filter): every
output is read back on the host, then two canaries read deliberately uninitialized memory
(CANARY-1: post residual, plain loads; CANARY-2: GEMM x via cp.async and norm ld.cg), which
shows what the tool reports for each load kind.
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/repo/docker/patch")
import mhc_det  # noqa: E402
import mhc_det_overlap as O  # noqa: E402


def main() -> int:
    init = "--init" in sys.argv
    dk = mhc_det.DetKernels()
    O._S.torch = torch
    O._S.armed = O._S.on = True
    g = torch.Generator(device="cuda").manual_seed(1)
    fns = {k: (torch.randn(24, k, device="cuda", generator=g) * 0.01) for k in (20480, 5120)}
    packed = {k: mhc_det.pack_fn(f) for k, f in fns.items()}
    scale = torch.tensor([0.5, 0.7, 0.9], device="cuda")
    base = torch.randn(24, device="cuda", generator=g) * 0.1
    nw = (torch.rand(5120, device="cuda", generator=g) + 0.5).bfloat16()
    eps = (1e-20, 1e-6, 1e-6, 2.0, 20)
    n = 0
    checksum = 0.0
    for t in ((1, 5, 8, 9, 16) if init else range(1, 17)):
        residual = (torch.randn(t, 4, 5120, device="cuda", generator=g) * 3).bfloat16()
        x = (torch.randn(t, 5120, device="cuda", generator=g)).bfloat16()
        post_mix = (2 * torch.sigmoid(torch.randn(t, 4, 1, device="cuda", generator=g))).contiguous()
        comb = torch.softmax(torch.randn(t, 4, 4, device="cuda", generator=g), -1).contiguous()
        pm = torch.softmax(torch.randn(t, 4, device="cuda", generator=g), -1).contiguous()
        r2 = mhc_det.det_post(dk, x, residual, post_mix, comb)
        outs = [r2]
        for pre_mix in (pm, None):
            for how in ("fused", "fork", "in_place"):
                o = mhc_det.det_pre_delayed(dk, packed[20480], r2, fns[20480], scale, base, *eps, pre_mix=pre_mix,
                                            norm_weight=nw, norm_eps=1e-20,
                                            defer=None if how == "fused" else O.defer)
                if how == "fork":
                    O._fork(O._S.pending)
                O.settle()
                outs += list(o)
                n += 1
        bres = x.unsqueeze(1).expand(-1, 4, -1).contiguous()
        for how in ("fused", "fork"):
            o = mhc_det.det_pre_delayed(dk, packed[5120], bres, fns[5120], scale, base, *eps, pre_mix=None, x=x,
                                        norm_weight=nw, norm_eps=1e-20, defer=None if how == "fused" else O.defer)
            if how == "fork":
                O._fork(O._S.pending)
            O.settle()
            outs += list(o)
            n += 1
        if init:  # consume every output on the host: no later report can be about them
            checksum += sum(float(v.float().sum()) for v in outs)
    torch.cuda.synchronize()
    print(f"ok: {n} det pre calls (fused, split forked, split in place) + posts; forks {O._S.forks}, "
          f"in place {O._S.in_place}; checksum-finite={checksum == checksum}", flush=True)
    if not init:
        return 0
    t = 4
    bad_res = torch.empty(t, 4, 5120, dtype=torch.bfloat16, device="cuda")  # never written
    zx = torch.zeros(t, 5120, dtype=torch.bfloat16, device="cuda")
    mhc_det.det_post(dk, zx, bad_res, torch.ones(t, 4, 1, device="cuda"), torch.full((t, 4, 4), 0.25, device="cuda"))
    torch.cuda.synchronize()
    print("CANARY-1-DONE", flush=True)
    bad_x = torch.empty(t, 4, 5120, dtype=torch.bfloat16, device="cuda")  # never written
    mhc_det.det_pre_delayed(dk, packed[20480], bad_x, fns[20480], scale, base, *eps, pre_mix=None, norm_weight=nw,
                            norm_eps=1e-20)
    torch.cuda.synchronize()
    print("CANARY-2-DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
