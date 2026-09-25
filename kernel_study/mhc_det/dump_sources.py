#!/usr/bin/env python3
"""Dump the stock decode mHC kernels as the serve image builds them (reference for bitwise work).

- DeepGEMM sm120_tf32_hc_prenorm_gemm: JIT K=20480 and K=5120 at 16 splits, copy kernel.cu +
  cubin, SASS via cuobjdump.
- TileLang: mhc_pre_big_fuse_with_norm (the three decode variants) and mhc_post; CUDA source
  of each compiled kernel.
Output: results/2026-09-25-kernels/mhc-det/stock_src/
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import torch

OUT = "/repo/results/2026-09-25-kernels/mhc-det/stock_src"


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    from vllm.model_executor.kernels.mhc import tilelang_kernels as tk
    from vllm.model_executor.kernels.mhc.warmup import MHC_PRE_NORM_KERNEL
    from vllm.utils import deep_gemm as vdg

    vdg._lazy_init()
    dev = "cuda"
    for k in (20480, 5120):
        x = torch.randn(4, k, device=dev).bfloat16()
        fn = torch.randn(24, k, device=dev)
        mixes = torch.empty(16, 4, 24, device=dev)
        sq = torch.empty(16, 4, device=dev)
        vdg.tf32_hc_prenorm_gemm(x, fn, mixes, sq, 16)
    torch.cuda.synchronize()
    home = os.environ.get("HOME", "/tmp/h")
    dirs = [os.path.join(r, n) for top in (home, "/root", "/tmp") for r, ds, _ in os.walk(top) for n in ds
            if n.startswith("kernel.sm120_tf32_hc_prenorm_gemm.")]
    for d in sorted(set(dirs)):
        name = os.path.basename(d)
        dst = os.path.join(OUT, name)
        os.makedirs(dst, exist_ok=True)
        for f in ("kernel.cu", "kernel.cubin"):
            if os.path.exists(os.path.join(d, f)):
                shutil.copy(os.path.join(d, f), dst)
        sass = subprocess.run(["/usr/local/cuda/bin/cuobjdump", "-sass", os.path.join(d, "kernel.cubin")],
                              capture_output=True, text=True)
        with open(os.path.join(dst, "kernel.sass"), "w") as fh:
            fh.write(sass.stdout + sass.stderr)
        print("deepgemm", name, open(os.path.join(d, "kernel.cu")).read().split("impl<")[1][:80].replace("\n", " "))

    common = dict(hidden_size=5120, rms_eps=1e-20, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6,
                  hc_post_mult_value=2.0, sinkhorn_repeat=20, norm_eps=1e-20, hc_mult=4, save_pre_mix=True)
    variants = {
        "pre_norm_s16_premix_k20480": dict(n_splits=16, use_pre_mix_in=True, rms_numel=20480),
        "pre_norm_s16_nopremix_k20480": dict(n_splits=16, use_pre_mix_in=False, rms_numel=20480),
        "pre_norm_s16_nopremix_k5120": dict(n_splits=16, use_pre_mix_in=False, rms_numel=5120),
    }
    for name, v in variants.items():
        kern = MHC_PRE_NORM_KERNEL.kernel.compile(**common, **v)
        with open(os.path.join(OUT, name + ".cu"), "w") as fh:
            fh.write(kern.get_kernel_source())
        print("tilelang", name, "ok")
    post = tk.mhc_post_tilelang.compile(hc=4, hidden=5120)
    with open(os.path.join(OUT, "mhc_post.cu"), "w") as fh:
        fh.write(post.get_kernel_source())
    print("tilelang mhc_post ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
