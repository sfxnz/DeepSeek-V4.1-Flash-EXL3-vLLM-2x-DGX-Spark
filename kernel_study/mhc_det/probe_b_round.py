#!/usr/bin/env python3
"""Which fp32 -> tf32 treatment of fn reproduces the stock DeepGEMM prenorm GEMM bitwise?

fn is pre-rounded on the host (truncate / round-to-nearest-away) or left to pack_fn's own
round-to-nearest-even, packed, run through the det GEMM, and compared with the stock kernel on
real fn / real embeddings. (A raw fp32 B operand is truncated by the mma itself: probe.py.)
First run (commit 61697f5, f32 kernel variant): rne 60/60, trunc/rna/raw 0/60.
"""
from __future__ import annotations

import json
import sys

import torch

sys.path.insert(0, "/repo/kernel_study/mhc_det")
import common as C  # noqa: E402
import mhc_det  # noqa: E402


def rounded(fn: torch.Tensor, mode: str) -> torch.Tensor:
    b = fn.contiguous().view(torch.int32)
    if mode == "trunc":
        r = b & ~0x1FFF
    elif mode == "rna":
        r = (b + 0x1000) & ~0x1FFF
    elif mode == "rne":
        r = (b + 0xFFF + ((b >> 13) & 1)) & ~0x1FFF
    else:
        r = b
    return r.view(torch.float32)


def main() -> int:
    from vllm.utils import deep_gemm as vdg

    vdg._lazy_init()
    fns = C.real_fns()
    emb = C.embeddings(64)
    dk = mhc_det.DetKernels()
    out = {}
    for mode in ("trunc", "rna", "rne"):
        n_eq = n = 0
        for li, (name, fn) in enumerate(fns[:20]):
            k = fn.shape[1]
            for t in (1, 4, 8):
                x = C.make_x("emb", t, k, seed=li * 31 + t, emb=emb)
                rm = torch.empty(16, t, 24, device="cuda")
                rs = torch.empty(16, t, device="cuda")
                vdg.tf32_hc_prenorm_gemm(x, fn, rm, rs, 16)
                m = torch.empty_like(rm)
                s = torch.empty_like(rs)
                dk.gemm_pk(x, mhc_det.pack_fn(rounded(fn, mode)), m, s)
                n += 1
                n_eq += int(C.bitwise_equal(m, rm))
        out[mode] = {"bitwise_equal_cases": n_eq, "cases": n}
        print(mode, out[mode], flush=True)
    # element-level: fraction of equal mixes elements per mode on one case
    name, fn = fns[5]
    x = C.make_x("emb", 4, fn.shape[1], seed=5, emb=emb)
    rm = torch.empty(16, 4, 24, device="cuda")
    rs = torch.empty(16, 4, device="cuda")
    vdg.tf32_hc_prenorm_gemm(x, fn, rm, rs, 16)
    for mode in ("trunc", "rna", "rne"):
        m = torch.empty_like(rm)
        s = torch.empty_like(rs)
        dk.gemm_pk(x, mhc_det.pack_fn(rounded(fn, mode)), m, s)
        out[f"elem_eq_frac_{mode}"] = float((m.view(torch.int32) == rm.view(torch.int32)).float().mean())
    print(json.dumps(out), flush=True)
    with open("/repo/results/2026-09-25-kernels/mhc-det/probe_b_round.json", "w") as fh:
        json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
