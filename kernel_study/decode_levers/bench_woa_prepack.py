#!/usr/bin/env python3
"""wo_a fp8_einsum: fp32 weight scale vs pre-packed UE8M0 scale (GPU, serve DOWN).

Lever: DSV41_WOA_PREPACK=1 (fix_o_proj_woa_fp8.py stage 2). This bench uses
the serve shapes G=4 local groups, N=o_lora_rank 1024, K=4096, recipe (1,1,32),
with power-of-2 (e8m0-exact) weight scales, the same as the requant path. It
checks two things:
  1. bitwise: torch.equal of the einsum output with the fp32 scale vs the
     packed scale, for each M and 5 random seeds, and
  2. timing: us/call from a CUDA graph of --iters einsums, for each variant.
     The fp32 variant includes the per-call transpose_and_pack kernel.
The activation scale stays fp32 in both variants, so both pay the same
activation pack and the delta is the weight-scale pack.

  docker run --rm --gpus all --ipc host --network none \
    -v "$PWD":/w -w /w --entrypoint python3 dsv41-flash-exl3-sm121:canonical-e12 \
    kernel_study/decode_levers/bench_woa_prepack.py \
    --json results/2026-09-24-review/decode-levers/woa_prepack.json
"""

from __future__ import annotations

import argparse
import json
import sys

from bench_mhc_prenorm import graph_us


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, nargs="+", default=[3, 4, 8])
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--gran-k", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    import torch
    from vllm.utils import deep_gemm as vdg

    dev = "cuda"
    g, n, k, gk = args.groups, args.n, args.k, args.gran_k
    recipe = (1, 1, gk)
    torch.manual_seed(0)
    w = (torch.randn(g, n, k, device=dev) * 0.5).to(torch.float8_e4m3fn)
    ws = torch.exp2(torch.randint(-12, -4, (g, n, k // gk), device=dev).float())
    sp = vdg.transform_sf_into_required_layout(ws, n, k, recipe, g, False)
    print(f"packed sf: dtype={sp.dtype} shape={tuple(sp.shape)} stride={tuple(sp.stride())}", flush=True)

    rows, all_equal = [], True
    for m in args.m:
        equal = True
        for seed in range(args.seeds):
            gen = torch.Generator(device=dev)
            gen.manual_seed(seed)
            x = torch.randn(m, g, k, device=dev, generator=gen).to(torch.float8_e4m3fn)
            xs = torch.exp2(torch.randint(-8, 8, (m, g, k // gk), device=dev, generator=gen).float())
            z1 = torch.empty(m, g, n, device=dev, dtype=torch.bfloat16)
            z2 = torch.empty_like(z1)
            vdg.fp8_einsum("bhr,hdr->bhd", (x, xs), (w, ws), z1, recipe=recipe)
            vdg.fp8_einsum("bhr,hdr->bhd", (x, xs), (w, sp), z2, recipe=recipe)
            torch.cuda.synchronize()
            equal &= torch.equal(z1, z2)
        z = torch.empty(m, g, n, device=dev, dtype=torch.bfloat16)
        fp32_us = graph_us(
            lambda: vdg.fp8_einsum("bhr,hdr->bhd", (x, xs), (w, ws), z, recipe=recipe),
            args.iters, args.reps,
        )
        packed_us = graph_us(
            lambda: vdg.fp8_einsum("bhr,hdr->bhd", (x, xs), (w, sp), z, recipe=recipe),
            args.iters, args.reps,
        )
        all_equal &= equal
        row = {"m": m, "bitwise_equal": equal, "fp32_sf_us": round(fp32_us, 2),
               "packed_sf_us": round(packed_us, 2), "saving_us": round(fp32_us - packed_us, 2)}
        rows.append(row)
        print(json.dumps(row), flush=True)

    print(f"BITWISE {'PASS' if all_equal else 'FAIL'}", flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"rows": rows, "bitwise_equal": all_equal}, fh, indent=1)
    return 0 if all_equal else 1


if __name__ == "__main__":
    sys.exit(main())
