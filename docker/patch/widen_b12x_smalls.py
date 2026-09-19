#!/usr/bin/env python3
"""Double the CTA count of b12x MXFP8 dense GEMMs at decode-sized m.

The flashinfer sm120 blockscaled dense GEMM has no K-split; at m<=8 the
default (16,128) tile yields n/128 CTAs. Decode projections are n<=~6k, so
grids of 14-48 CTAs x 96 threads land ~1 CTA/SM on the 48-SM GB10 and stream
weights at ~111 GB/s (measured, live profile) while the p2b MoE kernel
sustains ~205 GB/s with 192x256 threads. (16,64) is a valid tile in the same
family (the m==1 branch already uses it); halving tile_n doubles CTAs for
n<=8192 and leaves wider GEMMs and prefill-sized m untouched.

Applies to flashinfer/gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py.
Idempotent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

OLD_EM = """        if expected_m is not None:
            if expected_m == 1:
                return (16, 64)
            if expected_m <= 8:
                return (16, 128)
            if expected_m <= 128:
                return (32, 128)
            return (64, 128)
        if m == 1:
            return (16, 64)
        if m <= 8:
            return (16, 128)"""

NEW_EM = """        if expected_m is not None:
            if expected_m == 1:
                return (16, 64)
            if expected_m <= 8:
                # widen_b12x_smalls: n/128 CTAs underfill 48 SMs at decode n;
                # halve tile_n for n<=8192 (2x CTAs), keep wide-n tiles.
                return (16, 64) if n <= 8192 else (16, 128)
            if expected_m <= 128:
                return (32, 128)
            return (64, 128)
        if m == 1:
            return (16, 64)
        if m <= 8:
            return (16, 64) if n <= 8192 else (16, 128)"""


def patch(src: str) -> str:
    if "widen_b12x_smalls" in src:
        return src
    if src.count(OLD_EM) != 1:
        raise SystemExit(
            f"widen_b12x_smalls: expected-m block not found (count={src.count(OLD_EM)})"
        )
    return src.replace(OLD_EM, NEW_EM, 1)


def apply(path: Path) -> None:
    src = path.read_text()
    out = patch(src)
    if out != src:
        path.write_text(out)
    print(f"patched {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", type=Path)
    args = ap.parse_args()
    apply(args.target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
