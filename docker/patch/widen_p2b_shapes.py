#!/usr/bin/env python3
"""Widen vllm-exl3 p2b fused MoE from hidden=4096 / inter in {1024,2048}.

The CUDA kernel is parameterized by hidden and inter. Hadamard tiles are 128
wide, so 5120x1152 (V4.1 TP=2) is a legal geometry. Host TORCH_CHECK and the
Python dispatch gate still hardcode GLM-shaped widths.
"""

from __future__ import annotations

import argparse
from pathlib import Path

CU_HIDDEN_OLD = """    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1 && x.size(1) == 4096,
                "fused MoE requires one input row with hidden width 4096");"""
CU_HIDDEN_NEW = """    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1 && x.size(1) > 0 && x.size(1) % 128 == 0,
                "fused MoE requires one input row whose hidden width is a positive multiple of 128");"""

CU_INTER_OLD = """    TORCH_CHECK(intermediate_size == 1024 || intermediate_size == 2048,
                "fused MoE local intermediate width must be 1024 or 2048");"""
CU_INTER_NEW = """    TORCH_CHECK(intermediate_size > 0 && intermediate_size % 128 == 0,
                "fused MoE local intermediate width must be a positive multiple of 128");"""

CU_CONST_OLD = """    constexpr int m = 1, hidden = 4096;
    const int inter = static_cast<int>(intermediate_size);"""
CU_CONST_NEW = """    constexpr int m = 1;
    const int hidden = static_cast<int>(x.size(1));
    const int inter = static_cast<int>(intermediate_size);"""

PY_OLD = """        and int(x2d.shape[1]) == hidden_meta == 4096
        and inter_meta in (1024, 2048)"""
PY_NEW = """        and int(x2d.shape[1]) == hidden_meta
        and hidden_meta > 0
        and hidden_meta % 128 == 0
        and inter_meta > 0
        and inter_meta % 128 == 0"""


def patch_cu(src: str) -> str:
    if CU_HIDDEN_NEW in src and CU_CONST_NEW in src:
        return src
    out = src
    for old, new, label in (
        (CU_HIDDEN_OLD, CU_HIDDEN_NEW, "p2b hidden TORCH_CHECK"),
        (CU_INTER_OLD, CU_INTER_NEW, "p2b intermediate TORCH_CHECK"),
        (CU_CONST_OLD, CU_CONST_NEW, "p2b hidden constexpr"),
    ):
        if old not in out:
            raise SystemExit(f"widen_p2b_shapes: {label} not found")
        out = out.replace(old, new, 1)
    return out


def patch_py(src: str) -> str:
    if PY_NEW in src:
        return src
    if PY_OLD not in src:
        raise SystemExit("widen_p2b_shapes: python geometry gate not found")
    return src.replace(PY_OLD, PY_NEW, 1)


def apply(root: Path) -> None:
    cu = root / "csrc" / "p2b_moe.cu"
    py = root / "src" / "vllm_exl3" / "exl3.py"
    if not py.is_file():
        py = root / "vllm_exl3" / "exl3.py"
    cu.write_text(patch_cu(cu.read_text()))
    py.write_text(patch_py(py.read_text()))
    print(f"patched {cu}")
    print(f"patched {py}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    apply(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
