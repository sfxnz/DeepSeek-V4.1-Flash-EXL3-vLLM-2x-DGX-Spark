#!/usr/bin/env python3
"""setuptools rejects absolute CUDAExtension sources. Rewrite to repo-relative paths."""
from __future__ import annotations

import argparse
from pathlib import Path


def patch(src: str) -> str:
    old = '''                sources=[
                    str(ROOT / "csrc" / "bindings.cpp"),
                    str(ROOT / "csrc" / "exl3_gemv.cu"),
                    str(ROOT / "csrc" / "p2b_batched.cu"),
                    str(ROOT / "csrc" / "p2b_moe.cu"),
                    str(ROOT / "csrc" / "exl3_gemm.cu"),
                    str(ROOT / "csrc" / "exl3_fat_gemm.cu"),
                ],'''
    new = '''                sources=[
                    "csrc/bindings.cpp",
                    "csrc/exl3_gemv.cu",
                    "csrc/p2b_batched.cu",
                    "csrc/p2b_moe.cu",
                    "csrc/exl3_gemm.cu",
                    "csrc/exl3_fat_gemm.cu",
                ],'''
    if old not in src:
        if "csrc/bindings.cpp" in src:
            return src
        raise SystemExit("fix_vllm_exl3_setup: source list not found")
    return src.replace(old, new, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("setup_py", type=Path)
    args = ap.parse_args()
    args.setup_py.write_text(patch(args.setup_py.read_text()))
    print(f"patched {args.setup_py}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
