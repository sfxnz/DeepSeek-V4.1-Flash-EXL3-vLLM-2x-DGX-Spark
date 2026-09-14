#!/usr/bin/env python3
"""Prefer FlashInfer mm_mxfp8 auto (b12x on SM120) over hardcoded cutlass.

vLLM's FlashInferCutlassMxfp8LinearKernel always passes backend="cutlass".
On SM120/SM121, mm_mxfp8 auto puts b12x first: a warp-level MMA kernel with
small-M decode tiles. Cutlass large tiles pad skinny DSpark verify (m=6).
auto still falls back to cutlass when b12x requirements fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

OLD = """        output = vllm_flashinfer.mm_mxfp8(
            input_mxfp8,
            weight.t(),
            input_scale,
            weight_scale,
            out_dtype=out_dtype,
            backend="cutlass",
        )"""
NEW = """        output = vllm_flashinfer.mm_mxfp8(
            input_mxfp8,
            weight.t(),
            input_scale,
            weight_scale,
            out_dtype=out_dtype,
            backend="auto",
        )"""

REL = Path("model_executor/kernels/linear/mxfp8/flashinfer.py")


def patch_py(src: str) -> str:
    out = src
    if OLD in out:
        out = out.replace(OLD, NEW, 1)
    if NEW not in out:
        raise SystemExit("prefer_b12x_mxfp8: backend=auto not present")
    if "backend=\"cutlass\"" in out and OLD in out:
        raise SystemExit("prefer_b12x_mxfp8: cutlass backend still hardcoded")
    return out


def apply(tree: Path) -> bool:
    direct = tree / REL
    if direct.is_file():
        path = direct
    else:
        hits = list(tree.rglob("mxfp8/flashinfer.py"))
        if not hits:
            raise SystemExit(f"prefer_b12x_mxfp8: no mxfp8/flashinfer.py under {tree}")
        path = hits[0]
    src = path.read_text()
    out = patch_py(src)
    if out == src:
        return False
    path.write_text(out)
    print(f"patched {path}")
    return True


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: prefer_b12x_mxfp8.py TREE", file=sys.stderr)
        return 2
    apply(Path(argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
