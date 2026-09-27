#!/usr/bin/env python3
"""SM120 MXFP8 BMM: try b12x before BF16 emulation.

vLLM's init_mxfp8_linear_kernel(bmm_batch_size=...) only offers
DeepGemmMxfp8BmmLinearKernel then EmulationMxfp8LinearKernel. DeepGEMM BMM
requires SM100. GB10 is SM120/SM121, so MLA wo_a (is_bmm, n_local_groups)
falls through to EmulationMxfp8: dequant to BF16 at load, F.linear each step.

B12xMxfp8LinearKernel is already the SM120 dense path and accepts any config.
FlashInfer Cutlass is the next fallback. Emulation stays last.
"""

from __future__ import annotations

import sys
from pathlib import Path

BMM_OLD = """        possible = (
            [DeepGemmMxfp8BmmLinearKernel, EmulationMxfp8LinearKernel]
            if current_platform.is_cuda()
            else []
        )"""

BMM_NEW = """        possible = (
            [DeepGemmMxfp8BmmLinearKernel, FlashInferCutlassMxfp8LinearKernel, EmulationMxfp8LinearKernel]
            if current_platform.is_cuda()
            else []
        )"""

REL = Path("model_executor/kernels/linear/__init__.py")


def patch_py(src: str) -> str:
    out = src
    if BMM_OLD in out:
        out = out.replace(BMM_OLD, BMM_NEW, 1)
    if BMM_NEW not in out:
        raise SystemExit("prefer_b12x_bmm: BMM kernel list not present")
    if BMM_OLD in out:
        raise SystemExit("prefer_b12x_bmm: emulation-only BMM list still present")
    return out


def apply(tree: Path) -> bool:
    direct = tree / REL
    if direct.is_file():
        path = direct
    else:
        hits = list(tree.rglob("kernels/linear/__init__.py"))
        if not hits:
            raise SystemExit(f"prefer_b12x_bmm: no linear/__init__.py under {tree}")
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
        print("usage: prefer_b12x_bmm.py TREE", file=sys.stderr)
        return 2
    apply(Path(argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
