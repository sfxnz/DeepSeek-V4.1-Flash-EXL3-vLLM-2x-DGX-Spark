#!/usr/bin/env python3
"""SM120 DSV4 sparse-MLA: 32-candidate tiles, 4 math warps, keep KV double-buffer.

Stock is CAND_WINDOW=64, 8 math warps, COUNT=2, ~90 KiB smem, 1 block/SM.
COUNT=1 reached 2 blocks/SM but dropped IO/math overlap and did not beat CFG=1.

WINDOW=32 plus 4 math warps halves the KV smem tile. One w_fp8 slot (the vc
double-buffer is redundant with the existing bar_sync) lands the block under
51.2 KiB so two blocks can resident while COUNT=2 stays.

Python _BI must match CAND_WINDOW or num_splits disagrees with the kernel.
"""

from __future__ import annotations

import sys
from pathlib import Path

N_WARPS_OLD = "constexpr int DSV4_N_WARPS = 8;  // math warps"
N_WARPS_NEW = "constexpr int DSV4_N_WARPS = 4;  // math warps"

WIN_OLD = "constexpr int DSV4_CAND_WINDOW = 64;"
WIN_NEW = "constexpr int DSV4_CAND_WINDOW = 32;"

WFP8_OLD = "uint8_t* sm_w_fp8 = sm.w_fp8(vc & 1);"
WFP8_NEW = "uint8_t* sm_w_fp8 = sm.w_fp8(0);"

CU_WFP8_OLD = "+ 2 * HPB * (DSV4_BI + 16);                                     // sm_w_fp8 ×2 (vc parity)"
CU_WFP8_NEW = "+ 1 * HPB * (DSV4_BI + 16);                                     // sm_w_fp8 ×1 (bar_sync already orders vc)"

PY_BI_OLD = "_BI = 64  # KV partition tile size in candidates (BLOCK_SIZE_N)"
PY_BI_NEW = "_BI = 32  # KV partition tile size in candidates (BLOCK_SIZE_N)"

CORE_OLD = "    split_tile = 64"
CORE_NEW = "    split_tile = 32"

CUH_REL = Path(
    "include/flashinfer/attention/sparse_mla_sm120/decode_dsv4_kernel.cuh"
)
CU_REL = Path("csrc/sparse_mla_sm120_decode_dsv4.cu")
PY_REL = Path("mla/_sparse_mla_sm120.py")
CORE_REL = Path("mla/_core.py")


def patch_cuh(src: str) -> str:
    out = src
    if N_WARPS_OLD in out:
        out = out.replace(N_WARPS_OLD, N_WARPS_NEW, 1)
    if WIN_OLD in out:
        out = out.replace(WIN_OLD, WIN_NEW, 1)
    if WFP8_OLD in out:
        out = out.replace(WFP8_OLD, WFP8_NEW, 1)
    if N_WARPS_NEW not in out:
        raise SystemExit("widen_mla_tile32: DSV4_N_WARPS = 4 not present")
    if WIN_NEW not in out:
        raise SystemExit("widen_mla_tile32: DSV4_CAND_WINDOW = 32 not present")
    if WFP8_OLD in out:
        raise SystemExit("widen_mla_tile32: w_fp8(vc & 1) still present")
    if WFP8_NEW not in out:
        raise SystemExit("widen_mla_tile32: w_fp8(0) not present")
    if "constexpr int DSV4_KV_BUF_COUNT = 2;" not in out:
        raise SystemExit("widen_mla_tile32: KV double-buffer COUNT=2 missing")
    return out


def patch_cu(src: str) -> str:
    out = src
    if CU_WFP8_OLD in out:
        out = out.replace(CU_WFP8_OLD, CU_WFP8_NEW, 1)
    if CU_WFP8_NEW not in out:
        raise SystemExit("widen_mla_tile32: dyn smem w_fp8 x1 not present")
    if CU_WFP8_OLD in out:
        raise SystemExit("widen_mla_tile32: dyn smem w_fp8 x2 still present")
    return out


def patch_py(src: str) -> str:
    out = src
    if PY_BI_OLD in out:
        out = out.replace(PY_BI_OLD, PY_BI_NEW, 1)
    if PY_BI_NEW not in out:
        raise SystemExit("widen_mla_tile32: _BI = 32 not present")
    if PY_BI_OLD in out:
        raise SystemExit("widen_mla_tile32: _BI = 64 still present")
    return out


def patch_core(src: str) -> str:
    """Workspace mid_out splits must match CAND_WINDOW. Stock hardcodes 64."""
    out = src
    if CORE_OLD in out:
        out = out.replace(CORE_OLD, CORE_NEW, 1)
    if CORE_NEW not in out:
        raise SystemExit("widen_mla_tile32: split_tile = 32 not present")
    if CORE_OLD in out:
        raise SystemExit("widen_mla_tile32: split_tile = 64 still present")
    return out


def _find(tree: Path, rel: Path, glob_name: str) -> Path:
    direct = tree / rel
    if direct.is_file():
        return direct
    data = tree / "data" / rel
    if data.is_file():
        return data
    hits = list(tree.rglob(glob_name))
    if not hits:
        raise SystemExit(f"widen_mla_tile32: no {glob_name} under {tree}")
    return hits[0]


def apply(tree: Path) -> bool:
    cuh = _find(tree, CUH_REL, "decode_dsv4_kernel.cuh")
    cu = _find(tree, CU_REL, "sparse_mla_sm120_decode_dsv4.cu")
    py = _find(tree, PY_REL, "_sparse_mla_sm120.py")
    core = _find(tree, CORE_REL, "_core.py")
    changed = False
    for path, patch in (
        (cuh, patch_cuh),
        (cu, patch_cu),
        (py, patch_py),
        (core, patch_core),
    ):
        src = path.read_text()
        out = patch(src)
        if out != src:
            path.write_text(out)
            print(f"patched {path}")
            changed = True
    return changed


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: widen_mla_tile32.py TREE", file=sys.stderr)
        return 2
    apply(Path(argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
