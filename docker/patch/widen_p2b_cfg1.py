#!/usr/bin/env python3
"""Switch vllm-exl3 p2b fused MoE decode tiles from CFG=0 to CFG=1.

CFG=0 is WK=16, WNT=2, COLS=32, THREADS=512. CFG=1 is WK=8, WNT=4, COLS=64,
THREADS=256. Flipping only the template arg is not enough. num_groups, sh_red,
launch_bounds, occupancy, and the cooperative launch still assume COLS=32 and
512 threads.

Apply after widen_p2b_shapes.py and widen_p2b_mrow.py. Idempotent. Keeps BITS
and m-row work lists. Does not regroup tiles over m (that path was MMA).
"""

from __future__ import annotations

import argparse
from pathlib import Path

CFG_OLD = "run_gemv_tile<BITS, 1, 0>"
CFG_NEW = "run_gemv_tile<BITS, 1, 1>"

NUM_GROUPS_GATE_OLD = "    const int num_groups_gate = inter / 32;"
NUM_GROUPS_GATE_NEW = "    const int num_groups_gate = inter / 64;"

NUM_GROUPS_DOWN_OLD = "    const int num_groups_down = hidden / 32;"
NUM_GROUPS_DOWN_NEW = "    const int num_groups_down = hidden / 64;"

SH_RED_SHARED_OLD = "    __shared__ float sh_red[16][1][32];"
SH_RED_SHARED_NEW = "    __shared__ float sh_red[8][1][64];"

SH_RED_PTR_OLD = "float (*sh_red)[1][32]"
SH_RED_PTR_NEW = "float (*sh_red)[1][64]"

LAUNCH_BOUNDS_OLD = "__launch_bounds__(512)"
LAUNCH_BOUNDS_NEW = "__launch_bounds__(256, 4)"

OCCUPANCY_OLD = "cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, 512, 0);"
OCCUPANCY_NEW = "cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, 256, 0);"

LAUNCH_OLD = "cudaLaunchCooperativeKernel(kernel, dim3(grid), dim3(512), args, 0, stream)"
LAUNCH_NEW = "cudaLaunchCooperativeKernel(kernel, dim3(grid), dim3(256), args, 0, stream)"

DONE_MARKERS = (
    CFG_NEW,
    LAUNCH_BOUNDS_NEW,
    OCCUPANCY_NEW,
    LAUNCH_NEW,
    NUM_GROUPS_GATE_NEW,
    NUM_GROUPS_DOWN_NEW,
    SH_RED_SHARED_NEW,
    SH_RED_PTR_NEW,
)

LEFTOVERS = (
    CFG_OLD,
    NUM_GROUPS_GATE_OLD,
    NUM_GROUPS_DOWN_OLD,
    SH_RED_SHARED_OLD,
    SH_RED_PTR_OLD,
    LAUNCH_BOUNDS_OLD,
    OCCUPANCY_OLD,
    LAUNCH_OLD,
)


def _already(src: str) -> bool:
    return all(marker in src for marker in DONE_MARKERS) and not any(
        old in src for old in LEFTOVERS
    )


def _replace_all(src: str, old: str, new: str, label: str) -> str:
    if old not in src:
        if new in src:
            return src
        raise SystemExit(f"widen_p2b_cfg1: {label} not found")
    return src.replace(old, new)


def _replace_one(src: str, old: str, new: str, label: str) -> str:
    if new in src and old not in src:
        return src
    if old not in src:
        raise SystemExit(f"widen_p2b_cfg1: {label} not found")
    return src.replace(old, new, 1)


def patch_cu(src: str) -> str:
    if _already(src):
        return src
    had_mrow = "m * experts * warps_per_exp" in src
    out = _replace_all(src, CFG_OLD, CFG_NEW, "CFG=1 GEMV tile")
    out = _replace_one(out, NUM_GROUPS_GATE_OLD, NUM_GROUPS_GATE_NEW, "num_groups_gate")
    out = _replace_one(out, NUM_GROUPS_DOWN_OLD, NUM_GROUPS_DOWN_NEW, "num_groups_down")
    out = _replace_one(out, SH_RED_SHARED_OLD, SH_RED_SHARED_NEW, "sh_red shared")
    out = _replace_one(out, SH_RED_PTR_OLD, SH_RED_PTR_NEW, "sh_red pointer")
    out = _replace_one(out, LAUNCH_BOUNDS_OLD, LAUNCH_BOUNDS_NEW, "launch_bounds")
    out = _replace_one(out, OCCUPANCY_OLD, OCCUPANCY_NEW, "occupancy threads")
    out = _replace_one(out, LAUNCH_OLD, LAUNCH_NEW, "cooperative launch threads")
    if CFG_OLD in out:
        raise SystemExit("widen_p2b_cfg1: CFG=0 GEMV tile still present")
    if LAUNCH_BOUNDS_OLD in out:
        raise SystemExit("widen_p2b_cfg1: launch_bounds(512) still present")
    if "__launch_bounds__(256)" in out and LAUNCH_BOUNDS_NEW not in out:
        raise SystemExit("widen_p2b_cfg1: launch_bounds missing minBlocks=4")
    if OCCUPANCY_OLD in out:
        raise SystemExit("widen_p2b_cfg1: occupancy 512 still present")
    if LAUNCH_OLD in out:
        raise SystemExit("widen_p2b_cfg1: cooperative launch 512 still present")
    if "sh_red[16]" in out:
        raise SystemExit("widen_p2b_cfg1: sh_red WK=16 still present")
    if "inter / 32" in out or "hidden / 32" in out:
        raise SystemExit("widen_p2b_cfg1: num_groups still divides by 32")
    if "for (int row = 0; row < m; ++row)" in out:
        raise SystemExit("widen_p2b_cfg1: serial per-row moe loop is the reverted path")
    if had_mrow and "m * experts * warps_per_exp" not in out:
        raise SystemExit("widen_p2b_cfg1: m-row work lists were undone")
    if "template <int BITS>" not in out:
        raise SystemExit("widen_p2b_cfg1: BITS template missing")
    return out


def apply(root: Path) -> None:
    cu = root / "csrc" / "p2b_moe.cu"
    cu.write_text(patch_cu(cu.read_text()))
    print(f"patched {cu}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    apply(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
