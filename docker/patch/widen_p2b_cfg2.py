#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

CFG_OLD = "run_gemv_tile<BITS, 1, 1>"
CFG_NEW = "run_gemv_tile<BITS, 1, 2>"

WNT_OLD = "    constexpr int WNT = CFG == 0 ? 2 : 4;"
WNT_NEW = "    constexpr int WNT = CFG == 0 ? 2 : (CFG == 1 ? 4 : 8);"

NUM_GROUPS_GATE_OLD = "    const int num_groups_gate = inter / 64;"
NUM_GROUPS_GATE_NEW = "    const int num_groups_gate = inter / 128;"

NUM_GROUPS_DOWN_OLD = "    const int num_groups_down = hidden / 64;"
NUM_GROUPS_DOWN_NEW = "    const int num_groups_down = hidden / 128;"

SH_RED_SHARED_OLD = "    __shared__ float sh_red[8][1][64];"
SH_RED_SHARED_NEW = "    __shared__ float sh_red[8][1][128];"

SH_RED_PTR_OLD = "float (*sh_red)[1][64]"
SH_RED_PTR_NEW = "float (*sh_red)[1][128]"

LAUNCH_BOUNDS = "__launch_bounds__(256, 4)"
OCCUPANCY = "cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, 256, 0);"
LAUNCH = "cudaLaunchCooperativeKernel(kernel, dim3(grid), dim3(256), args, 0, stream)"

DONE_MARKERS = (
    CFG_NEW,
    WNT_NEW,
    LAUNCH_BOUNDS,
    OCCUPANCY,
    LAUNCH,
    NUM_GROUPS_GATE_NEW,
    NUM_GROUPS_DOWN_NEW,
    SH_RED_SHARED_NEW,
    SH_RED_PTR_NEW,
)

LEFTOVERS = (
    CFG_OLD,
    WNT_OLD,
    NUM_GROUPS_GATE_OLD,
    NUM_GROUPS_DOWN_OLD,
    SH_RED_SHARED_OLD,
    SH_RED_PTR_OLD,
)


def _already(src: str) -> bool:
    return all(marker in src for marker in DONE_MARKERS) and not any(
        old in src for old in LEFTOVERS
    )


def _replace_all(src: str, old: str, new: str, label: str) -> str:
    if old not in src:
        if new in src:
            return src
        raise SystemExit(f"widen_p2b_cfg2: {label} not found")
    return src.replace(old, new)


def _replace_one(src: str, old: str, new: str, label: str) -> str:
    if new in src and old not in src:
        return src
    if old not in src:
        raise SystemExit(f"widen_p2b_cfg2: {label} not found")
    return src.replace(old, new, 1)


def patch_cu(src: str) -> str:
    if _already(src):
        return src
    had_mrow = "m * experts * warps_per_exp" in src
    if "run_gemv_tile<BITS, 1, 0>" in src:
        raise SystemExit("widen_p2b_cfg2: apply widen_p2b_cfg1.py first")
    out = _replace_all(src, CFG_OLD, CFG_NEW, "CFG=2 GEMV tile")
    out = _replace_one(out, WNT_OLD, WNT_NEW, "WNT ternary")
    out = _replace_one(out, NUM_GROUPS_GATE_OLD, NUM_GROUPS_GATE_NEW, "num_groups_gate")
    out = _replace_one(out, NUM_GROUPS_DOWN_OLD, NUM_GROUPS_DOWN_NEW, "num_groups_down")
    out = _replace_one(out, SH_RED_SHARED_OLD, SH_RED_SHARED_NEW, "sh_red shared")
    out = _replace_one(out, SH_RED_PTR_OLD, SH_RED_PTR_NEW, "sh_red pointer")
    if CFG_OLD in out:
        raise SystemExit("widen_p2b_cfg2: CFG=1 GEMV tile still present")
    if "run_gemv_tile<BITS, 2," in out:
        raise SystemExit("widen_p2b_cfg2: cb=2 tile is MUL1, not this patcher")
    if WNT_OLD in out:
        raise SystemExit("widen_p2b_cfg2: binary WNT ternary still present")
    if LAUNCH_BOUNDS not in out:
        raise SystemExit("widen_p2b_cfg2: launch_bounds(256, 4) missing")
    if "__launch_bounds__(512)" in out:
        raise SystemExit("widen_p2b_cfg2: launch_bounds(512) still present")
    if OCCUPANCY not in out:
        raise SystemExit("widen_p2b_cfg2: occupancy 256 missing")
    if LAUNCH not in out:
        raise SystemExit("widen_p2b_cfg2: cooperative launch 256 missing")
    if "sh_red[16]" in out:
        raise SystemExit("widen_p2b_cfg2: sh_red WK=16 still present")
    if "inter / 64" in out or "hidden / 64" in out:
        raise SystemExit("widen_p2b_cfg2: num_groups still divides by 64")
    if "inter / 32" in out or "hidden / 32" in out:
        raise SystemExit("widen_p2b_cfg2: num_groups still divides by 32")
    if "for (int row = 0; row < m; ++row)" in out:
        raise SystemExit("widen_p2b_cfg2: serial per-row moe loop is the reverted path")
    if had_mrow and "m * experts * warps_per_exp" not in out:
        raise SystemExit("widen_p2b_cfg2: m-row work lists were undone")
    if "template <int BITS>" not in out:
        raise SystemExit("widen_p2b_cfg2: BITS template missing")
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
