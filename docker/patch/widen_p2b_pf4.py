#!/usr/bin/env python3
"""CFG=1 p2b GEMV software pipeline PF=2 → 4.

CFG=1 already uses 256-thread tiles and m-row work lists. PF=2 issues the
next B slice one MMA behind. PF=4 matches CFG=0's pipeline depth so more
DRAM/dequant overlaps mma_ab_h. Not a load opcode change (__ldcs stays).
Not MMA-over-m. launch_bounds(256, 4) stays.

Apply after widen_p2b_cfg1.py. Idempotent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

PF_OLD = "    constexpr int PF = CFG == 0 ? 4 : 2;"
PF_NEW = "    constexpr int PF = 4;"

DONE_MARKERS = (PF_NEW, "run_gemv_tile<BITS, 1, 1>", "__launch_bounds__(256, 4)")
LEFTOVERS = (PF_OLD,)


def _already(src: str) -> bool:
    return all(m in src for m in DONE_MARKERS) and PF_OLD not in src


def patch_cu(src: str) -> str:
    if _already(src):
        return src
    if "run_gemv_tile<BITS, 1, 1>" not in src:
        raise SystemExit("widen_p2b_pf4: apply widen_p2b_cfg1 first")
    if PF_OLD not in src:
        raise SystemExit("widen_p2b_pf4: CFG-dependent PF not found")
    out = src.replace(PF_OLD, PF_NEW, 1)
    if PF_OLD in out:
        raise SystemExit("widen_p2b_pf4: leftover CFG-dependent PF")
    if "run_gemv_tile<BITS, 1, 1>" not in out:
        raise SystemExit("widen_p2b_pf4: CFG=1 tile undone")
    if "__launch_bounds__(256, 4)" not in out:
        raise SystemExit("widen_p2b_pf4: launch_bounds undone")
    if "m * experts * warps_per_exp" not in out:
        raise SystemExit("widen_p2b_pf4: m-row work lists undone")
    if "grid.sync()" not in out:
        raise SystemExit("widen_p2b_pf4: cooperative fused kernel undone")
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
