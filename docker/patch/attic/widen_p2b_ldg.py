#!/usr/bin/env python3
"""Use L2-cached __ldg for p2b trellis B instead of streaming __ldcs.

CFG=1 + m-row still re-reads the same expert trellis once per activation
row. DSpark verify is m=6, so six DRAM trips if __ldcs bypasses L2.
__ldg keeps the 2bpw tiles in L2 for the later rows. Does not change CFG,
thread count, or m-row work lists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

OLD = "__ldcs("
NEW = "__ldg("


def patch_cu(src: str) -> str:
    if OLD not in src:
        if NEW in src and "ld_b" in src:
            return src
        raise SystemExit("widen_p2b_ldg: no __ldcs( and no __ldg( in p2b_moe.cu")
    out = src.replace(OLD, NEW)
    if OLD in out:
        raise SystemExit("widen_p2b_ldg: __ldcs( still present")
    if out.count(NEW) < 2:
        raise SystemExit("widen_p2b_ldg: expected at least two __ldg( B loads")
    if "run_gemv_tile" not in out:
        raise SystemExit("widen_p2b_ldg: run_gemv_tile missing")
    return out


def apply(root: Path) -> None:
    cu = root / "csrc" / "p2b_moe.cu"
    cu.write_text(patch_cu(cu.read_text()))
    print(f"patched {cu}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tree")
    args = ap.parse_args(argv)
    apply(Path(args.tree))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
