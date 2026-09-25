#!/usr/bin/env python3
"""Compare p2b kernel .text of the live chain (srcsort) and the coop build.

  text_identity.py chain_srcsort.cubin chain_coop.cubin

Exit 0 when every <BITS, CB, SORT> kernel of the live chain (SORT 0 and 1,
all six BITS/CB) has byte-identical machine code in the coop build, and the
coop build adds exactly one SORT=2 kernel, <2, 1, 2> (K=2 MCG). So
DSV41_P2B_COOP unset cannot change what the serve runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p2b_srcsort"))
from text_identity import kernels  # noqa: E402


def main(base: str, coop: str) -> int:
    a, b = kernels(base), kernels(coop)
    if not a:
        raise SystemExit(f"{base}: no p2b_moe_batched_kernel .text sections")
    bad = 0
    for key in sorted(a):
        same = a[key] == b.get(key)
        bad += not same
        print(f"BITS={key[0]} CB={key[1]} SORT={key[2]}: {len(a[key])} vs {len(b.get(key, b''))} bytes "
              f"{'IDENTICAL' if same else 'DIFFERENT'}")
    added = sorted(k for k in b if k not in a)
    for key in added:
        print(f"BITS={key[0]} CB={key[1]} SORT={key[2]}: {len(b[key])} bytes (new)")
    if added != [("2", "1", "2")]:
        print(f"expected exactly one new kernel <2, 1, 2>, got {added}")
        bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))
