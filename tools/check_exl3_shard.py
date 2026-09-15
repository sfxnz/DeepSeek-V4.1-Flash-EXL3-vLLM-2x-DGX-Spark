#!/usr/bin/env python3
"""Assert an output shard is mixed EXL3 (trellis experts, no leftover MXFP4 routed weights)."""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from pack_meta import is_routed_expert_weight  # noqa: E402


def tensor_names(path: Path) -> list[str]:
    with path.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    return [k for k in hdr if k != "__metadata__"]


def check_shard(path: Path) -> list[str]:
    names = tensor_names(path)
    leftover = [n for n in names if is_routed_expert_weight(n)]
    trellis = [n for n in names if n.endswith(".trellis")]
    suh = [n for n in names if n.endswith(".suh")]
    svh = [n for n in names if n.endswith(".svh")]
    mcg = [n for n in names if n.endswith(".mcg")]
    mul1 = [n for n in names if n.endswith(".mul1")]
    markers = mcg + mul1
    problems: list[str] = []
    if leftover:
        problems.append(f"leftover routed MXFP4 weights: {len(leftover)}")
    if not trellis:
        problems.append("no .trellis tensors")
    if trellis and not (len(trellis) == len(suh) == len(svh) == len(markers)):
        problems.append(
            "suffix counts "
            f"trellis={len(trellis)} suh={len(suh)} svh={len(svh)} "
            f"mcg={len(mcg)} mul1={len(mul1)}"
        )
    if mcg and mul1:
        problems.append("shard mixes .mcg and .mul1 markers")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("shard", type=Path, nargs="+")
    args = ap.parse_args()
    bad = 0
    for p in args.shard:
        probs = check_shard(p)
        if probs:
            print(f"FAIL {p}: {'; '.join(probs)}")
            bad += 1
        else:
            print(f"OK {p}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
