#!/usr/bin/env python3
"""Drop SM120 DSV4 sparse-MLA KV smem double-buffer from 2 to 1.

The decode kernel is smem-bound at 1 block/SM (KV fp8 double-buffer ~58 KiB
of ~100 KiB GB10 smem). One buffer halves that so two blocks can resident.
Stock indexes the slot with `(chunk_idx - chunk_lo) & 1`; that OOB-writes
slot 1 when COUNT=1. Index with `% DSV4_KV_BUF_COUNT` instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

OLD = "constexpr int DSV4_KV_BUF_COUNT = 2;"
NEW = "constexpr int DSV4_KV_BUF_COUNT = 1;"
BUF_OLD = "const int buf = (chunk_idx - chunk_lo) & 1;"
BUF_NEW = "const int buf = (chunk_idx - chunk_lo) % DSV4_KV_BUF_COUNT;"

REL = Path(
    "include/flashinfer/attention/sparse_mla_sm120/decode_dsv4_kernel.cuh"
)


def patch_cuh(src: str) -> str:
    """COUNT=1 plus buf index. Stock uses `& 1`, which OOB-writes slot 1
    when only one KV smem buffer exists (CUDA launch failure)."""
    out = src
    if OLD in out:
        out = out.replace(OLD, NEW, 1)
    if BUF_OLD in out:
        out = out.replace(BUF_OLD, BUF_NEW)
    if NEW not in out:
        raise SystemExit("widen_mla_kv_buf: DSV4_KV_BUF_COUNT = 1 not present")
    if BUF_OLD in out:
        raise SystemExit("widen_mla_kv_buf: hardcoded buf & 1 still present")
    if out.count(BUF_NEW) < 2:
        raise SystemExit("widen_mla_kv_buf: expected 2 buf % COUNT sites")
    return out


def _find_cuh(tree: Path) -> Path:
    direct = tree / REL
    if direct.is_file():
        return direct
    hits = list(tree.rglob("decode_dsv4_kernel.cuh"))
    if not hits:
        raise SystemExit(f"widen_mla_kv_buf: no decode_dsv4_kernel.cuh under {tree}")
    return hits[0]


def apply(tree: Path) -> bool:
    path = _find_cuh(tree)
    src = path.read_text()
    out = patch_cuh(src)
    if out == src:
        return False
    path.write_text(out)
    print(f"patched {path}")
    return True


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: widen_mla_kv_buf.py TREE", file=sys.stderr)
        return 2
    apply(Path(argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
