#!/usr/bin/env python3
"""SM120 DSV4 decode: two IO warps with linear io_tid, one expect_tx leader.

Stock DSV4_IO_WARPS=1 uses lane as the gather index. A naive IO_WARPS=4
duplicates entries 0-31 and fires expect_tx once per warp. Two warps cover
the 64-candidate window once: io_tid = threadIdx.x - MATH_THREADS, and
only thread MATH_THREADS arrives expect_tx. COUNT=2 and 8 math warps stay.
"""

from __future__ import annotations

import sys
from pathlib import Path

IO_OLD = "constexpr int DSV4_IO_WARPS = 1;"
IO_NEW = "constexpr int DSV4_IO_WARPS = 2;"

ENTRY_OLD = "      const int entry_idx = eo + lane;"
ENTRY_NEW = "      const int entry_idx = eo + io_tid;"

IO_TID_OLD = """    uint8_t* kv_fp8_dst = sm.kv_fp8(buf);
    bf16* kv_rope_dst = sm.kv_rope(buf);
    uint8_t* kv_sc_dst = sm.kv_sc(buf);

#pragma unroll
    for (int eo = 0; eo < DSV4_BI; eo += DSV4_IO_THREADS) {
      const int entry_idx = eo + lane;
"""

IO_TID_NEW = """    uint8_t* kv_fp8_dst = sm.kv_fp8(buf);
    bf16* kv_rope_dst = sm.kv_rope(buf);
    uint8_t* kv_sc_dst = sm.kv_sc(buf);
    const int io_tid = threadIdx.x - DSV4_MATH_THREADS;

#pragma unroll
    for (int eo = 0; eo < DSV4_BI; eo += DSV4_IO_THREADS) {
      const int entry_idx = eo + io_tid;
"""

EXPECT_OLD = """    if (lane == 0) {
      mbarrier_arrive_expect_tx(sm.mbar_full(buf), DSV4_BULK_TX_BYTES);
    }
"""
EXPECT_NEW = """    if (threadIdx.x == DSV4_MATH_THREADS) {
      mbarrier_arrive_expect_tx(sm.mbar_full(buf), DSV4_BULK_TX_BYTES);
    }
"""

REL = Path(
    "include/flashinfer/attention/sparse_mla_sm120/decode_dsv4_kernel.cuh"
)


def patch_cuh(src: str) -> str:
    if IO_NEW in src and "eo + io_tid" in src and EXPECT_NEW in src:
        return src
    out = src
    if IO_OLD in out:
        out = out.replace(IO_OLD, IO_NEW, 1)
    if IO_TID_OLD in out:
        out = out.replace(IO_TID_OLD, IO_TID_NEW, 1)
    out = out.replace(ENTRY_OLD, ENTRY_NEW)
    if EXPECT_OLD in out:
        out = out.replace(EXPECT_OLD, EXPECT_NEW, 1)
    if IO_NEW not in out:
        raise SystemExit("widen_mla_io2: DSV4_IO_WARPS = 2 not present")
    if "eo + io_tid" not in out:
        raise SystemExit("widen_mla_io2: linear io_tid gather missing")
    if ENTRY_OLD in out:
        raise SystemExit("widen_mla_io2: eo + lane gather still present")
    if EXPECT_OLD in out:
        raise SystemExit("widen_mla_io2: per-warp lane==0 expect_tx still present")
    if EXPECT_NEW not in out:
        raise SystemExit("widen_mla_io2: single expect_tx leader missing")
    if "constexpr int DSV4_N_WARPS = 8;" not in out:
        raise SystemExit("widen_mla_io2: 8 math warps undone")
    if "constexpr int DSV4_KV_BUF_COUNT = 2;" not in out:
        raise SystemExit("widen_mla_io2: KV COUNT=2 undone")
    if "constexpr int DSV4_CAND_WINDOW = 64;" not in out:
        raise SystemExit("widen_mla_io2: WINDOW=64 undone")
    return out


def _find_cuh(tree: Path) -> Path:
    direct = tree / REL
    if direct.is_file():
        return direct
    hits = list(tree.rglob("decode_dsv4_kernel.cuh"))
    if not hits:
        raise SystemExit(f"widen_mla_io2: no decode_dsv4_kernel.cuh under {tree}")
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
        print("usage: widen_mla_io2.py TREE", file=sys.stderr)
        return 2
    apply(Path(argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
