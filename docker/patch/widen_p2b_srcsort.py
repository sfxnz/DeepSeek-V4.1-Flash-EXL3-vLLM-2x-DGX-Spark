#!/usr/bin/env python3
"""Src-sorted p2b work order: duplicate experts across verify rows run together.

DSpark-3 verify calls p2b with m=4 rows x top-6. The m-row work lists
(widen_p2b_mrow.py GEMV_GATE_NEW / GEMV_DOWN_NEW) are row-major, so the
same expert picked by rows r and r+1 is streamed twice about one grid wave
apart (~17.7 MB of gate/up traffic in between).

DSV41_P2B_SRC_SORT=1 launches a SORT=1 instantiation instead. Each block
sorts the m*K (row, expert) pairs by src in shared memory (stable, pair
index breaks ties). The gate/up and down work items then walk that order.
Inside a run of equal src, the items for one (group, gate|up) sit at
consecutive item indices, so neighbouring blocks stream the same trellis
lines at about the same time. Item math, scratch slots, reduce and
grid.sync count are unchanged, so the output is bit-exact.

Why this is not evidence/p2b-ldg (REVERT, L.A.I.L 22.87/21.22 vs
23.44/22.37): ldg kept the row-major order and made every B load L2-cached,
hoping a line survives ~18 MB of other streaming until the next row asks
for it. Here the loads stay __ldcs. The duplicate read is issued while the
first is in flight or just filled, so the reuse distance is ~0 instead of
~1 wave. It also adds no launches or barriers (evidence/p2b-nocoop lost 5%
on 9 launches); the only cost is a per-block prologue over m*K <= 64 ids.

Off (unset or not "1"), and when m*K > 64, the host launches the SORT=0
instantiation, whose work lists are the mrow code unchanged.

Apply after widen_p2b_fshift.py. Idempotent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MARK = "// --- widen_p2b_srcsort"

INCLUDE_OLD = "#include <cmath>\n"
INCLUDE_NEW = "#include <cmath>\n#include <cstdlib>\n#include <cstring>\n"

HELPERS = r'''
// --- widen_p2b_srcsort: src-sorted work order (bit-exact, DSV41_P2B_SRC_SORT=1) ---
constexpr int P2B_SORT_CAP = 64;

static bool p2b_src_sort_enabled()
{
    static const bool on = [] {
        const char* v = std::getenv("DSV41_P2B_SRC_SORT");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}

// [0, CAP) sorted slot -> pair (row * experts + e); [CAP, 2CAP) run start; [2CAP, 3CAP) run length.
__device__ __forceinline__ int* p2b_sort_smem()
{
    __shared__ int s[3 * P2B_SORT_CAP];
    return s;
}

__device__ __forceinline__ void p2b_sort_build(const int32_t* __restrict__ ids, int pairs)
{
    int* perm = p2b_sort_smem();
    int* run_start = perm + P2B_SORT_CAP;
    int* run_len = run_start + P2B_SORT_CAP;
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) {
        const int v = ids[p];
        int rank = 0;
        for (int q = 0; q < pairs; ++q) {
            const int w = ids[q];
            rank += (w < v) || (w == v && q < p);
        }
        perm[rank] = p;
    }
    __syncthreads();
    for (int s = threadIdx.x; s < pairs; s += blockDim.x) {
        const int v = ids[perm[s]];
        int a = s, b = s + 1;
        while (a > 0 && ids[perm[a - 1]] == v) --a;
        while (b < pairs && ids[perm[b]] == v) ++b;
        run_start[s] = a;
        run_len[s] = b - a;
    }
    __syncthreads();
}

// Item -> pair. Items of a run [s0, s0 + len) are [s0 * per_pair, (s0 + len) * per_pair);
// the run member varies fastest, so duplicates of one (sub) item are adjacent.
__device__ __forceinline__ int p2b_sort_pair(int item, int per_pair, int& sub)
{
    const int* perm = p2b_sort_smem();
    const int s = item / per_pair;
    const int s0 = perm[P2B_SORT_CAP + s];
    const int len = perm[2 * P2B_SORT_CAP + s];
    const int li = item - s0 * per_pair;
    sub = li / len;
    return perm[s0 + li % len];
}
'''

TILE_ANCHOR = "template <int bits, int cb, int CFG>\n__device__ __forceinline__ void run_gemv_tile("

KERNEL_TPL_OLD = "template <int BITS, int CB>\n__global__ __launch_bounds__(256, 4)"
KERNEL_TPL_NEW = "template <int BITS, int CB, int SORT = 0>\n__global__ __launch_bounds__(256, 4)"

BUILD_OLD = "    __shared__ float sh_red[8][1][64];\n"
BUILD_NEW = (
    "    __shared__ float sh_red[8][1][64];\n"
    "    if constexpr (SORT) p2b_sort_build(ids, m * experts);\n"
)

# Anchors are widen_p2b_mrow.py GEMV_GATE_NEW / GEMV_DOWN_NEW item decodes.
GATE_OLD = """            int is_up = item & 1;
            int rem = item >> 1;
            int row = rem / (experts * num_groups_gate);
            int rem2 = rem % (experts * num_groups_gate);
            int e = rem2 / num_groups_gate;
            int group = rem2 % num_groups_gate;
"""
GATE_NEW = """            int is_up, row, e, group;
            if constexpr (SORT) {
                int sub;
                const int pair = p2b_sort_pair(item, 2 * num_groups_gate, sub);
                is_up = sub & 1;
                group = sub >> 1;
                row = pair / experts;
                e = pair % experts;
            } else {
                is_up = item & 1;
                int rem = item >> 1;
                row = rem / (experts * num_groups_gate);
                int rem2 = rem % (experts * num_groups_gate);
                e = rem2 / num_groups_gate;
                group = rem2 % num_groups_gate;
            }
"""

DOWN_OLD = """            int row = item / (experts * num_groups_down);
            int rest = item % (experts * num_groups_down);
            int e = rest / num_groups_down;
            int group = rest % num_groups_down;
"""
DOWN_NEW = """            int row, e, group;
            if constexpr (SORT) {
                const int pair = p2b_sort_pair(item, num_groups_down, group);
                row = pair / experts;
                e = pair % experts;
            } else {
                row = item / (experts * num_groups_down);
                int rest = item % (experts * num_groups_down);
                e = rest / num_groups_down;
                group = rest % num_groups_down;
            }
"""

LAUNCH_OLD = "    void* kernel = (void*) p2b_moe_batched_kernel<BITS, CB>;\n"
LAUNCH_NEW = """    void* kernel = p2b_src_sort_enabled() && m * e <= P2B_SORT_CAP
        ? (void*) p2b_moe_batched_kernel<BITS, CB, 1>
        : (void*) p2b_moe_batched_kernel<BITS, CB, 0>;
"""

EDITS = (
    (INCLUDE_OLD, INCLUDE_NEW, "cmath include"),
    (TILE_ANCHOR, HELPERS.lstrip("\n") + "\n\n" + TILE_ANCHOR, "run_gemv_tile template"),
    (KERNEL_TPL_OLD, KERNEL_TPL_NEW, "p2b kernel template"),
    (BUILD_OLD, BUILD_NEW, "sh_red declaration"),
    (GATE_OLD, GATE_NEW, "gate/up m-row item decode (apply widen_p2b_mrow.py first)"),
    (DOWN_OLD, DOWN_NEW, "down m-row item decode (apply widen_p2b_mrow.py first)"),
    (LAUNCH_OLD, LAUNCH_NEW, "launch kernel pointer"),
)


def patch_cu(src: str) -> str:
    if MARK in src:
        return src
    out = src
    for old, new, label in EDITS:
        if out.count(old) != 1:
            raise SystemExit(f"widen_p2b_srcsort: {label}: expected 1 match, found {out.count(old)}")
        out = out.replace(old, new, 1)
    if out.count("grid.sync()") != src.count("grid.sync()"):
        raise SystemExit("widen_p2b_srcsort: barrier count changed")
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
