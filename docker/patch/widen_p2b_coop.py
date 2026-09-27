#!/usr/bin/env python3
"""Cooperative p2b decode MoE: stream each unique expert's trellis once.

DSpark-3 verify calls p2b with m=4 rows x top-6. The s10 census measured
29.87% of those (row, expert) pairs repeating an expert of another row in the
same call. p2b runs one GEMV tile per (row, expert, group) with the row in MMA
row 0, so a repeated expert streams its trellis once per row. Src-sort
(widen_p2b_srcsort.py) only moved the repeats next to each other and hoped
for L2 hits; it measured +0.8..1.4%.

DSV41_P2B_COOP=1 launches a SORT=2 instantiation instead (K=2 MCG only,
the serving format). Every block builds the src-sorted pair order
(p2b_sort_build) and splits each run of equal src into chunks of <= 8
members. The gate/up and down phases then run one tile per
(chunk, group[, gate|up]): lane group r = lane / 4 loads member r's
activation into MMA row r, so one B stream feeds up to 8 rows. p2b already
issues the m16n8k16 MMA with rows 1..15 zero, so the MMA count, the
registers and the prefetch ring are unchanged. This is not
widen_p2b_mma.py (REVERT, 10.2 vs 15.1 tok/s), which decoded B once and
then looped a separate MMA and fp32 accumulator set per row.

The final slot sum runs in fixed slot order with no atomics, so SORT=2 is
deterministic. Per (row, expert) the gate/up/down values are the same
arithmetic as p2b. The slot sum uses the same products (__fmul_rn) but a
fixed order instead of atomic arrival order, so with one-hot routing
weights the output equals p2b bit for bit, and with full weights it can
differ by fp32 reordering before the fp16 store (<= 1 fp16 ulp). The
launch, grid size, scratch and host checks are p2b's: no new allocation,
no host sync, one cooperative launch (CUDA-graph capture as today).

Off (unset or not "1"), for K != 2 or cb != 1, and when m*K > 64, the host
launches the SORT=0/1 instantiations, whose code this patch leaves as is
(kernel_study/p2b_coop/sass_identity.sh checks their machine code).

Apply after widen_p2b_srcsort.py. Idempotent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MARK = "// --- widen_p2b_coop"
ROWS = 8  # MMA rows 0..7 (lane / 4); m <= 8 and distinct top-k keep runs <= 8

HELPERS = r'''
// --- widen_p2b_coop: one B stream per unique expert (DSV41_P2B_COOP=1, SORT=2) ---
constexpr int P2B_COOP_ROWS = %d;

static bool p2b_coop_enabled()
{
    static const bool on = [] {
        const char* v = std::getenv("DSV41_P2B_COOP");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}

// [0, CAP) chunk start (sorted slot); [CAP, 2CAP) chunk length; [2CAP] chunk count.
__device__ __forceinline__ int* p2b_coop_smem()
{
    __shared__ int s[2 * P2B_SORT_CAP + 1];
    return s;
}

// Cross-warp reduction rows for the coop tile (WK = 8 warps at CFG 1).
typedef float P2bCoopRed[P2B_COOP_ROWS][64];
__device__ __forceinline__ P2bCoopRed* p2b_coop_red()
{
    __shared__ P2bCoopRed s[8];
    return s;
}

// After p2b_sort_build: runs of equal src cut into chunks of <= P2B_COOP_ROWS members.
__device__ __forceinline__ void p2b_coop_build(int pairs)
{
    const int* run_start = p2b_sort_smem() + P2B_SORT_CAP;
    const int* run_len = run_start + P2B_SORT_CAP;
    int* chunk = p2b_coop_smem();
    if (threadIdx.x == 0) {
        int u = 0;
        for (int s = 0; s < pairs; ++s) {
            const int off = s - run_start[s];
            if (off %% P2B_COOP_ROWS == 0) {
                chunk[u] = s;
                chunk[P2B_SORT_CAP + u] = min(P2B_COOP_ROWS, run_len[s] - off);
                ++u;
            }
        }
        chunk[2 * P2B_SORT_CAP] = u;
    }
    __syncthreads();
}

// Pair (row * experts + e) -> scratch slot (e * m + row), as the p2b phases index it.
__device__ __forceinline__ size_t p2b_coop_em(int pair, int m, int experts)
{
    return (size_t) (pair %% experts) * m + pair / experts;
}
''' % ROWS

# run_gemv_tile (after the whole chain) -> run_gemv_tile_coop. Everything not
# listed here (B stream, prefetch ring, decode, MMA, fold) is copied verbatim.
TILE_START = "template <int bits, int cb, int CFG>\n__device__ __forceinline__ void run_gemv_tile("
TILE_END = "template <int BITS, int CB, int SORT = 0>\n__global__ __launch_bounds__(256, 4)"
TILE_EDITS = (
    (
        "void run_gemv_tile(",
        "void run_gemv_tile_coop(",
        "tile name",
    ),
    (
        "    const half2* __restrict__ A2,\n",
        "    const half* __restrict__ A,\n",
        "A operand",
    ),
    (
        "    float (*sh_red)[1][64])\n",
        "    P2bCoopRed* sh_red,\n"
        "    const int* __restrict__ mem,\n"
        "    int len,\n"
        "    int m,\n"
        "    int experts)\n",
        "sh_red parameter",
    ),
    (
        "    const size_t a_row0 = 0;\n    const bool r0_ok = lane < 4;\n",
        "    // Lane group r0 = lane / 4 carries chunk member r0 in MMA row r0 (p2b: row 0 only).\n"
        "    const int r0 = lane >> 2;\n"
        "    const bool r0_ok = r0 < len;\n"
        "    const size_t a_row0 = 0;\n"
        "    const half2* A2 = reinterpret_cast<const half2*>(A + p2b_coop_em(mem[r0_ok ? r0 : 0], m, experts) * size_k);\n",
        "A row select",
    ),
    (
        "            a01[0] = r0_ok ? A2[a_row0 + a_col] : hzero;\n"
        "            a23[0] = r0_ok ? A2[a_row0 + a_col + 4] : hzero;\n",
        "            // Masked lanes read member 0's row, so the loads are unconditional global loads\n"
        "            // (sm_121a: 56/140 B spill like p2b instead of 64/156 with the guarded form).\n"
        "            const half2 a0v = A2[a_row0 + a_col];\n"
        "            const half2 a2v = A2[a_row0 + a_col + 4];\n"
        "            a01[0] = r0_ok ? a0v : hzero;\n"
        "            a23[0] = r0_ok ? a2v : hzero;\n",
        "A fragment load",
    ),
    (
        "    if (lane < 4) {\n",
        "    if (r0_ok) {\n",
        "warp reduction guard",
    ),
    (
        "                sh_red[warp][0][col + 0] = acc0[t][f].x;\n"
        "                sh_red[warp][0][col + 1] = acc0[t][f].y;\n",
        "                sh_red[warp][r0][col + 0] = acc0[t][f].x;\n"
        "                sh_red[warp][r0][col + 1] = acc0[t][f].y;\n",
        "warp reduction row",
    ),
    (
        "    for (int idx = threadIdx.x; idx < COLS; idx += THREADS) {\n"
        "        float sum = 0.0f;\n"
        "        #pragma unroll\n"
        "        for (int j = 0; j < WK; ++j)\n"
        "            sum += sh_red[j][0][idx];\n"
        "        const int col = group * COLS + idx;\n"
        "        C[col] = __float2half_rn(sum);\n"
        "    }\n",
        "    for (int idx = threadIdx.x; idx < COLS * len; idx += THREADS) {\n"
        "        const int r = idx / COLS;\n"
        "        const int c = idx % COLS;\n"
        "        float sum = 0.0f;\n"
        "        #pragma unroll\n"
        "        for (int j = 0; j < WK; ++j)\n"
        "            sum += sh_red[j][r][c];\n"
        "        const int col = group * COLS + c;\n"
        "        C[p2b_coop_em(mem[r], m, experts) * (size_t) (ntiles * 16) + col] = __float2half_rn(sum);\n"
        "    }\n",
        "block reduction store",
    ),
)

BUILD_OLD = "    if constexpr (SORT) p2b_sort_build(ids, m * experts);\n"
BUILD_NEW = BUILD_OLD + "    if constexpr (SORT == 2) p2b_coop_build(m * experts);\n"

GATE_OLD = "        int total_work = 2 * m * experts * num_groups_gate;\n"
GATE_NEW = """        if constexpr (SORT == 2) {
            // widen_p2b_coop: one tile per (chunk, group, gate|up); the chunk's rows share the B stream.
            const int* coop = p2b_coop_smem();
            const int units = coop[2 * P2B_SORT_CAP];
            for (int item = blockIdx.x; item < 2 * units * num_groups_gate; item += gridDim.x) {
                const int is_up = item & 1;
                const int u = (item >> 1) / num_groups_gate;
                const int group = (item >> 1) % num_groups_gate;
                const int* mem = p2b_sort_smem() + coop[u];
                const int src = ids[mem[0]];
                const uint32_t* B32 = reinterpret_cast<const uint32_t*>(is_up ? ut_ptrs[src] : gt_ptrs[src]);
                run_gemv_tile_coop<BITS, CB, 1>(B32, is_up ? had_up : had_gate, is_up ? up : gate, kslices_gate, hidden,
                                                group, ntiles_gate, warp, lane, p2b_coop_red(),
                                                mem, coop[P2B_SORT_CAP + u], m, experts);
            }
        }
        int total_work = SORT == 2 ? 0 : 2 * m * experts * num_groups_gate;
"""

DOWN_OLD = "        int total_work = m * experts * num_groups_down;\n"
DOWN_NEW = """        if constexpr (SORT == 2) {
            // widen_p2b_coop: one tile per (chunk, group).
            const int* coop = p2b_coop_smem();
            const int units = coop[2 * P2B_SORT_CAP];
            for (int item = blockIdx.x; item < units * num_groups_down; item += gridDim.x) {
                const int u = item / num_groups_down;
                const int group = item % num_groups_down;
                const int* mem = p2b_sort_smem() + coop[u];
                const int src = ids[mem[0]];
                const uint32_t* B32 = reinterpret_cast<const uint32_t*>(dt_ptrs[src]);
                run_gemv_tile_coop<BITS, CB, 1>(B32, had_down, down, kslices_down, inter,
                                                group, ntiles_down, warp, lane, p2b_coop_red(),
                                                mem, coop[P2B_SORT_CAP + u], m, experts);
            }
        }
        int total_work = SORT == 2 ? 0 : m * experts * num_groups_down;
"""

REDUCE_OLD = "        // Weighted reduction into accum\n"
REDUCE_NEW = """        if constexpr (SORT == 2) {
            // widen_p2b_coop: slots summed in fixed order, no atomics (deterministic). Same
            // products as the atomic path; one-hot routing weights give the same bits.
            for (int j = tid; j < m * hidden; j += total_threads) {
                const int row = j / hidden;
                const int col = j % hidden;
                float sum = 0.0f;
                for (int e = 0; e < experts; ++e)
                    sum = __fadd_rn(sum, __fmul_rn(__half2float(rw[row * experts + e]),
                                                   __half2float(down[((size_t) e * m + row) * hidden + col])));
                out[j] = __float2half(sum);
            }
            return;
        }
""" + REDUCE_OLD

LAUNCH_OLD = "        : (void*) p2b_moe_batched_kernel<BITS, CB, 0>;\n"
LAUNCH_NEW = LAUNCH_OLD + """    if constexpr (BITS == 2 && CB == 1) {
        if (p2b_coop_enabled() && m * e <= P2B_SORT_CAP)
            kernel = (void*) p2b_moe_batched_kernel<BITS, CB, 2>;
    }
"""


def _sub1(src: str, old: str, new: str, label: str) -> str:
    if src.count(old) != 1:
        raise SystemExit(f"widen_p2b_coop: {label}: expected 1 match, found {src.count(old)}")
    return src.replace(old, new, 1)


def coop_tile(src: str) -> str:
    """run_gemv_tile of src rewritten as run_gemv_tile_coop (TILE_EDITS only)."""
    if src.count(TILE_START) != 1 or src.count(TILE_END) != 1:
        raise SystemExit("widen_p2b_coop: run_gemv_tile / SORT kernel template not found (apply widen_p2b_srcsort.py first)")
    tile = src[src.index(TILE_START) : src.index(TILE_END)].rstrip("\n") + "\n"
    for old, new, label in TILE_EDITS:
        tile = _sub1(tile, old, new, f"run_gemv_tile {label}")
    return tile


def patch_cu(src: str) -> str:
    if MARK in src:
        return src
    out = _sub1(src, TILE_END, HELPERS.lstrip("\n") + "\n" + coop_tile(src) + "\n" + TILE_END, "kernel template")
    for old, new, label in (
        (BUILD_OLD, BUILD_NEW, "p2b_sort_build call (apply widen_p2b_srcsort.py first)"),
        (GATE_OLD, GATE_NEW, "gate/up work count"),
        (DOWN_OLD, DOWN_NEW, "down work count"),
        (REDUCE_OLD, REDUCE_NEW, "weighted reduction"),
        (LAUNCH_OLD, LAUNCH_NEW, "launch kernel pointer"),
    ):
        out = _sub1(out, old, new, label)
    if out.count("grid.sync()") != src.count("grid.sync()"):
        raise SystemExit("widen_p2b_coop: barrier count changed")
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
