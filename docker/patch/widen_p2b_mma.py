#!/usr/bin/env python3
"""MMA over m rows in p2b run_gemv_tile: decode B once, reuse across rows.

Work-list scaling (widen_p2b_mrow.py) still launched one GEMV tile per
(row, expert, group), so expert weights were re-read m times. This regroups
pairs that share an expert id and runs one tile with m_loc activation rows.

Apply after widen_p2b_mrow.py. Idempotent. m_loc cap 8.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MAX_M = 8

GEMV_TILE_OLD = """template <int bits, int cb, int CFG>
__device__ __forceinline__ void run_gemv_tile(
    const uint32_t* __restrict__ B32,
    const half2* __restrict__ A2,
    half* __restrict__ C,
    int kslices,
    int size_k,
    int group,
    int ntiles,
    int warp,
    int lane,
    float (*sh_red)[1][32])
{
    constexpr int WK = CFG == 0 ? 16 : 8;
    constexpr int WNT = CFG == 0 ? 2 : 4;
    constexpr int PF = CFG == 0 ? 4 : 2;
    constexpr int FOLD = CFG == 0 ? 4 : 2;
    constexpr int THREADS = WK * 32;
    constexpr int COLS = WNT * 16;
    constexpr int TWORDS = 8 * bits;
    constexpr int LOADS = bits == 2 ? WNT / 2 : WNT;
    constexpr int LSTRIDE = bits == 3 ? 24 : 32;

    const int chunk = CEIL_DIVIDE(kslices, WK);
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));
    const size_t slice_stride = (size_t) ntiles * TWORDS;

    const size_t a_row0 = 0;
    const bool r0_ok = lane < 4;
    const half2 hzero = __half2half2(__ushort_as_half(0));
"""

GEMV_TILE_NEW = f"""template <int bits, int cb, int CFG>
__device__ __forceinline__ void run_gemv_tile(
    const uint32_t* __restrict__ B32,
    const half2* __restrict__ A2,
    const int* __restrict__ a_off,
    half* __restrict__ C,
    const int* __restrict__ c_off,
    int m_loc,
    int kslices,
    int size_k,
    int group,
    int ntiles,
    int warp,
    int lane,
    float (*sh_red)[1][32])
{{
    constexpr int WK = CFG == 0 ? 16 : 8;
    constexpr int WNT = CFG == 0 ? 2 : 4;
    constexpr int PF = CFG == 0 ? 4 : 2;
    constexpr int THREADS = WK * 32;
    constexpr int COLS = WNT * 16;
    constexpr int TWORDS = 8 * bits;
    constexpr int LOADS = bits == 2 ? WNT / 2 : WNT;
    constexpr int LSTRIDE = bits == 3 ? 24 : 32;
    constexpr int MAX_M = {MAX_M};

    const int chunk = CEIL_DIVIDE(kslices, WK);
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));
    const size_t slice_stride = (size_t) ntiles * TWORDS;
    (void)size_k;

    m_loc = min(m_loc, MAX_M);
    const bool r0_ok = lane < 4;
    const half2 hzero = __half2half2(__ushort_as_half(0));
"""

# The rest of the original tile from x_src through ld_b/prefetch is kept,
# then we replace the A-load/MMA/reduce tail.

TILE_A_MMA_OLD = """    FragC_h ch[WNT][2] = {};
    float2 acc0[WNT][2] = {};

    for (int ib = 0; ib < myn; ib += PF) {
        #pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int i = ib + d;
            if (i >= myn) break;

            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                bw[l] = pf[d][l];

            if (i + PF < myn) {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(i + PF, l);
            }

            const size_t a_col = (size_t) (ks0 + i) * 8 + (lane & 3);
            FragB a01, a23;
            a01[0] = r0_ok ? A2[a_row0 + a_col] : hzero;
            a23[0] = r0_ok ? A2[a_row0 + a_col + 4] : hzero;
            a01[1] = hzero;
            a23[1] = hzero;

            #pragma unroll
            for (int t = 0; t < WNT; ++t) {
                FragB f0, f1;
                if constexpr (bits == 4) {
                    uint32_t aw = __shfl_sync(0xffffffffu, bw[t], (lane + 31) & 31);
                    exl3_gemv_ns::dq8_regs_4bits<cb>(aw, bw[t], f0, f1);
                } else if constexpr (bits == 2) {
                    const uint32_t w = bw[t >> 1];
                    const int base = (t & 1) << 4;
                    uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                    uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    exl3_gemv_ns::dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);
                } else {
                    uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);
                    uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);
                    exl3_gemv_ns::dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);
                }

                exl3_gemv_ns::mma_ab_h(a01, a23, f0, ch[t][0]);
                exl3_gemv_ns::mma_ab_h(a01, a23, f1, ch[t][1]);
            }

            if ((d + 1) % FOLD == 0 || i + 1 == myn) {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        acc0[t][f].x += __low2float(ch[t][f][0]);
                        acc0[t][f].y += __high2float(ch[t][f][0]);
                        ch[t][f][0] = hzero;
                    }
            }
        }
    }

    // Warp reduction
    if (lane < 4) {
        #pragma unroll
        for (int t = 0; t < WNT; ++t) {
            #pragma unroll
            for (int f = 0; f < 2; ++f) {
                const int col = t * 16 + f * 8 + (lane & 3) * 2;
                sh_red[warp][0][col + 0] = acc0[t][f].x;
                sh_red[warp][0][col + 1] = acc0[t][f].y;
            }
        }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < COLS; idx += THREADS) {
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < WK; ++j)
            sum += sh_red[j][0][idx];
        const int col = group * COLS + idx;
        C[col] = __float2half_rn(sum);
    }
    __syncthreads();
}"""

TILE_A_MMA_NEW = """    float2 acc0[MAX_M][WNT][2] = {};

    for (int ib = 0; ib < myn; ib += PF) {
        #pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int i = ib + d;
            if (i >= myn) break;

            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                bw[l] = pf[d][l];

            if (i + PF < myn) {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(i + PF, l);
            }

            FragB f0s[WNT], f1s[WNT];
            #pragma unroll
            for (int t = 0; t < WNT; ++t) {
                if constexpr (bits == 4) {
                    uint32_t aw = __shfl_sync(0xffffffffu, bw[t], (lane + 31) & 31);
                    exl3_gemv_ns::dq8_regs_4bits<cb>(aw, bw[t], f0s[t], f1s[t]);
                } else if constexpr (bits == 2) {
                    const uint32_t w = bw[t >> 1];
                    const int base = (t & 1) << 4;
                    uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                    uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    exl3_gemv_ns::dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0s[t], f1s[t]);
                } else {
                    uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);
                    uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);
                    exl3_gemv_ns::dq8_regs_3bits<cb>(awv, bwv, x_s2, f0s[t], f1s[t]);
                }
            }

            const size_t a_col = (size_t) (ks0 + i) * 8 + (lane & 3);
            #pragma unroll
            for (int row = 0; row < MAX_M; ++row) {
                if (row >= m_loc) break;
                FragB a01, a23;
                const size_t a_row = (size_t) a_off[row];
                a01[0] = r0_ok ? A2[a_row + a_col] : hzero;
                a23[0] = r0_ok ? A2[a_row + a_col + 4] : hzero;
                a01[1] = hzero;
                a23[1] = hzero;
                FragC_h ch[WNT][2] = {};
                #pragma unroll
                for (int t = 0; t < WNT; ++t) {
                    exl3_gemv_ns::mma_ab_h(a01, a23, f0s[t], ch[t][0]);
                    exl3_gemv_ns::mma_ab_h(a01, a23, f1s[t], ch[t][1]);
                }
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        acc0[row][t][f].x += __low2float(ch[t][f][0]);
                        acc0[row][t][f].y += __high2float(ch[t][f][0]);
                    }
            }
        }
    }

    for (int row = 0; row < MAX_M; ++row) {
        if (row >= m_loc) break;
        if (lane < 4) {
            #pragma unroll
            for (int t = 0; t < WNT; ++t) {
                #pragma unroll
                for (int f = 0; f < 2; ++f) {
                    const int col = t * 16 + f * 8 + (lane & 3) * 2;
                    sh_red[warp][0][col + 0] = acc0[row][t][f].x;
                    sh_red[warp][0][col + 1] = acc0[row][t][f].y;
                }
            }
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < COLS; idx += THREADS) {
            float sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < WK; ++j)
                sum += sh_red[j][0][idx];
            const int col = group * COLS + idx;
            C[c_off[row] + col] = __float2half_rn(sum);
        }
        __syncthreads();
    }
}"""

GEMV_GATE_MROW = """        int total_work = 2 * m * experts * num_groups_gate;
        for (int item = blockIdx.x; item < total_work; item += gridDim.x) {
            int is_up = item & 1;
            int rem = item >> 1;
            int row = rem / (experts * num_groups_gate);
            int rem2 = rem % (experts * num_groups_gate);
            int e = rem2 / num_groups_gate;
            int group = rem2 % num_groups_gate;
            int src = ids[row * experts + e];
            size_t em = (size_t) e * m + row;

            const uint32_t* B32 = reinterpret_cast<const uint32_t*>(is_up ? ut_ptrs[src] : gt_ptrs[src]);
            const half2* A2 = reinterpret_cast<const half2*>((is_up ? had_up : had_gate) + em * hidden);
            half* C = (is_up ? up : gate) + em * inter;

            run_gemv_tile<BITS, 1, 0>(B32, A2, C, kslices_gate, hidden, group, ntiles_gate, warp, lane, sh_red);
"""

GEMV_GATE_STOCK = """        int total_work = 2 * experts * num_groups_gate;
        for (int item = blockIdx.x; item < total_work; item += gridDim.x) {
            int is_up = item & 1;
            int rem = item >> 1;
            int e = rem / num_groups_gate;
            int group = rem % num_groups_gate;
            int src = ids[e];

            const uint32_t* B32 = reinterpret_cast<const uint32_t*>(is_up ? ut_ptrs[src] : gt_ptrs[src]);
            const half2* A2 = reinterpret_cast<const half2*>((is_up ? had_up : had_gate) + e * hidden);
            half* C = (is_up ? up : gate) + e * inter;

            run_gemv_tile<BITS, 1, 0>(B32, A2, C, kslices_gate, hidden, group, ntiles_gate, warp, lane, sh_red);
"""

GEMV_GATE_MMA = f"""        int n_pairs = m * experts;
        int total_work = 2 * n_pairs * num_groups_gate;
        for (int item = blockIdx.x; item < total_work; item += gridDim.x) {{
            int is_up = item & 1;
            int rem = item >> 1;
            int group = rem / n_pairs;
            int leader = rem % n_pairs;
            int src = ids[leader];
            bool first = true;
            for (int p = 0; p < leader; ++p) {{
                if (ids[p] == src) {{ first = false; break; }}
            }}
            if (!first) continue;

            int a_off[{MAX_M}];
            int c_off[{MAX_M}];
            int m_loc = 0;
            const half* had = is_up ? had_up : had_gate;
            half* outp = is_up ? up : gate;
            const uint32_t* B32 = reinterpret_cast<const uint32_t*>(is_up ? ut_ptrs[src] : gt_ptrs[src]);
            const half2* A2 = reinterpret_cast<const half2*>(had);
            for (int p = 0; p < n_pairs; ++p) {{
                if (ids[p] != src) continue;
                int row = p / experts;
                int slot = p % experts;
                a_off[m_loc] = (int)(((size_t) slot * m + row) * (hidden / 2));
                c_off[m_loc] = (int)(((size_t) slot * m + row) * inter);
                m_loc += 1;
                if (m_loc == {MAX_M}) {{
                    run_gemv_tile<BITS, 1, 0>(B32, A2, a_off, outp, c_off, m_loc, kslices_gate, hidden, group, ntiles_gate, warp, lane, sh_red);
                    m_loc = 0;
                }}
            }}
            if (m_loc > 0)
                run_gemv_tile<BITS, 1, 0>(B32, A2, a_off, outp, c_off, m_loc, kslices_gate, hidden, group, ntiles_gate, warp, lane, sh_red);
"""

GEMV_DOWN_MROW = """        int total_work = m * experts * num_groups_down;
        for (int item = blockIdx.x; item < total_work; item += gridDim.x) {
            int row = item / (experts * num_groups_down);
            int rest = item % (experts * num_groups_down);
            int e = rest / num_groups_down;
            int group = rest % num_groups_down;
            int src = ids[row * experts + e];
            size_t em = (size_t) e * m + row;

            const uint32_t* B32 = reinterpret_cast<const uint32_t*>(dt_ptrs[src]);
            const half2* A2 = reinterpret_cast<const half2*>(had_down + em * inter);
            half* C = down + em * hidden;

            run_gemv_tile<BITS, 1, 0>(B32, A2, C, kslices_down, inter, group, ntiles_down, warp, lane, sh_red);
"""

GEMV_DOWN_STOCK = """        int total_work = experts * num_groups_down;
        for (int item = blockIdx.x; item < total_work; item += gridDim.x) {
            int e = item / num_groups_down;
            int group = item % num_groups_down;
            int src = ids[e];

            const uint32_t* B32 = reinterpret_cast<const uint32_t*>(dt_ptrs[src]);
            const half2* A2 = reinterpret_cast<const half2*>(had_down + e * inter);
            half* C = down + e * hidden;

            run_gemv_tile<BITS, 1, 0>(B32, A2, C, kslices_down, inter, group, ntiles_down, warp, lane, sh_red);
"""

GEMV_DOWN_MMA = f"""        int n_pairs = m * experts;
        int total_work = n_pairs * num_groups_down;
        for (int item = blockIdx.x; item < total_work; item += gridDim.x) {{
            int group = item / n_pairs;
            int leader = item % n_pairs;
            int src = ids[leader];
            bool first = true;
            for (int p = 0; p < leader; ++p) {{
                if (ids[p] == src) {{ first = false; break; }}
            }}
            if (!first) continue;

            int a_off[{MAX_M}];
            int c_off[{MAX_M}];
            int m_loc = 0;
            const uint32_t* B32 = reinterpret_cast<const uint32_t*>(dt_ptrs[src]);
            const half2* A2 = reinterpret_cast<const half2*>(had_down);
            for (int p = 0; p < n_pairs; ++p) {{
                if (ids[p] != src) continue;
                int row = p / experts;
                int slot = p % experts;
                a_off[m_loc] = (int)(((size_t) slot * m + row) * (inter / 2));
                c_off[m_loc] = (int)(((size_t) slot * m + row) * hidden);
                m_loc += 1;
                if (m_loc == {MAX_M}) {{
                    run_gemv_tile<BITS, 1, 0>(B32, A2, a_off, down, c_off, m_loc, kslices_down, inter, group, ntiles_down, warp, lane, sh_red);
                    m_loc = 0;
                }}
            }}
            if (m_loc > 0)
                run_gemv_tile<BITS, 1, 0>(B32, A2, a_off, down, c_off, m_loc, kslices_down, inter, group, ntiles_down, warp, lane, sh_red);
"""

DONE_MARKERS = (
    "const int* __restrict__ a_off",
    "FragB f0s[WNT], f1s[WNT]",
    "for (int row = 0; row < MAX_M; ++row)",
    "if (ids[p] == src)",
)


def _already(src: str) -> bool:
    return all(m in src for m in DONE_MARKERS)


def _replace_one(src: str, old: str, new: str, label: str) -> str:
    if new in src and old not in src:
        return src
    if old not in src:
        raise SystemExit(f"widen_p2b_mma: {label} not found")
    return src.replace(old, new, 1)


def _replace_one_of(src: str, pairs: tuple[tuple[str, str], ...], label: str) -> str:
    for old, new in pairs:
        if new in src and old not in src:
            return src
        if old in src:
            return src.replace(old, new, 1)
    raise SystemExit(f"widen_p2b_mma: {label} not found")


def patch_cu(src: str) -> str:
    if _already(src):
        return src
    out = _replace_one(src, GEMV_TILE_OLD, GEMV_TILE_NEW, "run_gemv_tile signature")
    out = _replace_one(out, TILE_A_MMA_OLD, TILE_A_MMA_NEW, "run_gemv_tile MMA body")
    out = _replace_one_of(
        out,
        ((GEMV_GATE_MROW, GEMV_GATE_MMA), (GEMV_GATE_STOCK, GEMV_GATE_MMA)),
        "gate/up GEMV grouping",
    )
    out = _replace_one_of(
        out,
        ((GEMV_DOWN_MROW, GEMV_DOWN_MMA), (GEMV_DOWN_STOCK, GEMV_DOWN_MMA)),
        "down GEMV grouping",
    )
    if "const size_t a_row0 = 0" in out:
        raise SystemExit("widen_p2b_mma: a_row0=0 still present")
    if "FragB f0s[WNT], f1s[WNT]" not in out:
        raise SystemExit("widen_p2b_mma: B decode-once missing")
    if "for (int row = 0; row < MAX_M; ++row)" not in out:
        raise SystemExit("widen_p2b_mma: MMA row loop missing")
    if "if (ids[p] == src)" not in out:
        raise SystemExit("widen_p2b_mma: expert grouping missing")
    if "run_gemv_tile<BITS, 1, 0>(B32, A2, C," in out:
        raise SystemExit("widen_p2b_mma: old single-row GEMV call still present")
    return out


def patch_py(src: str) -> str:
    return src


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
