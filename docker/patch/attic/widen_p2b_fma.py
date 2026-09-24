#!/usr/bin/env python3
"""CFG=1 p2b GEMV: drop padded m16 MMA, FMA the one live A row.

mma.m16n8k16 still runs with a01[1]=a23[1]=0 and r0_ok=lane<4, so 15 of 16
M-rows are zeros. Unique experts at m=6 are m_loc≈1. After shuffle plus
dq8_regs_2bits, FMA that row into float2 acc0 and xor-reduce K across the
warp. Keep __ldcs, PF=2, launch_bounds(256, 4), cooperative fused launch.

Apply after widen_p2b_cfg1.py. Idempotent. Not MMA-over-m.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MMA_OLD = """    FragC_h ch[WNT][2] = {};
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
"""

MMA_NEW = """    float2 acc0[WNT][2] = {};

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

            // PTX m16n8k16.row.col: A lives on groupID=0 (lanes 0-3). B's N is
            // groupID. C's N is threadID_in_group*2. FMA at B's home thread,
            // xor-reduce K in the 4-thread group, pack n-pairs into lanes 0-3.
            const size_t a_col = (size_t) (ks0 + i) * 8 + (lane & 3);
            const half2 a_k01_lane = r0_ok ? A2[a_row0 + a_col] : hzero;
            const half2 a_k89_lane = r0_ok ? A2[a_row0 + a_col + 4] : hzero;
            const int a_src = lane & 3;
            const half2 a_k01 = __shfl_sync(0xffffffffu, a_k01_lane, a_src);
            const half2 a_k89 = __shfl_sync(0xffffffffu, a_k89_lane, a_src);
            const float a0 = __half2float(__low2half(a_k01));
            const float a1 = __half2float(__high2half(a_k01));
            const float a8 = __half2float(__low2half(a_k89));
            const float a9 = __half2float(__high2half(a_k89));

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

                float p0 =
                    a0 * __half2float(__low2half(f0[0])) +
                    a1 * __half2float(__high2half(f0[0])) +
                    a8 * __half2float(__low2half(f0[1])) +
                    a9 * __half2float(__high2half(f0[1]));
                float p1 =
                    a0 * __half2float(__low2half(f1[0])) +
                    a1 * __half2float(__high2half(f1[0])) +
                    a8 * __half2float(__low2half(f1[1])) +
                    a9 * __half2float(__high2half(f1[1]));
                p0 += __shfl_xor_sync(0xffffffffu, p0, 1);
                p0 += __shfl_xor_sync(0xffffffffu, p0, 2);
                p1 += __shfl_xor_sync(0xffffffffu, p1, 1);
                p1 += __shfl_xor_sync(0xffffffffu, p1, 2);
                const int pack_src = (lane & 3) << 3;
                const float n0 = __shfl_sync(0xffffffffu, p0, pack_src);
                const float n1 = __shfl_sync(0xffffffffu, p0, pack_src + 4);
                const float n8 = __shfl_sync(0xffffffffu, p1, pack_src);
                const float n9 = __shfl_sync(0xffffffffu, p1, pack_src + 4);
                if (lane < 4) {
                    acc0[t][0].x += n0;
                    acc0[t][0].y += n1;
                    acc0[t][1].x += n8;
                    acc0[t][1].y += n9;
                }
            }
        }
    }
"""

DONE_MARKERS = (
    "const int a_src = lane & 3;",
    "__shfl_xor_sync(0xffffffffu, p0, 1)",
    "const int pack_src = (lane & 3) << 3;",
    "run_gemv_tile<BITS, 1, 1>",
)
LEFTOVERS = (
    "mma_ab_h",
    "FragC_h ch[WNT][2]",
)


def fma_row(a16: list[float], b16x8: list[list[float]]) -> tuple[float, ...]:
    """1-row product: out[n] = sum_k a16[k] * b16x8[k][n] for n in 0..7."""
    return tuple(
        sum(float(a16[k]) * float(b16x8[k][n]) for k in range(16)) for n in range(8)
    )


def fma_warp_pack(a16: list[float], b16x8: list[list[float]]) -> tuple[tuple[float, float], ...]:
    """Lanes 0-3 dump pairs (n0,n1)..(n6,n7) matching ch[0] of m16n8k16."""
    row = fma_row(a16, b16x8)
    return tuple((row[2 * i], row[2 * i + 1]) for i in range(4))


def fma_warp_simulate(a16: list[float], b16x8: list[list[float]]) -> tuple[tuple[float, float], ...]:
    """PTX mapping: A from lane%4, B at groupID, xor K with 1 then 2, pack 8*L."""
    partial = [0.0] * 32
    for lane in range(32):
        g = lane >> 2
        t = lane & 3
        partial[lane] = (
            float(a16[2 * t]) * float(b16x8[2 * t][g])
            + float(a16[2 * t + 1]) * float(b16x8[2 * t + 1][g])
            + float(a16[2 * t + 8]) * float(b16x8[2 * t + 8][g])
            + float(a16[2 * t + 9]) * float(b16x8[2 * t + 9][g])
        )
    for mask in (1, 2):
        partial = [partial[i] + partial[i ^ mask] for i in range(32)]
    return tuple((partial[8 * L], partial[8 * L + 4]) for L in range(4))


def fma_warp_simulate_collapsed(
    a16: list[float], b16x8: list[list[float]]
) -> tuple[tuple[float, float], ...]:
    """The reverted mapping: a_src=lane>>3 and xor 4/8/16. Must not equal fma_row."""
    acc_x = [0.0] * 32
    acc_y = [0.0] * 32
    for lane in range(32):
        a_src = lane >> 3
        t = a_src
        fa = float(a16[2 * t + ((lane >> 2) & 1)])
        fa8 = float(a16[2 * t + 8 + ((lane >> 2) & 1)])
        g = lane >> 2
        acc_x[lane] = fa * float(b16x8[2 * (lane & 3)][g]) + fa8 * float(
            b16x8[2 * (lane & 3) + 8][g]
        )
        acc_y[lane] = fa * float(b16x8[2 * (lane & 3) + 1][g]) + fa8 * float(
            b16x8[2 * (lane & 3) + 9][g]
        )
    for mask in (4, 8, 16):
        acc_x = [acc_x[i] + acc_x[i ^ mask] for i in range(32)]
        acc_y = [acc_y[i] + acc_y[i ^ mask] for i in range(32)]
    return tuple((acc_x[L], acc_y[L]) for L in range(4))


def _already(src: str) -> bool:
    return all(m in src for m in DONE_MARKERS) and not any(x in src for x in LEFTOVERS)


def _replace_one(src: str, old: str, new: str, label: str) -> str:
    if new in src and old not in src:
        return src
    if old not in src:
        raise SystemExit(f"widen_p2b_fma: {label} not found")
    return src.replace(old, new, 1)


def patch_cu(src: str) -> str:
    if _already(src):
        return src
    if "run_gemv_tile<BITS, 1, 1>" not in src:
        raise SystemExit("widen_p2b_fma: apply widen_p2b_cfg1 first")
    had_mrow = "m * experts * warps_per_exp" in src
    had_bounds = "__launch_bounds__(256, 4)" in src
    out = _replace_one(src, MMA_OLD, MMA_NEW, "padded m16 MMA loop")
    if "mma_ab_h" in out:
        raise SystemExit("widen_p2b_fma: mma_ab_h still present")
    if "FragC_h ch[WNT][2]" in out:
        raise SystemExit("widen_p2b_fma: FragC_h accumulator still present")
    if "dq8_regs_2bits" not in out:
        raise SystemExit("widen_p2b_fma: 2-bit dequant undone")
    if "__ldcs(" not in out:
        raise SystemExit("widen_p2b_fma: __ldcs load undone")
    if "constexpr int PF = CFG == 0 ? 4 : 2;" not in out:
        raise SystemExit("widen_p2b_fma: PF=2 CFG=1 contract undone")
    if "run_gemv_tile<BITS, 1, 1>" not in out:
        raise SystemExit("widen_p2b_fma: CFG=1 tile undone")
    if had_mrow and "m * experts * warps_per_exp" not in out:
        raise SystemExit("widen_p2b_fma: m-row work lists undone")
    if had_bounds and "__launch_bounds__(256, 4)" not in out:
        raise SystemExit("widen_p2b_fma: launch_bounds undone")
    if "a_src = lane >> 3" in out:
        raise SystemExit("widen_p2b_fma: collapsed a_src=lane>>3 mapping still present")
    if "mask = 4" in out:
        raise SystemExit("widen_p2b_fma: collapsed xor-4/8/16 reduce still present")
    if "(t & 1) << 4" not in out and "bits == 2" in out:
        raise SystemExit("widen_p2b_fma: 2-bit shuffle base undone")
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
