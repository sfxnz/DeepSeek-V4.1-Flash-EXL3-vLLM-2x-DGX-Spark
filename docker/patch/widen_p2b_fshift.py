#!/usr/bin/env python3
"""Faster trellis window extraction in vllm-exl3 p2b fused MoE (K=2, MCG).

The stock bits==2 decode merges the lane's word with its neighbour through a
64-bit funnel (fshift: 64-bit merge + shift, 2-3 SASS ops) before the eight
16-bit window extractions. The merge shift is the lane-parity constant
((~(lane<<3))&8)<<1, i.e. exactly 0 or 16, so __funnelshift_r (single SHF)
computes the same 32 bits. Bit-identical windows, one fewer ALU op per tile
iteration: -6.1% kernel time at e=30 warm (670.7 -> 629.7 us), neutral cold.

Apply after widen_p2b_cfg1.py (and widen_p2b_codebook.py when present).
Idempotent. Only the MCG cb=1 path is rewritten; cb=2 keeps stock decode.
"""

from __future__ import annotations

import argparse
from pathlib import Path

VENDORED = r'''
// --- widen_p2b_fshift: single-instruction window merge (bit-identical) ---
namespace bench_fshift {
__device__ __forceinline__ uint32_t fshift1(const uint32_t b, const uint32_t a, int shift)
{
    return __funnelshift_r(b, a, shift);
}
template <int cb>
__device__ __forceinline__ half2 decode_pair_mcg(uint32_t x0, uint32_t x1)
{
    x0 *= 0xCBAC1FEDu;
    x1 *= 0xCBAC1FEDu;
    asm ("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x0));
    asm ("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x1));
    half2_uint32 xu0(x0);
    half2_uint32 xu1(x1);
    half2 d0 = __lows2half2(xu0.as_half2, xu1.as_half2);
    half2 d1 = __highs2half2(xu0.as_half2, xu1.as_half2);
    return __hadd2(d0, d1);
}
template <int cb>
__device__ __forceinline__ void dq8_regs_2bits_fs(uint32_t a, uint32_t b, int t_offset, FragB& f0, FragB& f1)
{
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    b = fshift1(b, a, ((~t_offset) & 8) << 1);
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 2);
    BFE16_IMM(w5, b, 4);
    BFE16_IMM(w4, b, 6);
    BFE16_IMM(w3, b, 8);
    BFE16_IMM(w2, b, 10);
    BFE16_IMM(w1, b, 12);
    BFE16_IMM(w0, b, 14);
    f0[0] = decode_pair_mcg<cb>(w0, w1);
    f0[1] = decode_pair_mcg<cb>(w2, w3);
    f1[0] = decode_pair_mcg<cb>(w4, w5);
    f1[1] = decode_pair_mcg<cb>(w6, w7);
}
}  // namespace bench_fshift
'''

INSERT_ANCHOR = "namespace cg = cooperative_groups;"

OLD_CALL = """                    uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                    uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    exl3_gemv_ns::dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);"""
NEW_CALL = """                    uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                    uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    if constexpr (cb == 1)
                        bench_fshift::dq8_regs_2bits_fs<cb>(awv, bwv, lane << 3, f0, f1);
                    else
                        exl3_gemv_ns::dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);"""


def patch_cu(src: str) -> str:
    if "bench_fshift::dq8_regs_2bits_fs" in src:
        return src  # idempotent
    if src.count(INSERT_ANCHOR) != 1:
        raise SystemExit("widen_p2b_fshift: cooperative_groups anchor not found")
    if src.count(OLD_CALL) != 1:
        raise SystemExit("widen_p2b_fshift: bits==2 decode call not found (patch order?)")
    out = src.replace(INSERT_ANCHOR, INSERT_ANCHOR + "\n" + VENDORED, 1)
    out = out.replace(OLD_CALL, NEW_CALL, 1)
    if "bench_fshift::dq8_regs_2bits_fs<cb>(awv, bwv, lane << 3" not in out:
        raise SystemExit("widen_p2b_fshift: rewrite failed")
    return out


def apply(root: Path) -> None:
    cu = root / "csrc" / "p2b_moe.cu"
    if not cu.is_file():
        raise SystemExit(f"widen_p2b_fshift: {cu} not found")
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
