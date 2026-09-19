#!/usr/bin/env python3
"""Prefetch p2b trellis B with cp.async into smem instead of __ldcs/__ldg.

CFG=1 still re-reads the same expert trellis once per activation row.
__ldg did not beat streaming (22.87/21.22 vs KEEP 23.44/22.37). Issue the
next 4B B load into the same smem slot after copying bw[], then MMA, then
wait so DRAM overlaps mma_ab_h. Does not change CFG, thread count, or m-row
work lists. Does not regroup tiles over m.
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROLOGUE_OLD_LDCS = """    auto ld_b = [&] (int i, int l) -> uint32_t {
        if constexpr (bits == 3)
            return lane < 24 ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
        else
            return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE);
    };

    uint32_t pf[PF][LOADS];
    #pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < myn)
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                pf[d][l] = ld_b(d, l);
"""

PROLOGUE_OLD_LDG = """    auto ld_b = [&] (int i, int l) -> uint32_t {
        if constexpr (bits == 3)
            return lane < 24 ? __ldg(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
        else
            return __ldg(bp + (size_t) i * slice_stride + l * LSTRIDE);
    };

    uint32_t pf[PF][LOADS];
    #pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < myn)
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                pf[d][l] = ld_b(d, l);
"""

PROLOGUE_NEW = """    __shared__ uint32_t sh_b[PF][LOADS][THREADS];

    auto ld_b = [&] (int i, int l, int slot) {
        uint32_t* dst = &sh_b[slot][l][threadIdx.x];
        if constexpr (bits == 3) {
            if (lane >= 24) {
                *dst = 0;
                return;
            }
        }
        const uint32_t* src = bp + (size_t) i * slice_stride + l * LSTRIDE;
        const unsigned long long smem_ptr = __cvta_generic_to_shared(dst);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "l"(smem_ptr), "l"(src) : "memory");
    };

    #pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < myn)
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                ld_b(d, l, d);
    if (myn > 0) {
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group 0;" ::: "memory");
    }
"""

INNER_OLD = """            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                bw[l] = pf[d][l];

            if (i + PF < myn) {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(i + PF, l);
            }
"""

INNER_NEW = """            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                bw[l] = sh_b[d][l][threadIdx.x];

            if (i + PF < myn) {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    ld_b(i + PF, l, d);
                asm volatile("cp.async.commit_group;" ::: "memory");
            }
"""

FOLD_OLD = """            if ((d + 1) % FOLD == 0 || i + 1 == myn) {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        acc0[t][f].x += __low2float(ch[t][f][0]);
                        acc0[t][f].y += __high2float(ch[t][f][0]);
                        ch[t][f][0] = hzero;
                    }
            }
"""

FOLD_NEW = """            if ((d + 1) % FOLD == 0 || i + 1 == myn) {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        acc0[t][f].x += __low2float(ch[t][f][0]);
                        acc0[t][f].y += __high2float(ch[t][f][0]);
                        ch[t][f][0] = hzero;
                    }
            }
            if (i + PF < myn)
                asm volatile("cp.async.wait_group 0;" ::: "memory");
"""

DONE_MARKERS = (
    "cp.async",
    "__shared__ uint32_t sh_b[PF][LOADS][THREADS]",
    "cp.async.commit_group",
    "cp.async.wait_group",
    "ld_b(i + PF, l, d)",
)

LEFTOVERS = (
    "__ldcs(",
    "uint32_t pf[PF][LOADS]",
    "pf[d][l] = ld_b",
)


def _already(src: str) -> bool:
    return all(marker in src for marker in DONE_MARKERS) and not any(
        old in src for old in LEFTOVERS
    )


def _replace_one(src: str, old: str, new: str, label: str) -> str:
    if new in src and old not in src:
        return src
    if old not in src:
        raise SystemExit(f"widen_p2b_cpasync: {label} not found")
    return src.replace(old, new, 1)


def _replace_one_of(src: str, pairs: tuple[tuple[str, str], ...], label: str) -> str:
    for old, new in pairs:
        if new in src and old not in src:
            return src
        if old in src:
            return src.replace(old, new, 1)
    raise SystemExit(f"widen_p2b_cpasync: {label} not found")


def patch_cu(src: str) -> str:
    if _already(src):
        return src
    had_cfg1 = "run_gemv_tile<BITS, 1, 1>" in src
    had_mrow = "m * experts * warps_per_exp" in src
    had_bounds = "__launch_bounds__(256, 4)" in src
    had_mma = "FragB f0s[WNT], f1s[WNT]" in src
    if "run_gemv_tile" not in src:
        raise SystemExit("widen_p2b_cpasync: run_gemv_tile missing")
    out = _replace_one_of(
        src,
        ((PROLOGUE_OLD_LDCS, PROLOGUE_NEW), (PROLOGUE_OLD_LDG, PROLOGUE_NEW)),
        "ld_b / prefetch prologue",
    )
    out = _replace_one(out, INNER_OLD, INNER_NEW, "ld_b inner prefetch")
    out = _replace_one(out, FOLD_OLD, FOLD_NEW, "cp.async wait after MMA")
    if "__ldcs(" in out:
        raise SystemExit("widen_p2b_cpasync: __ldcs( still present")
    if "uint32_t pf[PF][LOADS]" in out:
        raise SystemExit("widen_p2b_cpasync: register pf[] prefetch still present")
    if "cp.async" not in out:
        raise SystemExit("widen_p2b_cpasync: cp.async missing")
    if "__shared__ uint32_t sh_b[PF][LOADS][THREADS]" not in out:
        raise SystemExit("widen_p2b_cpasync: smem B staging missing")
    if "cp.async.wait_group" not in out:
        raise SystemExit("widen_p2b_cpasync: cp.async.wait_group missing")
    if had_cfg1 and "run_gemv_tile<BITS, 1, 1>" not in out:
        raise SystemExit("widen_p2b_cpasync: CFG=1 GEMV tile was undone")
    if had_mrow and "m * experts * warps_per_exp" not in out:
        raise SystemExit("widen_p2b_cpasync: m-row work lists were undone")
    if had_bounds and "__launch_bounds__(256, 4)" not in out:
        raise SystemExit("widen_p2b_cpasync: launch_bounds(256, 4) was undone")
    if "for (int row = 0; row < m; ++row)" in out:
        raise SystemExit("widen_p2b_cpasync: serial per-row moe loop is the reverted path")
    if not had_mma and "FragB f0s[WNT], f1s[WNT]" in out:
        raise SystemExit("widen_p2b_cpasync: MMA-over-m regroup is the reverted path")
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
