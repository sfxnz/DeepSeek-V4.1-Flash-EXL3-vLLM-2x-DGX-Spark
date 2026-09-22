#!/usr/bin/env python3
"""PF-G8 kernel port — vllm-exl3 side (components 7 + 8).

Applies to a fresh clone of vllm-exl3 @ VLLM_EXL3_REF (the canonical-e12 pin,
d3cfd39) AFTER the repo's widen chain AND after exllamav3 has been patched
(pfg8_kernels_exllamav3.py) and pip-installed: the vllm-exl3 build pulls
exllamav3_ext headers via EXL3_EXT_INCLUDE/include_dirs, so `pf_g8` and the
`pfg8::` host registry (pf_g8_host.h in exllamav3_ext/quant/) are already in
scope in every TU that includes quant/exl3_gemv_kernel.cuh. ONE pf_g8_host.h
per .so — no duplicate-definition hazard. Building against an unpatched
exllamav3 fails to compile (loud) — the desired guard.

Runtime behavior is env-gated by DSV41_LOAD_PF_G8 (default off = exact stock
arithmetic; the TU-local __constant__ stays 0 when the env is unset).

Patches (repo root argument): p2b_moe.cu decode reader (DEC5 G8 adapted,
CFG=1 WNT=4 -> pack group = group>>1) + apply-once; p2b_batched.cu
p2b_run_gemv_tile_2 (WNT=2 -> group>>2) + apply-once; exl3_gemv.cu /
exl3_gemm.cu host G8 dims branches (test/tool entries); exl3_fat_gemm.cu
fail-loud G8 guards (audit R3; loader also gates fat routing OFF via
VLLM_EXL3_FAT_THRESHOLD — belt and braces); bindings.cpp pf_g8_set/pf_g8_get
exports + build marker (boot-g8.sh strings guard).

Idempotent: re-running on a patched tree is a no-op (marker check per file).
"""
from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "pfg8-kernel-port"  # matches both "# ---" and "// ---" comment forms

# ---- p2b_moe.cu ----
MOE_BP_OLD = """    const uint32_t* bp = B32 + (size_t) ks0 * slice_stride + group * WNT * TWORDS + lane;

    auto ld_b = [&] (int i, int l) -> uint32_t {
        if constexpr (bits == 3)
            return lane < 24 ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
        else
            return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE);
    };
"""
MOE_BP_NEW = """    // --- pfg8-kernel-port ---
    // PF-G8: trellis stored [group of 8 tiles][k-slice][8*TWORDS uint32].
    // This block's run of WNT tiles (CFG=1 -> WNT=4; 8 % WNT == 0) lives in
    // pack group group/(8/WNT) at tile offset (group % (8/WNT))*WNT; a
    // k-slice row of a pack group is 8*TWORDS contiguous uint32 — the same
    // word stream as stock, so prefetch ring, fragment math and reduction
    // order are untouched (bench5.cu DEC5 pattern, G8 adapted).
    const size_t g8_krow = (size_t) 8 * TWORDS;
    const size_t b_i_stride = pf_g8 ? g8_krow : slice_stride;
    const uint32_t* bp = pf_g8
        ? B32 + (size_t) (group / (8 / WNT)) * kslices * g8_krow
                + (size_t) ks0 * g8_krow
                + (size_t) (group % (8 / WNT)) * WNT * TWORDS + lane
        : B32 + (size_t) ks0 * slice_stride + group * WNT * TWORDS + lane;

    auto ld_b = [&] (int i, int l) -> uint32_t {
        if constexpr (bits == 3)
            return lane < 24 ? __ldcs(bp + (size_t) i * b_i_stride + l * LSTRIDE) : 0;
        else
            return __ldcs(bp + (size_t) i * b_i_stride + l * LSTRIDE);
    };
"""

MOE_HOST_OLD = """at::Tensor p2b_fused_moe_cuda(const at::Tensor& x, at::Tensor& out,
    const at::Tensor& gt, const at::Tensor& gu, const at::Tensor& gv,
    const at::Tensor& ut, const at::Tensor& uu, const at::Tensor& uv,
    const at::Tensor& dt, const at::Tensor& du, const at::Tensor& dv,
    const at::Tensor& ids, const at::Tensor& rw, int64_t kg, int64_t ku,
    int64_t kd, bool mcg, int64_t intermediate_size, float swiglu_limit) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "fused MoE requires CUDA fp16 input");"""
MOE_HOST_NEW = """at::Tensor p2b_fused_moe_cuda(const at::Tensor& x, at::Tensor& out,
    const at::Tensor& gt, const at::Tensor& gu, const at::Tensor& gv,
    const at::Tensor& ut, const at::Tensor& uu, const at::Tensor& uv,
    const at::Tensor& dt, const at::Tensor& du, const at::Tensor& dv,
    const at::Tensor& ids, const at::Tensor& rw, int64_t kg, int64_t ku,
    int64_t kd, bool mcg, int64_t intermediate_size, float swiglu_limit) {
    pfg8::apply();  // --- pfg8-kernel-port --- (call_once; no per-call cost)
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "fused MoE requires CUDA fp16 input");"""

# ---- p2b_batched.cu ----
BATCH_TILE2_OLD = """    const uint32_t* bp = B32 + (size_t)ks0 * slice_stride + group * WNT * TWORDS + lane;

    auto ld_b = [&](int i) -> uint32_t {
        return __ldcs(bp + (size_t)i * slice_stride);
    };
"""
BATCH_TILE2_NEW = """    // --- pfg8-kernel-port ---
    // PF-G8 group-major, WNT=2: pack group = group/4, tile offset (group%4)*2.
    // k-slice row of a pack group = 8*TWORDS contiguous uint32 (k=2: TWORDS=16).
    const size_t g8_krow = (size_t)8 * TWORDS;
    const size_t b_i_stride = pf_g8 ? g8_krow : slice_stride;
    const uint32_t* bp = pf_g8
        ? B32 + (size_t)(group / (8 / WNT)) * kslices * g8_krow
                + (size_t)ks0 * g8_krow
                + (size_t)(group % (8 / WNT)) * WNT * TWORDS + lane
        : B32 + (size_t)ks0 * slice_stride + group * WNT * TWORDS + lane;

    auto ld_b = [&](int i) -> uint32_t {
        return __ldcs(bp + (size_t)i * b_i_stride);
    };
"""

BATCH_ENTRY_OLD = """at::Tensor p2b_gemv_batched_cuda(const at::Tensor& x,
                                 const at::Tensor& trellis_ptrs,
                                 const at::Tensor& suh_ptrs,
                                 const at::Tensor& svh_ptrs,
                                 const at::Tensor& expert_indices,
                                 int64_t bits, bool mcg, int64_t mmode) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "batched GEMV requires CUDA fp16");"""
BATCH_ENTRY_NEW = """at::Tensor p2b_gemv_batched_cuda(const at::Tensor& x,
                                 const at::Tensor& trellis_ptrs,
                                 const at::Tensor& suh_ptrs,
                                 const at::Tensor& svh_ptrs,
                                 const at::Tensor& expert_indices,
                                 int64_t bits, bool mcg, int64_t mmode) {
    pfg8::apply();  // --- pfg8-kernel-port ---
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "batched GEMV requires CUDA fp16 input");"""

# ---- csrc/exl3_gemv.cu (vllm_exl3_c's own QTIP gemv entry) ----
VGEVV_INCLUDE_OLD = '#include "quant/exl3_gemv_kernel.cuh"\n'
VGEVV_INCLUDE_NEW = '''#include "quant/exl3_gemv_kernel.cuh"
#include "pf_g8_host.h"  // --- pfg8-kernel-port ---
'''

VGEVV_ENTRY_OLD = """    const int m = static_cast<int>(x.numel() / x.size(-1));
    const int k = static_cast<int>(x.size(-1));
    const int n = static_cast<int>(trellis.size(1) * 16);
    const int cb = mcg ? 1 : 2;
"""
VGEVV_ENTRY_NEW = """    pfg8::apply();  // --- pfg8-kernel-port ---
    const int m = static_cast<int>(x.numel() / x.size(-1));
    const int k = static_cast<int>(x.size(-1));
    // --- pfg8-kernel-port --- G8 trellis [NG][KT][128*K]: n on dim 0, k on dim 1
    const int n = static_cast<int>(
        pfg8::env_value() ? trellis.size(0) * 128 : trellis.size(1) * 16);
    const int cb = mcg ? 1 : 2;
"""

# ---- csrc/exl3_gemm.cu (vllm_exl3_c chunked gemm) ----
VGEMM_HEAD_OLD = '#include "exl3_gemv.cuh"\n'
VGEMM_HEAD_NEW = '''#include "exl3_gemv.cuh"
#include "pf_g8_host.h"  // --- pfg8-kernel-port ---
'''

VGEMM_N_OLD = """    TORCH_CHECK(x.dim() == 2, "GEMM input must be [m,k]");
    const int64_t m = x.size(0);
    const int64_t n = trellis.size(1) * 16;
"""
VGEMM_N_NEW = """    TORCH_CHECK(x.dim() == 2, "GEMM input must be [m,k]");
    const int64_t m = x.size(0);
    const int64_t n = pfg8::env_value() ? trellis.size(0) * 128 : trellis.size(1) * 16;  // --- pfg8-kernel-port ---
"""

# ---- csrc/exl3_fat_gemm.cu ----
FAT_HEAD_OLD = '#include "exl3_fat_gemm.cuh"\n'
FAT_HEAD_NEW = '''#include "exl3_fat_gemm.cuh"
#include "pf_g8_host.h"  // --- pfg8-kernel-port ---
'''

FAT_GUARD_ANCHORS = ("void exl3_fat_gemm(", "void exl3_fat_gemm_scatter(")
FAT_GUARD_ADD = """    TORCH_CHECK(!pfg8::env_value(),
                "exl3_fat_gemm: stock-layout reader called on a PF-G8 pack "
                "(set DSV41_LOAD_PF_G8=0 or route fat experts elsewhere)");
"""


def fat_guards(text: str) -> str:
    # insert the guard as the first statement of both fat entry bodies
    out = text
    for anchor in FAT_GUARD_ANCHORS:
        i = out.find(anchor)
        if i < 0:
            raise SystemExit(f"pfg8_kernels_vllm: fat entry anchor missing: {anchor}")
        j = out.find("{", i)
        k = j + 1
        out = out[:k] + "\n" + FAT_GUARD_ADD + out[k:]
    return out


# ---- bindings.cpp ----
VBIND_MODULE_OLD = "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n"
VBIND_MODULE_NEW = """// --- pfg8-kernel-port ---
void pf_g8_set(int64_t v)
{
    pfg8::force((int) v);
}

int64_t pf_g8_get()
{
    return pfg8::current_value();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pf_g8_set", &pf_g8_set, "set PF-G8 trellis layout flag (0 stock, 1 group-major)");
    m.def("pf_g8_get", &pf_g8_get, "get PF-G8 trellis layout flag");
    m.attr("pf_g8_build") = "pfg8-port-v1";
"""

VBIND_INCLUDE_OLD = '#include "p2b_moe.cuh"\n'
VBIND_INCLUDE_NEW = '''#include "p2b_moe.cuh"
#include "pf_g8_host.h"  // --- pfg8-kernel-port ---
'''


def sub(text: str, old: str, new: str, label: str, count: int = 1) -> str:
    n = text.count(old)
    if n != count:
        raise SystemExit(f"pfg8_kernels_vllm: [{label}] anchor count={n} (expected {count})")
    return text.replace(old, new)


def patch_file(root: Path, rel: str, subs: list, marker_in_head: bool = True) -> bool:
    p = root / rel
    t = p.read_text()
    if MARKER in t:
        return False
    for old, new, label, *cnt in subs:
        c = cnt[0] if cnt else 1
        t = sub(t, old, new, label, c)
    p.write_text(t)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_root", type=Path, help="vllm-exl3 checkout root (e.g. /opt/vllm-exl3)")
    args = ap.parse_args()
    root = args.repo_root

    jobs = [
        ("csrc/p2b_moe.cu", [
            (MOE_BP_OLD, MOE_BP_NEW, "p2b_moe bp/ld_b"),
            (MOE_HOST_OLD, MOE_HOST_NEW, "p2b_moe host apply"),
        ]),
        ("csrc/p2b_batched.cu", [
            (BATCH_TILE2_OLD, BATCH_TILE2_NEW, "p2b_batched tile2"),
            (BATCH_ENTRY_OLD, BATCH_ENTRY_NEW, "p2b_batched entry apply"),
        ]),
        ("csrc/exl3_gemv.cu", [
            (VGEVV_INCLUDE_OLD, VGEVV_INCLUDE_NEW, "exl3_gemv.cu include"),
            (VGEVV_ENTRY_OLD, VGEVV_ENTRY_NEW, "exl3_gemv.cu dims+apply"),
        ]),
        ("csrc/exl3_gemm.cu", [
            (VGEMM_HEAD_OLD, VGEMM_HEAD_NEW, "exl3_gemm.cu include"),
            (VGEMM_N_OLD, VGEMM_N_NEW, "exl3_gemm.cu n"),
        ]),
        ("csrc/bindings.cpp", [
            (VBIND_INCLUDE_OLD, VBIND_INCLUDE_NEW, "bindings include"),
            (VBIND_MODULE_OLD, VBIND_MODULE_NEW, "bindings module"),
        ]),
    ]
    for rel, subs in jobs:
        did = patch_file(root, rel, subs)
        print(f"dsv41: pfg8 vllm-exl3 {rel}: {'patched' if did else 'already present'}")

    # fat gemm: guard both entries (marker = the guard string itself)
    fat = root / "csrc/exl3_fat_gemm.cu"
    t = fat.read_text()
    if "pfg8-kernel-port" not in t:
        t2 = sub(t, FAT_HEAD_OLD, FAT_HEAD_NEW, "fat include")
        t2 = fat_guards(t2)
        fat.write_text(t2)
        print("dsv41: pfg8 vllm-exl3 csrc/exl3_fat_gemm.cu: patched")
    else:
        print("dsv41: pfg8 vllm-exl3 csrc/exl3_fat_gemm.cu: already present")


if __name__ == "__main__":
    main()
