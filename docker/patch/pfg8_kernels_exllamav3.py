#!/usr/bin/env python3
"""PF-G8 kernel port — exllamav3 side (components 6 + the LinearEXL3 reader).

Applies to a fresh clone of exllamav3 @ EXLLAMAV3_REF (the canonical-e12 pin;
repo layout /opt/exllamav3 with package exllamav3/exllamav3_ext, per the
Dockerfile's patch_exllamav3_aarch64.py call) in the image build. Runtime
behavior is env-gated by DSV41_LOAD_PF_G8 (default off = exact stock
arithmetic; the flag is a __constant__ that stays 0 when the env is unset, so
no symbol writes ever happen on a stock boot).

What it patches (repo root argument):

  exllamav3/exllamav3_ext/quant/pf_g8_host.h        NEW header-only host registry
  exllamav3/exllamav3_ext/quant/exl3_gemm_inner.cuh PREFILL GEMM B-load remap — the exact
      substitutions proven bit-exact 20/20 by the harness twin
      (kernel_study/gemv_bench/build_prefill/exl3_gemm_inner_pf.cuh, generated
      from this same file; k-stride fix 9f71b6e included). Covers exl3_gemm,
      exl3_mgemm (non-sliced) and exl3_moe: all include this header.
  exllamav3_ext/quant/exl3_gemv_kernel.cuh QTIP small-m GEMV reader (K=2 is
      always eligible at m<=8, so this is a real decode reader) — DEC5
      group-major addressing adapted to 8-tile pack groups.
  exllamav3_ext/quant/exl3_gemm.cu        host: G8 dims branch (K/size_n from
      [NG][KT][128*K] instead of [KT][NT][16*K]), pf_g8 apply, mgemm n-slice
      guard under G8.
  exllamav3_ext/quant/exl3_gemv.cu        host: G8 dims branch + apply.
  exllamav3_ext/quant/exl3_moe.cu         host: pf_g8 apply.
  exllamav3_ext/bindings.cpp              pf_g8_set/pf_g0_get exports + build
      marker string (boot-g8.sh strings guard).
  exllamav3/modules/quant/exl3.py         LinearEXL3.forward: under G8 never
      take the reconstruct branch (rows > 144 would decode trellis with the
      stock-layout reconstruct kernel -> garbage on a G8 pack).

Idempotent: re-running on a patched tree is a no-op (marker check per file).
"""
from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "pfg8-kernel-port"  # matches both "# ---" and "// ---" comment forms

PF_G8_HOST_H = '''#pragma once
// --- pfg8-kernel-port ---
// PF-G8 layout flag plumbing (header-only, one instance per shared object).
//
// Device side: every translation unit that includes a patched kernel header
// gets its own `static __constant__ int pf_g8` (TU-local; no RDC needed) and
// registers a setter lambda here at static init. Host side: DSV41_LOAD_PF_G8
// is read ONCE per process (cached); pfg8::apply() (call_once, first kernel
// dispatch) writes the value into every registered TU constant on the current
// device. When the env is unset the value is 0 and nothing is ever written —
// the constant is initialized to 0 at module load, so the stock arithmetic
// path is bit-identical to the unpatched build.
#include <cstdlib>
#include <functional>
#include <mutex>
#include <vector>
#include <cuda_runtime.h>

namespace pfg8 {

inline int env_value()
{
    static const int v = []
    {
        const char* e = std::getenv("DSV41_LOAD_PF_G8");
        return (e && e[0] == '1') ? 1 : 0;
    }();
    return v;
}

inline int& forced_value()
{
    static int v = -1;
    return v;
}

inline int current_value()
{
    return forced_value() >= 0 ? forced_value() : env_value();
}

inline std::vector<std::function<void(int)>>& setters()
{
    static std::vector<std::function<void(int)>> v;
    return v;
}

inline void register_setter(std::function<void(int)> s)
{
    setters().push_back(std::move(s));
}

inline void apply_now(int v)
{
    for (auto& s : setters()) s(v);
}

inline std::once_flag& once_flag()
{
    static std::once_flag f;
    return f;
}

// Call before the first kernel launch of the process (host entries do).
inline void apply()
{
    std::call_once(once_flag(), [] { apply_now(current_value()); });
}

// Test/A-B lever: force the flag and push it to every registered constant.
inline void force(int v)
{
    forced_value() = v;
    apply_now(v);
}

}  // namespace pfg8

// Per-TU registrar: `PFG8_REGISTER(pf_g8);` after a `static __constant__` decl.
// Anonymous namespace => TU-local, no ODR issues across the ~60 kernel TUs.
#define PFG8_REGISTER(sym)                                                    \\
    namespace                                                                 \\
    {                                                                         \\
    struct pfg8_reg_t                                                         \\
    {                                                                         \\
        pfg8_reg_t()                                                          \\
        {                                                                     \\
            pfg8::register_setter([](int _v)                                  \\
            {                                                                 \\
                cudaError_t _e = cudaMemcpyToSymbol(sym, &_v, sizeof(int));   \\
                (void) _e;                                                    \\
            });                                                               \\
        }                                                                     \\
    } pfg8_reg_inst_;                                                         \\
    }
'''

INNER_HEAD_OLD = "#define EXL3_GEMM_BASE_THREADS 256\n"
INNER_HEAD_NEW = """#define EXL3_GEMM_BASE_THREADS 256

// --- pfg8-kernel-port ---
// PF-G8 layout flag: 1 = trellis stored group-major
// [n-group of 8 tiles][k-block][8*16*bits words]. Host-settable via
// DSV41_LOAD_PF_G8 (read once; see pf_g8_host.h). TU-local: each kernel TU
// gets its own copy and registers a setter with the host registry.
#include "pf_g8_host.h"
static __constant__ int pf_g8 = 0;
PFG8_REGISTER(pf_g8);
"""

INNER_PTR_OLD = """    int gl_b_stride_k = blocks_n_full * TILEBLOCKS_K * 256 / 16 * bits;
    const int gl_b_stride_n = TILEBLOCKS_N * 256 / 16 * bits;
    const int sh0_b_stride_k = TILEBLOCKS_K * TILEBLOCKS_N * 256 / 16 * bits;
    const uint16_t* gl_b_ptr = B + slice0_k * gl_b_stride_k + slice0_n * gl_b_stride_n;
"""
INNER_PTR_NEW = """    // --- pfg8-kernel-port ---
    // PF-G8: B stored [group of 8 n-tiles][k-block][8*16*bits words].
    // Group g covers n-blocks 8g..8g+7; a k-block row inside a group is
    // 8*16*bits contiguous words (512B at bits=2). Any n-tile window
    // (TILEBLOCKS_N in {8,16,32}) is whole groups. load_b_gl addresses
    // within the window (its (blk/8) term jumps groups INSIDE the window),
    // so the window base advances per k-tile by TILEBLOCKS_K group k-rows
    // and per n-window by whole groups (KB_FULL * 8 * 16 * bits words).
    const int pf_kb_full = size_k / 16;
    int gl_b_stride_k = blocks_n_full * TILEBLOCKS_K * 256 / 16 * bits;
    const int gl_b_stride_n = TILEBLOCKS_N * 256 / 16 * bits;
    const int sh0_b_stride_k = TILEBLOCKS_K * TILEBLOCKS_N * 256 / 16 * bits;
    if (pf_g8) gl_b_stride_k = TILEBLOCKS_K * 8 * 16 * bits;
    auto pf_b_n_off = [&] (int sn) -> size_t
    {
        return pf_g8
            ? (size_t)(sn * TILEBLOCKS_N / 8) * ((size_t) pf_kb_full * 8 * 16 * bits)
            : (size_t) sn * gl_b_stride_n;
    };
    const uint16_t* gl_b_ptr = B + slice0_k * gl_b_stride_k + pf_b_n_off(slice0_n);
"""

INNER_LOADB_OLD = """        int n = (i * EXL3_GEMM_BASE_THREADS + t) % (gl_b_stride_n / 8);
        int k = (i * EXL3_GEMM_BASE_THREADS + t) / (gl_b_stride_n / 8);
        load_b_gl[i] = k * (blocks_n_full * 256 / 16 * bits / 8) + n;
"""
INNER_LOADB_NEW = """        int n = (i * EXL3_GEMM_BASE_THREADS + t) % (gl_b_stride_n / 8);
        int k = (i * EXL3_GEMM_BASE_THREADS + t) / (gl_b_stride_n / 8);
        if (pf_g8)
        {
            // n = int4 column in the window (4 int4s per 16-col block),
            // k = k-block in the tile. G8 address (int4 units):
            //   (blk/8) * KB_FULL*16*bits   group-to-group (TILEBLOCKS_N > 8 only)
            // + k      * 16*bits            k-block row inside the group
            // + (blk%8)* 2*bits             block inside the k-row
            // + n%4                          int4 inside the block
            int blk = n / 4;
            load_b_gl[i] = (blk / 8) * (pf_kb_full * 16 * bits)
                         + k * (16 * bits)
                         + (blk % 8) * (2 * bits)
                         + (n % 4);
        }
        else
            load_b_gl[i] = k * (blocks_n_full * 256 / 16 * bits / 8) + n;
"""

INNER_WRAP_OLD = """            slice0_k = 0;
            slice0_n++;
            gl_a_ptr = A + slice_m * gl_a_stride_m + slice0_k * gl_a_stride_k;
            gl_b_ptr = B + slice0_k * gl_b_stride_k + slice0_n * gl_b_stride_n;
"""
INNER_WRAP_NEW = """            slice0_k = 0;
            slice0_n++;
            gl_a_ptr = A + slice_m * gl_a_stride_m + slice0_k * gl_a_stride_k;
            gl_b_ptr = B + slice0_k * gl_b_stride_k + pf_b_n_off(slice0_n);
"""

GEMV_KERNEL_MAXM_OLD = "#define EXL3_GEMV_MAX_M 8\n"
GEMV_KERNEL_MAXM_NEW = """#define EXL3_GEMV_MAX_M 8

// --- pfg8-kernel-port ---
#include "pf_g8_host.h"
static __constant__ int pf_g8 = 0;
PFG8_REGISTER(pf_g8);
"""

GEMV_KERNEL_BP_OLD = """        const uint32_t* bp = B32 + (size_t) ks0 * slice_stride + group * WNT * TWORDS + lane;

        // Prefetch ring (indices must be compile-time or pf lands in local memory)
        auto ld_b = [&] (int i, int l) -> uint32_t
        {
            if constexpr (bits == 3)
                return lane < 24 ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
            else
                return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE);
        };
"""
GEMV_KERNEL_BP_NEW = """        // --- pfg8-kernel-port ---
        // PF-G8: trellis stored [group of 8 tiles][k-slice][8*TWORDS uint32].
        // This block's run of WNT tiles (WNT in {2,4}; 8 % WNT == 0) lives in
        // pack group group/(8/WNT) at tile offset (group % (8/WNT))*WNT; a
        // k-slice row of a pack group is 8*TWORDS contiguous uint32 — the
        // same word stream as stock, so the prefetch ring, fragment math and
        // reduction order are untouched (bench5.cu DEC5 pattern, G8 adapted).
        const size_t g8_krow = (size_t) 8 * TWORDS;
        const size_t b_i_stride = pf_g8 ? g8_krow : slice_stride;
        const uint32_t* bp = pf_g8
            ? B32 + (size_t) (group / (8 / WNT)) * kslices * g8_krow
                    + (size_t) ks0 * g8_krow
                    + (size_t) (group % (8 / WNT)) * WNT * TWORDS + lane
            : B32 + (size_t) ks0 * slice_stride + group * WNT * TWORDS + lane;

        // Prefetch ring (indices must be compile-time or pf lands in local memory)
        auto ld_b = [&] (int i, int l) -> uint32_t
        {
            if constexpr (bits == 3)
                return lane < 24 ? __ldcs(bp + (size_t) i * b_i_stride + l * LSTRIDE) : 0;
            else
                return __ldcs(bp + (size_t) i * b_i_stride + l * LSTRIDE);
        };
"""

# ---- host: exl3_gemm.cu ----
GEMM_CU_INCLUDES_OLD = "#include \"exl3_gemm_kernel.cuh\"\n"
GEMM_CU_INCLUDES_NEW = "#include \"exl3_gemm_kernel.cuh\"\n#include \"pf_g8_host.h\"  // --- pfg8-kernel-port ---\n"

GEMM_CU_CHECKS_OLD = """    TORCH_CHECK_DIM(B, 3);
    TORCH_CHECK_SHAPES(A, -1, B, 0, 16);
    TORCH_CHECK_SHAPES(C, -1, B, 1, 16);
"""
GEMM_CU_CHECKS_NEW = """    TORCH_CHECK_DIM(B, 3);
    // --- pfg8-kernel-port --- G8 trellis [NG][KT][128*K]: k on dim 1, n on dim 0
    if (pfg8::env_value())
    {
        TORCH_CHECK_SHAPES(A, -1, B, 1, 16);
        TORCH_CHECK_SHAPES(C, -1, B, 0, 128);
    }
    else
    {
        TORCH_CHECK_SHAPES(A, -1, B, 0, 16);
        TORCH_CHECK_SHAPES(C, -1, B, 1, 16);
    }
"""

GEMM_CU_K_OLD = "    int K = B.size(2) / 16;\n"
GEMM_CU_K_NEW = "    int K = pfg8::env_value() ? B.size(2) / 128 : B.size(2) / 16;  // --- pfg8-kernel-port ---\n"

GEMM_CU_N_OLD = "    int size_n = B.size(1) * 16;\n"
GEMM_CU_N_NEW = "    int size_n = pfg8::env_value() ? B.size(0) * 128 : B.size(1) * 16;  // --- pfg8-kernel-port ---\n"

GEMM_CU_LOCKS_OLD = "    int* locks = DevCtx::instance().get_locks(device);\n"
GEMM_CU_LOCKS_NEW = "    pfg8::apply();  // --- pfg8-kernel-port --- (once per process)\n    int* locks = DevCtx::instance().get_locks(device);\n"

GEMM_CU_SLICE_OLD = "    TORCH_CHECK(!n_stride_list, \"exl3_mgemm: n_stride_list requires had_src_list\");\n"
GEMM_CU_SLICE_NEW = """    TORCH_CHECK(!n_stride_list, "exl3_mgemm: n_stride_list requires had_src_list");
    // --- pfg8-kernel-port --- column slices index B at a window offset; the G8
    // group-major base is only defined for whole groups (full-width matrices).
    TORCH_CHECK(!(pfg8::env_value() && size_n_list),
                "pf_g8: exl3_mgemm n-slices are not supported on G8 packs");
"""

# ---- host: exl3_gemv.cu ----
GEVV_CU_INCLUDES_OLD = "#include \"exl3_gemv_kernel.cuh\"\n"
GEVV_CU_INCLUDES_NEW = "#include \"exl3_gemv_kernel.cuh\"\n#include \"pf_g8_host.h\"  // --- pfg8-kernel-port ---\n"

GEVV_CU_CHECKS_OLD = """    TORCH_CHECK_DIM(B, 3);
    TORCH_CHECK_SHAPES(A, -1, B, 0, 16);
    TORCH_CHECK_SHAPES(C, -1, B, 1, 16);
"""
GEVV_CU_CHECKS_NEW = """    TORCH_CHECK_DIM(B, 3);
    // --- pfg8-kernel-port --- G8 trellis [NG][KT][128*K]
    if (pfg8::env_value())
    {
        TORCH_CHECK_SHAPES(A, -1, B, 1, 16);
        TORCH_CHECK_SHAPES(C, -1, B, 0, 128);
    }
    else
    {
        TORCH_CHECK_SHAPES(A, -1, B, 0, 16);
        TORCH_CHECK_SHAPES(C, -1, B, 1, 16);
    }
"""

GEVV_CU_DIMS_OLD = """    int size_n = B.size(1) * 16;
    int K = B.size(2) / 16;
"""
GEVV_CU_DIMS_NEW = """    // --- pfg8-kernel-port ---
    int size_n = pfg8::env_value() ? B.size(0) * 128 : B.size(1) * 16;
    int K = pfg8::env_value() ? B.size(2) / 128 : B.size(2) / 16;
"""

GEVV_CU_LOCKS_OLD = "    int* locks = DevCtx::instance().get_locks(device);\n"
GEVV_CU_LOCKS_NEW = "    pfg8::apply();  // --- pfg8-kernel-port --- (once per process)\n    int* locks = DevCtx::instance().get_locks(device);\n"

# ---- host: exl3_moe.cu ----
MOE_CU_INCLUDE_OLD = "#include \"comp_units/exl3_moe_instances.cuh\"\n"
MOE_CU_INCLUDE_NEW = "#include \"comp_units/exl3_moe_instances.cuh\"\n#include \"pf_g8_host.h\"  // --- pfg8-kernel-port ---\n"

MOE_CU_LOCKS_OLD = "    int* locks = DevCtx::instance().get_locks(device);\n"
MOE_CU_LOCKS_NEW = "    pfg8::apply();  // --- pfg8-kernel-port --- (once per process)\n    int* locks = DevCtx::instance().get_locks(device);\n"

# ---- bindings.cpp ----
BIND_MODULE_OLD = "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)\n{\n"
BIND_MODULE_NEW = """// --- pfg8-kernel-port ---
void pf_g8_set(int64_t v)
{
    pfg8::force((int) v);
}

int64_t pf_g8_get()
{
    return pfg8::current_value();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("pf_g8_set", &pf_g8_set, "set PF-G8 trellis layout flag (0 stock, 1 group-major)");
    m.def("pf_g8_get", &pf_g8_get, "get PF-G8 trellis layout flag");
    m.attr("pf_g8_build") = "pfg8-port-v1";
"""
BIND_INCLUDE_OLD = "#include \"sam.h\"\n"
BIND_INCLUDE_NEW = "#include \"sam.h\"\n#include \"quant/pf_g8_host.h\"  // --- pfg8-kernel-port ---\n"

# ---- python: modules/quant/exl3.py ----
PY_K_OLD = """        self.trellis = trellis
        self.K = trellis.shape[-1] // 16
"""
PY_K_NEW = """        self.trellis = trellis
        # --- pfg8-kernel-port --- G8 trellis [NG][KT][128*K]: last dim is 128*K
        self.K = (
            trellis.shape[-1] // 128
            if __import__("os").environ.get("DSV41_LOAD_PF_G8", "0") == "1"
            and trellis.shape[-1] % 128 == 0
            and trellis.shape[-1] // 128 in (2, 3, 4)
            else trellis.shape[-1] // 16
        )
"""

PY_RECON_OLD = """            rows = x.numel() // x.shape[-1]
            if rows <= AUTO_RECONSTRUCT_THRESHOLD or self.config.infer_params.no_reconstruct:
"""
PY_RECON_NEW = """            rows = x.numel() // x.shape[-1]
            # --- pfg8-kernel-port --- reconstruct decodes the stock trellis
            # layout; a G8 pack must stay on the (G8-aware) gemm/gemv path.
            _pf_g8 = __import__("os").environ.get("DSV41_LOAD_PF_G8", "0") == "1"
            if _pf_g8 or rows <= AUTO_RECONSTRUCT_THRESHOLD or self.config.infer_params.no_reconstruct:
"""


def sub(text: str, old: str, new: str, label: str, count: int = 1) -> str:
    n = text.count(old)
    if n != count:
        raise SystemExit(f"pfg8_kernels_exllamav3: [{label}] anchor count={n} (expected {count})")
    return text.replace(old, new)


def patch_inner(root: Path) -> bool:
    p = root / "exllamav3/exllamav3_ext/quant/exl3_gemm_inner.cuh"
    t = p.read_text()
    if MARKER in t:
        return False
    t = sub(t, INNER_HEAD_OLD, INNER_HEAD_NEW, "inner head")
    t = sub(t, INNER_PTR_OLD, INNER_PTR_NEW, "inner b ptr")
    t = sub(t, INNER_LOADB_OLD, INNER_LOADB_NEW, "inner load_b_gl")
    t = sub(t, INNER_WRAP_OLD, INNER_WRAP_NEW, "inner advance0 wrap")
    p.write_text(t)
    return True


def patch_gemv_kernel(root: Path) -> bool:
    p = root / "exllamav3/exllamav3_ext/quant/exl3_gemv_kernel.cuh"
    t = p.read_text()
    if MARKER in t:
        return False
    t = sub(t, GEMV_KERNEL_MAXM_OLD, GEMV_KERNEL_MAXM_NEW, "gemv kernel head")
    t = sub(t, GEMV_KERNEL_BP_OLD, GEMV_KERNEL_BP_NEW, "gemv kernel bp/ld_b")
    p.write_text(t)
    return True


def patch_gemm_cu(root: Path) -> bool:
    p = root / "exllamav3/exllamav3_ext/quant/exl3_gemm.cu"
    t = p.read_text()
    if MARKER in t:
        return False
    t = sub(t, GEMM_CU_INCLUDES_OLD, GEMM_CU_INCLUDES_NEW, "gemm.cu include")
    t = sub(t, GEMM_CU_CHECKS_OLD, GEMM_CU_CHECKS_NEW, "gemm.cu checks")
    t = sub(t, GEMM_CU_K_OLD, GEMM_CU_K_NEW, "gemm.cu K")
    t = sub(t, GEMM_CU_N_OLD, GEMM_CU_N_NEW, "gemm.cu size_n")
    # exactly two host entries (exl3_gemm_gr + exl3_mgemm_gr)
    t = sub(t, GEMM_CU_LOCKS_OLD, GEMM_CU_LOCKS_NEW, "gemm.cu locks/apply", count=2)
    t = sub(t, GEMM_CU_SLICE_OLD, GEMM_CU_SLICE_NEW, "gemm.cu mgemm slice guard")
    p.write_text(t)
    return True


def patch_gemv_cu(root: Path) -> bool:
    p = root / "exllamav3/exllamav3_ext/quant/exl3_gemv.cu"
    t = p.read_text()
    if MARKER in t:
        return False
    t = sub(t, GEVV_CU_INCLUDES_OLD, GEVV_CU_INCLUDES_NEW, "gemv.cu include")
    t = sub(t, GEVV_CU_CHECKS_OLD, GEVV_CU_CHECKS_NEW, "gemv.cu checks")
    t = sub(t, GEVV_CU_DIMS_OLD, GEVV_CU_DIMS_NEW, "gemv.cu dims")
    t = sub(t, GEVV_CU_LOCKS_OLD, GEVV_CU_LOCKS_NEW, "gemv.cu locks/apply")
    p.write_text(t)
    return True


def patch_moe_cu(root: Path) -> bool:
    p = root / "exllamav3/exllamav3_ext/quant/exl3_moe.cu"
    t = p.read_text()
    if MARKER in t:
        return False
    t = sub(t, MOE_CU_INCLUDE_OLD, MOE_CU_INCLUDE_NEW, "moe.cu include")
    t = sub(t, MOE_CU_LOCKS_OLD, MOE_CU_LOCKS_NEW, "moe.cu locks/apply")
    p.write_text(t)
    return True


def patch_bindings(root: Path) -> bool:
    p = root / "exllamav3/exllamav3_ext/bindings.cpp"
    t = p.read_text()
    if MARKER in t:
        return False
    t = sub(t, BIND_INCLUDE_OLD, BIND_INCLUDE_NEW, "bindings include")
    t = sub(t, BIND_MODULE_OLD, BIND_MODULE_NEW, "bindings module")
    p.write_text(t)
    return True


def patch_py_linear(root: Path) -> bool:
    p = root / "exllamav3/modules/quant/exl3.py"
    t = p.read_text()
    if MARKER in t:
        return False
    t = sub(t, PY_K_OLD, PY_K_NEW, "linear K derive")
    t = sub(t, PY_RECON_OLD, PY_RECON_NEW, "linear reconstruct gate")
    p.write_text(t)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_root", type=Path, help="exllamav3 checkout root (e.g. /opt/exllamav3)")
    args = ap.parse_args()
    root = args.repo_root

    host_h = root / "exllamav3/exllamav3_ext/quant/pf_g8_host.h"
    if not host_h.exists():
        host_h.write_text(PF_G8_HOST_H)
        print("dsv41: pfg8 wrote exllamav3_ext/quant/pf_g8_host.h")

    changed = [
        ("exl3_gemm_inner.cuh", patch_inner(root)),
        ("exl3_gemv_kernel.cuh", patch_gemv_kernel(root)),
        ("exl3_gemm.cu", patch_gemm_cu(root)),
        ("exl3_gemv.cu", patch_gemv_cu(root)),
        ("exl3_moe.cu", patch_moe_cu(root)),
        ("bindings.cpp", patch_bindings(root)),
        ("modules/quant/exl3.py", patch_py_linear(root)),
    ]
    for name, did in changed:
        print(f"dsv41: pfg8 exllamav3 {name}: {'patched' if did else 'already present'}")


if __name__ == "__main__":
    main()
