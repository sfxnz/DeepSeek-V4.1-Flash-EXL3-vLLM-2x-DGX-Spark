#!/usr/bin/env python3
"""Generate bench5.cu: deployed kernel + DEC=5 transposed-layout tile variant.

DEC 5: assumes the trellis is stored group-major: [group][k_slice][GWORDS]
where GWORDS = WNT*TWORDS words per (group, k) — the warp's whole k-chunk
stream becomes fully contiguous (stride 256B per i instead of jumping
slice_stride=4.6/20.5KB). Tests the DRAM-locality hypothesis without
touching the pack. Same math order -> bit-exact vs stock on permuted data.
"""
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "vllm-exl3-patched" / "csrc" / "p2b_moe.cu"
DST = Path(__file__).resolve().parent / "bench5.cu"

src = SRC.read_text()

def sub1(s, old, new, label):
    if s.count(old) != 1:
        raise SystemExit(f"[{label}] count={s.count(old)}")
    return s.replace(old, new)

src = sub1(src, "template <int bits, int cb, int CFG>\n__device__ __forceinline__ void run_gemv_tile(",
               "template <int bits, int cb, int CFG, int DEC = 0>\n__device__ __forceinline__ void run_gemv_tile(", "tile template")

OLD_BP = "    const uint32_t* bp = B32 + (size_t) ks0 * slice_stride + group * WNT * TWORDS + lane;"
NEW_BP = """    const uint32_t* bp;
    if constexpr (DEC == 5) {
        // group-major layout: [group][kslices][WNT*TWORDS]; warp chunk contiguous
        bp = B32 + (size_t) group * kslices * WNT * TWORDS + (size_t) ks0 * WNT * TWORDS + lane;
    } else {
        bp = B32 + (size_t) ks0 * slice_stride + group * WNT * TWORDS + lane;
    }"""
src = sub1(src, OLD_BP, NEW_BP, "bp init")

OLD_LDB = """    auto ld_b = [&] (int i, int l) -> uint32_t {
        if constexpr (bits == 3)
            return lane < 24 ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
        else
            return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE);
    };"""
NEW_LDB = """    auto ld_b = [&] (int i, int l) -> uint32_t {
        if constexpr (DEC == 5)
            return __ldcs(bp + (size_t) i * WNT * TWORDS + l * LSTRIDE);
        else if constexpr (bits == 3)
            return lane < 24 ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
        else
            return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE);
    };"""
src = sub1(src, OLD_LDB, NEW_LDB, "ld_b")

OLD_PFREFILL = """            if (i + PF < myn) {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(i + PF, l);
            }"""
NEW_PFREFILL = """            if (i + PF < myn) {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(i + PF, l);
            }"""
# unchanged; ld_b handles DEC

src = sub1(src, "template <int BITS, int CB>\n__global__ __launch_bounds__(256, 4)",
               "template <int BITS, int CB, int DEC = 0>\n__global__ __launch_bounds__(256, 4)", "kernel template")
src = sub1(src, "run_gemv_tile<BITS, CB, 1>(B32, A2, C, kslices_gate, hidden, group, ntiles_gate, warp, lane, sh_red);",
               "run_gemv_tile<BITS, CB, 1, DEC>(B32, A2, C, kslices_gate, hidden, group, ntiles_gate, warp, lane, sh_red);", "gate call")
src = sub1(src, "run_gemv_tile<BITS, CB, 1>(B32, A2, C, kslices_down, inter, group, ntiles_down, warp, lane, sh_red);",
               "run_gemv_tile<BITS, CB, 1, DEC>(B32, A2, C, kslices_down, inter, group, ntiles_down, warp, lane, sh_red);", "down call")
src = sub1(src, "template <int BITS, int CB>\nstatic void launch_moe_batched(",
               "template <int BITS, int CB, int DEC = 0>\nstatic void launch_moe_batched(", "launch template")
src = sub1(src, "    void* kernel = (void*) p2b_moe_batched_kernel<BITS, CB>;",
               "    void* kernel = (void*) p2b_moe_batched_kernel<BITS, CB, DEC>;", "kernel ptr")

BENCH = r'''

// ---------------------------------------------------------------------------
// Bench additions
// ---------------------------------------------------------------------------
#include <pybind11/pybind11.h>
namespace py = pybind11;

static double g_last_ms = 0.0;

template <int DEC>
static void run_bench_variant(
    const at::Tensor& x, at::Tensor& out,
    const at::Tensor& gt, const at::Tensor& gu, const at::Tensor& gv,
    const at::Tensor& ut, const at::Tensor& uu, const at::Tensor& uv,
    const at::Tensor& dt, const at::Tensor& du, const at::Tensor& dv,
    const at::Tensor& ids, const at::Tensor& rw,
    int64_t inter, double swiglu_limit, int64_t iters, int64_t warmup)
{
    const int e = static_cast<int>(ids.numel());
    constexpr int m = 1, hidden = 5120, bits = 2;
    const int inter_i = static_cast<int>(inter);

    static int cached_e = -1, cached_inter = -1;
    static at::Tensor gate, up, down, had_gate, had_up, had_down, accum;
    if (cached_e != e || cached_inter != inter_i) {
        auto opts = x.options();
        gate = at::empty({e, m, inter_i}, opts);
        up = at::empty({e, m, inter_i}, opts);
        down = at::empty({e, m, hidden}, opts);
        had_gate = at::empty({e, m, hidden}, opts);
        had_up = at::empty({e, m, hidden}, opts);
        had_down = at::empty({e, m, inter_i}, opts);
        accum = at::zeros({m, hidden}, opts.dtype(at::kFloat));
        cached_e = e; cached_inter = inter_i;
    }
    auto run1 = [&] {
        launch_moe_batched<bits, 1, DEC>(x, gt, gu, gv, ut, uu, uv, dt, du, dv,
            ids, rw, out, gate, up, down, had_gate, had_up, had_down, accum,
            e, m, hidden, inter_i, (float)swiglu_limit);
    };
    for (int64_t i = 0; i < warmup; ++i) run1();
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    cudaEventRecord(ev0, stream);
    for (int64_t i = 0; i < iters; ++i) run1();
    cudaEventRecord(ev1, stream);
    cudaEventSynchronize(ev1);
    float ms = 0; cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    g_last_ms = (double)ms / (double)iters;
}

at::Tensor bench5(
    const at::Tensor& x, at::Tensor& out,
    const at::Tensor& gt, const at::Tensor& gu, const at::Tensor& gv,
    const at::Tensor& ut, const at::Tensor& uu, const at::Tensor& uv,
    const at::Tensor& dt, const at::Tensor& du, const at::Tensor& dv,
    const at::Tensor& ids, const at::Tensor& rw,
    int64_t kg, int64_t ku, int64_t kd, bool mcg,
    int64_t intermediate_size, double swiglu_limit,
    int64_t vdec, int64_t iters, int64_t warmup)
{
    TORCH_CHECK(kg == 2 && ku == 2 && kd == 2 && mcg, "bench5 supports K=2 MCG only");
    TORCH_CHECK(vdec == 0 || vdec == 5, "variant must be 0 (stock layout) or 5 (group-major layout)");
    if (vdec == 0) run_bench_variant<0>(x, out, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, intermediate_size, swiglu_limit, iters, warmup);
    else run_bench_variant<5>(x, out, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, intermediate_size, swiglu_limit, iters, warmup);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("bench5", &bench5, "p2b bench v5");
    m.def("last_ms", []() { return g_last_ms; });
}
'''
src += BENCH
DST.write_text(src)
print(f"wrote {DST} ({len(src.splitlines())} lines)")
