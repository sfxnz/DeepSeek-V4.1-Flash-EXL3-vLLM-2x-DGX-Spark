#!/usr/bin/env python3
"""Generate bench6.cu: deployed kernel + DEC=6 N-split warp decomposition.

DEC 6: instead of warps splitting K (each warp a k-chunk of ONE 4-tile
group), each warp owns ONE group and iterates the FULL k range. The 8 warps
of a work item cover 8 adjacent groups, so per k-slice the block reads
8 x 256B = 2KB contiguous instead of 8 scattered 256B chunks. The cross-warp
reduction disappears (each warp owns its group's columns end to end).
Numerics: same per-column fp16->fp32 fold cadence, but one summation tree
instead of 8 partials summed in fp32 — equivalent-or-better rounding, NOT
bit-exact vs stock (gated by tolerance in the driver, e2e gates in serve).
"""
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "vllm-exl3-patched" / "csrc" / "p2b_moe.cu"
DST = Path(__file__).resolve().parent / "bench6.cu"

src = SRC.read_text()

def sub1(s, old, new, label):
    if s.count(old) != 1:
        raise SystemExit(f"[{label}] count={s.count(old)}")
    return s.replace(old, new)

src = sub1(src, "template <int bits, int cb, int CFG>\n__device__ __forceinline__ void run_gemv_tile(",
               "template <int bits, int cb, int CFG, int DEC = 0>\n__device__ __forceinline__ void run_gemv_tile(", "tile template")

# Whole k-range per warp; group from warp id; no sh_red reduction.
OLD_HEAD = """    const int chunk = CEIL_DIVIDE(kslices, WK);
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));
    const size_t slice_stride = (size_t) ntiles * TWORDS;"""
NEW_HEAD = """    int chunk, ks0, myn, group_;
    if constexpr (DEC == 6) {
        // N-split: warp w owns group (item_group * WK + warp), full k range.
        chunk = kslices;
        ks0 = 0;
        myn = kslices;
        group_ = group * WK + warp;
        if (group_ >= ntiles / WNT) myn = 0;
        group = group_;
    } else {
        chunk = CEIL_DIVIDE(kslices, WK);
        ks0 = warp * chunk;
        myn = max(0, min(chunk, kslices - ks0));
        group_ = group;
    }
    const size_t slice_stride = (size_t) ntiles * TWORDS;"""
src = sub1(src, OLD_HEAD, NEW_HEAD, "head")

# Epilogue: DEC6 warps own their columns; direct store, no reduction.
OLD_EPI = """    // Warp reduction
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
    __syncthreads();"""
NEW_EPI = """    // Warp reduction
    if constexpr (DEC == 6) {
        if (myn > 0) {
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
            __syncwarp();
            for (int idx = lane; idx < COLS; idx += 32) {
                const int col = group * COLS + idx;
                C[col] = __float2half_rn(sh_red[warp][0][idx]);
            }
            __syncwarp();
        }
    } else {
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
src = sub1(src, OLD_EPI, NEW_EPI, "epilogue")

src = sub1(src, "template <int BITS, int CB>\n__global__ __launch_bounds__(256, 4)",
               "template <int BITS, int CB, int DEC = 0>\n__global__ __launch_bounds__(256, 4)", "kernel template")
src = sub1(src, "run_gemv_tile<BITS, CB, 1>(B32, A2, C, kslices_gate, hidden, group, ntiles_gate, warp, lane, sh_red);",
               "run_gemv_tile<BITS, CB, 1, DEC>(B32, A2, C, kslices_gate, hidden, group, ntiles_gate, warp, lane, sh_red);", "gate call")
src = sub1(src, "run_gemv_tile<BITS, CB, 1>(B32, A2, C, kslices_down, inter, group, ntiles_down, warp, lane, sh_red);",
               "run_gemv_tile<BITS, CB, 1, DEC>(B32, A2, C, kslices_down, inter, group, ntiles_down, warp, lane, sh_red);", "down call")

# Work-item counts: super-groups of WK groups.
src = sub1(src, "        int total_work = 2 * m * experts * num_groups_gate;\n        for (int item = blockIdx.x; item < total_work; item += gridDim.x) {\n            int is_up = item & 1;\n            int rem = item >> 1;\n            int row = rem / (experts * num_groups_gate);\n            int rem2 = rem % (experts * num_groups_gate);\n            int e = rem2 / num_groups_gate;\n            int group = rem2 % num_groups_gate;",
               "        const int item_groups_gate = DEC == 6 ? CEIL_DIVIDE(num_groups_gate, 8) : num_groups_gate;\n        int total_work = 2 * m * experts * item_groups_gate;\n        for (int item = blockIdx.x; item < total_work; item += gridDim.x) {\n            int is_up = item & 1;\n            int rem = item >> 1;\n            int row = rem / (experts * item_groups_gate);\n            int rem2 = rem % (experts * item_groups_gate);\n            int e = rem2 / item_groups_gate;\n            int group = rem2 % item_groups_gate;", "gate items+group")
src = sub1(src, "        int total_work = m * experts * num_groups_down;\n        for (int item = blockIdx.x; item < total_work; item += gridDim.x) {\n            int row = item / (experts * num_groups_down);\n            int rest = item % (experts * num_groups_down);\n            int e = rest / num_groups_down;\n            int group = rest % num_groups_down;",
               "        const int item_groups_down = DEC == 6 ? CEIL_DIVIDE(num_groups_down, 8) : num_groups_down;\n        int total_work = m * experts * item_groups_down;\n        for (int item = blockIdx.x; item < total_work; item += gridDim.x) {\n            int row = item / (experts * item_groups_down);\n            int rest = item % (experts * item_groups_down);\n            int e = rest / item_groups_down;\n            int group = rest % item_groups_down;", "down items+group")

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

at::Tensor bench6(
    const at::Tensor& x, at::Tensor& out,
    const at::Tensor& gt, const at::Tensor& gu, const at::Tensor& gv,
    const at::Tensor& ut, const at::Tensor& uu, const at::Tensor& uv,
    const at::Tensor& dt, const at::Tensor& du, const at::Tensor& dv,
    const at::Tensor& ids, const at::Tensor& rw,
    int64_t kg, int64_t ku, int64_t kd, bool mcg,
    int64_t intermediate_size, double swiglu_limit,
    int64_t vdec, int64_t iters, int64_t warmup)
{
    TORCH_CHECK(kg == 2 && ku == 2 && kd == 2 && mcg, "bench6 supports K=2 MCG only");
    TORCH_CHECK(vdec == 0 || vdec == 6, "variant must be 0 (stock) or 6 (N-split warps)");
    if (vdec == 0) run_bench_variant<0>(x, out, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, intermediate_size, swiglu_limit, iters, warmup);
    else run_bench_variant<6>(x, out, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, intermediate_size, swiglu_limit, iters, warmup);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("bench6", &bench6, "p2b bench v6");
    m.def("last_ms", []() { return g_last_ms; });
}
'''
src += BENCH
DST.write_text(src)
print(f"wrote {DST} ({len(src.splitlines())} lines)")
