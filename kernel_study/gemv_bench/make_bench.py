#!/usr/bin/env python3
"""Generate bench.cu from the patched p2b_moe.cu (deployed kernel source).

Transformations:
  1. run_gemv_tile gains a PFMUL template param (prefetch ring depth multiplier).
  2. Kernel/launch templates gain PFMUL; dispatch passes 1.
  3. Appends bench host entry: scratch cached statically, warmup + CUDA-event
     timed loop, variant dispatch over (PFMUL), ms reported via last_ms().
Everything else is byte-identical to the deployed kernel so variant 0 is the
serving path.
"""
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parent.parent / "vllm-exl3-patched" / "csrc" / "p2b_moe.cu"
DST = Path(__file__).resolve().parent / "bench.cu"

src = SRC.read_text()

def sub1(s, old, new):
    if s.count(old) != 1:
        raise SystemExit(f"expected exactly one occurrence of {old!r}, got {s.count(old)}")
    return s.replace(old, new)

src = sub1(src, "template <int bits, int cb, int CFG>\n__device__ __forceinline__ void run_gemv_tile(",
               "template <int bits, int cb, int CFG, int PFMUL = 1>\n__device__ __forceinline__ void run_gemv_tile(")
src = sub1(src, "    constexpr int PF = CFG == 0 ? 4 : 2;",
               "    constexpr int PF = (CFG == 0 ? 4 : 2) * PFMUL;")
# The tail-fold guard `(d + 1) % FOLD == 0` still divides PF; FOLD unchanged, fine.
src = sub1(src, "template <int BITS, int CB>\n__global__ __launch_bounds__(256, 4)",
               "template <int BITS, int CB, int PFMUL = 1>\n__global__ __launch_bounds__(256, 4)")
assert src.count("run_gemv_tile<BITS, CB, 1>(") == 2
src = src.replace("run_gemv_tile<BITS, CB, 1>(", "run_gemv_tile<BITS, CB, 1, PFMUL>(")
src = sub1(src, "template <int BITS, int CB>\nstatic void launch_moe_batched(",
               "template <int BITS, int CB, int PFMUL = 1>\nstatic void launch_moe_batched(")
src = sub1(src, "    void* kernel = (void*) p2b_moe_batched_kernel<BITS, CB>;",
               "    void* kernel = (void*) p2b_moe_batched_kernel<BITS, CB, PFMUL>;")

# Variant dispatch for the bench entry (kg==2, mcg only; serving shapes).
BENCH = r'''

// ---------------------------------------------------------------------------
// Bench additions (not in the serving binary)
// ---------------------------------------------------------------------------
#include <pybind11/pybind11.h>

namespace py = pybind11;

template <int PFMUL>
static void launch_variant(
    const at::Tensor& x, const at::Tensor& gt, const at::Tensor& gu,
    const at::Tensor& gv, const at::Tensor& ut, const at::Tensor& uu,
    const at::Tensor& uv, const at::Tensor& dt, const at::Tensor& du,
    const at::Tensor& dv, const at::Tensor& ids, const at::Tensor& rw,
    at::Tensor& out, at::Tensor& gate, at::Tensor& up, at::Tensor& down,
    at::Tensor& had_gate, at::Tensor& had_up, at::Tensor& had_down,
    at::Tensor& accum, int e, int m, int hidden, int inter, float swiglu_limit)
{
    launch_moe_batched<2, 1, PFMUL>(x, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw,
                                    out, gate, up, down, had_gate, had_up, had_down,
                                    accum, e, m, hidden, inter, swiglu_limit);
}

static double g_last_ms = 0.0;
static int g_last_occupancy = 0;

at::Tensor bench_p2b(
    const at::Tensor& x, at::Tensor& out,
    const at::Tensor& gt, const at::Tensor& gu, const at::Tensor& gv,
    const at::Tensor& ut, const at::Tensor& uu, const at::Tensor& uv,
    const at::Tensor& dt, const at::Tensor& du, const at::Tensor& dv,
    const at::Tensor& ids, const at::Tensor& rw,
    int64_t kg, int64_t ku, int64_t kd, bool mcg,
    int64_t intermediate_size, double swiglu_limit,
    int64_t variant, int64_t iters, int64_t warmup)
{
    TORCH_CHECK(kg == 2 && ku == 2 && kd == 2 && mcg, "bench supports K=2 MCG only");
    const int e = static_cast<int>(ids.numel());
    constexpr int m = 1, hidden = 5120;
    const int inter = static_cast<int>(intermediate_size);

    // Scratch cached by e (avoids allocator noise inside the timed loop).
    static int cached_e = -1, cached_inter = -1;
    static at::Tensor gate, up, down, had_gate, had_up, had_down, accum;
    if (cached_e != e || cached_inter != inter) {
        auto opts = x.options();
        gate = at::empty({e, m, inter}, opts);
        up = at::empty({e, m, inter}, opts);
        down = at::empty({e, m, hidden}, opts);
        had_gate = at::empty({e, m, hidden}, opts);
        had_up = at::empty({e, m, hidden}, opts);
        had_down = at::empty({e, m, inter}, opts);
        accum = at::zeros({m, hidden}, opts.dtype(at::kFloat));
        cached_e = e; cached_inter = inter;
    }

    int dev = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    void* kptr = variant == 0 ? (void*) p2b_moe_batched_kernel<2, 1, 1>
               : variant == 1 ? (void*) p2b_moe_batched_kernel<2, 1, 2>
               : variant == 2 ? (void*) p2b_moe_batched_kernel<2, 1, 4>
               : variant == 3 ? (void*) p2b_moe_batched_kernel<2, 1, 8>
               : nullptr;
    TORCH_CHECK(kptr, "unknown variant");
    int resident = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kptr, 256, 0);
    g_last_occupancy = resident;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    for (int64_t i = 0; i < warmup; ++i) {
        switch (variant) {
        case 0: launch_variant<1>(x, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, out, gate, up, down, had_gate, had_up, had_down, accum, e, m, hidden, inter, (float)swiglu_limit); break;
        case 1: launch_variant<2>(x, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, out, gate, up, down, had_gate, had_up, had_down, accum, e, m, hidden, inter, (float)swiglu_limit); break;
        case 2: launch_variant<4>(x, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, out, gate, up, down, had_gate, had_up, had_down, accum, e, m, hidden, inter, (float)swiglu_limit); break;
        case 3: launch_variant<8>(x, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, out, gate, up, down, had_gate, had_up, had_down, accum, e, m, hidden, inter, (float)swiglu_limit); break;
        }
    }
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0, stream);
    for (int64_t i = 0; i < iters; ++i) {
        switch (variant) {
        case 0: launch_variant<1>(x, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, out, gate, up, down, had_gate, had_up, had_down, accum, e, m, hidden, inter, (float)swiglu_limit); break;
        case 1: launch_variant<2>(x, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, out, gate, up, down, had_gate, had_up, had_down, accum, e, m, hidden, inter, (float)swiglu_limit); break;
        case 2: launch_variant<4>(x, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, out, gate, up, down, had_gate, had_up, had_down, accum, e, m, hidden, inter, (float)swiglu_limit); break;
        case 3: launch_variant<8>(x, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, out, gate, up, down, had_gate, had_up, had_down, accum, e, m, hidden, inter, (float)swiglu_limit); break;
        }
    }
    cudaEventRecord(ev1, stream);
    cudaEventSynchronize(ev1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    g_last_ms = (double) ms / (double) iters;
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("bench", &bench_p2b, "p2b bench");
    m.def("last_ms", []() { return g_last_ms; });
    m.def("last_occupancy", []() { return g_last_occupancy; });
}
'''

src = src + BENCH
DST.write_text(src)
print(f"wrote {DST} ({len(src.splitlines())} lines)")
