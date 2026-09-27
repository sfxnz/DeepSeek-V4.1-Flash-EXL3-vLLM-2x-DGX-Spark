#!/usr/bin/env python3
"""Generate bench3.cu: deployed kernel + cp.async smem-pipeline tile variant.

DEC 3: stage each k-slice's warp words through shared memory with cp.async
       (4B per lane), a 4-slice ring over 4 buffers (safe: buffer = i & 3,
       RING = NBUF). Replaces both SHFL gathers with smem reads and decouples
       prefetch depth from register pressure. Same math order -> bit-exact.
"""
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "vllm-exl3-patched" / "csrc" / "p2b_moe.cu"
DST = Path(__file__).resolve().parent / "bench3.cu"

src = SRC.read_text()

def sub1(s, old, new, label):
    if s.count(old) != 1:
        raise SystemExit(f"[{label}] expected one occurrence, got {s.count(old)}")
    return s.replace(old, new)

VENDORED = r'''
// --- bench3: cp.async pipeline helpers (sm_90+) ---
namespace bench3ns {

__device__ __forceinline__ void cp_async4(void* smem, const void* gmem)
{
    unsigned smem_int = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                 :: "r"(smem_int), "l"(gmem));
}

__device__ __forceinline__ void cp_commit()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_wait3()
{
    asm volatile("cp.async.wait_group 3;\n" ::);
}

}  // namespace bench3ns
'''

src = sub1(src, "namespace cg = cooperative_groups;",
               "namespace cg = cooperative_groups;\n" + VENDORED, "vendored")

src = sub1(src, "template <int bits, int cb, int CFG>\n__device__ __forceinline__ void run_gemv_tile(",
               "template <int bits, int cb, int CFG, int DEC = 0>\n__device__ __forceinline__ void run_gemv_tile(", "tile template")

src = sub1(src, "float (*sh_red)[1][64])",
               "float (*sh_red)[1][64], uint32_t* cp_words = nullptr)", "tile args")

OLD_PF = """    uint32_t pf[PF][LOADS];
    #pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < myn)
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                pf[d][l] = ld_b(d, l);
"""
NEW_PF = """    uint32_t pf[PF][LOADS];
    if constexpr (DEC == 3) {
        // cp.async ring: 4 buffers, one k-slice per buffer. Prologue fills
        // slices 0..3; the main loop waits <=3 pending (slice i landed),
        // consumes buffer i&3, then issues slice i+4 into the same buffer
        // (its reads for this warp already happened this iteration).
        const uint32_t* bpv = bp;
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            if (p < myn) {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    bench3ns::cp_async4(cp_words + ((p & 3) * LOADS + l) * 32 + lane,
                                        bpv + (size_t) p * slice_stride + l * LSTRIDE);
                bench3ns::cp_commit();
            }
        }
    } else {
        #pragma unroll
        for (int d = 0; d < PF; ++d)
            if (d < myn)
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(d, l);
    }"""
src = sub1(src, OLD_PF, NEW_PF, "prefetch")

OLD_BODY = """            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                bw[l] = pf[d][l];

            if (i + PF < myn) {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(i + PF, l);
            }"""
NEW_BODY = """            uint32_t bw[LOADS];
            if constexpr (DEC == 3) {
                bench3ns::cp_wait3();
                __syncwarp();
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    bw[l] = cp_words[((i & 3) * LOADS + l) * 32 + lane];
                if (i + 4 < myn) {
                    #pragma unroll
                    for (int l = 0; l < LOADS; ++l)
                        bench3ns::cp_async4(cp_words + (((i + 4) & 3) * LOADS + l) * 32 + lane,
                                            bp + (size_t)(i + 4) * slice_stride + l * LSTRIDE);
                    bench3ns::cp_commit();
                }
            } else {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    bw[l] = pf[d][l];
            }

            if (i + PF < myn && DEC != 3) {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(i + PF, l);
            }"""
src = sub1(src, OLD_BODY, NEW_BODY, "body")

OLD_DECODE = """                } else if constexpr (bits == 2) {
                    const uint32_t w = bw[t >> 1];
                    const int base = (t & 1) << 4;
                    uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                    uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    exl3_gemv_ns::dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);
                } else {"""
NEW_DECODE = """                } else if constexpr (bits == 2) {
                    const int base = (t & 1) << 4;
                    uint32_t bwv, awv;
                    if constexpr (DEC == 3) {
                        const uint32_t* tp = cp_words + ((i & 3) * LOADS + (t >> 1)) * 32;
                        bwv = tp[base + x_src_b];
                        awv = tp[base + x_src_a];
                    } else {
                        const uint32_t w = bw[t >> 1];
                        bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                        awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    }
                    exl3_gemv_ns::dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);
                } else {"""
src = sub1(src, OLD_DECODE, NEW_DECODE, "decode")

src = sub1(src, "template <int BITS, int CB>\n__global__ __launch_bounds__(256, 4)",
               "template <int BITS, int CB, int DEC = 0>\n__global__ __launch_bounds__(256, 4)", "kernel template")
src = sub1(src, "    __shared__ float sh_red[8][1][64];",
               "    __shared__ float sh_red[8][1][64];\n    __shared__ uint32_t cp_sh[8][256];", "shared decl")

OLD_CALL_G = "run_gemv_tile<BITS, CB, 1>(B32, A2, C, kslices_gate, hidden, group, ntiles_gate, warp, lane, sh_red);"
NEW_CALL_G = "run_gemv_tile<BITS, CB, 1, DEC>(B32, A2, C, kslices_gate, hidden, group, ntiles_gate, warp, lane, sh_red, (uint32_t*)cp_sh[warp]);"
src = sub1(src, OLD_CALL_G, NEW_CALL_G, "gate call")
OLD_CALL_D = "run_gemv_tile<BITS, CB, 1>(B32, A2, C, kslices_down, inter, group, ntiles_down, warp, lane, sh_red);"
NEW_CALL_D = "run_gemv_tile<BITS, CB, 1, DEC>(B32, A2, C, kslices_down, inter, group, ntiles_down, warp, lane, sh_red, (uint32_t*)cp_sh[warp]);"
src = sub1(src, OLD_CALL_D, NEW_CALL_D, "down call")

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

at::Tensor bench3(
    const at::Tensor& x, at::Tensor& out,
    const at::Tensor& gt, const at::Tensor& gu, const at::Tensor& gv,
    const at::Tensor& ut, const at::Tensor& uu, const at::Tensor& uv,
    const at::Tensor& dt, const at::Tensor& du, const at::Tensor& dv,
    const at::Tensor& ids, const at::Tensor& rw,
    int64_t kg, int64_t ku, int64_t kd, bool mcg,
    int64_t intermediate_size, double swiglu_limit,
    int64_t vdec, int64_t iters, int64_t warmup)
{
    TORCH_CHECK(kg == 2 && ku == 2 && kd == 2 && mcg, "bench3 supports K=2 MCG only");
    TORCH_CHECK(vdec == 0 || vdec == 3, "variant must be 0 (stock) or 3 (cp.async)");
    if (vdec == 0) run_bench_variant<0>(x, out, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, intermediate_size, swiglu_limit, iters, warmup);
    else run_bench_variant<3>(x, out, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, intermediate_size, swiglu_limit, iters, warmup);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("bench3", &bench3, "p2b bench v3");
    m.def("last_ms", []() { return g_last_ms; });
}
'''
src += BENCH
DST.write_text(src)
print(f"wrote {DST} ({len(src.splitlines())} lines)")
