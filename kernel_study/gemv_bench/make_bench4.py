#!/usr/bin/env python3
"""Generate bench4.cu: deployed kernel + CFG=2 (WNT=8) wide-tile variant.

DEC 4: CFG=2 tile — WK=8, WNT=8, PF=2, FOLD=2 at 256 threads. Doubles the
contiguous bytes per k-slice stride (512B vs 256B per 4.6KB step) without
touching the pack. Same decode math -> bit-exact modulo fp16 accumulation
order per output column (identical per-column order, actually: each column's
k-reduction order is unchanged; only the work partitioning differs).
"""
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "vllm-exl3-patched" / "csrc" / "p2b_moe.cu"
DST = Path(__file__).resolve().parent / "bench4.cu"

src = SRC.read_text()

def sub1(s, old, new, label):
    if s.count(old) != 1:
        raise SystemExit(f"[{label}] count={s.count(old)}")
    return s.replace(old, new)

# 1. CFG constants: add CFG==2 branch.
OLD_CONSTS = """    constexpr int WK = CFG == 0 ? 16 : 8;
    constexpr int WNT = CFG == 0 ? 2 : 4;
    constexpr int PF = CFG == 0 ? 4 : 2;
    constexpr int FOLD = CFG == 0 ? 4 : 2;"""
NEW_CONSTS = """    constexpr int WK = CFG == 0 ? 16 : 8;
    constexpr int WNT = CFG == 0 ? 2 : (CFG == 2 ? 8 : 4);
    constexpr int PF = CFG == 0 ? 4 : 2;
    constexpr int FOLD = CFG == 0 ? 4 : 2;"""
src = sub1(src, OLD_CONSTS, NEW_CONSTS, "cfg consts")

# 2. tile template gains DEC knob (unused decode paths unchanged).
src = sub1(src, "template <int bits, int cb, int CFG>\n__device__ __forceinline__ void run_gemv_tile(",
               "template <int bits, int cb, int CFG>\n__device__ __forceinline__ void run_gemv_tile(", "noop")

# 3. kernel: CFGT template param.
src = sub1(src, "template <int BITS, int CB>\n__global__ __launch_bounds__(256, 4)",
               "template <int BITS, int CB, int CFGT = 1>\n__global__ __launch_bounds__(256, 4)", "kernel template")

src = sub1(src, "run_gemv_tile<BITS, CB, 1>(B32, A2, C, kslices_gate, hidden, group, ntiles_gate, warp, lane, sh_red);",
               "run_gemv_tile<BITS, CB, CFGT>(B32, A2, C, kslices_gate, hidden, group, ntiles_gate, warp, lane, sh_red);", "gate call")
src = sub1(src, "run_gemv_tile<BITS, CB, 1>(B32, A2, C, kslices_down, inter, group, ntiles_down, warp, lane, sh_red);",
               "run_gemv_tile<BITS, CB, CFGT>(B32, A2, C, kslices_down, inter, group, ntiles_down, warp, lane, sh_red);", "down call")

# 4. sh_red must fit WNT=8 -> COLS=128. Declare max-size and size by template.
src = sub1(src, "    __shared__ float sh_red[8][1][64];",
               "    __shared__ float sh_red[8][1][128];", "shared decl")
# tile's sh_red param type: [1][64] -> [1][128] (uses COLS dynamic bound)
src = sub1(src, "float (*sh_red)[1][64])", "float (*sh_red)[1][128])", "tile param")

# 5. num_groups must divide by COLS=WNT*16: gate inter/64 -> inter/CFGT-divisor.
src = sub1(src, "    const int num_groups_gate = inter / 64;",
               "    const int num_groups_gate = inter / (CFGT == 2 ? 128 : 64);", "groups gate")
src = sub1(src, "    const int num_groups_down = hidden / 64;",
               "    const int num_groups_down = hidden / (CFGT == 2 ? 128 : 64);", "groups down")

# 6. launcher + host bench.
src = sub1(src, "template <int BITS, int CB>\nstatic void launch_moe_batched(",
               "template <int BITS, int CB, int CFGT = 1>\nstatic void launch_moe_batched(", "launch template")
src = sub1(src, "    void* kernel = (void*) p2b_moe_batched_kernel<BITS, CB>;",
               "    void* kernel = (void*) p2b_moe_batched_kernel<BITS, CB, CFGT>;", "kernel ptr")

BENCH = r'''

// ---------------------------------------------------------------------------
// Bench additions
// ---------------------------------------------------------------------------
#include <pybind11/pybind11.h>
namespace py = pybind11;

static double g_last_ms = 0.0;

template <int CFGT>
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
        launch_moe_batched<bits, 1, CFGT>(x, gt, gu, gv, ut, uu, uv, dt, du, dv,
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

at::Tensor bench4(
    const at::Tensor& x, at::Tensor& out,
    const at::Tensor& gt, const at::Tensor& gu, const at::Tensor& gv,
    const at::Tensor& ut, const at::Tensor& uu, const at::Tensor& uv,
    const at::Tensor& dt, const at::Tensor& du, const at::Tensor& dv,
    const at::Tensor& ids, const at::Tensor& rw,
    int64_t kg, int64_t ku, int64_t kd, bool mcg,
    int64_t intermediate_size, double swiglu_limit,
    int64_t vcfg, int64_t iters, int64_t warmup)
{
    TORCH_CHECK(kg == 2 && ku == 2 && kd == 2 && mcg, "bench4 supports K=2 MCG only");
    TORCH_CHECK(vcfg == 1 || vcfg == 2, "variant must be 1 (stock WNT=4) or 2 (WNT=8)");
    if (vcfg == 1) run_bench_variant<1>(x, out, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, intermediate_size, swiglu_limit, iters, warmup);
    else run_bench_variant<2>(x, out, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, intermediate_size, swiglu_limit, iters, warmup);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("bench4", &bench4, "p2b bench v4");
    m.def("last_ms", []() { return g_last_ms; });
}
'''
src += BENCH
DST.write_text(src)
print(f"wrote {DST} ({len(src.splitlines())} lines)")
