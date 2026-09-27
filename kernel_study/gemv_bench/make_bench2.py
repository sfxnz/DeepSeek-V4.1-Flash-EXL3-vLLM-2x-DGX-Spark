#!/usr/bin/env python3
"""Generate bench2.cu: deployed kernel + parameterized decode variant knobs.

Variants (all must stay bit-exact vs stock on K=2 MCG):
  DEC 0: stock exl3_gemv_ns::dq8_regs_2bits
  DEC 1: funnelshift extraction (__funnelshift_r instead of 64-bit merge)
  DEC 2: warp-smem staging (2 LDS replace the 2 SHFL gathers)
  CFG 0/1 tile shape (stock serving uses CFG=1; CFG=0 = 512-thread narrow)
"""
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "vllm-exl3-patched" / "csrc" / "p2b_moe.cu"
DST = Path(__file__).resolve().parent / "bench2.cu"

src = SRC.read_text()

def sub1(s, old, new, label):
    if s.count(old) != 1:
        raise SystemExit(f"[{label}] expected one occurrence, got {s.count(old)}")
    return s.replace(old, new)

# 1. Vendored decode variants, inserted before run_gemv_tile.
VENDORED = r'''
// ---------------------------------------------------------------------------
// Vendored decode variants (bench-only). All produce bit-identical FragB to
// exl3_gemv_ns::dq8_regs_2bits<1> on the MCG (cb=1) codebook.
// ---------------------------------------------------------------------------
namespace bench_ns {

__device__ __forceinline__ uint32_t fshift_f(const uint32_t b, const uint32_t a, int shift)
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

// DEC 1: funnelshift extraction
template <int cb>
__device__ __forceinline__ void dq8_regs_2bits_fs(uint32_t a, uint32_t b, int t_offset, FragB& f0, FragB& f1)
{
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    b = fshift_f(b, a, ((~t_offset) & 8) << 1);
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

}  // namespace bench_ns
'''

src = sub1(src, "namespace cg = cooperative_groups;",
               "namespace cg = cooperative_groups;\n" + VENDORED, "vendored insert")

# 2. run_gemv_tile: add DEC knob.
src = sub1(src, "template <int bits, int cb, int CFG>\n__device__ __forceinline__ void run_gemv_tile(",
               "template <int bits, int cb, int CFG, int DEC = 0>\n__device__ __forceinline__ void run_gemv_tile(", "tile template")

# 3. bits==2 decode call switches on DEC.
OLD_DECODE = """                } else if constexpr (bits == 2) {
                    const uint32_t w = bw[t >> 1];
                    const int base = (t & 1) << 4;
                    uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                    uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    exl3_gemv_ns::dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);
                } else {"""
NEW_DECODE = """                } else if constexpr (bits == 2) {
                    const uint32_t w = bw[t >> 1];
                    const int base = (t & 1) << 4;
                    uint32_t bwv, awv;
                    if constexpr (DEC == 2) {
                        __syncwarp();
                        sh_words[warp][0][lane] = w;
                        __syncwarp();
                        const uint32_t* tp = sh_words[warp][0];
                        bwv = tp[base + x_src_b];
                        awv = tp[base + x_src_a];
                    } else {
                        bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                        awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    }
                    if constexpr (DEC == 1)
                        bench_ns::dq8_regs_2bits_fs<cb>(awv, bwv, lane << 3, f0, f1);
                    else
                        exl3_gemv_ns::dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);
                } else {"""
src = sub1(src, OLD_DECODE, NEW_DECODE, "decode switch")

# 4. smem for DEC=2 staging.
src = sub1(src, "float (*sh_red)[1][64])",
               "float (*sh_red)[1][64],\n    uint32_t (*sh_words)[1][32] = nullptr)", "tile args")

# 5. kernel: template DEC + CFG knob + shared words.
src = sub1(src, "template <int BITS, int CB>\n__global__ __launch_bounds__(256, 4)",
               "template <int BITS, int CB, int DEC = 0, int CFGT = 1>\n__global__ __launch_bounds__(256, 4)", "kernel template")
src = sub1(src, "    __shared__ float sh_red[8][1][64];",
               "    __shared__ float sh_red[8][1][64];\n    __shared__ uint32_t sh_words[8][1][32];", "shared decl")

OLD_CALL_G = "            run_gemv_tile<BITS, CB, 1>(B32, A2, C, kslices_gate, hidden, group, ntiles_gate, warp, lane, sh_red);"
NEW_CALL_G = "            run_gemv_tile<BITS, CB, CFGT, DEC>(B32, A2, C, kslices_gate, hidden, group, ntiles_gate, warp, lane, sh_red, sh_words);"
src = sub1(src, OLD_CALL_G, NEW_CALL_G, "gate call")
OLD_CALL_D = "            run_gemv_tile<BITS, CB, 1>(B32, A2, C, kslices_down, inter, group, ntiles_down, warp, lane, sh_red);"
NEW_CALL_D = "            run_gemv_tile<BITS, CB, CFGT, DEC>(B32, A2, C, kslices_down, inter, group, ntiles_down, warp, lane, sh_red, sh_words);"
src = sub1(src, OLD_CALL_D, NEW_CALL_D, "down call")

# 6. launcher + host entry with variant dispatch.
src = sub1(src, "template <int BITS, int CB>\nstatic void launch_moe_batched(",
               "template <int BITS, int CB, int DEC = 0, int CFGT = 1>\nstatic void launch_moe_batched(", "launch template")
src = sub1(src, "    void* kernel = (void*) p2b_moe_batched_kernel<BITS, CB>;",
               "    void* kernel = (void*) p2b_moe_batched_kernel<BITS, CB, DEC, CFGT>;", "kernel ptr")

src = sub1(src, "    if (mcg && kg == 2) launch_moe_batched<2, 1>(",
               "    if (mcg && kg == 2 && vdec == 0 && vcfg == 1) launch_moe_batched<2, 1, 0, 1>(", "dispatch guard")
src = src.replace("launch_moe_batched<2, 1>(", "launch_moe_batched<2, 1, 0, 1>(")
src = src.replace("launch_moe_batched<3, 1>(", "launch_moe_batched<3, 1, 0, 1>(")
src = src.replace("launch_moe_batched<4, 1>(", "launch_moe_batched<4, 1, 0, 1>(")
src = src.replace("launch_moe_batched<2, 2>(", "launch_moe_batched<2, 2, 0, 1>(")
src = src.replace("launch_moe_batched<3, 2>(", "launch_moe_batched<3, 2, 0, 1>(")
src = src.replace("launch_moe_batched<4, 2>(", "launch_moe_batched<4, 2, 0, 1>(")

# host signature gains vdec/vcfg
src = sub1(src, "int64_t kd, bool mcg, int64_t intermediate_size, float swiglu_limit) {",
               "int64_t kd, bool mcg, int64_t intermediate_size, float swiglu_limit, int64_t vdec = 0, int64_t vcfg = 1) {", "host sig")

BENCH = r'''

// ---------------------------------------------------------------------------
// Bench additions
// ---------------------------------------------------------------------------
#include <pybind11/pybind11.h>
namespace py = pybind11;

static double g_last_ms = 0.0;

template <int DEC, int CFGT>
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
        launch_moe_batched<bits, 1, DEC, CFGT>(x, gt, gu, gv, ut, uu, uv, dt, du, dv,
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

at::Tensor bench2(
    const at::Tensor& x, at::Tensor& out,
    const at::Tensor& gt, const at::Tensor& gu, const at::Tensor& gv,
    const at::Tensor& ut, const at::Tensor& uu, const at::Tensor& uv,
    const at::Tensor& dt, const at::Tensor& du, const at::Tensor& dv,
    const at::Tensor& ids, const at::Tensor& rw,
    int64_t kg, int64_t ku, int64_t kd, bool mcg,
    int64_t intermediate_size, double swiglu_limit,
    int64_t vdec, int64_t vcfg, int64_t iters, int64_t warmup)
{
    TORCH_CHECK(kg == 2 && ku == 2 && kd == 2 && mcg, "bench2 supports K=2 MCG only");
    TORCH_CHECK(vdec >= 0 && vdec <= 2 && vcfg == 1, "bad variant");
    if (vdec == 0) run_bench_variant<0, 1>(x, out, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, intermediate_size, swiglu_limit, iters, warmup);
    else if (vdec == 1) run_bench_variant<1, 1>(x, out, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, intermediate_size, swiglu_limit, iters, warmup);
    else if (vdec == 2) run_bench_variant<2, 1>(x, out, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, intermediate_size, swiglu_limit, iters, warmup);
    else TORCH_CHECK(false, "variant not built");
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("bench2", &bench2, "p2b bench v2");
    m.def("last_ms", []() { return g_last_ms; });
}
'''
src += BENCH
DST.write_text(src)
print(f"wrote {DST} ({len(src.splitlines())} lines)")
