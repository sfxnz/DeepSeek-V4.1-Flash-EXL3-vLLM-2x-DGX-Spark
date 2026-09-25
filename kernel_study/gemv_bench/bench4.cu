#include <cuda_fp16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cmath>

#include "util.h"
#include "util.cuh"
#include "quant/exl3_gemv_kernel.cuh"

namespace cg = cooperative_groups;

template <int bits, int cb, int CFG>
__device__ __forceinline__ void run_gemv_tile(
    const uint32_t* __restrict__ B32,
    const half2* __restrict__ A2,
    half* __restrict__ C,
    int kslices,
    int size_k,
    int group,
    int ntiles,
    int warp,
    int lane,
    float (*sh_red)[1][128])
{
    constexpr int WK = CFG == 0 ? 16 : 8;
    constexpr int WNT = CFG == 0 ? 2 : (CFG == 2 ? 8 : 4);
    constexpr int PF = CFG == 0 ? 4 : 2;
    constexpr int FOLD = CFG == 0 ? 4 : 2;
    constexpr int THREADS = WK * 32;
    constexpr int COLS = WNT * 16;
    constexpr int TWORDS = 8 * bits;
    constexpr int LOADS = bits == 2 ? WNT / 2 : WNT;
    constexpr int LSTRIDE = bits == 3 ? 24 : 32;

    const int chunk = CEIL_DIVIDE(kslices, WK);
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));
    const size_t slice_stride = (size_t) ntiles * TWORDS;

    const size_t a_row0 = 0;
    const bool r0_ok = lane < 4;
    const half2 hzero = __half2half2(__ushort_as_half(0));

    int x_src_a = 0, x_src_b = 0, x_s2 = 0;
    if constexpr (bits == 2) {
        int i1 = lane >> 1;
        x_src_b = i1;
        x_src_a = (i1 + 15) & 15;
    } else if constexpr (bits == 3) {
        int t_offset = lane << 3;
        int b1 = (t_offset + 257) * 3;
        int b2 = b1 + 21;
        int i0 = (b1 - 16) / 32;
        int i2 = (b2 - 1) / 32;
        x_s2 = (i2 + 1) * 32 - b2;
        x_src_a = i0 % 24;
        x_src_b = i2 % 24;
    }

    const uint32_t* bp = B32 + (size_t) ks0 * slice_stride + group * WNT * TWORDS + lane;

    auto ld_b = [&] (int i, int l) -> uint32_t {
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

    FragC_h ch[WNT][2] = {};
    float2 acc0[WNT][2] = {};

    for (int ib = 0; ib < myn; ib += PF) {
        #pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int i = ib + d;
            if (i >= myn) break;

            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                bw[l] = pf[d][l];

            if (i + PF < myn) {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(i + PF, l);
            }

            const size_t a_col = (size_t) (ks0 + i) * 8 + (lane & 3);
            FragB a01, a23;
            a01[0] = r0_ok ? A2[a_row0 + a_col] : hzero;
            a23[0] = r0_ok ? A2[a_row0 + a_col + 4] : hzero;
            a01[1] = hzero;
            a23[1] = hzero;

            #pragma unroll
            for (int t = 0; t < WNT; ++t) {
                FragB f0, f1;
                if constexpr (bits == 4) {
                    uint32_t aw = __shfl_sync(0xffffffffu, bw[t], (lane + 31) & 31);
                    exl3_gemv_ns::dq8_regs_4bits<cb>(aw, bw[t], f0, f1);
                } else if constexpr (bits == 2) {
                    const uint32_t w = bw[t >> 1];
                    const int base = (t & 1) << 4;
                    uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                    uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    exl3_gemv_ns::dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);
                } else {
                    uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);
                    uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);
                    exl3_gemv_ns::dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);
                }

                exl3_gemv_ns::mma_ab_h(a01, a23, f0, ch[t][0]);
                exl3_gemv_ns::mma_ab_h(a01, a23, f1, ch[t][1]);
            }

            if ((d + 1) % FOLD == 0 || i + 1 == myn) {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        acc0[t][f].x += __low2float(ch[t][f][0]);
                        acc0[t][f].y += __high2float(ch[t][f][0]);
                        ch[t][f][0] = hzero;
                    }
            }
        }
    }

    // Warp reduction
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
}

template <int BITS, int CB, int CFGT = 1>
__global__ __launch_bounds__(256, 4)
void p2b_moe_batched_kernel(
    const half* __restrict__ x,
    const int64_t* __restrict__ gt_ptrs,
    const int64_t* __restrict__ gu_ptrs,
    const int64_t* __restrict__ gv_ptrs,
    const int64_t* __restrict__ ut_ptrs,
    const int64_t* __restrict__ uu_ptrs,
    const int64_t* __restrict__ uv_ptrs,
    const int64_t* __restrict__ dt_ptrs,
    const int64_t* __restrict__ du_ptrs,
    const int64_t* __restrict__ dv_ptrs,
    const int32_t* __restrict__ ids,
    const half* __restrict__ rw,
    half* __restrict__ gate,
    half* __restrict__ up,
    half* __restrict__ down,
    half* __restrict__ out,
    half* __restrict__ had_gate,
    half* __restrict__ had_up,
    half* __restrict__ had_down,
    float* __restrict__ accum,
    int experts,
    int m,
    int hidden,
    int inter,
    float swiglu_limit)
{
    auto grid = cg::this_grid();
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    const int ntiles_gate = inter / 16;
    const int kslices_gate = hidden / 16;
    const int num_groups_gate = inter / (CFGT == 2 ? 128 : 64);

    const int ntiles_down = hidden / 16;
    const int kslices_down = inter / 16;
    const int num_groups_down = hidden / (CFGT == 2 ? 128 : 64);

    __shared__ float sh_red[8][1][128];

    // Zero accum
    for (int j = tid; j < m * hidden; j += total_threads)
        accum[j] = 0.0f;

    // Phase 1: Input Hadamard for Gate and Up across all active experts
    {
        int warps_per_exp = hidden / 128;
        int total_warps = m * experts * warps_per_exp;
        int this_warp = warp + (blockDim.x / 32) * blockIdx.x;
        int grid_warps = gridDim.x * (blockDim.x / 32);

        for (; this_warp < total_warps; this_warp += grid_warps) {
            int row = this_warp / (experts * warps_per_exp);
            int rem = this_warp % (experts * warps_per_exp);
            int e = rem / warps_per_exp;
            int w = rem % warps_per_exp;
            int src = ids[row * experts + e];
            const half* x_row = x + (size_t) row * hidden;
            const half* gu_e = reinterpret_cast<const half*>(gu_ptrs[src]);
            const half* uu_e = reinterpret_cast<const half*>(uu_ptrs[src]);
            half* hg_e = had_gate + ((size_t) e * m + row) * hidden;
            half* hu_e = had_up + ((size_t) e * m + row) * hidden;

            had_hf_r_128_inner<true, false>(x_row + w * 128, hg_e + w * 128, gu_e + (w * 128) % hidden, 0.088388347648f);
            had_hf_r_128_inner<true, false>(x_row + w * 128, hu_e + w * 128, uu_e + (w * 128) % hidden, 0.088388347648f);
        }
        grid.sync();
    }

    // Phase 2: Batched Gate & Up GEMV across all active experts
    {
        int total_work = 2 * m * experts * num_groups_gate;
        for (int item = blockIdx.x; item < total_work; item += gridDim.x) {
            int is_up = item & 1;
            int rem = item >> 1;
            int row = rem / (experts * num_groups_gate);
            int rem2 = rem % (experts * num_groups_gate);
            int e = rem2 / num_groups_gate;
            int group = rem2 % num_groups_gate;
            int src = ids[row * experts + e];
            size_t em = (size_t) e * m + row;

            const uint32_t* B32 = reinterpret_cast<const uint32_t*>(is_up ? ut_ptrs[src] : gt_ptrs[src]);
            const half2* A2 = reinterpret_cast<const half2*>((is_up ? had_up : had_gate) + em * hidden);
            half* C = (is_up ? up : gate) + em * inter;

            run_gemv_tile<BITS, CB, CFGT>(B32, A2, C, kslices_gate, hidden, group, ntiles_gate, warp, lane, sh_red);
        }
        grid.sync();
    }

    // Epilogue Hadamard on Gate and Up
    {
        int warps_per_exp = inter / 128;
        int total_warps = m * experts * warps_per_exp;
        int this_warp = warp + (blockDim.x / 32) * blockIdx.x;
        int grid_warps = gridDim.x * (blockDim.x / 32);

        for (; this_warp < total_warps; this_warp += grid_warps) {
            int row = this_warp / (experts * warps_per_exp);
            int rem = this_warp % (experts * warps_per_exp);
            int e = rem / warps_per_exp;
            int w = rem % warps_per_exp;
            int src = ids[row * experts + e];
            const half* x_row = x + (size_t) row * hidden;
            const half* gv_e = reinterpret_cast<const half*>(gv_ptrs[src]);
            const half* uv_e = reinterpret_cast<const half*>(uv_ptrs[src]);
            half* gp_e = gate + ((size_t) e * m + row) * inter;
            half* up_e = up + ((size_t) e * m + row) * inter;

            had_hf_r_128_inner<false, true>(gp_e + w * 128, gp_e + w * 128, gv_e + (w * 128) % inter, 0.088388347648f);
            had_hf_r_128_inner<false, true>(up_e + w * 128, up_e + w * 128, uv_e + (w * 128) % inter, 0.088388347648f);
        }
        grid.sync();
    }

    // Phase 3: SwiGLU activation + Down input Hadamard across all active experts
    {
        // Match vLLM's input-clipped SwiGLU. Zero preserves the plain activation.
        int total_elements = m * experts * inter;
        for (int j = tid; j < total_elements; j += total_threads) {
            float g = __half2float(gate[j]);
            float u = __half2float(up[j]);
            if (swiglu_limit > 0.0f) {
                g = fminf(g, swiglu_limit);
                u = fminf(fmaxf(u, -swiglu_limit), swiglu_limit);
            }
            float s = g / (1.0f + expf(-g));
            had_down[j] = __float2half(s * u);
        }
        grid.sync();

        // Down input Hadamard on had_down
        int warps_per_exp = inter / 128;
        int total_warps = m * experts * warps_per_exp;
        int this_warp = warp + (blockDim.x / 32) * blockIdx.x;
        int grid_warps = gridDim.x * (blockDim.x / 32);

        for (; this_warp < total_warps; this_warp += grid_warps) {
            int row = this_warp / (experts * warps_per_exp);
            int rem = this_warp % (experts * warps_per_exp);
            int e = rem / warps_per_exp;
            int w = rem % warps_per_exp;
            int src = ids[row * experts + e];
            const half* x_row = x + (size_t) row * hidden;
            const half* du_e = reinterpret_cast<const half*>(du_ptrs[src]);
            half* hd_e = had_down + ((size_t) e * m + row) * inter;

            had_hf_r_128_inner<true, false>(hd_e + w * 128, hd_e + w * 128, du_e + (w * 128) % inter, 0.088388347648f);
        }
        grid.sync();
    }

    // Phase 4: Batched Down GEMV across all active experts
    {
        int total_work = m * experts * num_groups_down;
        for (int item = blockIdx.x; item < total_work; item += gridDim.x) {
            int row = item / (experts * num_groups_down);
            int rest = item % (experts * num_groups_down);
            int e = rest / num_groups_down;
            int group = rest % num_groups_down;
            int src = ids[row * experts + e];
            size_t em = (size_t) e * m + row;

            const uint32_t* B32 = reinterpret_cast<const uint32_t*>(dt_ptrs[src]);
            const half2* A2 = reinterpret_cast<const half2*>(had_down + em * inter);
            half* C = down + em * hidden;

            run_gemv_tile<BITS, CB, CFGT>(B32, A2, C, kslices_down, inter, group, ntiles_down, warp, lane, sh_red);
        }
        grid.sync();
    }

    // Down output Hadamard and atomic accumulation into accum
    {
        int warps_per_exp = hidden / 128;
        int total_warps = m * experts * warps_per_exp;
        int this_warp = warp + (blockDim.x / 32) * blockIdx.x;
        int grid_warps = gridDim.x * (blockDim.x / 32);

        for (; this_warp < total_warps; this_warp += grid_warps) {
            int row = this_warp / (experts * warps_per_exp);
            int rem = this_warp % (experts * warps_per_exp);
            int e = rem / warps_per_exp;
            int w = rem % warps_per_exp;
            int src = ids[row * experts + e];
            const half* x_row = x + (size_t) row * hidden;
            const half* dv_e = reinterpret_cast<const half*>(dv_ptrs[src]);
            half* dp_e = down + ((size_t) e * m + row) * hidden;

            had_hf_r_128_inner<false, true>(dp_e + w * 128, dp_e + w * 128, dv_e + (w * 128) % hidden, 0.088388347648f);
        }
        grid.sync();

        // Weighted reduction into accum
        int total_elements = m * experts * hidden;
        for (int j = tid; j < total_elements; j += total_threads) {
            int e = j / (m * hidden);
            int rem = j % (m * hidden);
            int row = rem / hidden;
            int col = rem % hidden;
            float w = __half2float(rw[row * experts + e]);
            atomicAdd(accum + row * hidden + col, w * __half2float(down[j]));
        }
        grid.sync();
    }

    // Write back to out
    for (int j = tid; j < m * hidden; j += total_threads) {
        out[j] = __float2half(accum[j]);
    }
}

template <int BITS, int CB, int CFGT = 1>
static void launch_moe_batched(
    const at::Tensor& x, const at::Tensor& gt, const at::Tensor& gu,
    const at::Tensor& gv, const at::Tensor& ut, const at::Tensor& uu,
    const at::Tensor& uv, const at::Tensor& dt, const at::Tensor& du,
    const at::Tensor& dv, const at::Tensor& ids, const at::Tensor& rw,
    at::Tensor& out, at::Tensor& gate, at::Tensor& up, at::Tensor& down,
    at::Tensor& had_gate, at::Tensor& had_up, at::Tensor& had_down,
    at::Tensor& accum, int e, int m, int hidden, int inter, float swiglu_limit)
{
    int dev = 0, sms = 0, resident = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    void* kernel = (void*) p2b_moe_batched_kernel<BITS, CB, CFGT>;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, 256, 0);
    const int grid = std::max(1, resident * sms);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const half* xp = reinterpret_cast<const half*>(x.data_ptr<c10::Half>());
    const int64_t* gtp = gt.data_ptr<int64_t>();
    const int64_t* gup = gu.data_ptr<int64_t>();
    const int64_t* gvp = gv.data_ptr<int64_t>();
    const int64_t* utp = ut.data_ptr<int64_t>();
    const int64_t* uup = uu.data_ptr<int64_t>();
    const int64_t* uvp = uv.data_ptr<int64_t>();
    const int64_t* dtp = dt.data_ptr<int64_t>();
    const int64_t* dup = du.data_ptr<int64_t>();
    const int64_t* dvp = dv.data_ptr<int64_t>();
    const int32_t* idp = ids.data_ptr<int32_t>();
    const half* rwp = reinterpret_cast<const half*>(rw.data_ptr<c10::Half>());

    half* gp = reinterpret_cast<half*>(gate.data_ptr<c10::Half>());
    half* up_p = reinterpret_cast<half*>(up.data_ptr<c10::Half>());
    half* dp = reinterpret_cast<half*>(down.data_ptr<c10::Half>());
    half* op = reinterpret_cast<half*>(out.data_ptr<c10::Half>());
    half* hg_p = reinterpret_cast<half*>(had_gate.data_ptr<c10::Half>());
    half* hu_p = reinterpret_cast<half*>(had_up.data_ptr<c10::Half>());
    half* hd_p = reinterpret_cast<half*>(had_down.data_ptr<c10::Half>());
    float* accp = accum.data_ptr<float>();

    void* args[] = {
        (void*)&xp, (void*)&gtp, (void*)&gup, (void*)&gvp,
        (void*)&utp, (void*)&uup, (void*)&uvp,
        (void*)&dtp, (void*)&dup, (void*)&dvp,
        (void*)&idp, (void*)&rwp,
        (void*)&gp, (void*)&up_p, (void*)&dp, (void*)&op,
        (void*)&hg_p, (void*)&hu_p, (void*)&hd_p, (void*)&accp,
        (void*)&e, (void*)&m, (void*)&hidden, (void*)&inter, (void*)&swiglu_limit
    };

    cuda_check(cudaLaunchCooperativeKernel(kernel, dim3(grid), dim3(256), args, 0, stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor p2b_fused_moe_cuda(const at::Tensor& x, at::Tensor& out,
    const at::Tensor& gt, const at::Tensor& gu, const at::Tensor& gv,
    const at::Tensor& ut, const at::Tensor& uu, const at::Tensor& uv,
    const at::Tensor& dt, const at::Tensor& du, const at::Tensor& dv,
    const at::Tensor& ids, const at::Tensor& rw, int64_t kg, int64_t ku,
    int64_t kd, bool mcg, int64_t intermediate_size, float swiglu_limit) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "fused MoE requires CUDA fp16 input");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf, "fused MoE output must be CUDA fp16");
    TORCH_CHECK(x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= 8 && x.size(1) > 0 && x.size(1) % 128 == 0,
                "fused MoE requires 1-8 input rows whose hidden width is a positive multiple of 128");
    TORCH_CHECK(out.sizes() == x.sizes(), "fused MoE output shape must match input");
    TORCH_CHECK(intermediate_size > 0 && intermediate_size % 128 == 0,
                "fused MoE local intermediate width must be a positive multiple of 128");
    TORCH_CHECK(std::isfinite(swiglu_limit) && swiglu_limit >= 0.0f,
                "fused MoE SwiGLU limit must be finite and nonnegative (0 disables clipping)");
    TORCH_CHECK(kg == ku && ku == kd && (kg == 2 || kg == 3 || kg == 4), "unsupported fused MoE K");
    TORCH_CHECK(ids.scalar_type() == at::kInt && ids.numel() > 0,
                "fused MoE expert indices must be a nonempty int32 routing tensor");
    TORCH_CHECK((ids.dim() == 1 && x.size(0) == 1)
                    || (ids.dim() == 2 && ids.size(0) == x.size(0) && ids.size(1) > 0),
                "fused MoE expert indices must be [K] for one row or [m, K] matching the input rows");
    TORCH_CHECK(rw.scalar_type() == at::kHalf && rw.numel() == ids.numel(),
                "fused MoE requires one fp16 routing weight per expert index");
    // Pointer tables describe already-loaded tensors. Their pointee shapes and
    // expert IDs are validated/prepared by the Python caller, without a host sync.
    const at::Tensor* tensors[] = {&x, &out, &ids, &rw, &gt, &gu, &gv, &ut, &uu, &uv, &dt, &du, &dv};
    for (const auto* tensor : tensors) {
        TORCH_CHECK(tensor->device() == x.device() && tensor->is_contiguous(),
                    "fused MoE tensors must be contiguous and on the input CUDA device");
    }
    for (const auto* ptrs : {&gt, &gu, &gv, &ut, &uu, &uv, &dt, &du, &dv}) {
        TORCH_CHECK(ptrs->dim() == 1 && ptrs->scalar_type() == at::kLong &&
                    ptrs->numel() == gt.numel() && ptrs->numel() > 0,
                    "fused MoE pointer tables must be equally sized nonempty int64 vectors");
    }
    const c10::cuda::CUDAGuard device_guard(x.device());
    const int e = static_cast<int>(ids.dim() == 1 ? ids.size(0) : ids.size(1));
    const int m = static_cast<int>(x.size(0));
    const int hidden = static_cast<int>(x.size(1));
    const int inter = static_cast<int>(intermediate_size);

    auto gate = at::empty({e, m, inter}, x.options());
    auto up = at::empty({e, m, inter}, x.options());
    auto down = at::empty({e, m, hidden}, x.options());
    auto had_gate = at::empty({e, m, hidden}, x.options());
    auto had_up = at::empty({e, m, hidden}, x.options());
    auto had_down = at::empty({e, m, inter}, x.options());
    auto accum = at::zeros({m, hidden}, x.options().dtype(at::kFloat));

    if (mcg && kg == 2) launch_moe_batched<2, 1>(x, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, out, gate, up, down, had_gate, had_up, had_down, accum, e, m, hidden, inter, swiglu_limit);
    else if (mcg && kg == 3) launch_moe_batched<3, 1>(x, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, out, gate, up, down, had_gate, had_up, had_down, accum, e, m, hidden, inter, swiglu_limit);
    else if (mcg && kg == 4) launch_moe_batched<4, 1>(x, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, out, gate, up, down, had_gate, had_up, had_down, accum, e, m, hidden, inter, swiglu_limit);
    else if (!mcg && kg == 2) launch_moe_batched<2, 2>(x, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, out, gate, up, down, had_gate, had_up, had_down, accum, e, m, hidden, inter, swiglu_limit);
    else if (!mcg && kg == 3) launch_moe_batched<3, 2>(x, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, out, gate, up, down, had_gate, had_up, had_down, accum, e, m, hidden, inter, swiglu_limit);
    else if (!mcg && kg == 4) launch_moe_batched<4, 2>(x, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, out, gate, up, down, had_gate, had_up, had_down, accum, e, m, hidden, inter, swiglu_limit);

    return out;
}


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
