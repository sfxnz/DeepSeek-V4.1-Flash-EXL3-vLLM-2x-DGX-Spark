// ---------------------------------------------------------------------------
// bench_prefill.cu — prefill no-regression harness (generated, do not edit)
//
// exllamav3 exl3_gemm prefill kernels (K=2, cb=1 MCG, fp16 C) with a runtime
// PF-G8 group-major B-layout flag. Same kernels, same math order; only the
// B-tile global->shared load addresses differ -> bit-exact vs stock on
// permuted data.
// ---------------------------------------------------------------------------
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <set>
namespace cg = cooperative_groups;

#include "util.h"
#include "util.cuh"
#include "exl3_gemm_kernel_pf.cuh"

#include <pybind11/pybind11.h>
namespace py = pybind11;

static int* g_locks = nullptr;
static double g_last_ms = 0.0;

static void ensure_locks()
{
    if (!g_locks)
    {
        cuda_check(cudaMalloc(&g_locks, WORKSPACE_SIZE));
        cuda_check(cudaMemset(g_locks, 0, WORKSPACE_SIZE));
    }
}

// shape_idx 2..4 (shape 1 unused), K=2, cb=1 (MCG), C fp16
static fp_exl3_gemm_kernel pick_kernel(int shape_idx)
{
    static fp_exl3_gemm_kernel table[] = {
        nullptr, nullptr,
        exl3_gemm_kernel<2, false, 1, EXL3_GEMM_SHAPE_2>,
        exl3_gemm_kernel<2, false, 1, EXL3_GEMM_SHAPE_3>,
        exl3_gemm_kernel<2, false, 1, EXL3_GEMM_SHAPE_4>,
    };
    TORCH_CHECK(shape_idx >= 2 && shape_idx <= 4, "shape_idx must be 2..4");
    return table[shape_idx];
}

static int blockdim_for(int shape_idx)
{
    static int bd[] = {0, 0, 512, 512, 256};  // EXL3_GEMM_BLOCKDIM = 0,256,512,512,256
    return bd[shape_idx];
}

static void launch_gemm(
    const half* A, const uint16_t* B, void* C,
    const half* suh, half* A_had, const half* svh,
    int size_m, int size_k, int size_n,
    int shape_idx, cudaStream_t stream)
{
    fp_exl3_gemm_kernel kernel = pick_kernel(shape_idx);
    int block_dim = blockdim_for(shape_idx);

    static std::set<void*> attr_set;
    if (attr_set.find((void*) kernel) == attr_set.end())
    {
        cuda_check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX));
        attr_set.insert((void*) kernel);
    }

    int dev = 0, sms = 0, resident = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, (void*) kernel, block_dim, SMEM_MAX);
    int grid = std::max(1, std::min(resident * sms, sms));

    int* locks = g_locks;
    void* args[] = {
        (void*)&A, (void*)&B, (void*)&C,
        (void*)&size_m, (void*)&size_k, (void*)&size_n,
        (void*)&locks, (void*)&suh, (void*)&A_had, (void*)&svh
    };
    cuda_check(cudaLaunchCooperativeKernel((void*) kernel, dim3(grid), dim3(block_dim), args, SMEM_MAX, stream));
}

at::Tensor gemm(
    const at::Tensor& x,        // [m, k] fp16
    const at::Tensor& trellis,  // int16, stock [k/16, n/16, 32] or G8 [n/128, k/16, 256]
    const at::Tensor& suh,      // [k] fp16 (packed scales/flips; ones for the bench)
    const at::Tensor& svh,      // [n] fp16
    int64_t shape_idx, int64_t iters, int64_t warmup)
{
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 2, "x must be [m,k] CUDA fp16");
    TORCH_CHECK(trellis.is_cuda() && trellis.scalar_type() == at::kShort && trellis.is_contiguous(),
                "trellis must be contiguous CUDA int16");
    const int m = x.size(0);
    const int k = x.size(1);
    const int n = (int) (trellis.numel() / (k / 16) / 32) * 16;  // n = (words / (16*bits)) * 16
    const int64_t tw = trellis.numel();
    TORCH_CHECK(tw == (int64_t)(k / 16) * (n / 16) * 32, "trellis size mismatch for K=2");
    TORCH_CHECK(suh.numel() >= k && svh.numel() >= n, "suh/svh too short");

    const c10::cuda::CUDAGuard guard(x.device());
    ensure_locks();

    auto out = at::empty({m, n}, x.options());
    auto a_had = at::empty_like(x);

    const half* A = (const half*) x.data_ptr<c10::Half>();
    const uint16_t* B = (const uint16_t*) trellis.data_ptr();
    void* C = (void*) out.data_ptr<c10::Half>();
    const half* SUH = (const half*) suh.data_ptr<c10::Half>();
    half* AHAD = (half*) a_had.data_ptr<c10::Half>();
    const half* SVH = (const half*) svh.data_ptr<c10::Half>();

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto run1 = [&] { launch_gemm(A, B, C, SUH, AHAD, SVH, m, k, n, (int) shape_idx, stream); };

    for (int64_t i = 0; i < warmup; ++i) run1();
    if (iters > 0)
    {
        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0); cudaEventCreate(&ev1);
        cudaEventRecord(ev0, stream);
        for (int64_t i = 0; i < iters; ++i) run1();
        cudaEventRecord(ev1, stream);
        cudaEventSynchronize(ev1);
        float ms = 0; cudaEventElapsedTime(&ms, ev0, ev1);
        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
        g_last_ms = (double) ms / (double) iters;
    }
    return out;
}

void set_layout(int64_t g8)
{
    int v = (int) g8;
    cuda_check(cudaMemcpyToSymbol(pf_g8, &v, sizeof(int)));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gemm", &gemm, "exl3 prefill gemm bench (stock vs PF-G8 layout)");
    m.def("set_layout", &set_layout, "0 = stock trellis layout, 1 = group-major G8");
    m.def("last_ms", []() { return g_last_ms; });
}
