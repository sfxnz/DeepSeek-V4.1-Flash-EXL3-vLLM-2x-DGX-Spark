#!/usr/bin/env python3
"""Generate the prefill no-regression harness sources (bench_prefill).

Extends the driver5/make_bench5 pattern from the p2b decode GEMV to the
exllamav3 PREFILL GEMM (`exl3_gemm` family, kernel_study/cb2 =
image `exllamav3_ext/quant` sources).

PF-G8 layout: trellis stored [n-group of 8 tiles][k-block][8*16*bits words]
(stock: [k-block][n-tile][16*bits words], i.e. the pack's [KT][NT][16*K]).
For every trellis reader:
  - prefill GEMM k-block rows become 8*16*bits contiguous words (512B at
    bits=2, 512B-aligned),
  - every aligned 4-tile run (the p2b decode warp stream, driver5 DEC5)
    stays fully contiguous, because 8 % 4 == 0.
So ONE permuted pack serves both the decode and prefill readers.

Patch is a pure load-address remap (`load_b_gl` / `gl_b_ptr` / k-stride in
`exl3_gemm_kernel_inner`): shared-memory staging, fragment loads, mma order
and reductions are untouched -> outputs are bit-exact vs stock on permuted
data. Layout is a runtime `__constant__ int pf_g8` so one compiled module
contains both variants.

Generated files land in build_prefill/ and are compiled inside the serve
image WITHOUT GPUs by build_prefill.sh (CPU-only nvcc; never JIT on the
host while the serve is resident).
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
CB2 = HERE.parent / "cb2"
DST = HERE / "build_prefill"

SRC_INNER = CB2 / "exl3_gemm_inner.cuh"
SRC_KERNEL = CB2 / "exl3_gemm_kernel.cuh"


def sub1(s, old, new, label):
    if s.count(old) != 1:
        raise SystemExit(f"[{label}] count={s.count(old)}")
    return s.replace(old, new)


def make_inner() -> str:
    src = SRC_INNER.read_text()

    # Constant-memory layout flag (set from the host between bench phases).
    src = sub1(
        src,
        "#define EXL3_GEMM_BASE_THREADS 256\n",
        "#define EXL3_GEMM_BASE_THREADS 256\n\n"
        "// PF-G8 bench flag: 1 = trellis stored group-major\n"
        "// [n-group of 8 tiles][k-block][8*16*bits words]. Host-settable.\n"
        "__constant__ int pf_g8 = 0;\n",
        "pf_g8 decl",
    )

    # B global pointer + k-tile stride: G8 addresses by (group, k-block).
    # load_b_gl addresses are relative to the WINDOW BASE (gl_b_ptr), not
    # fully-addressing: stock load_b_gl[k] skips blocks_n_full rows because the
    # stock window base only ever moves one k-tile forward inside one n-window,
    # so the G8 window base must likewise advance per k-tile — by
    # TILEBLOCKS_K group k-rows (8*16*bits words each) — and by whole groups
    # when the n-window wraps (pf_b_n_off).
    src = sub1(
        src,
        """    int gl_b_stride_k = blocks_n_full * TILEBLOCKS_K * 256 / 16 * bits;
    const int gl_b_stride_n = TILEBLOCKS_N * 256 / 16 * bits;
    const int sh0_b_stride_k = TILEBLOCKS_K * TILEBLOCKS_N * 256 / 16 * bits;
    const uint16_t* gl_b_ptr = B + slice0_k * gl_b_stride_k + slice0_n * gl_b_stride_n;
""",
        """    // PF-G8: B stored [group of 8 n-tiles][k-block][8*16*bits words].
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
""",
        "b ptr init",
    )

    # Per-thread cp_async source offsets for the B tile.
    src = sub1(
        src,
        """        int n = (i * EXL3_GEMM_BASE_THREADS + t) % (gl_b_stride_n / 8);
        int k = (i * EXL3_GEMM_BASE_THREADS + t) / (gl_b_stride_n / 8);
        load_b_gl[i] = k * (blocks_n_full * 256 / 16 * bits / 8) + n;
""",
        """        int n = (i * EXL3_GEMM_BASE_THREADS + t) % (gl_b_stride_n / 8);
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
""",
        "load_b_gl",
    )

    # k-tile wrap in advance0(): stride is 0 in G8 mode, so the flat pointer
    # already equals the new window base; the stock expression is kept as-is.
    src = sub1(
        src,
        """            slice0_k = 0;
            slice0_n++;
            gl_a_ptr = A + slice_m * gl_a_stride_m + slice0_k * gl_a_stride_k;
            gl_b_ptr = B + slice0_k * gl_b_stride_k + slice0_n * gl_b_stride_n;
""",
        """            slice0_k = 0;
            slice0_n++;
            gl_a_ptr = A + slice_m * gl_a_stride_m + slice0_k * gl_a_stride_k;
            gl_b_ptr = B + slice0_k * gl_b_stride_k + pf_b_n_off(slice0_n);
""",
        "advance0 wrap",
    )
    return src


def make_kernel(inner_src: str) -> str:
    src = SRC_KERNEL.read_text()
    src = sub1(src, '#include "exl3_gemm_inner.cuh"',
               '#include "exl3_gemm_inner_pf.cuh"', "inner include")
    # The bench only needs the plain exl3_gemm_kernel; drop the mgemm kernel
    # (its EXL3_MGEMM_ARGS reference kernel-map internals not included here).
    cut = src.index("#define MAX_INDICES 128")
    src = src[:cut].rstrip() + "\n"
    return src


BENCH_CU = r'''
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
'''


def main():
    DST.mkdir(exist_ok=True)
    inner = make_inner()
    kernel = make_kernel(inner)
    # the patched inner travels next to the patched kernel header
    kernel = kernel.replace('#include "exl3_gemm_inner_pf.cuh"',
                            '#include "exl3_gemm_inner_pf.cuh"')
    (DST / "exl3_gemm_inner_pf.cuh").write_text(inner)
    (DST / "exl3_gemm_kernel_pf.cuh").write_text(kernel)
    (DST / "bench_prefill.cu").write_text(BENCH_CU.lstrip("\n"))
    print(f"wrote {DST}/exl3_gemm_inner_pf.cuh")
    print(f"wrote {DST}/exl3_gemm_kernel_pf.cuh")
    print(f"wrote {DST}/bench_prefill.cu")


if __name__ == "__main__":
    main()
