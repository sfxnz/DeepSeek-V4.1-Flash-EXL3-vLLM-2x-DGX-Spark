// Study extension: v3 GEMV (TMA-engine weight stream), see dense_gemv_v3.cuh.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "dense_gemv_v3.cuh"

namespace {

template <int W, int STAGES, int KSPAN, int SM, int IM, int MR>
void launch3(const dgemv::Params3& p, int grid, bool pdl) {
  auto kern = dgemv::gemv3_kernel<W, STAGES, KSPAN, SM, IM, true, MR>;
  const int smem = dgemv::smem3_bytes<W, STAGES, KSPAN, MR>(p.K);
  static int configured = 0;
  if (configured < smem) {
    TORCH_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess,
                "smem attr ", smem);
    configured = smem;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(W * 32);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, kern, p) == cudaSuccess, "launch failed");
}

using Fn3 = void (*)(const dgemv::Params3&, int, bool);

template <int W, int STAGES, int KSPAN>
Fn3 pick3m(int sm, int im) {
  if (sm == 0) return im == 0 ? launch3<W, STAGES, KSPAN, 0, 0, 8> : launch3<W, STAGES, KSPAN, 0, 1, 8>;
  return im == 0 ? launch3<W, STAGES, KSPAN, 1, 0, 8> : launch3<W, STAGES, KSPAN, 1, 1, 8>;
}

Fn3 pick3(int W, int stages, int kspan, int sm, int im) {
#define CASE(w_, s_, k_) \
  if (W == w_ && stages == s_ && kspan == k_) return pick3m<w_, s_, k_>(sm, im);
  CASE(4, 3, 256) CASE(4, 4, 256) CASE(2, 6, 256) CASE(4, 6, 128) CASE(8, 3, 128) CASE(8, 4, 128) CASE(4, 8, 128)
  CASE(8, 6, 128) CASE(16, 3, 128) CASE(16, 2, 256)
#undef CASE
  TORCH_CHECK(false, "unsupported v3 config W=", W, " stages=", stages, " kspan=", kspan);
  return nullptr;
}

}  // namespace

int64_t smem3(int64_t W, int64_t stages, int64_t kspan, int64_t K) {
  const int mr = 8;
  return 16 * W * stages + 16 + mr * (K + 16) + mr * (K / 32) + 80 +
         W * stages * (16 * (kspan + 16) + 16 * (kspan / 32));
}

void gemv3(c10::optional<torch::Tensor> x, c10::optional<torch::Tensor> xq, c10::optional<torch::Tensor> xs,
           torch::Tensor w, torch::Tensor wscale, int64_t smode, torch::Tensor y, int64_t W, int64_t stages,
           int64_t kspan, int64_t grid, bool pdl) {
  dgemv::Params3 p = {};
  p.w = (const uint8_t*)w.data_ptr();
  p.wscale = (const uint8_t*)wscale.data_ptr();
  p.N = w.size(0);
  p.K = w.size(1);
  p.y = (__nv_bfloat16*)y.data_ptr();
  p.ldy = y.stride(0);
  int im;
  if (x.has_value()) {
    im = 0;
    p.x = (const __nv_bfloat16*)x->data_ptr();
    p.M = x->size(0);
    p.ldx = x->stride(0);
    TORCH_CHECK(x->stride(1) == 1 && (p.ldx % 8) == 0 && ((uintptr_t)p.x % 16) == 0, "x layout");
  } else {
    im = 1;
    p.xq = (const uint8_t*)xq->data_ptr();
    p.xs = (const uint8_t*)xs->data_ptr();
    p.M = xq->size(0);
    p.ldxq = xq->stride(0);
    TORCH_CHECK((p.ldxq % 16) == 0 && ((uintptr_t)p.xq % 16) == 0, "xq layout");
  }
  TORCH_CHECK(p.M >= 1 && p.M <= 8, "M must be 1..8");
  TORCH_CHECK(p.K % kspan == 0 && p.N % 16 == 0 && p.K % 128 == 0, "shape");
  TORCH_CHECK(((uintptr_t)p.w % 16) == 0 && ((uintptr_t)p.wscale % 16) == 0, "alignment");
  pick3(W, stages, kspan, smode, im)(p, grid, pdl);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemv3", &gemv3);
  m.def("smem3", &smem3);
}
