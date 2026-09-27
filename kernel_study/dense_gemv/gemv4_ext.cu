// Study extension: v4 GEMV (long per-row bursts), see dense_gemv_v4.cuh.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "dense_gemv_v4.cuh"

namespace {

template <int W, int S, int KC, int SM, int IM, int MR>
void launch4(const dgemv::Params& p, int grid, bool pdl) {
  auto kern = dgemv::gemv4_kernel<W, S, KC, SM, IM, MR>;
  const int smem = dgemv::smem4_bytes<W, S, KC, SM, MR>(p.K);
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

using Fn4 = void (*)(const dgemv::Params&, int, bool);

template <int W, int S, int KC, int MR>
Fn4 pick4m(int sm, int im) {
  if (sm == 0) return im == 0 ? launch4<W, S, KC, 0, 0, MR> : launch4<W, S, KC, 0, 1, MR>;
  return im == 0 ? launch4<W, S, KC, 1, 0, MR> : launch4<W, S, KC, 1, 1, MR>;
}

Fn4 pick4(int W, int S, int KC, int MR, int sm, int im) {
#define CASE(w_, s_, k_, mr_) \
  if (W == w_ && S == s_ && KC == k_ && MR == mr_) return pick4m<w_, s_, k_, mr_>(sm, im);
  CASE(4, 2, 512, 4) CASE(2, 2, 512, 8) CASE(3, 2, 512, 8) CASE(2, 3, 512, 8) CASE(2, 2, 1024, 4)
  CASE(4, 2, 512, 8) CASE(4, 2, 640, 8) CASE(2, 2, 1280, 8) CASE(4, 2, 384, 8) CASE(2, 2, 1152, 8)
  CASE(2, 2, 768, 8)
#undef CASE
  TORCH_CHECK(false, "unsupported v4 config W=", W, " S=", S, " KC=", KC, " MR=", MR);
  return nullptr;
}

int smem_of(int W, int S, int KC, int MR, int sm, int K) {
  const int kbs = KC / 32, scb = (kbs + 15) / 16 * 16;
  return MR * K + ((MR * (K / 32) + 15) & ~15) + W * S * ((16 * KC + (sm == 0 ? scb : 16 * kbs) + 127) & ~127);
}

}  // namespace

int64_t smem4(int64_t W, int64_t S, int64_t KC, int64_t MR, int64_t sm, int64_t K) {
  return smem_of(W, S, KC, MR, sm, K);
}

void gemv4(c10::optional<torch::Tensor> x, c10::optional<torch::Tensor> xq, c10::optional<torch::Tensor> xs,
           torch::Tensor w, torch::Tensor wscale, int64_t smode, torch::Tensor y, int64_t W, int64_t S, int64_t KC,
           int64_t MR, int64_t grid, bool pdl, int64_t diag) {
  dgemv::Params p = {};
  p.diag = (int)diag;
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
    TORCH_CHECK((p.ldxq % 8) == 0 && ((uintptr_t)p.xq % 8) == 0, "xq layout");
  }
  TORCH_CHECK(p.M >= 1 && p.M <= MR, "M must be 1..MR");
  TORCH_CHECK(p.K % KC == 0 && KC % 128 == 0 && p.N % 32 == 0, "shape");
  TORCH_CHECK(((uintptr_t)p.w % 16) == 0 && ((uintptr_t)p.wscale % 16) == 0, "alignment");
  TORCH_CHECK(smem_of(W, S, KC, MR, smode, p.K) <= 101376, "smem");
  pick4(W, S, KC, MR, smode, im)(p, grid, pdl);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemv4", &gemv4, py::arg("x"), py::arg("xq"), py::arg("xs"), py::arg("w"), py::arg("wscale"),
        py::arg("smode"), py::arg("y"), py::arg("W"), py::arg("S"), py::arg("KC"), py::arg("MR"), py::arg("grid"),
        py::arg("pdl"), py::arg("diag") = 0);
  m.def("smem4", &smem4);
}
