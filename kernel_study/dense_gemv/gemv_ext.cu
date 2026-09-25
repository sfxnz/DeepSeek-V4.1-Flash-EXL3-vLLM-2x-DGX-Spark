// Bench/test extension for the dense-gemv study: the GEMV kernel variants,
// the fused-quant check kernel, and the timing helpers (flush, spin, read).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "bench_common.cuh"
#include "dense_gemv.cuh"

namespace {

cudaStream_t cur() { return at::cuda::getCurrentCUDAStream(); }

// Quantize [M, K] bf16 with the in-kernel rule into plain layouts (test only).
__global__ void quant_check_kernel(const __nv_bfloat16* x, int M, int K, int ldx, uint8_t* xq, uint8_t* xs) {
  const int KB = K >> 5;
  const int total = M * KB * 4;
  for (int base = blockIdx.x * blockDim.x; base < total; base += gridDim.x * blockDim.x) {
    const int s = base + threadIdx.x;
    const bool valid = s < total;
    const int m = valid ? s / (KB * 4) : 0;
    const int rem = valid ? s - m * (KB * 4) : 0;
    const int kb = rem >> 2, t = rem & 3;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (valid) v = *reinterpret_cast<const uint4*>(x + (size_t)m * ldx + kb * 32 + t * 8);
    float a = fmaxf(fmaxf(dgemv::bf16x2_absmax(v.x), dgemv::bf16x2_absmax(v.y)),
                    fmaxf(dgemv::bf16x2_absmax(v.z), dgemv::bf16x2_absmax(v.w)));
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 1));
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 2));
    if (valid) {
      const float kInv448 = (float)(1.0 / 448.0);
      const uint32_t ue = dgemv::float_to_ue8m0(__fmul_rn(a, kInv448));
      const float inv = dgemv::ue8m0_to_inv_scale(ue);
      uint2 o;
      o.x = dgemv::bf16x2_to_e4m3x2(v.x, inv) | (dgemv::bf16x2_to_e4m3x2(v.y, inv) << 16);
      o.y = dgemv::bf16x2_to_e4m3x2(v.z, inv) | (dgemv::bf16x2_to_e4m3x2(v.w, inv) << 16);
      *reinterpret_cast<uint2*>(xq + (size_t)m * K + kb * 32 + t * 8) = o;
      if (t == 0) xs[m * KB + kb] = (uint8_t)ue;
    }
  }
}

template <int W, int STAGES, int KSPAN, int SM, int IM, bool PDL, int DIAG = 0>
void launch_one(const dgemv::Params& p, int grid) {
  auto kern = dgemv::gemv_mxfp8_kernel<W, STAGES, KSPAN, SM, IM, PDL, DIAG>;
  const int smem = dgemv::smem_bytes_for<W, STAGES, KSPAN, SM>(p.K);
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
  cfg.stream = cur();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = PDL ? 1 : 0;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, kern, p) == cudaSuccess, "launch failed");
}

using LaunchFn = void (*)(const dgemv::Params&, int);

template <int W, int STAGES, int KSPAN>
LaunchFn pick_mode(int sm, int im, bool pdl) {
  if (sm == 0 && im == 0) return pdl ? launch_one<W, STAGES, KSPAN, 0, 0, true> : launch_one<W, STAGES, KSPAN, 0, 0, false>;
  if (sm == 0 && im == 1) return pdl ? launch_one<W, STAGES, KSPAN, 0, 1, true> : launch_one<W, STAGES, KSPAN, 0, 1, false>;
  if (sm == 1 && im == 0) return pdl ? launch_one<W, STAGES, KSPAN, 1, 0, true> : launch_one<W, STAGES, KSPAN, 1, 0, false>;
  return pdl ? launch_one<W, STAGES, KSPAN, 1, 1, true> : launch_one<W, STAGES, KSPAN, 1, 1, false>;
}

LaunchFn pick(int W, int stages, int kspan, int sm, int im, bool pdl) {
  if (sm >= 10) {  // study diagnostics: compact scales, bf16 in, W=4 STAGES=6 KSPAN=128, DIAG = sm - 10
    if (sm == 11) return launch_one<4, 6, 128, 0, 0, false, 1>;
    if (sm == 12) return launch_one<4, 6, 128, 0, 0, false, 2>;
  }
#define CASE(w_, s_, k_) \
  if (W == w_ && stages == s_ && kspan == k_) return pick_mode<w_, s_, k_>(sm, im, pdl);
  CASE(4, 4, 128) CASE(4, 6, 128) CASE(8, 3, 128) CASE(8, 4, 128) CASE(4, 3, 256) CASE(2, 4, 128) CASE(2, 6, 128)
  CASE(2, 8, 128) CASE(2, 12, 128) CASE(4, 8, 128) CASE(2, 4, 256) CASE(2, 6, 256) CASE(1, 8, 128) CASE(1, 16, 128)
#undef CASE
  TORCH_CHECK(false, "unsupported config W=", W, " stages=", stages, " kspan=", kspan);
  return nullptr;
}

}  // namespace

// x: bf16 [M, K] (IN_BF16) or None; xq/xs: e4m3 [M, K] + swizzled scales (IN_QUANT) or None.
void gemv(c10::optional<torch::Tensor> x, c10::optional<torch::Tensor> xq, c10::optional<torch::Tensor> xs,
          torch::Tensor w, torch::Tensor wscale, int64_t scale_mode, torch::Tensor y, int64_t W, int64_t stages,
          int64_t kspan, int64_t grid, bool pdl) {
  dgemv::Params p = {};
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
  }
  TORCH_CHECK(p.M >= 1 && p.M <= dgemv::MPAD, "M must be 1..8");
  TORCH_CHECK(p.K % kspan == 0 && p.N % 16 == 0, "shape");
  pick(W, stages, kspan, scale_mode, im, pdl)(p, grid);
}

void quant_check(torch::Tensor x, torch::Tensor xq, torch::Tensor xs) {
  const int M = x.size(0), K = x.size(1);
  quant_check_kernel<<<64, 256, 0, cur()>>>((const __nv_bfloat16*)x.data_ptr(), M, K, x.stride(0),
                                            (uint8_t*)xq.data_ptr(), (uint8_t*)xs.data_ptr());
}

void flush_perm(torch::Tensor buf, torch::Tensor out) {
  size_t nchunk = buf.numel() / 4096;
  TORCH_CHECK((nchunk & (nchunk - 1)) == 0, "power-of-two 4 KiB chunks");
  dgemv_bench::flush_perm_kernel<<<48 * 8, 256, 0, cur()>>>((const uint4*)buf.data_ptr(), nchunk,
                                                             (unsigned*)out.data_ptr());
}

void spin(int64_t cycles) { dgemv_bench::spin_kernel<<<1, 32, 0, cur()>>>(cycles); }

void read_flat(torch::Tensor w, torch::Tensor out, int64_t grid, int64_t u) {
  size_t n16 = w.numel() * w.element_size() / 16;
  if (u == 2) dgemv_bench::read_flat<2><<<grid, 256, 0, cur()>>>((const uint4*)w.data_ptr(), n16, (unsigned*)out.data_ptr());
  else if (u == 4) dgemv_bench::read_flat<4><<<grid, 256, 0, cur()>>>((const uint4*)w.data_ptr(), n16, (unsigned*)out.data_ptr());
  else dgemv_bench::read_flat<8><<<grid, 256, 0, cur()>>>((const uint4*)w.data_ptr(), n16, (unsigned*)out.data_ptr());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemv", &gemv);
  m.def("quant_check", &quant_check);
  m.def("flush_perm", &flush_perm);
  m.def("spin", &spin);
  m.def("read_flat", &read_flat);
}
