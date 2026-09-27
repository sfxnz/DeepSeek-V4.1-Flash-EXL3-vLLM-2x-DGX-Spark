// Small-M (M <= 8) MXFP8 decode GEMV for sm_121a (GB10).
//
// y[m, n] = bf16( sum_k  xq[m, k] * 2^(xs[m, k/32] - 127) * w[n, k] * 2^(ws[n, k/32] - 127) )
//
// Numerics are those of the production FlashInfer b12x MXFP8 GEMM, by
// construction:
//  - the activation is MXFP8-quantized with the rule of FlashInfer's CuTe-DSL
//    mxfp8_quantize (amax*fp32(1/448) -> ue8m0 round-up, x*2^(127-ue8m0)
//    clamped to +-448 -> cvt.rn.satfinite.e4m3x2), either fused here from bf16
//    or taken pre-quantized (the serve's fused q/kv-norm quant for wq_b);
//  - each 32-wide k block is one mma.sync m16n8k32 kind::mxf8f6f4
//    block_scale scale_vec::1X e4m3 x e4m3 with ue8m0 scales, chained into one
//    fp32 accumulator in increasing k order from 0.0 (no split-K);
//  - output is the round-to-nearest bf16 of the fp32 accumulator.
// The MMA is issued "swap-AB": weights are the 16-row A operand, the M <= 8
// activation rows are the n=8 B operand, so no MMA row is wasted on padding.
//
// Weights stay in the stock row-major [N, K] layout shared with b12x (no second
// copy). Weight scales: SCALE_COMPACT32 reads [N/32, K/32] (the checkpoint's
// 32x32 blocks; built once at load), SCALE_SWZ reads the stock 128x4-swizzled
// per-row scale (lm_head mxfp8 pack).
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace dgemv {

constexpr int MPAD = 8;
enum ScaleMode : int { SCALE_COMPACT32 = 0, SCALE_SWZ = 1 };
enum InMode : int { IN_BF16 = 0, IN_QUANT = 1 };

struct Params {
  const uint8_t* w;        // e4m3 [N, K] row-major
  const uint8_t* wscale;   // COMPACT32: [N/32, K/32]; SWZ: 128x4-swizzled per-row scales
  const __nv_bfloat16* x;  // IN_BF16: [M, K], row stride ldx elements
  const uint8_t* xq;       // IN_QUANT: e4m3 [M, K], row stride ldxq bytes
  const uint8_t* xs;       // IN_QUANT: 128x4-swizzled scales of [M, K/32]
  __nv_bfloat16* y;        // [M, N], row stride ldy elements
  int M, N, K;
  int ldx, ldxq, ldy;
  int diag;  // study only: bit 0 skip MMA, bit 1 skip activation staging
};

// ---------------------------------------------------------------- PTX helpers

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}

__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(dst), "l"(src));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void mma_mxf8(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1,
                                         uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "h"((uint16_t)0),
        "h"((uint16_t)0), "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }

// ------------------------------------------- MXFP8 activation quant (bit-exact)

// FlashInfer quantization_cute_dsl_utils.float_to_ue8m0_fast, verbatim.
__device__ __forceinline__ uint32_t float_to_ue8m0(float v) {
  uint32_t r;
  asm("{\n"
      ".reg .pred p_zero, p_has_mant, p_exp_zero, p_tiny_sub, p_ovf;\n"
      ".reg .u32 bits, exp_biased, mantissa, bump, result;\n"
      "setp.le.f32 p_zero, %1, 0f00000000;\n"
      "mov.b32 bits, %1;\n"
      "shr.b32 exp_biased, bits, 23;\n"
      "and.b32 exp_biased, exp_biased, 255;\n"
      "and.b32 mantissa, bits, 0x7FFFFF;\n"
      "setp.ne.u32 p_has_mant, mantissa, 0;\n"
      "selp.u32 bump, 1, 0, p_has_mant;\n"
      "setp.eq.u32 p_exp_zero, exp_biased, 0;\n"
      "setp.le.u32 p_tiny_sub, mantissa, 0x400000;\n"
      "and.pred p_tiny_sub, p_exp_zero, p_tiny_sub;\n"
      "@p_tiny_sub mov.u32 bump, 0;\n"
      "add.u32 result, exp_biased, bump;\n"
      "setp.gt.u32 p_ovf, result, 254;\n"
      "selp.u32 result, 254, result, p_ovf;\n"
      "selp.u32 %0, 0, result, p_zero;\n"
      "}\n"
      : "=r"(r)
      : "f"(v));
  return r;
}

// FlashInfer ue8m0_to_inv_scale_fast, verbatim: 2^(127 - ue8m0), 0 for ue8m0 == 0.
__device__ __forceinline__ float ue8m0_to_inv_scale(uint32_t ue) {
  float r;
  asm("{\n"
      ".reg .s32 new_exp;\n"
      ".reg .b32 float_bits;\n"
      ".reg .pred p_zero;\n"
      "setp.eq.u32 p_zero, %1, 0;\n"
      "sub.s32 new_exp, 254, %1;\n"
      "max.s32 new_exp, new_exp, 0;\n"
      "shl.b32 float_bits, new_exp, 23;\n"
      "mov.b32 %0, float_bits;\n"
      "@p_zero mov.b32 %0, 0;\n"
      "}\n"
      : "=f"(r)
      : "r"(ue));
  return r;
}

// FlashInfer bfloat2_to_fp8x2_scaled, verbatim: two bf16 -> two e4m3 (low byte = low bf16).
__device__ __forceinline__ uint32_t bf16x2_to_e4m3x2(uint32_t bf2, float inv) {
  uint32_t r;
  asm("{\n"
      ".reg .b32 lo, hi;\n"
      ".reg .f32 f0, f1;\n"
      ".reg .b16 fp8_pair;\n"
      "and.b32 lo, %1, 0xFFFF;\n"
      "shr.b32 hi, %1, 16;\n"
      "shl.b32 lo, lo, 16;\n"
      "shl.b32 hi, hi, 16;\n"
      "mov.b32 f0, lo;\n"
      "mov.b32 f1, hi;\n"
      "mul.f32 f0, f0, %2;\n"
      "mul.f32 f1, f1, %2;\n"
      "min.f32 f0, f0, 0f43E00000;\n"
      "max.f32 f0, f0, 0fC3E00000;\n"
      "min.f32 f1, f1, 0f43E00000;\n"
      "max.f32 f1, f1, 0fC3E00000;\n"
      "cvt.rn.satfinite.e4m3x2.f32 fp8_pair, f1, f0;\n"
      "cvt.u32.u16 %0, fp8_pair;\n"
      "}\n"
      : "=r"(r)
      : "r"(bf2), "f"(inv));
  return r;
}

__device__ __forceinline__ float bf16x2_absmax(uint32_t v) {
  float lo = __uint_as_float((v & 0x7FFFu) << 16);
  float hi = __uint_as_float(v & 0x7FFF0000u);
  return fmaxf(lo, hi);
}

// 128x4-swizzled scale offset (vLLM swizzle_mxfp8_scale / FlashInfer layout_128x4).
__device__ __forceinline__ int swz_offset(int row, int kb, int nkt) {
  return ((row >> 7) * nkt + (kb >> 2)) * 512 + (row & 31) * 16 + ((row & 127) >> 5) * 4 + (kb & 3);
}

// Stage the M <= 8 activation rows into shared memory in MMA-B-fragment order:
// xf[kb][lane] = {b0, b1} (8 bytes), xs[kb][m] = ue8m0. Only rows < M are
// written; the MMA masks B-fragment lanes of rows >= M to zero (scale 127).
// Loads are batched NB per thread per round so the L2 round trips overlap.
template <int IN_MODE, int NB>
__device__ __forceinline__ void stage_activation(const Params& p, uint8_t* s_xf, uint8_t* s_xs) {
  const int KB = p.K >> 5;
  const int total = p.M * KB * 4;  // 8-element sub-chunks of the real rows
  const int nthr = blockDim.x;
  for (int base = 0; base < total; base += nthr * NB) {
    uint4 v[NB];
    uint32_t ue_in[NB];
#pragma unroll
    for (int i = 0; i < NB; ++i) {
      const int s = base + i * nthr + threadIdx.x;
      v[i] = make_uint4(0, 0, 0, 0);
      ue_in[i] = 127;
      if (s < total) {
        const int m = s / (KB * 4);
        const int rem = s - m * (KB * 4);
        const int kb = rem >> 2, t = rem & 3;
        if constexpr (IN_MODE == IN_BF16) {
          v[i] = *reinterpret_cast<const uint4*>(p.x + (size_t)m * p.ldx + kb * 32 + t * 8);
        } else {
          const uint2 u = *reinterpret_cast<const uint2*>(p.xq + (size_t)m * p.ldxq + kb * 32 + t * 8);
          v[i].x = u.x;
          v[i].y = u.y;
          if (t == 0) ue_in[i] = p.xs[swz_offset(m, kb, (KB + 3) >> 2)];
        }
      }
    }
#pragma unroll
    for (int i = 0; i < NB; ++i) {
      const int s = base + i * nthr + threadIdx.x;
      uint32_t w0 = v[i].x, w1 = v[i].y, ue = ue_in[i];
      if constexpr (IN_MODE == IN_BF16) {
        float a = fmaxf(fmaxf(bf16x2_absmax(v[i].x), bf16x2_absmax(v[i].y)),
                        fmaxf(bf16x2_absmax(v[i].z), bf16x2_absmax(v[i].w)));
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 1));
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 2));
        const float kInv448 = (float)(1.0 / 448.0);
        ue = float_to_ue8m0(__fmul_rn(a, kInv448));
        const float inv = ue8m0_to_inv_scale(ue);
        w0 = bf16x2_to_e4m3x2(v[i].x, inv) | (bf16x2_to_e4m3x2(v[i].y, inv) << 16);
        w1 = bf16x2_to_e4m3x2(v[i].z, inv) | (bf16x2_to_e4m3x2(v[i].w, inv) << 16);
      }
      if (s < total) {
        const int m = s / (KB * 4);
        const int rem = s - m * (KB * 4);
        const int kb = rem >> 2, t = rem & 3;
        // element k = 8t + [0..8) of block kb goes to B-fragment lanes 4m + 2(t&1) + {0,1}, word t>>1
        uint32_t* dst = reinterpret_cast<uint32_t*>(s_xf + kb * 256 + (4 * m + 2 * (t & 1)) * 8 + (t >> 1) * 4);
        dst[0] = w0;
        dst[2] = w1;  // next lane: +8 bytes
        if (t == 0) s_xs[kb * 8 + m] = (uint8_t)ue;
      }
    }
  }
}

// ------------------------------------------------------------------ kernel

// W warps per CTA, each warp streams whole 16-row tiles (full K, exact chain)
// through a private STAGES-deep cp.async ring of 16 x KSPAN-byte stages.
// DIAG (study only): 0 normal; 1 skip ldmatrix+MMA (data movement only);
// 2 skip the activation staging (weights + MMA only, garbage activation).
template <int W, int STAGES, int KSPAN, int SCALE_MODE, int IN_MODE, bool PDL, int DIAG = 0>
__global__ void __launch_bounds__(W * 32) gemv_mxfp8_kernel(const Params p) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int KBS = KSPAN / 32;                                   // k-blocks per stage
  constexpr int A_BYTES = 16 * KSPAN;
  constexpr int SC_BYTES = SCALE_MODE == SCALE_COMPACT32 ? 16 : 16 * KBS;
  constexpr int STAGE_BYTES = A_BYTES + SC_BYTES;
  constexpr int CH = KSPAN / 16;                                    // 16-B chunks per row per stage

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int K = p.K, KB = K >> 5;
  const int nspan = K / KSPAN;
  const int T = p.N >> 4;
  const int gw = blockIdx.x * W + warp, GW = gridDim.x * W;
  const int ntile = gw < T ? (T - 1 - gw) / GW + 1 : 0;
  const int S = ntile * nspan;

  uint8_t* s_xf = smem;
  uint8_t* s_xs = smem + KB * 256;
  uint8_t* ring = smem + KB * 264 + warp * STAGES * STAGE_BYTES;
  const uint32_t ring_u32 = smem_u32(ring);

  auto issue = [&](int s) {
    if (s < S) {
      const int tile = gw + (s / nspan) * GW, span = s % nspan;
      const int n0 = tile * 16, k0 = span * KSPAN;
      const uint32_t st = ring_u32 + (s % STAGES) * STAGE_BYTES;
#pragma unroll
      for (int i = 0; i < (16 * CH) / 32; ++i) {
        const int c = lane + 32 * i, row = c / CH, ch = c % CH;
        cp_async16(st + row * KSPAN + ((ch ^ (row & 7)) << 4), p.w + (size_t)(n0 + row) * K + k0 + ch * 16);
      }
      if constexpr (SCALE_MODE == SCALE_COMPACT32) {
        if (lane < KBS / 4)
          cp_async4(st + A_BYTES + lane * 4, p.wscale + (size_t)(n0 >> 5) * KB + (k0 >> 5) + lane * 4);
      } else {
        const int nkt = (KB + 3) >> 2;
#pragma unroll
        for (int i = 0; i < (16 * KBS / 4 + 31) / 32; ++i) {
          const int c = lane + 32 * i;
          if (c < 16 * KBS / 4) {
            const int row = c % 16, g = c / 16;
            cp_async4(st + A_BYTES + row * KBS + g * 4, p.wscale + swz_offset(n0 + row, (k0 >> 5) + g * 4, nkt));
          }
        }
      }
    }
    cp_async_commit();
  };

  // Weights are static: start streaming before waiting on the producer kernel.
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) issue(s);

  if constexpr (PDL) pdl_wait();
  if constexpr (DIAG != 2) stage_activation<IN_MODE, 8>(p, s_xf, s_xs);
  __syncthreads();

  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  const int r = lane >> 2, q = lane & 3;
  // ldmatrix row address pieces: matrix mi = lane/8 -> rows (mi&1)*8 + lane%8, chunk (mi>>1)
  const int lm_row = ((lane >> 3) & 1) * 8 + (lane & 7);
  const int lm_half = lane >> 4;
  const int sfa_row = r + 8 * (lane & 1);
  const bool col_live = r < p.M;

  for (int s = 0; s < S; ++s) {
    cp_async_wait<STAGES - 2>();
    __syncwarp();
    const int span = s % nspan;
    const uint32_t st = ring_u32 + (s % STAGES) * STAGE_BYTES;
    const uint8_t* stp = ring + (s % STAGES) * STAGE_BYTES;
#pragma unroll
    for (int j = 0; j < KBS && DIAG != 1; ++j) {
      const int kb = span * KBS + j;
      uint32_t a[4];
      const int ch = 2 * j + lm_half;
      ldmatrix_x4(a, st + lm_row * KSPAN + ((ch ^ (lm_row & 7)) << 4));
      uint2 b = *reinterpret_cast<const uint2*>(s_xf + kb * 256 + lane * 8);
      uint32_t sfb = s_xs[kb * 8 + r];
      if (!col_live) {  // activation rows >= M: zero B fragment, finite scale
        b = make_uint2(0, 0);
        sfb = 127;
      }
      uint32_t sfa;
      if constexpr (SCALE_MODE == SCALE_COMPACT32) sfa = stp[A_BYTES + j];
      else sfa = stp[A_BYTES + sfa_row * KBS + j];
      mma_mxf8(acc, a, b.x, b.y, sfa, sfb);
    }
    __syncwarp();
    issue(s + STAGES - 1);
    if (span == nspan - 1) {
      const int n0 = (gw + (s / nspan) * GW) * 16;
      const int m0 = 2 * q, m1 = 2 * q + 1;
      if (m0 < p.M) {
        p.y[(size_t)m0 * p.ldy + n0 + r] = __float2bfloat16_rn(acc[0]);
        p.y[(size_t)m0 * p.ldy + n0 + r + 8] = __float2bfloat16_rn(acc[2]);
      }
      if (m1 < p.M) {
        p.y[(size_t)m1 * p.ldy + n0 + r] = __float2bfloat16_rn(acc[1]);
        p.y[(size_t)m1 * p.ldy + n0 + r + 8] = __float2bfloat16_rn(acc[3]);
      }
      acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
    }
  }
  cp_async_wait<0>();
  if constexpr (PDL) pdl_launch();
}

template <int W, int STAGES, int KSPAN, int SCALE_MODE>
constexpr int smem_bytes_for(int K) {
  return (K >> 5) * 264 + W * STAGES * (16 * KSPAN + (SCALE_MODE == SCALE_COMPACT32 ? 16 : 16 * (KSPAN / 32)));
}

}  // namespace dgemv
