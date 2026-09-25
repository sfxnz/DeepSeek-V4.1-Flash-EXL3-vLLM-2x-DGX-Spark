// dsv41 dense GEMV: small-M (M <= 8) MXFP8 decode GEMM for sm_121a (GB10).
// Loaded by docker/patch/dense_gemv.py (torch cpp_extension JIT, opt-in
// DSV41_DENSE_GEMV=1). Study, iteration log and benchmarks:
// kernel_study/dense_gemv/, results/2026-09-25-kernels/dense-gemv/.
//
// y[m, n] = bf16( sum_k xq[m, k] 2^(xs[m, k/32]-127) * w[n, k] 2^(ws[n, k/32]-127) )
//
// Bit-exact with the production FlashInfer b12x MXFP8 GEMM by construction:
//  - activation MXFP8 quant with FlashInfer's cute-dsl mxfp8_quantize rule
//    (PTX copied verbatim below), fused from bf16 or taken pre-quantized
//    (a QuantizedActivation: e4m3 rows + 128x4-swizzled ue8m0 scales);
//  - one mma.sync m16n8k32 kind::mxf8f6f4 block_scale scale_vec::1X e4m3 x e4m3
//    ue8m0 per 32-wide k block, chained into one fp32 accumulator in increasing
//    k from 0.0 (no split-K), bf16 RN output. Issued swap-AB: weights are the
//    16-row A operand, the M <= 8 activation rows the n8 B operand.
// Weights stay in the stock row-major [N, K] e4m3 tensor shared with b12x.
//
// Stream design (each point measured, see the iteration log):
//  - one warp owns whole 16-row tiles (full K) and streams them through a
//    private S-stage cp.async ring of 16 rows x KC contiguous bytes (KC >= 384:
//    short per-row bursts lose 15-30% to DRAM page misses);
//  - stages are 128-B aligned in shared memory (misaligned cost 20%);
//  - the whole ring is issued before griddepcontrol.wait (weights are static);
//  - two extra staging warps per CTA quantize/stage the activation one KC-span
//    at a time (next span's loads in flight while the current one is
//    quantized) and publish progress through a shared counter that the MMA
//    warps poll only in their first tile, so the weight stream never waits for
//    the whole activation and MMA warps never barrier with each other;
//  - persistent grid: a warp's tiles stream back to back through one ring.
// Weight scales (built once at load by dense_gemv.py):
//  COMPACT32 [N/32][nspan][SCB]  one scale per 32 rows (checkpoint 32x32
//            blocks); bytes 0..KBS-1 of a cell = k blocks span*KBS + j
//  TILE      [N/16][nspan][16][KBS]  per-row scales (lm_head mxfp8 pack)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace dsv41_gemv {

enum ScaleMode : int { COMPACT32 = 0, TILE = 1 };
enum InMode : int { IN_BF16 = 0, IN_QUANT = 1 };

enum XsMode : int { XS_SWZ128X4 = 0, XS_DG_PACKED = 1 };

struct Params {
  const uint8_t* w;        // e4m3 [N, K] row-major (grouped: G x [N/G, K], group-major)
  const uint8_t* wscale;   // COMPACT32 or TILE layout
  const __nv_bfloat16* x;  // IN_BF16: [M, K], row stride ldx elements
  const uint8_t* xq;       // IN_QUANT: e4m3 [M, K] per group, row stride ldxq bytes
  const uint8_t* xs;       // IN_QUANT: scales, see xs_mode
  __nv_bfloat16* y;        // [M, N], row stride ldy elements (grouped: group g at column g*N/G)
  int M, N, K;
  int ldx, ldxq, ldy;
  // Grouped GEMV (wo_a: y[:, g] = x[:, g] @ W[g]^T): CTA b serves group b % G,
  // group g's activation starts at xq + g*xq_gstride. G = 1 is a plain GEMV.
  int G;
  long long xq_gstride;
  // xs_mode XS_SWZ128X4: FlashInfer/vLLM 128x4-swizzled ue8m0 bytes (b12x inputs).
  // xs_mode XS_DG_PACKED: DeepGEMM MN-major INT32-packed ue8m0 (fused_inv_rope_fp8_quant
  // on SM12x): int32 at [g*xs_gstride + m + (kb/4)*xs_tal], byte kb%4.
  int xs_mode, xs_tal;
  long long xs_gstride;
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
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

template <int NBS>
struct ActRegs {
  uint4 v[NBS];
  uint32_t ue[NBS];
};

// Load this thread's 8-element sub-chunks of one KC-span of the M activation rows.
template <int IN_MODE, int NBS, int KBS>
__device__ __forceinline__ void act_load(const Params& p, ActRegs<NBS>& ar, int span, int tid, int nthr, int g) {
  const int subs = p.M * KBS * 4;
  const int KB = p.K >> 5;
#pragma unroll
  for (int j = 0; j < NBS; ++j) {
    const int i = tid + j * nthr;
    ar.v[j] = make_uint4(0, 0, 0, 0);
    ar.ue[j] = 127;
    if (i < subs) {
      const int m = i / (KBS * 4), rem = i - m * (KBS * 4);
      const int kb = span * KBS + (rem >> 2), t = rem & 3;
      if constexpr (IN_MODE == IN_BF16) {
        ar.v[j] = *reinterpret_cast<const uint4*>(p.x + (size_t)m * p.ldx + kb * 32 + t * 8);
      } else {
        const uint2 u = *reinterpret_cast<const uint2*>(p.xq + g * p.xq_gstride + (size_t)m * p.ldxq + kb * 32 +
                                                        t * 8);
        ar.v[j].x = u.x;
        ar.v[j].y = u.y;
        if (t == 0) {
          if (p.xs_mode == XS_SWZ128X4) {
            ar.ue[j] = p.xs[swz_offset(m, kb, (KB + 3) >> 2)];
          } else {
            const uint32_t w = reinterpret_cast<const uint32_t*>(p.xs)[g * p.xs_gstride + m + (kb >> 2) * p.xs_tal];
            ar.ue[j] = (w >> (8 * (kb & 3))) & 0xFFu;
          }
        }
      }
    }
  }
}

// Quantize (bf16 input) and store one span in MMA-B-fragment order:
// xf[kb][lane < 4*MR][8 B] (lane 4m+q holds k 4q..4q+3 and 16+4q..16+4q+3 of
// row m), xs[kb][MR]. Rows >= M are never written (the MMA masks them).
template <int IN_MODE, int NBS, int KBS, int MR>
__device__ __forceinline__ void act_store(const Params& p, const ActRegs<NBS>& ar, int span, uint8_t* s_xf,
                                          uint8_t* s_xs, int tid, int nthr) {
  const int subs = p.M * KBS * 4;
#pragma unroll
  for (int j = 0; j < NBS; ++j) {
    const int i = tid + j * nthr;
    uint32_t w0 = ar.v[j].x, w1 = ar.v[j].y, ue = ar.ue[j];
    if constexpr (IN_MODE == IN_BF16) {
      float a = fmaxf(fmaxf(bf16x2_absmax(ar.v[j].x), bf16x2_absmax(ar.v[j].y)),
                      fmaxf(bf16x2_absmax(ar.v[j].z), bf16x2_absmax(ar.v[j].w)));
      a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 1));
      a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 2));
      const float kInv448 = (float)(1.0 / 448.0);
      ue = float_to_ue8m0(__fmul_rn(a, kInv448));
      const float inv = ue8m0_to_inv_scale(ue);
      w0 = bf16x2_to_e4m3x2(ar.v[j].x, inv) | (bf16x2_to_e4m3x2(ar.v[j].y, inv) << 16);
      w1 = bf16x2_to_e4m3x2(ar.v[j].z, inv) | (bf16x2_to_e4m3x2(ar.v[j].w, inv) << 16);
    }
    if (i < subs) {
      const int m = i / (KBS * 4), rem = i - m * (KBS * 4);
      const int kb = span * KBS + (rem >> 2), t = rem & 3;
      uint32_t* dst = reinterpret_cast<uint32_t*>(s_xf + kb * (MR * 32) + (4 * m + 2 * (t & 1)) * 8 + (t >> 1) * 4);
      dst[0] = w0;
      dst[2] = w1;
      if (t == 0) s_xs[kb * MR + m] = (uint8_t)ue;
    }
  }
}

template <int KC, int SMODE>
struct Geom {
  static constexpr int KBS = KC / 32;
  static constexpr int CPR = KC / 16;
  static constexpr int A_BYTES = 16 * KC;
  static constexpr int SC_BYTES = SMODE == COMPACT32 ? (KBS + 15) / 16 * 16 : 16 * KBS;
  static constexpr int STAGE_BYTES = (A_BYTES + SC_BYTES + 127) & ~127;
};

template <int W, int S, int KC, int SMODE, int MR>
constexpr int smem_bytes(int K) {
  return W * S * Geom<KC, SMODE>::STAGE_BYTES + MR * K + ((MR * (K >> 5) + 15) & ~15);
}

constexpr int SW = 2;  // staging warps per CTA (measured: 1-2 us/call better than a CTA-wide staging barrier)

// W MMA warps (warps 0..W-1) own tiles; warps W, W+1 stage the activation.
template <int W, int S, int KC, int SMODE, int IN_MODE, int MR>
__global__ void __launch_bounds__((W + SW) * 32) gemv_kernel(const Params p) {
  using GM = Geom<KC, SMODE>;
  constexpr int KBS = GM::KBS, CPR = GM::CPR, A_BYTES = GM::A_BYTES, SC_BYTES = GM::SC_BYTES;
  constexpr int STAGE_BYTES = GM::STAGE_BYTES;
  constexpr int PER_LANE = 16 * CPR / 32;
  constexpr int NBS = (MR * KBS * 4 + SW * 32 - 1) / (SW * 32);
  extern __shared__ __align__(128) uint8_t smem[];
  // spans published by each staging warp. One counter per warp: a shared sum
  // lets a fast warp's span s+1 stand in for a slow warp's span s (measured:
  // garbage reads in 14 of 564 fused cases before this was per warp).
  __shared__ int s_ready[SW];

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int grp = blockIdx.x % p.G;  // this CTA's group (0 for a plain GEMV)
  if (threadIdx.x < SW) s_ready[threadIdx.x] = 0;
  __syncthreads();
  if (warp >= W) {  // staging warps: every span in k order, next span's loads in flight
    uint8_t* s_xf = smem + W * S * STAGE_BYTES;
    uint8_t* s_xs = s_xf + MR * p.K;
    const int tid = threadIdx.x - W * 32, nthr = SW * 32, nspan = p.K / KC;
    pdl_wait();
    ActRegs<NBS> cur, nxt;
    act_load<IN_MODE, NBS, KBS>(p, cur, 0, tid, nthr, grp);
    for (int span = 0; span < nspan; ++span) {
      if (span + 1 < nspan) act_load<IN_MODE, NBS, KBS>(p, nxt, span + 1, tid, nthr, grp);
      act_store<IN_MODE, NBS, KBS, MR>(p, cur, span, s_xf, s_xs, tid, nthr);
      __threadfence_block();  // span data before the counter (release)
      __syncwarp();
      if (lane == 0) atomicAdd(&s_ready[warp - W], 1);
      cur = nxt;
    }
    return;
  }
  const int K = p.K, nspan = K / KC;
  // tiles of this CTA's group: global tile = grp*Tg + (gw + ti*GW), Tg = N/G/16
  const int Tg = (p.N / p.G) >> 4;
  const int gw = (blockIdx.x / p.G) * W + warp, GW = (gridDim.x / p.G) * W;
  const int ntile = gw < Tg ? (Tg - 1 - gw) / GW + 1 : 0;
  const int nsteps = ntile * nspan;
  const int tile0 = grp * Tg;

  uint8_t* ring = smem + warp * S * STAGE_BYTES;
  uint8_t* s_xf = smem + W * S * STAGE_BYTES;
  uint8_t* s_xs = s_xf + MR * K;
  const uint32_t ring_u32 = smem_u32(ring);

  auto issue = [&](int s) {
    if (s < nsteps) {
      const int ti = s / nspan, span = s - ti * nspan;
      const int tile = tile0 + gw + ti * GW, n0 = tile * 16, k0 = span * KC;
      const uint32_t st = ring_u32 + (s % S) * STAGE_BYTES;
#pragma unroll
      for (int i = 0; i < PER_LANE; ++i) {
        const int c = lane + 32 * i, row = c / CPR, ch = c % CPR;
        cp_async16(st + row * KC + ((ch ^ (row & 7)) << 4), p.w + (size_t)(n0 + row) * K + k0 + ch * 16);
      }
      if constexpr (SMODE == COMPACT32) {
        if (lane < SC_BYTES / 16)
          cp_async16(st + A_BYTES + lane * 16, p.wscale + ((size_t)(n0 >> 5) * nspan + span) * SC_BYTES + lane * 16);
      } else {
#pragma unroll
        for (int i = 0; i < (SC_BYTES / 16 + 31) / 32; ++i) {
          const int c = lane + 32 * i;
          if (c < SC_BYTES / 16)
            cp_async16(st + A_BYTES + c * 16, p.wscale + ((size_t)tile * nspan + span) * SC_BYTES + c * 16);
        }
      }
    }
    cp_async_commit();
  };

  // Weights are static: fill the ring before waiting on the producer kernel.
#pragma unroll
  for (int s = 0; s < S; ++s) issue(s);

  pdl_wait();  // no-op unless launched as a PDL dependent (y is written after it)

  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  const int r = lane >> 2, q = lane & 3;
  const int lm_row = ((lane >> 3) & 1) * 8 + (lane & 7);
  const int lm_half = lane >> 4;
  const int sfa_row = r + 8 * (lane & 1);
  const bool col_live = r < p.M;
  const uint8_t* xf = s_xf + (col_live ? lane : 0) * 8;
  const uint8_t* xsr = s_xs + (col_live ? r : 0);

  for (int s = 0; s < nsteps; ++s) {
    if (s < nspan) {  // first tile: wait until every staging warp published span s (acquire)
#pragma unroll
      for (int w = 0; w < SW; ++w)
        while (*reinterpret_cast<volatile int*>(&s_ready[w]) < s + 1) __nanosleep(20);
      __threadfence_block();
    }
    cp_async_wait<S - 1>();
    __syncwarp();
    const int ti = s / nspan, span = s - ti * nspan;
    const uint32_t st = ring_u32 + (s % S) * STAGE_BYTES;
    const uint8_t* stp = ring + (s % S) * STAGE_BYTES;
#pragma unroll
    for (int j = 0; j < KBS; ++j) {
      const int kb = span * KBS + j;
      uint32_t a[4];
      const int ch = 2 * j + lm_half;
      ldmatrix_x4(a, st + lm_row * KC + ((ch ^ (lm_row & 7)) << 4));
      uint2 b = *reinterpret_cast<const uint2*>(xf + kb * (MR * 32));
      uint32_t sfb = xsr[kb * MR];
      if (!col_live) {  // activation rows >= M: zero B fragment, finite scale
        b = make_uint2(0, 0);
        sfb = 127;
      }
      uint32_t sfa;
      if constexpr (SMODE == COMPACT32) sfa = stp[A_BYTES + j];
      else sfa = stp[A_BYTES + sfa_row * KBS + j];
      mma_mxf8(acc, a, b.x, b.y, sfa, sfb);
    }
    __syncwarp();
    issue(s + S);
    if (span == nspan - 1) {
      const int n0 = (tile0 + gw + ti * GW) * 16;
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
  pdl_launch();
}

struct Cfg {
  int W, S, KC, MR, smode, in_mode;
};
using LaunchFn = void (*)(const Params&, int grid, bool pdl);
using SmemFn = int (*)(int K);

template <int W, int S, int KC, int SMODE, int IN_MODE, int MR>
void launch(const Params& p, int grid, bool pdl) {
  auto kern = gemv_kernel<W, S, KC, SMODE, IN_MODE, MR>;
  const int smem = smem_bytes<W, S, KC, SMODE, MR>(p.K);
  static int configured = 0;
  if (configured < smem) {
    TORCH_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess,
                "dense_gemv: cannot set ", smem, " B of dynamic smem");
    configured = smem;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3((W + SW) * 32);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, kern, p) == cudaSuccess, "dense_gemv: launch failed");
}

template <int W, int S, int KC, int SMODE, int IN_MODE, int MR>
int occupancy(int K) {
  auto kern = gemv_kernel<W, S, KC, SMODE, IN_MODE, MR>;
  const int smem = smem_bytes<W, S, KC, SMODE, MR>(K);
  if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess) return 0;
  int n = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kern, (W + SW) * 32, smem) != cudaSuccess) return 0;
  return n;
}

struct Entry {
  Cfg c;
  LaunchFn launch;
  int (*occ)(int);
  int (*smem)(int);
};

template <int W, int S, int KC, int SMODE, int IN_MODE, int MR>
constexpr Entry entry() {
  return Entry{{W, S, KC, MR, SMODE, IN_MODE},
               launch<W, S, KC, SMODE, IN_MODE, MR>,
               occupancy<W, S, KC, SMODE, IN_MODE, MR>,
               smem_bytes<W, S, KC, SMODE, MR>};
}

// Every (W, S, KC, MR, scale mode, input mode) the serve's shape table uses
// (dense_gemv.py CONFIGS); anything else is refused, never guessed.
#define DSV41_GEMV_BOTH_INPUTS(W, S, KC, SM, MR) entry<W, S, KC, SM, IN_BF16, MR>(), entry<W, S, KC, SM, IN_QUANT, MR>()
static const Entry kTable[] = {
    DSV41_GEMV_BOTH_INPUTS(4, 2, 512, COMPACT32, 4),   // K 5120 / 4096, M <= 4
    DSV41_GEMV_BOTH_INPUTS(3, 2, 512, COMPACT32, 8),   // K 5120, M <= 8
    DSV41_GEMV_BOTH_INPUTS(4, 2, 512, COMPACT32, 8),   // K 4096, M <= 8
    DSV41_GEMV_BOTH_INPUTS(2, 2, 640, COMPACT32, 8),   // K 1280
    DSV41_GEMV_BOTH_INPUTS(4, 2, 384, COMPACT32, 8),   // K 1152
    DSV41_GEMV_BOTH_INPUTS(2, 2, 512, COMPACT32, 4),   // K 6144 / 15360, M <= 4
    DSV41_GEMV_BOTH_INPUTS(2, 2, 512, COMPACT32, 8),   // K 6144, M <= 8
    DSV41_GEMV_BOTH_INPUTS(3, 2, 512, TILE, 8),        // lm_head (per-row scales)
};
#undef DSV41_GEMV_BOTH_INPUTS

const Entry* find(int W, int S, int KC, int MR, int smode, int in_mode) {
  for (const Entry& e : kTable)
    if (e.c.W == W && e.c.S == S && e.c.KC == KC && e.c.MR == MR && e.c.smode == smode && e.c.in_mode == in_mode)
      return &e;
  return nullptr;
}

}  // namespace dsv41_gemv

namespace {

using dsv41_gemv::Entry;

const Entry& must_find(int64_t W, int64_t S, int64_t KC, int64_t MR, int64_t smode, int64_t in_mode) {
  const Entry* e = dsv41_gemv::find(W, S, KC, MR, smode, in_mode);
  TORCH_CHECK(e != nullptr, "dense_gemv: config W=", W, " S=", S, " KC=", KC, " MR=", MR, " smode=", smode,
              " in=", in_mode, " is not compiled");
  return *e;
}

// Persistent grid: G x min(tiles per group / W, SMs x resident CTAs per SM / G);
// 0 = cannot run. N is the total row count (all groups).
int64_t plan_grid(int64_t W, int64_t S, int64_t KC, int64_t MR, int64_t smode, int64_t in_mode, int64_t N, int64_t K,
                  int64_t G) {
  const Entry& e = must_find(W, S, KC, MR, smode, in_mode);
  int dev = 0, sms = 0, optin = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  if (G < 1 || K % KC || N % (32 * G) || e.smem(K) > optin) return 0;
  const int occ = e.occ(K);
  if (occ < 1 || (int64_t)sms * occ < G) return 0;
  const int64_t tiles_g = N / G / 16;
  return G * std::min<int64_t>((tiles_g + W - 1) / W, (int64_t)sms * occ / G);
}

void gemv(c10::optional<torch::Tensor> x, c10::optional<torch::Tensor> xq, c10::optional<torch::Tensor> xs,
          torch::Tensor w, torch::Tensor wscale, int64_t smode, torch::Tensor y, int64_t W, int64_t S, int64_t KC,
          int64_t MR, int64_t grid, bool pdl) {
  dsv41_gemv::Params p = {};
  p.G = 1;
  p.xs_mode = dsv41_gemv::XS_SWZ128X4;
  TORCH_CHECK(w.dim() == 2 && w.is_contiguous() && w.element_size() == 1, "dense_gemv: w must be [N, K] 1-byte");
  p.w = (const uint8_t*)w.data_ptr();
  p.wscale = (const uint8_t*)wscale.data_ptr();
  p.N = w.size(0);
  p.K = w.size(1);
  TORCH_CHECK(y.scalar_type() == at::kBFloat16 && y.stride(1) == 1 && y.size(1) == p.N, "dense_gemv: y layout");
  p.y = (__nv_bfloat16*)y.data_ptr();
  p.ldy = y.stride(0);
  int in_mode;
  if (x.has_value()) {
    in_mode = dsv41_gemv::IN_BF16;
    TORCH_CHECK(x->scalar_type() == at::kBFloat16 && x->dim() == 2 && x->size(1) == p.K && x->stride(1) == 1,
                "dense_gemv: x must be bf16 [M, K]");
    p.x = (const __nv_bfloat16*)x->data_ptr();
    p.M = x->size(0);
    p.ldx = x->stride(0);
    TORCH_CHECK((p.ldx % 8) == 0 && ((uintptr_t)p.x % 16) == 0, "dense_gemv: x rows must be 16-B aligned");
  } else {
    in_mode = dsv41_gemv::IN_QUANT;
    TORCH_CHECK(xq.has_value() && xs.has_value(), "dense_gemv: need x or (xq, xs)");
    TORCH_CHECK(xq->dim() == 2 && xq->size(1) == p.K && xq->stride(1) == 1 && xq->element_size() == 1,
                "dense_gemv: xq must be e4m3 [M, K]");
    p.xq = (const uint8_t*)xq->data_ptr();
    p.xs = (const uint8_t*)xs->data_ptr();
    p.M = xq->size(0);
    p.ldxq = xq->stride(0);
    TORCH_CHECK((p.ldxq % 8) == 0 && ((uintptr_t)p.xq % 8) == 0, "dense_gemv: xq rows must be 8-B aligned");
    const int64_t need = ((p.M + 127) / 128) * 128 * (((p.K / 32) + 3) / 4) * 4;
    TORCH_CHECK(xs->numel() * xs->element_size() >= need, "dense_gemv: xs is not a 128x4-swizzled scale");
  }
  TORCH_CHECK(p.M >= 1 && p.M <= MR && y.size(0) == p.M, "dense_gemv: M must be 1..MR");
  TORCH_CHECK(p.K % KC == 0 && p.N % 32 == 0, "dense_gemv: shape");
  TORCH_CHECK(((uintptr_t)p.w % 16) == 0 && ((uintptr_t)p.wscale % 16) == 0, "dense_gemv: alignment");
  TORCH_CHECK(grid >= 1, "dense_gemv: grid");
  must_find(W, S, KC, MR, smode, in_mode).launch(p, (int)grid, pdl);
}

// Grouped pre-quantized GEMV for o_proj wo_a (bit-exact with DeepGEMM's
// fp8_einsum "bhr,hdr->bhd", recipe (1,1,32), split_k=1: same MMA chain).
// xq: e4m3 [M, G, K] (row stride, group stride, 1) as fused_inv_rope_fp8_quant
// returns it; xs: int32 [M, G, K/128] MN-major packed ue8m0 (strides (1, gs, tal));
// w: e4m3 [G*D, K] group-major; y: bf16 [M, G*D] (the caller's [M, G, D] output).
void gemv_grouped(torch::Tensor xq, torch::Tensor xs, torch::Tensor w, torch::Tensor wscale, int64_t smode,
                  torch::Tensor y, int64_t W, int64_t S, int64_t KC, int64_t MR, int64_t grid, bool pdl, int64_t G) {
  dsv41_gemv::Params p = {};
  TORCH_CHECK(w.dim() == 2 && w.is_contiguous() && w.element_size() == 1, "dense_gemv: w must be [G*D, K] 1-byte");
  TORCH_CHECK(xq.dim() == 3 && xq.element_size() == 1 && xq.size(1) == G && xq.size(2) == w.size(1) &&
                  xq.stride(2) == 1,
              "dense_gemv: xq must be e4m3 [M, G, K]");
  TORCH_CHECK(xs.dim() == 3 && xs.scalar_type() == at::kInt && xs.size(0) == xq.size(0) && xs.size(1) == G &&
                  xs.size(2) * 128 == w.size(1) && xs.stride(0) == 1,
              "dense_gemv: xs must be the MN-major int32-packed ue8m0 scale [M, G, K/128]");
  TORCH_CHECK(y.scalar_type() == at::kBFloat16 && y.dim() == 2 && y.stride(1) == 1 && y.size(1) == w.size(0),
              "dense_gemv: y must be bf16 [M, G*D]");
  p.G = G;
  p.w = (const uint8_t*)w.data_ptr();
  p.wscale = (const uint8_t*)wscale.data_ptr();
  p.N = w.size(0);
  p.K = w.size(1);
  p.y = (__nv_bfloat16*)y.data_ptr();
  p.ldy = y.stride(0);
  p.xq = (const uint8_t*)xq.data_ptr();
  p.ldxq = xq.stride(0);
  p.xq_gstride = xq.stride(1);
  p.xs = (const uint8_t*)xs.data_ptr();
  p.xs_mode = dsv41_gemv::XS_DG_PACKED;
  p.xs_tal = xs.stride(2);
  p.xs_gstride = xs.stride(1);
  p.M = xq.size(0);
  TORCH_CHECK((p.ldxq % 8) == 0 && (p.xq_gstride % 8) == 0 && ((uintptr_t)p.xq % 8) == 0,
              "dense_gemv: xq rows must be 8-B aligned");
  TORCH_CHECK(p.M >= 1 && p.M <= MR && y.size(0) == p.M && p.xs_tal >= p.M, "dense_gemv: M must be 1..MR");
  TORCH_CHECK(p.K % KC == 0 && p.N % (32 * G) == 0, "dense_gemv: shape");
  TORCH_CHECK(((uintptr_t)p.w % 16) == 0 && ((uintptr_t)p.wscale % 16) == 0, "dense_gemv: alignment");
  TORCH_CHECK(grid >= G && grid % G == 0, "dense_gemv: grid must be a multiple of G");
  must_find(W, S, KC, MR, smode, dsv41_gemv::IN_QUANT).launch(p, (int)grid, pdl);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemv", &gemv, "dense small-M MXFP8 GEMV (bit-exact with b12x)");
  m.def("plan_grid", &plan_grid, "persistent grid for a config, 0 if it cannot run", py::arg("W"), py::arg("S"),
        py::arg("KC"), py::arg("MR"), py::arg("smode"), py::arg("in_mode"), py::arg("N"), py::arg("K"),
        py::arg("G") = 1);
  m.def("gemv_grouped", &gemv_grouped, "grouped pre-quantized GEMV (o_proj wo_a, bit-exact with fp8_einsum)");
}
