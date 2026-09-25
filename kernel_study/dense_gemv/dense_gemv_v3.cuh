// v3: the v2 numerics (dense_gemv.cuh) with a TMA-engine weight stream.
//
// v2 fed each warp's ring with per-lane 16-B cp.async; with 28 SMs busy
// (qkv_a has 112 row tiles) it streamed at ~167 GB/s even with the activation
// staging removed, i.e. per-SM LSU request capacity, not DRAM, was the limit.
// v3 moves the weight stream to cp.async.bulk (the TMA engine, like b12x's
// loads): each stage is 16 per-row bulk copies of KSPAN bytes plus one 16-B
// aligned bulk copy of that stage's scales, completing on a per-stage
// mbarrier. Rows are padded +16 B in shared memory so ldmatrix is
// bank-conflict free. The grid is persistent: a warp streams its tiles back
// to back through one ring, so the pipeline never drains between tiles.
//
// Scale layouts are built once at load (both tiny or equal to the stock
// per-row scale size):
//   COMPACT32  [N/32][nspan][16]: the checkpoint's 32x32 blocks, bytes 0..KBS-1
//              of each 16-B cell hold kb = span*KBS + j (one scale per 32 rows)
//   TILE       [N/16][nspan][16 rows][KBS]: per-row scales (lm_head mxfp8 pack)
// The activation is staged row-major into smem ([MR][K+16] e4m3 + [MR][KB]
// ue8m0): pre-quantized rows by bulk copy, bf16 rows by the fused quant.
#pragma once
#include "dense_gemv.cuh"

namespace dgemv {

enum ScaleMode3 : int { S3_COMPACT32 = 0, S3_TILE = 1 };

struct Params3 {
  const uint8_t* w;        // e4m3 [N, K]
  const uint8_t* wscale;   // COMPACT32 or TILE layout (see above)
  const __nv_bfloat16* x;  // IN_BF16: [M, K], row stride ldx elements
  const uint8_t* xq;       // IN_QUANT: e4m3 [M, K], row stride ldxq bytes (16-B aligned rows)
  const uint8_t* xs;       // IN_QUANT: 128x4-swizzled scales of [M, K/32]
  __nv_bfloat16* y;        // [M, N], row stride ldy
  int M, N, K;
  int ldx, ldxq, ldy;
};

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_fence_init() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t phase) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "LAB_WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra LAB_WAIT_%=;\n"
      "}\n" ::"r"(bar),
      "r"(phase)
      : "memory");
}

__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(dst),
               "l"(src), "r"(bytes), "r"(bar)
               : "memory");
}

__device__ __forceinline__ void fence_proxy_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// Stage MR activation rows (rows >= M are never read: the MMA masks them).
template <int IN_MODE, int NB>
__device__ __forceinline__ void stage_activation3(const Params3& p, uint8_t* s_act, uint8_t* s_asc, uint32_t bar_act) {
  const int K = p.K, KB = K >> 5, ld = K + 16;
  if constexpr (IN_MODE == IN_QUANT) {
    if (threadIdx.x == 0) {
      mbar_expect_tx(bar_act, (uint32_t)(p.M * K));
      for (int m = 0; m < p.M; ++m)
        bulk_g2s(smem_u32(s_act + m * ld), p.xq + (size_t)m * p.ldxq, (uint32_t)K, bar_act);
    }
    const int nkt = (KB + 3) >> 2;
    for (int i = threadIdx.x; i < p.M * KB; i += blockDim.x) {
      const int m = i / KB, kb = i - m * KB;
      s_asc[m * KB + kb] = p.xs[swz_offset(m, kb, nkt)];
    }
    mbar_wait(bar_act, 0);
  } else {
    const int total = p.M * KB * 4;  // 8-element sub-chunks
    const int nthr = blockDim.x;
    for (int base = 0; base < total; base += nthr * NB) {
      uint4 v[NB];
#pragma unroll
      for (int i = 0; i < NB; ++i) {
        const int s = base + i * nthr + threadIdx.x;
        v[i] = make_uint4(0, 0, 0, 0);
        if (s < total) {
          const int m = s / (KB * 4), rem = s - m * (KB * 4);
          v[i] = *reinterpret_cast<const uint4*>(p.x + (size_t)m * p.ldx + (rem >> 2) * 32 + (rem & 3) * 8);
        }
      }
#pragma unroll
      for (int i = 0; i < NB; ++i) {
        const int s = base + i * nthr + threadIdx.x;
        float a = fmaxf(fmaxf(bf16x2_absmax(v[i].x), bf16x2_absmax(v[i].y)),
                        fmaxf(bf16x2_absmax(v[i].z), bf16x2_absmax(v[i].w)));
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 1));
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 2));
        const float kInv448 = (float)(1.0 / 448.0);
        const uint32_t ue = float_to_ue8m0(__fmul_rn(a, kInv448));
        const float inv = ue8m0_to_inv_scale(ue);
        if (s < total) {
          const int m = s / (KB * 4), rem = s - m * (KB * 4), kb = rem >> 2, t = rem & 3;
          uint2 o;
          o.x = bf16x2_to_e4m3x2(v[i].x, inv) | (bf16x2_to_e4m3x2(v[i].y, inv) << 16);
          o.y = bf16x2_to_e4m3x2(v[i].z, inv) | (bf16x2_to_e4m3x2(v[i].w, inv) << 16);
          *reinterpret_cast<uint2*>(s_act + m * ld + kb * 32 + t * 8) = o;
          if (t == 0) s_asc[m * KB + kb] = (uint8_t)ue;
        }
      }
    }
  }
}

template <int W, int STAGES, int KSPAN, int MR>
constexpr int smem3_bytes(int K) {
  return 16 * W * STAGES /*mbarriers, padded*/ + 16 /*act mbarrier*/ + MR * (K + 16) + MR * (K >> 5) + 80 +
         W * STAGES * (16 * (KSPAN + 16) + 16 * (KSPAN / 32));
}

// W warps per CTA; each warp owns whole 16-row tiles (full K: exact chain).
template <int W, int STAGES, int KSPAN, int SMODE, int IN_MODE, bool PDL, int MR>
__global__ void __launch_bounds__(W * 32) gemv3_kernel(const Params3 p) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int KBS = KSPAN / 32;
  constexpr int ROWB = KSPAN + 16;                               // padded row stride in smem
  constexpr int A_BYTES = 16 * ROWB;
  constexpr int SC_BYTES = SMODE == S3_COMPACT32 ? 16 : 16 * KBS;
  constexpr int STAGE_BYTES = A_BYTES + ((SC_BYTES + 15) & ~15);

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int K = p.K, KB = K >> 5, nspan = K / KSPAN;
  const int T = p.N >> 4;
  const int gw = blockIdx.x * W + warp, GW = gridDim.x * W;
  const int ntile = gw < T ? (T - 1 - gw) / GW + 1 : 0;
  const int S = ntile * nspan;

  // carve: [W*STAGES mbarriers x16 B][act mbarrier 16 B][s_act MR*(K+16)][s_asc MR*KB, pad 64][rings]
  uint8_t* bars = smem;
  const uint32_t bars_u32 = smem_u32(bars);
  const uint32_t bar_act = bars_u32 + 16 * W * STAGES;
  uint8_t* s_act = smem + 16 * W * STAGES + 16;
  uint8_t* s_asc = s_act + MR * (K + 16);
  uint8_t* rings = s_asc + ((MR * KB + 64 + 15) & ~15);
  uint8_t* ring = rings + warp * STAGES * STAGE_BYTES;
  const uint32_t ring_u32 = smem_u32(ring);
  const uint32_t my_bars = bars_u32 + 16 * warp * STAGES;

  if (lane == 0) {
    for (int i = 0; i < STAGES; ++i) mbar_init(my_bars + 16 * i, 1);
    if (warp == 0) mbar_init(bar_act, 1);
    mbar_fence_init();
  }
  __syncwarp();

  auto issue = [&](int s) {
    if (s >= S) return;
    const int ti = s / nspan, span = s - ti * nspan;
    const int tile = gw + ti * GW, n0 = tile * 16;
    const int slot = s % STAGES;
    const uint32_t st = ring_u32 + slot * STAGE_BYTES, bar = my_bars + 16 * slot;
    if (lane == 0) mbar_expect_tx(bar, 16 * KSPAN + SC_BYTES);
    __syncwarp();
    if (lane < 16) {
      bulk_g2s(st + lane * ROWB, p.w + (size_t)(n0 + lane) * K + span * KSPAN, KSPAN, bar);
    } else if (lane == 16) {
      const uint8_t* src = SMODE == S3_COMPACT32 ? p.wscale + ((size_t)(n0 >> 5) * nspan + span) * 16
                                                 : p.wscale + ((size_t)tile * nspan + span) * (16 * KBS);
      bulk_g2s(st + A_BYTES, src, SC_BYTES, bar);
    }
  };

  // Weights are static: start the stream before waiting on the producer kernel.
  for (int s = 0; s < STAGES - 1; ++s) issue(s);

  if constexpr (PDL) pdl_wait();
  __syncthreads();  // bar_act init (warp 0) visible to every thread before staging
  stage_activation3<IN_MODE, 8>(p, s_act, s_asc, bar_act);
  __syncthreads();

  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  const int r = lane >> 2, q = lane & 3;
  const int lm_row = ((lane >> 3) & 1) * 8 + (lane & 7);
  const int lm_half = lane >> 4;
  const int sfa_row = r + 8 * (lane & 1);
  const bool col_live = r < p.M;
  const uint8_t* xrow = s_act + (col_live ? r : 0) * (K + 16) + 4 * q;
  const uint8_t* srow = s_asc + (col_live ? r : 0) * KB;

  for (int s = 0; s < S; ++s) {
    const int slot = s % STAGES;
    mbar_wait(my_bars + 16 * slot, (s / STAGES) & 1);
    const int ti = s / nspan, span = s - ti * nspan;
    const uint32_t st = ring_u32 + slot * STAGE_BYTES;
    const uint8_t* stp = ring + slot * STAGE_BYTES;
#pragma unroll
    for (int j = 0; j < KBS; ++j) {
      const int kb = span * KBS + j;
      uint32_t a[4];
      ldmatrix_x4(a, st + lm_row * ROWB + (2 * j + lm_half) * 16);
      uint32_t b0 = *reinterpret_cast<const uint32_t*>(xrow + kb * 32);
      uint32_t b1 = *reinterpret_cast<const uint32_t*>(xrow + kb * 32 + 16);
      uint32_t sfb = srow[kb];
      if (!col_live) {
        b0 = b1 = 0;
        sfb = 127;
      }
      uint32_t sfa;
      if constexpr (SMODE == S3_COMPACT32) sfa = stp[A_BYTES + j];
      else sfa = stp[A_BYTES + sfa_row * KBS + j];
      mma_mxf8(acc, a, b0, b1, sfa, sfb);
    }
    __syncwarp();
    fence_proxy_async_smem();
    issue(s + STAGES - 1);
    if (span == nspan - 1) {
      const int n0 = (gw + ti * GW) * 16;
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
  if constexpr (PDL) pdl_launch();
}

}  // namespace dgemv
