// v4: v2 numerics (dense_gemv.cuh: bit-exact b12x math) with a DRAM-friendly
// weight stream.
//
// Measured (ringcal.py, pure stream, cold): a per-warp cp.async ring that
// fetches 16 rows x KC contiguous bytes per stage streams at the flat-read
// roofline once KC >= 512 (qkv_a 43.1 us vs flat 43.0; wo_b 93-94 vs 92.2),
// while v2's KC = 128 lost 15-30%: short per-row bursts spread the requests of
// ~1800 row streams over too many DRAM pages. So each warp stage is
// 16 x KC bytes (KC = 384..1280, a divisor of K), with the prologue issuing
// the whole ring so the activation staging hides under the first stages.
//
// Scale layouts, built once at load:
//   COMPACT32 [N/32][nspan][SCB]   32-row groups share scales (checkpoint 32x32
//                                  blocks); bytes 0..KBS-1 of each cell = kb
//                                  span*KBS + j; SCB = KBS rounded up to 16
//   TILE      [N/16][nspan][16][KBS]  per-row scales (lm_head mxfp8 pack)
// Activation rows are staged once per CTA in B-fragment order for MR (4 or 8)
// rows: xf[kb][lane < 4*MR][8 B], xs[kb][MR].
#pragma once
#include "dense_gemv.cuh"

namespace dgemv {

enum ScaleMode4 : int { S4_COMPACT32 = 0, S4_TILE = 1 };

template <int KC>
struct V4Geom {
  static constexpr int KBS = KC / 32;              // k-blocks per stage
  static constexpr int CPR = KC / 16;              // 16-B chunks per row
  static constexpr int A_BYTES = 16 * KC;
  static constexpr int SCB = (KBS + 15) / 16 * 16; // compact scale cell
};

template <int W, int S, int KC, int SMODE, int MR>
constexpr int smem4_bytes(int K) {
  using G = V4Geom<KC>;
  return MR * K + ((MR * (K >> 5) + 15) & ~15) +
         W * S * ((G::A_BYTES + (SMODE == S4_COMPACT32 ? G::SCB : 16 * G::KBS) + 127) & ~127);
}

template <int IN_MODE, int NB, int MR>
__device__ __forceinline__ void stage_activation4(const Params& p, uint8_t* s_xf, uint8_t* s_xs) {
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
        const int m = s / (KB * 4), rem = s - m * (KB * 4), kb = rem >> 2, t = rem & 3;
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
        const int m = s / (KB * 4), rem = s - m * (KB * 4), kb = rem >> 2, t = rem & 3;
        uint32_t* dst =
            reinterpret_cast<uint32_t*>(s_xf + kb * (MR * 32) + (4 * m + 2 * (t & 1)) * 8 + (t >> 1) * 4);
        dst[0] = w0;
        dst[2] = w1;
        if (t == 0) s_xs[kb * MR + m] = (uint8_t)ue;
      }
    }
  }
}

// W warps per CTA; every warp streams whole 16-row tiles (full K: the exact
// b12x accumulation chain) through a private S-stage ring of 16 x KC bytes.
template <int W, int S, int KC, int SMODE, int IN_MODE, int MR>
__global__ void __launch_bounds__(W * 32) gemv4_kernel(const Params p) {
  using G = V4Geom<KC>;
  constexpr int KBS = G::KBS, CPR = G::CPR, A_BYTES = G::A_BYTES;
  constexpr int SC_BYTES = SMODE == S4_COMPACT32 ? G::SCB : 16 * KBS;
  constexpr int STAGE_BYTES = (A_BYTES + SC_BYTES + 127) & ~127;  // 128-B aligned stages
  constexpr int PER_LANE = 16 * CPR / 32;
  extern __shared__ __align__(128) uint8_t smem[];

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int K = p.K, KB = K >> 5, nspan = K / KC;
  const int T = p.N >> 4;
  const int gw = blockIdx.x * W + warp, GW = gridDim.x * W;
  const int ntile = gw < T ? (T - 1 - gw) / GW + 1 : 0;
  const int nsteps = ntile * nspan;

  // rings first (1024-B aligned base), then the staged activation
  uint8_t* ring = smem + warp * S * STAGE_BYTES;
  uint8_t* s_xf = smem + W * S * STAGE_BYTES;
  uint8_t* s_xs = s_xf + MR * K;
  const uint32_t ring_u32 = smem_u32(ring);

  auto issue = [&](int s) {
    if (s < nsteps) {
      const int ti = s / nspan, span = s - ti * nspan;
      const int tile = gw + ti * GW, n0 = tile * 16, k0 = span * KC;
      const uint32_t st = ring_u32 + (s % S) * STAGE_BYTES;
#pragma unroll
      for (int i = 0; i < PER_LANE; ++i) {
        const int c = lane + 32 * i, row = c / CPR, ch = c % CPR;
        const int sw = (p.diag & 16) ? 0 : (row & 7);  // study: bit 4 disables the XOR swizzle
        cp_async16(st + row * KC + ((ch ^ sw) << 4), p.w + (size_t)(n0 + row) * K + k0 + ch * 16);
      }
      if (p.diag & 4) {
        // study: no scale loads
      } else if constexpr (SMODE == S4_COMPACT32) {
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

  // Weights are static: fill the whole ring before waiting on the producer.
  const bool shallow = p.diag & 8;  // study: prologue S-1, wait S-2 (ring_read's schedule)
  const bool act_first = p.diag & 64;  // study: stage the activation before the weight prologue
  if (act_first) {
    if (!(p.diag & 32)) pdl_wait();
    if (!(p.diag & 2)) stage_activation4<IN_MODE, 8, MR>(p, s_xf, s_xs);
  }
#pragma unroll
  for (int s = 0; s < S; ++s)
    if (!shallow || s < S - 1) issue(s);

  if (!act_first) {
    if (!(p.diag & 32)) pdl_wait();  // no-op unless launched as a PDL dependent
    if (!(p.diag & 2)) stage_activation4<IN_MODE, 8, MR>(p, s_xf, s_xs);
  }
  __syncthreads();

  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  const int r = lane >> 2, q = lane & 3;
  const int lm_row = ((lane >> 3) & 1) * 8 + (lane & 7);
  const int lm_half = lane >> 4;
  const int sfa_row = r + 8 * (lane & 1);
  const bool col_live = r < p.M;
  const uint8_t* xf = s_xf + (col_live ? lane : 0) * 8;
  const uint8_t* xsr = s_xs + (col_live ? r : 0);

  for (int s = 0; s < nsteps; ++s) {
    if (shallow) cp_async_wait<(S >= 2 ? S - 2 : 0)>();
    else cp_async_wait<S - 1>();
    __syncwarp();
    const int ti = s / nspan, span = s - ti * nspan;
    const uint32_t st = ring_u32 + (s % S) * STAGE_BYTES;
    const uint8_t* stp = ring + (s % S) * STAGE_BYTES;
#pragma unroll
    for (int j = 0; j < KBS && !(p.diag & 1); ++j) {
      const int kb = span * KBS + j;
      uint32_t a[4];
      const int ch = 2 * j + lm_half;
      ldmatrix_x4(a, st + lm_row * KC + ((ch ^ (lm_row & 7)) << 4));
      uint2 b = *reinterpret_cast<const uint2*>(xf + kb * (MR * 32));
      uint32_t sfb = xsr[kb * MR];
      if (!col_live) {
        b = make_uint2(0, 0);
        sfb = 127;
      }
      uint32_t sfa;
      if constexpr (SMODE == S4_COMPACT32) sfa = stp[A_BYTES + j];
      else sfa = stp[A_BYTES + sfa_row * KBS + j];
      mma_mxf8(acc, a, b.x, b.y, sfa, sfb);
    }
    __syncwarp();
    issue(shallow ? s + S - 1 : s + S);
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
  cp_async_wait<0>();
  if (!(p.diag & 32)) pdl_launch();
}

}  // namespace dgemv
