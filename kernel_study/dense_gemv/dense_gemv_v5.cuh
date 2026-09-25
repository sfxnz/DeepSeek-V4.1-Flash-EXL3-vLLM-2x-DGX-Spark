// v5: v4 (long per-row bursts, 128-B aligned stages) + progressive activation
// staging.
//
// v4 measured (ringcal4.py): the weight stream alone (no staging) runs at the
// read roofline (qkv_a 43.9 us), but staging the whole activation before the
// first MMA added 6.3 us at M=4 (W=4) and 24.6 us at M=8 (W=2): the ring is
// only S stages deep, it fills in ~2 us and then the stream stalls behind the
// quant compute. v5 stages the activation one KC-span at a time, one span
// ahead of use, inside the first tile's loop (CTA-wide, a __syncthreads per
// span, spans loaded one iteration early so the load latency overlaps the
// weight wait). Later tiles find the activation fully staged.
#pragma once
#include "dense_gemv_v4.cuh"

namespace dgemv {

template <int IN_MODE, int NBS>
struct ActRegs {
  uint4 v[NBS];
  uint32_t ue[NBS];
};

template <int IN_MODE, int NBS, int KBS>
__device__ __forceinline__ void act_load(const Params& p, ActRegs<IN_MODE, NBS>& ar, int span) {
  const int subs = p.M * KBS * 4;
  const int KB = p.K >> 5;
#pragma unroll
  for (int j = 0; j < NBS; ++j) {
    const int i = threadIdx.x + j * blockDim.x;
    ar.v[j] = make_uint4(0, 0, 0, 0);
    ar.ue[j] = 127;
    if (i < subs) {
      const int m = i / (KBS * 4), rem = i - m * (KBS * 4);
      const int kb = span * KBS + (rem >> 2), t = rem & 3;
      if constexpr (IN_MODE == IN_BF16) {
        ar.v[j] = *reinterpret_cast<const uint4*>(p.x + (size_t)m * p.ldx + kb * 32 + t * 8);
      } else {
        const uint2 u = *reinterpret_cast<const uint2*>(p.xq + (size_t)m * p.ldxq + kb * 32 + t * 8);
        ar.v[j].x = u.x;
        ar.v[j].y = u.y;
        if (t == 0) ar.ue[j] = p.xs[swz_offset(m, kb, (KB + 3) >> 2)];
      }
    }
  }
}

template <int IN_MODE, int NBS, int KBS, int MR>
__device__ __forceinline__ void act_store(const Params& p, const ActRegs<IN_MODE, NBS>& ar, int span, uint8_t* s_xf,
                                          uint8_t* s_xs) {
  const int subs = p.M * KBS * 4;
#pragma unroll
  for (int j = 0; j < NBS; ++j) {
    const int i = threadIdx.x + j * blockDim.x;
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

template <int W, int S, int KC, int SMODE, int IN_MODE, int MR>
__global__ void __launch_bounds__(W * 32) gemv5_kernel(const Params p) {
  using G = V4Geom<KC>;
  constexpr int KBS = G::KBS, CPR = G::CPR, A_BYTES = G::A_BYTES;
  constexpr int SC_BYTES = SMODE == S4_COMPACT32 ? G::SCB : 16 * KBS;
  constexpr int STAGE_BYTES = (A_BYTES + SC_BYTES + 127) & ~127;  // 128-B aligned stages (measured: -20% if not)
  constexpr int PER_LANE = 16 * CPR / 32;
  constexpr int NBS = (MR * KBS * 4 + W * 32 - 1) / (W * 32);
  extern __shared__ __align__(128) uint8_t smem[];

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int K = p.K, nspan = K / KC;
  const int T = p.N >> 4;
  const int gw = blockIdx.x * W + warp, GW = gridDim.x * W;
  const int ntile = gw < T ? (T - 1 - gw) / GW + 1 : 0;
  const int nsteps = ntile * nspan;

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
        cp_async16(st + row * KC + ((ch ^ (row & 7)) << 4), p.w + (size_t)(n0 + row) * K + k0 + ch * 16);
      }
      if constexpr (SMODE == S4_COMPACT32) {
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

  pdl_wait();  // no-op unless launched as a PDL dependent
  ActRegs<IN_MODE, NBS> ar;
  act_load<IN_MODE, NBS, KBS>(p, ar, 0);
  act_store<IN_MODE, NBS, KBS, MR>(p, ar, 0, s_xf, s_xs);
  if (nspan > 1) act_load<IN_MODE, NBS, KBS>(p, ar, 1);
  __syncthreads();

  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  const int r = lane >> 2, q = lane & 3;
  const int lm_row = ((lane >> 3) & 1) * 8 + (lane & 7);
  const int lm_half = lane >> 4;
  const int sfa_row = r + 8 * (lane & 1);
  const bool col_live = r < p.M;
  const uint8_t* xf = s_xf + (col_live ? lane : 0) * 8;
  const uint8_t* xsr = s_xs + (col_live ? r : 0);

  const int nloop = nsteps > nspan ? nsteps : nspan;  // every warp runs the first-tile staging schedule
  for (int s = 0; s < nloop; ++s) {
    if (s >= 1 && s < nspan) {
      act_store<IN_MODE, NBS, KBS, MR>(p, ar, s, s_xf, s_xs);
      if (s + 1 < nspan) act_load<IN_MODE, NBS, KBS>(p, ar, s + 1);
      __syncthreads();
    }
    if (s >= nsteps) continue;
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
    issue(s + S);
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
  pdl_launch();
}

}  // namespace dgemv
