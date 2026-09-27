// mHC det kernels (DSV41_MHC_DET_SPLITS). NVRTC-compiled at runtime by mhc_det.py.
//
// Bitwise replica of the stock decode prenorm GEMM, DeepGEMM
// sm120_tf32_hc_prenorm_gemm_impl<24, K, 128, 32, 64, kNumSplits=16, 4, 256, 128>:
//   partial[s][m][n] = TF32 mma.sync m16n8k8 chain over the k-blocks of split s, k-steps 0..7,
//                      A = bf16 x widened to fp32 bits, B = fn rounded fp32 -> tf32 to nearest
//                      even (DeepGEMM's B tensor map converts on load: probe_b_round.json, RNE
//                      60/60 cases bitwise, truncation and round-away 0/60),
//   sqr[s][m]        = per-lane acc += fma(a_lo, a_lo, a_hi * a_hi) over the same k order,
//                      then + shfl_xor 2, + shfl_xor 1 (DeepGEMM warp_reduce_sum<4>).
// The chain of every (split, 8-column n-tile) is unchanged; only the work layout differs:
// one warp per (split, n-tile) = 48 CTAs instead of 16 CTAs x 8 warps x 128 rows, so all
// 48 SMs stream fn and no warp multiplies zero rows. T <= 16 (one m16 tile).
//
// No atomics. Every output element is written by exactly one thread with plain stores.

typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned long long u64;

#define DEV __device__ __forceinline__

DEV u32 smem_u32(const void* p) {
  u64 a;
  asm("cvta.to.shared.u64 %0, %1;" : "=l"(a) : "l"(p));
  return (u32)a;
}
DEV void mbar_init(u64* bar, u32 count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
DEV void mbar_arrive_expect_tx(u64* bar, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes)
               : "memory");
}
DEV bool mbar_try_wait(u64* bar, u32 parity) {
  u32 ok;
  asm volatile(
      "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
      : "=r"(ok)
      : "r"(smem_u32(bar)), "r"(parity)
      : "memory");
  return ok != 0;
}
DEV void mbar_wait(u64* bar, u32 parity) {
  while (!mbar_try_wait(bar, parity)) {
  }
}
DEV void fence_barrier_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
// 1-D bulk async copy global -> this CTA's shared memory, completes on an mbarrier.
DEV void bulk_g2s(void* dst, const void* src, u32 bytes, u64* bar) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                   smem_u32(dst)),
               "l"(src), "r"(bytes), "r"(smem_u32(bar))
               : "memory");
}
DEV void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src) : "memory");
}
DEV void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
DEV void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }
DEV void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
DEV void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

DEV void mma_tf32(float (&d)[4], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
DEV float bf16f(u16 v) { return __uint_as_float(((u32)v) << 16); }

// Optional phase timestamps (kernel_study profiling only: compiled with -DMHC_DET_PROF).
#ifdef MHC_DET_PROF
__device__ unsigned long long g_prof[64 * 16];
#define PROF(i)                                                                    \
  do {                                                                             \
    if (threadIdx.x == 0) {                                                        \
      unsigned long long c_;                                                       \
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(c_)::"memory");            \
      g_prof[(blockIdx.y * gridDim.x + blockIdx.x) * 16 + (i)] = c_;               \
    }                                                                              \
  } while (0)
#define PROFN(i)                                                                   \
  do {                                                                             \
    unsigned long long c_;                                                         \
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(c_)::"memory");              \
    g_prof[(48 + blockIdx.x) * 16 + (i)] = c_;                                     \
  } while (0)
#else
#define PROF(i)
#define PROFN(i)
#endif

#define MHC_SPLITS 16
#define STAGE_KB 2
#define XS_PAD 8           // bf16 x row pad in smem: rows start 4 banks apart
#define PK_KB_BYTES 1280   // packed fn bytes per (split, n-tile, k-block)

// Packed fn: per (split, n-tile, k-block) the RNE tf32 bits (top 19 bits of fp32) of the
// 32 lanes' 16 B-fragment values: hi [2][32 lanes][8 x u16] = bits 31..16, lo [32 lanes][16 x
// 4 bit] = bits 15..13. Lane (g, t) value i: ks = i >> 1, k = kb*64 + ks*8 + t + 4*(i & 1),
// row = n-tile*8 + g. 1280 B per k-block instead of 2048 B; the MMA sees identical bits.
// Stage-major: [stage][cta = split*3 + n-tile][nkb(stage) x 1280 B], so each stage of all 48
// CTAs is one contiguous region (stream_probe.json: 6.6 vs 7.5 us for 48 x 25.6 KB cold).
//
// grid (3 n-tiles, 16 splits), block 32. x [T, K] bf16, packed fn -> mixes [16, T, 24] f32,
// sqr [16, T] f32. ROW1: T may exceed 8 (rows g + 8 are live).
#ifndef PREFETCH_STAGES
#define PREFETCH_STAGES 4  // fn stages issued before griddepcontrol.wait (sweep_prefetch.json)
#endif
template <bool ROW1>
DEV void gemm_body(const u16* __restrict__ x, const unsigned char* __restrict__ fnp, float* __restrict__ mixes,
                   float* __restrict__ sqr, const int T, const int K) {
  extern __shared__ __align__(128) unsigned char smem[];
  const int nt = blockIdx.x, s = blockIdx.y;
  const int cta = s * 3 + nt;
  const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
  const int kbps = K / (64 * MHC_SPLITS);
  const int kspan = kbps * 64;
  const int k0 = s * kspan;
  const int xs = kspan + XS_PAD;
  const int fn_bytes = kbps * PK_KB_BYTES;
  unsigned char* fsm = smem;
  u16* xsm = (u16*)(smem + fn_bytes);
  const int nstage = (kbps + STAGE_KB - 1) / STAGE_KB;
  u64* bars = (u64*)(smem + fn_bytes + ((T * xs * 2 + 15) & ~15));
  auto issue_fn = [&](int st) {
    const int kb_lo = st * STAGE_KB, nkb = min(STAGE_KB, kbps - kb_lo);
    const unsigned char* src = fnp + ((size_t)MHC_SPLITS * 3 * kb_lo + (size_t)cta * nkb) * PK_KB_BYTES;
    mbar_arrive_expect_tx(&bars[st], (u32)(nkb * PK_KB_BYTES));
    bulk_g2s(fsm + kb_lo * PK_KB_BYTES, src, nkb * PK_KB_BYTES, &bars[st]);
  };
  PROF(0);
  if (lane == 0) {
    for (int st = 0; st < nstage; ++st) mbar_init(&bars[st], 1);
    fence_barrier_init();
    // fn is a weight: its first stages stream before waiting on the producer of x (PDL).
    for (int st = 0; st < min(PREFETCH_STAGES, nstage); ++st) issue_fn(st);
  }
  PROF(1);
  pdl_wait();
  // The producer of x (the post) is complete now: let the dependent fused norm launch; it reads
  // only the post output and weights before its own wait (see mhc_det_norm).
  pdl_trigger();
  PROF(2);
  // x is L2-hot (just written by the previous kernel): issue it before the remaining fn
  // stages so its requests do not queue behind them. All lanes, 16 B cp.async, one group.
  const int chunks = kspan / 8;
  for (int r = 0; r < T; ++r)
    for (int c = lane; c < chunks; c += 32) cp_async16(xsm + r * xs + c * 8, x + (size_t)r * K + k0 + c * 8);
  cp_async_commit();
  if (lane == 0)
    for (int st = PREFETCH_STAGES; st < nstage; ++st) issue_fn(st);
  cp_async_wait_all();
  __syncwarp();
  PROF(3);

  float d[4] = {0.f, 0.f, 0.f, 0.f};
  float acc0 = 0.f, acc1 = 0.f;
  const bool r0 = g < T, r1 = ROW1 && (g + 8 < T);
  const u16* xr0 = xsm + g * xs;
  const u16* xr1 = xsm + (g + 8) * xs;
  for (int st = 0; st < nstage; ++st) {
    mbar_wait(&bars[st], 0);
    if (st < 12) PROF(4 + (st >> 1));
    const int kb_lo = st * STAGE_KB, kb_hi = min(kb_lo + STAGE_KB, kbps);
#pragma unroll
    for (int j = 0; j < STAGE_KB; ++j) {
      const int kb = kb_lo + j;
      if (kb >= kb_hi) break;
      const unsigned char* blk = fsm + kb * PK_KB_BYTES;
      const uint4 h0 = *(const uint4*)(blk + lane * 16);
      const uint4 h1 = *(const uint4*)(blk + 512 + lane * 16);
      const uint2 lo = *(const uint2*)(blk + 1024 + lane * 8);
      const u32 hw[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const int k = kb * 64 + ks * 8 + t;
        const float fa0 = r0 ? bf16f(xr0[k]) : 0.f;
        const float fa2 = r0 ? bf16f(xr0[k + 4]) : 0.f;
        const float fa1 = r1 ? bf16f(xr1[k]) : 0.f;
        const float fa3 = r1 ? bf16f(xr1[k + 4]) : 0.f;
        const u32 w = hw[ks];                                      // values 2ks (low), 2ks+1 (high)
        const u32 nib = (ks < 4 ? lo.x : lo.y) >> (8 * (ks & 3));  // nibbles 2ks, 2ks+1
        const u32 b0 = (w << 16) | ((nib & 7u) << 13);
        const u32 b1 = (w & 0xffff0000u) | (((nib >> 4) & 7u) << 13);
        mma_tf32(d, __float_as_uint(fa0), __float_as_uint(fa1), __float_as_uint(fa2), __float_as_uint(fa3), b0, b1);
        if (nt == 0) {
          acc0 = __fadd_rn(acc0, __fmaf_rn(fa0, fa0, __fmul_rn(fa2, fa2)));
          if (ROW1) acc1 = __fadd_rn(acc1, __fmaf_rn(fa1, fa1, __fmul_rn(fa3, fa3)));
        }
      }
    }
  }
  PROF(10);
  if (r0) *(float2*)&mixes[((size_t)s * T + g) * 24 + nt * 8 + 2 * t] = make_float2(d[0], d[1]);
  if (r1) *(float2*)&mixes[((size_t)s * T + g + 8) * 24 + nt * 8 + 2 * t] = make_float2(d[2], d[3]);
  if (nt == 0) {
    acc0 = __fadd_rn(acc0, __shfl_xor_sync(0xffffffffu, acc0, 2));
    acc0 = __fadd_rn(acc0, __shfl_xor_sync(0xffffffffu, acc0, 1));
    if (ROW1) {
      acc1 = __fadd_rn(acc1, __shfl_xor_sync(0xffffffffu, acc1, 2));
      acc1 = __fadd_rn(acc1, __shfl_xor_sync(0xffffffffu, acc1, 1));
    }
    if (t == 0) {
      if (r0) sqr[s * T + g] = acc0;
      if (r1) sqr[s * T + g + 8] = acc1;
    }
  }
  PROF(11);
}

#define GEMM_KERNEL(NAME, ROW1)                                                                         \
  extern "C" __global__ void __launch_bounds__(32, 1)                                                  \
      NAME(const u16* __restrict__ x, const unsigned char* __restrict__ fnp, float* __restrict__ mixes, \
           float* __restrict__ sqr, int T, int K) {                                                    \
    gemm_body<ROW1>(x, fnp, mixes, sqr, T, K);                                                         \
  }
GEMM_KERNEL(mhc_det_gemm_t8, false)
GEMM_KERNEL(mhc_det_gemm_t16, true)

// ---------------------------------------------------------------------------------------------
// Bitwise replica of the TileLang mhc_post kernel (vLLM kernels/mhc/tilelang_kernels.py):
//   out[t][o][h] = bf16_rn(fma(a3,b3, fma(a2,b2, fma(a1,b1, fma(c, d, a0*b0)))))  -- the FMA
//   contraction nvcc applies to the TileLang source "x = c*d; x += a_i*b_i (i = 0..3)"
// a = comb [T,4,4], b = residual [T,4,H], c = post [T,4], d = x [T,H]; out [T,4,H] bf16.
// grid (H/256, T), block 64, 4 consecutive h per thread. It triggers its dependent (the
// prenorm GEMM) on entry, so that kernel launches now and streams its weights during this one.
DEV u32 bf16x2_rn(float lo, float hi) {
  u32 r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
extern "C" __global__ void __launch_bounds__(64)
    mhc_det_post(const float* __restrict__ comb, const u16* __restrict__ resid, const float* __restrict__ post,
                 const u16* __restrict__ xin, u16* __restrict__ out, const int H) {
  pdl_trigger();
  pdl_wait();
  const int t = blockIdx.y;
  const int h = (blockIdx.x * 64 + threadIdx.x) * 4;
  const float4 a0 = *(const float4*)(comb + t * 16 + 0);
  const float4 a1 = *(const float4*)(comb + t * 16 + 4);
  const float4 a2 = *(const float4*)(comb + t * 16 + 8);
  const float4 a3 = *(const float4*)(comb + t * 16 + 12);
  const float4 cc = *(const float4*)(post + t * 4);
  const float a[4][4] = {{a0.x, a0.y, a0.z, a0.w}, {a1.x, a1.y, a1.z, a1.w},
                         {a2.x, a2.y, a2.z, a2.w}, {a3.x, a3.y, a3.z, a3.w}};
  const float c[4] = {cc.x, cc.y, cc.z, cc.w};
  const uint2 dv = *(const uint2*)(xin + (size_t)t * H + h);
  float d[4] = {__uint_as_float(dv.x << 16), __uint_as_float(dv.x & 0xffff0000u),
                __uint_as_float(dv.y << 16), __uint_as_float(dv.y & 0xffff0000u)};
  float b[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint2 bv = *(const uint2*)(resid + ((size_t)t * 4 + i) * H + h);
    b[i][0] = __uint_as_float(bv.x << 16);
    b[i][1] = __uint_as_float(bv.x & 0xffff0000u);
    b[i][2] = __uint_as_float(bv.y << 16);
    b[i][3] = __uint_as_float(bv.y & 0xffff0000u);
  }
#pragma unroll
  for (int o = 0; o < 4; ++o) {
    float v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      // nvcc contracts (c*d) + (a0*b0) as fma(c, d, a0*b0): the left product is fused.
      float acc = __fmaf_rn(c[o], d[e], __fmul_rn(a[0][o], b[0][e]));
#pragma unroll
      for (int i = 1; i < 4; ++i) acc = __fmaf_rn(a[i][o], b[i][e], acc);
      v[e] = acc;
    }
    *(uint2*)(out + ((size_t)t * 4 + o) * H + h) = make_uint2(bf16x2_rn(v[0], v[1]), bf16x2_rn(v[2], v[3]));
  }
}

// ---------------------------------------------------------------------------------------------
// Bitwise replica of the TileLang mhc_pre_big_fuse_with_norm kernel (hc 4, hidden 5120, 16
// splits, save_pre_mix; carried pre-mix or the one-hot stream-0 pre-mix), one CTA per token:
//   warp 0     : coefficients after griddepcontrol.wait: serial split sums (rms, 24 mixes),
//                sigmoids, 20-step sinkhorn with the TileLang AllReduce butterflies (max over
//                xor 2,1; sums as rowsum4 / colsum4, the same additions as xor 2,1 for rows and
//                xor 8,4 for columns), IEEE divisions, expf/rsqrtf as nvcc 13.0.
//   warps 1..8 : layer_input = bf16(bf16(sum_j pre_j * resid_j) * rsqrt(sumsq/5120 + eps) * w),
//                sumsq in TileLang's order: 64 virtual threads l, slot i of l = position
//                hb*1024 + (i>>3)*512 + l*8 + (i&7), fma chain over hb = 0..4, slots summed as
//                0,8,1,9,..,7,15, then (p_l + p_{l+32}) and shfl_xor 16,8,4,2,1.
//                It needs only the post output and the carried pre-mix, so it runs BEFORE the
//                wait, while the prenorm GEMM (the primary) streams. Safe: the GEMM triggers only
//                after its own griddepcontrol.wait, so the post is complete when this starts;
//                its reads use ld.global.cg (L2, no stale L1).
// The two halves are device functions (norm_coef, norm_li) with three entries: mhc_det_norm (both,
// one launch after the GEMM) and, for DSV41_MHC_DET_OVERLAP, mhc_det_norm_li (layer_input alone,
// right after the post: waits on it first) and mhc_det_norm_coef (the coefficient half alone,
// after the GEMM on the side stream). Same device code, so the same bits in every entry.
#define NORM_H 5120
DEV uint2 ldcg_u2(const void* p) {
  uint2 v;
  asm volatile("ld.global.cg.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}
DEV float bf16_lo(u32 w) { return __uint_as_float(w << 16); }
DEV float bf16_hi(u32 w) { return __uint_as_float(w & 0xffff0000u); }
DEV void bar_named(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
// The TileLang AllReduce butterflies, bitwise, with independent shuffles. Rows (xor 2 then xor 1)
// give lane l (v_l + v_l^2) + (v_l^1 + v_l^3); columns (xor 8 then xor 4) give
// (v_l + v_l^8) + (v_l^4 + v_l^12). The same additions in the same order, but the three shuffles
// issue back to back instead of two dependent SHFL -> FADD steps.
DEV float rowsum4(float v) {
  const float a1 = __shfl_xor_sync(0xffffffffu, v, 1);
  const float a2 = __shfl_xor_sync(0xffffffffu, v, 2);
  const float a3 = __shfl_xor_sync(0xffffffffu, v, 3);
  return __fadd_rn(__fadd_rn(v, a2), __fadd_rn(a1, a3));
}
DEV float colsum4(float v) {
  const float a4 = __shfl_xor_sync(0xffffffffu, v, 4);
  const float a8 = __shfl_xor_sync(0xffffffffu, v, 8);
  const float a12 = __shfl_xor_sync(0xffffffffu, v, 12);
  return __fadd_rn(__fadd_rn(v, a8), __fadd_rn(a4, a12));
}

// Coefficient half (one warp per token): split sums, sigmoids, sinkhorn. Reads the GEMM partials
// after griddepcontrol.wait; the weights it needs are fetched before the wait.
DEV void norm_coef(float* mixes_s, const float* __restrict__ mixes_p, const float* __restrict__ sqr_p,
                   const float* __restrict__ hc_scale, const float* __restrict__ hc_base,
                   float* __restrict__ post_mix, float* __restrict__ comb_mix, float* __restrict__ pre_mix_out,
                   const int T, const float rms_numel, const float rms_eps, const float hc_pre_eps,
                   const float sk_eps, const float post_mult, const int sk_repeat, const int tok, const int lane) {
  if (lane == 0) PROFN(0);
  // weights: fetch before waiting on the GEMM
  const float s0 = hc_scale[0], s1 = hc_scale[1], s2 = hc_scale[2];
  const float bpre = hc_base[lane & 3], bpost = hc_base[4 + (lane & 3)];
  const float bsk = hc_base[(lane & 15) + 8];
  pdl_wait();
  if (lane == 0) PROFN(1);
  float rms = 0.f;
  for (int s = 0; s < MHC_SPLITS; ++s) rms = __fadd_rn(rms, sqr_p[s * T + tok]);
  rms = rsqrtf(__fadd_rn(__fdiv_rn(rms, rms_numel), rms_eps));
  const int j = lane % 24;
  float mix = 0.f;
  for (int s = 0; s < MHC_SPLITS; ++s) mix = __fadd_rn(mix, mixes_p[((size_t)s * T + tok) * 24 + j]);
  mix = __fmul_rn(mix, rms);
  if (lane < 24) mixes_s[lane] = mix;
  __syncwarp();
  if (lane == 0) PROFN(2);
  if (lane < 4) {
    const float e0 = expf(__fsub_rn(0.f, __fmaf_rn(mixes_s[lane], s0, bpre)));
    pre_mix_out[tok * 4 + lane] = __fadd_rn(__fdiv_rn(1.f, __fadd_rn(1.f, e0)), hc_pre_eps);
    const float e1 = expf(__fsub_rn(0.f, __fmaf_rn(mixes_s[lane + 4], s1, bpost)));
    post_mix[tok * 4 + lane] = __fmul_rn(__fdiv_rn(1.f, __fadd_rn(1.f, e1)), post_mult);
  }
  // Sinkhorn exactly as TileLang: one element per lane (lanes 16..31 compute copies), row sums
  // rowsum4, column sums colsum4, fmaxf over shfl_xor 2,1 for the row max. (One row per lane
  // with 4 divisions each measured 2x slower: iterations.txt item 9.)
  const int c = (lane & 15) + 8;
  float cm = __fmaf_rn(mixes_s[c], s2, bsk);
  float rmax = fmaxf(-__int_as_float(0x7f800000), cm);
  rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffffu, rmax, 2));
  rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffffu, rmax, 1));
  cm = expf(__fsub_rn(cm, rmax));
  cm = __fadd_rn(__fdiv_rn(cm, rowsum4(__fadd_rn(0.f, cm))), sk_eps);
  cm = __fdiv_rn(cm, __fadd_rn(colsum4(__fadd_rn(0.f, cm)), sk_eps));
  if (lane == 0) PROFN(3);
  for (int it = 0; it < sk_repeat - 1; ++it) {
    cm = __fdiv_rn(cm, __fadd_rn(rowsum4(__fadd_rn(0.f, cm)), sk_eps));
    cm = __fdiv_rn(cm, __fadd_rn(colsum4(__fadd_rn(0.f, cm)), sk_eps));
  }
  if (lane < 16) comb_mix[tok * 16 + lane] = cm;
  if (lane == 0) PROFN(4);
}

// layer_input half (256 threads per token, lt = 0..255, named barrier 1): needs only the post
// output (residual) and the carried pre-mix. The caller orders it after the post.
DEV void norm_li(u16* rounded_s, float* part_s, float* rsqrt_s, const u16* __restrict__ residual,
                 const float* __restrict__ pre_mix_in, const u16* __restrict__ norm_w, u16* __restrict__ layer_input,
                 const float norm_eps, const int tok, const int lt, const bool wait_first) {
  uint2 wv[5];  // norm weight, fetched first (a weight: independent of the primary)
#pragma unroll
  for (int q = 0; q < 5; ++q) wv[q] = ldcg_u2(norm_w + 4 * (lt + 256 * q));
  if (wait_first) pdl_wait();
  float pre[4] = {1.f, 0.f, 0.f, 0.f};
  if (pre_mix_in != nullptr) {
    const float4 pm = *(const float4*)(pre_mix_in + tok * 4);
    pre[0] = pm.x; pre[1] = pm.y; pre[2] = pm.z; pre[3] = pm.w;
  }
  const u16* rb = residual + (size_t)tok * 4 * NORM_H;
  // weighted stream sum, bf16 round: 5 chunks of 4 positions per thread (p = 4*(lt + 256q))
#pragma unroll
  for (int q = 0; q < 5; ++q) {
    const int p = 4 * (lt + 256 * q);
    uint2 xv[4];
#pragma unroll
    for (int hc = 0; hc < 4; ++hc) xv[hc] = ldcg_u2(rb + hc * NORM_H + p);
    float ol[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      float acc = 0.f;
#pragma unroll
      for (int hc = 0; hc < 4; ++hc) {
        const u32 w = (e < 2) ? xv[hc].x : xv[hc].y;
        acc = __fmaf_rn(pre[hc], (e & 1) ? bf16_hi(w) : bf16_lo(w), acc);
      }
      ol[e] = acc;
    }
    *(uint2*)(rounded_s + p) = make_uint2(bf16x2_rn(ol[0], ol[1]), bf16x2_rn(ol[2], ol[3]));
  }
  bar_named(1, 256);
  if (lt < 64) {
    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.f;
#pragma unroll
    for (int hb = 0; hb < 5; ++hb) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const uint4 v = *(const uint4*)(rounded_s + hb * 1024 + half * 512 + lt * 8);
        const u32 w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float f = (e & 1) ? bf16_hi(w[e >> 1]) : bf16_lo(w[e >> 1]);
          acc[half * 8 + e] = __fmaf_rn(f, f, acc[half * 8 + e]);
        }
      }
    }
    float sq = 0.f;
#pragma unroll
    for (int rv = 0; rv < 16; ++rv) sq = __fadd_rn(sq, acc[(rv & 1) * 8 + (rv >> 1)]);
    part_s[lt] = sq;
  }
  bar_named(1, 256);
  if (lt < 32) {
    float y = __fadd_rn(part_s[lt], part_s[lt + 32]);
    y = __fadd_rn(y, __shfl_xor_sync(0xffffffffu, y, 16));
    y = __fadd_rn(y, __shfl_xor_sync(0xffffffffu, y, 8));
    y = __fadd_rn(y, __shfl_xor_sync(0xffffffffu, y, 4));
    y = __fadd_rn(y, __shfl_xor_sync(0xffffffffu, y, 2));
    y = __fadd_rn(y, __shfl_xor_sync(0xffffffffu, y, 1));
    if (lt == 0) *rsqrt_s = rsqrtf(__fadd_rn(__fdiv_rn(y, (float)NORM_H), norm_eps));
  }
  bar_named(1, 256);
  const float r = *rsqrt_s;
  u16* lo = layer_input + (size_t)tok * NORM_H;
#pragma unroll
  for (int q = 0; q < 5; ++q) {
    const int p = 4 * (lt + 256 * q);
    const uint2 rv = *(const uint2*)(rounded_s + p);
    const float o0 = __fmul_rn(__fmul_rn(bf16_lo(rv.x), r), bf16_lo(wv[q].x));
    const float o1 = __fmul_rn(__fmul_rn(bf16_hi(rv.x), r), bf16_hi(wv[q].x));
    const float o2 = __fmul_rn(__fmul_rn(bf16_lo(rv.y), r), bf16_lo(wv[q].y));
    const float o3 = __fmul_rn(__fmul_rn(bf16_hi(rv.y), r), bf16_hi(wv[q].y));
    *(uint2*)(lo + p) = make_uint2(bf16x2_rn(o0, o1), bf16x2_rn(o2, o3));
  }
  if (lt == 0) PROFN(5);
  // layer_input is written: the next kernel may launch. PDL dependents still wait for this grid
  // before reading (the vLLM contract: PDL-launched kernels call griddepcontrol.wait first).
  pdl_trigger();
}

#define NORM_ARGS                                                                                        \
  const float *__restrict__ mixes_p, const float *__restrict__ sqr_p, const float *__restrict__ hc_scale, \
      const float *__restrict__ hc_base, const u16 *__restrict__ residual, const float *__restrict__ pre_mix_in, \
      const u16 *__restrict__ norm_w, float *__restrict__ post_mix, float *__restrict__ comb_mix,            \
      u16 *__restrict__ layer_input, float *__restrict__ pre_mix_out, const int T, const float rms_numel,     \
      const float rms_eps, const float hc_pre_eps, const float sk_eps, const float post_mult,                \
      const int sk_repeat, const float norm_eps

// Fused (one launch after the GEMM): warp 0 = coefficients, warps 1..8 = layer_input. The
// layer_input half runs BEFORE the wait, while the prenorm GEMM (the primary) streams. Safe: the
// GEMM triggers only after its own griddepcontrol.wait, so the post is complete when this starts;
// its reads use ld.global.cg (L2, no stale L1).
extern "C" __global__ void __launch_bounds__(288, 1) mhc_det_norm(NORM_ARGS) {
  __shared__ float mixes_s[24];
  __shared__ __align__(16) u16 rounded_s[NORM_H];
  __shared__ float part_s[64];
  __shared__ float rsqrt_s;
  const int tok = blockIdx.x, tid = threadIdx.x;
  if (tid < 32)
    norm_coef(mixes_s, mixes_p, sqr_p, hc_scale, hc_base, post_mix, comb_mix, pre_mix_out, T, rms_numel, rms_eps,
              hc_pre_eps, sk_eps, post_mult, sk_repeat, tok, tid);
  else
    norm_li(rounded_s, part_s, &rsqrt_s, residual, pre_mix_in, norm_w, layer_input, norm_eps, tok, tid - 32, false);
}

// Split launches (overlap mode): layer_input alone right after the post (its primary: waits
// before reading), and the coefficient half alone after the GEMM on the side stream. Same device
// code as the fused kernel, so the same bits.
extern "C" __global__ void __launch_bounds__(256, 1) mhc_det_norm_li(NORM_ARGS) {
  __shared__ __align__(16) u16 rounded_s[NORM_H];
  __shared__ float part_s[64];
  __shared__ float rsqrt_s;
  norm_li(rounded_s, part_s, &rsqrt_s, residual, pre_mix_in, norm_w, layer_input, norm_eps, blockIdx.x, threadIdx.x,
          true);
}
extern "C" __global__ void __launch_bounds__(32, 1) mhc_det_norm_coef(NORM_ARGS) {
  __shared__ float mixes_s[24];
  norm_coef(mixes_s, mixes_p, sqr_p, hc_scale, hc_base, post_mix, comb_mix, pre_mix_out, T, rms_numel, rms_eps,
            hc_pre_eps, sk_eps, post_mult, sk_repeat, blockIdx.x, threadIdx.x);
}
