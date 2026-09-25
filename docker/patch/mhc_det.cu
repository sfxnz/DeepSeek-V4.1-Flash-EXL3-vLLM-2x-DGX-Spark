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
// fp32 -> tf32 bits, round to nearest even (finite inputs; fn has no NaN/Inf/denormals).
DEV u32 rne_tf32(u32 b) { return (b + 0xFFFu + ((b >> 13) & 1u)) & ~0x1FFFu; }

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
#else
#define PROF(i)
#endif

#define MHC_SPLITS 16
#define STAGE_KB 2
#define FS_PAD 4           // f32 fn row pad in smem: rows start 4 banks apart
#define XS_PAD 8           // bf16 x row pad in smem: rows start 4 banks apart
#define PK_KB_BYTES 1280   // packed fn bytes per (split, n-tile, k-block)

// Packed fn (PK): per (split, n-tile, k-block) the RNE tf32 bits (top 19 bits of fp32) of the
// 32 lanes' 16 B-fragment values: hi [2][32 lanes][8 x u16] = bits 31..16, lo [32 lanes][16 x
// 4 bit] = bits 15..13. Lane (g, t) value i: ks = i >> 1, k = kb*64 + ks*8 + t + 4*(i & 1),
// row = n-tile*8 + g. 1280 B per k-block instead of 2048 B; the MMA sees identical bits.
// Stage-major: [stage][cta = split*3 + n-tile][nkb(stage) x 1280 B], so each stage of all 48
// CTAs is one contiguous region (stream_probe.json: 6.6 vs 7.5 us for 48 x 25.6 KB cold).
//
// grid (3 n-tiles, 16 splits), block 32. x [T, K] bf16, fn [24, K] f32 or packed
// -> mixes [16, T, 24] f32, sqr [16, T] f32. ROW1: T may exceed 8 (rows g + 8 are live).
#ifndef PREFETCH_STAGES
#define PREFETCH_STAGES 4  // fn stages issued before griddepcontrol.wait (sweep_prefetch.json)
#endif
template <bool PK, bool ROW1>
DEV void gemm_body(const u16* __restrict__ x, const void* __restrict__ fnsrc, float* __restrict__ mixes,
                   float* __restrict__ sqr, const int T, const int K) {
  extern __shared__ __align__(128) unsigned char smem[];
  const int nt = blockIdx.x, s = blockIdx.y;
  const int cta = s * 3 + nt;
  const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
  const int kbps = K / (64 * MHC_SPLITS);
  const int kspan = kbps * 64;
  const int k0 = s * kspan;
  const int fs = kspan + FS_PAD;
  const int xs = kspan + XS_PAD;
  const int fn_bytes = PK ? kbps * PK_KB_BYTES : 8 * fs * 4;
  unsigned char* fsm = smem;
  u16* xsm = (u16*)(smem + fn_bytes);
  const int nstage = (kbps + STAGE_KB - 1) / STAGE_KB;
  u64* bars = (u64*)(smem + fn_bytes + ((T * xs * 2 + 15) & ~15));
  auto issue_fn = [&](int st) {
    const int kb_lo = st * STAGE_KB, nkb = min(STAGE_KB, kbps - kb_lo);
    if (PK) {
      const unsigned char* src =
          (const unsigned char*)fnsrc + ((size_t)MHC_SPLITS * 3 * kb_lo + (size_t)cta * nkb) * PK_KB_BYTES;
      mbar_arrive_expect_tx(&bars[st], (u32)(nkb * PK_KB_BYTES));
      bulk_g2s(fsm + kb_lo * PK_KB_BYTES, src, nkb * PK_KB_BYTES, &bars[st]);
    } else {
      mbar_arrive_expect_tx(&bars[st], (u32)(8 * nkb * 256));
      for (int r = 0; r < 8; ++r)
        bulk_g2s(fsm + (r * fs + kb_lo * 64) * 4, (const float*)fnsrc + (size_t)(nt * 8 + r) * K + k0 + kb_lo * 64,
                 nkb * 256, &bars[st]);
    }
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
  const float* fr = (const float*)fsm + g * fs;
  for (int st = 0; st < nstage; ++st) {
    mbar_wait(&bars[st], 0);
    if (st < 12) PROF(4 + (st >> 1));
    const int kb_lo = st * STAGE_KB, kb_hi = min(kb_lo + STAGE_KB, kbps);
#pragma unroll
    for (int j = 0; j < STAGE_KB; ++j) {
      const int kb = kb_lo + j;
      if (kb >= kb_hi) break;
      u32 hw[8];
      uint2 lo = make_uint2(0u, 0u);
      if (PK) {
        const unsigned char* blk = fsm + kb * PK_KB_BYTES;
        const uint4 h0 = *(const uint4*)(blk + lane * 16);
        const uint4 h1 = *(const uint4*)(blk + 512 + lane * 16);
        lo = *(const uint2*)(blk + 1024 + lane * 8);
        hw[0] = h0.x; hw[1] = h0.y; hw[2] = h0.z; hw[3] = h0.w;
        hw[4] = h1.x; hw[5] = h1.y; hw[6] = h1.z; hw[7] = h1.w;
      }
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const int k = kb * 64 + ks * 8 + t;
        const float fa0 = r0 ? bf16f(xr0[k]) : 0.f;
        const float fa2 = r0 ? bf16f(xr0[k + 4]) : 0.f;
        const float fa1 = r1 ? bf16f(xr1[k]) : 0.f;
        const float fa3 = r1 ? bf16f(xr1[k + 4]) : 0.f;
        u32 b0, b1;
        if (PK) {
          const u32 w = hw[ks];                                      // values 2ks (low), 2ks+1 (high)
          const u32 nib = (ks < 4 ? lo.x : lo.y) >> (8 * (ks & 3));  // nibbles 2ks, 2ks+1
          b0 = (w << 16) | ((nib & 7u) << 13);
          b1 = (w & 0xffff0000u) | (((nib >> 4) & 7u) << 13);
        } else {
          b0 = rne_tf32(__float_as_uint(fr[k]));
          b1 = rne_tf32(__float_as_uint(fr[k + 4]));
        }
        mma_tf32(d, __float_as_uint(fa0), __float_as_uint(fa1), __float_as_uint(fa2), __float_as_uint(fa3), b0, b1);
        if (nt == 0) {
          acc0 = __fadd_rn(acc0, __fmaf_rn(fa0, fa0, __fmul_rn(fa2, fa2)));
          if (ROW1) acc1 = __fadd_rn(acc1, __fmaf_rn(fa1, fa1, __fmul_rn(fa3, fa3)));
        }
      }
    }
  }
  PROF(10);
  pdl_trigger();
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

#define GEMM_KERNEL(NAME, PK, ROW1)                                                                     \
  extern "C" __global__ void __launch_bounds__(32, 1)                                                  \
      NAME(const u16* __restrict__ x, const void* __restrict__ fn, float* __restrict__ mixes,          \
           float* __restrict__ sqr, int T, int K) {                                                    \
    gemm_body<PK, ROW1>(x, fn, mixes, sqr, T, K);                                                      \
  }
GEMM_KERNEL(mhc_det_gemm_f32_t8, false, false)
GEMM_KERNEL(mhc_det_gemm_f32_t16, false, true)
GEMM_KERNEL(mhc_det_gemm_pk_t8, true, false)
GEMM_KERNEL(mhc_det_gemm_pk_t16, true, true)

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
