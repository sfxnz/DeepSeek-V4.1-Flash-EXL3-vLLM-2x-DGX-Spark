#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/copy.h>
#include <math_constants.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void mhc_pre_big_fuse_with_norm_tilelang_kernel(float* comb_mix, const float* gemm_out_mul, const float* gemm_out_sqrsum, const float* hc_base, const float* hc_scale, bfloat16_t* layer_input, const bfloat16_t* norm_weight, float* post_mix, float* pre_mix_out, const bfloat16_t* residual, int num_tokens);
extern "C" __global__ void __launch_bounds__(96, 1) mhc_pre_big_fuse_with_norm_tilelang_kernel(float* comb_mix, const float* gemm_out_mul, const float* gemm_out_sqrsum, const float* hc_base, const float* hc_scale, bfloat16_t* layer_input, const bfloat16_t* norm_weight, float* post_mix, float* pre_mix_out, const bfloat16_t* residual, int num_tokens) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* mixes_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* xs = ((void*)((char*)buf_dyn_shmem + 96));
  void* output_shared = ((void*)((char*)buf_dyn_shmem + 16480));
  void* w_shared = ((void*)((char*)buf_dyn_shmem + 26720));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 30816));
  void* pre_mix_shared = ((void*)((char*)buf_dyn_shmem + 31072));
  float mixes[1];
  float rms[1];
  float cm[1];
  float row_max[1];
  float row_sum[1];
  float col_sum[1];
  float sumsq_per_pos[16];
  float xl[64];
  float ol[16];
  bfloat16_t rounded[16];
  float sumsq[1];
  float rsqrt_norm[1];
  float w_local[16];
  float ol_1[16];
  bfloat16_t xs_local_cast[8];
  bfloat16_t xs_local_cast_1[8];
  bfloat16_t xs_local_cast_2[8];
  bfloat16_t w_shared_local_cast_3[8];
  bfloat16_t output_shared_local_cast_4[8];
  bfloat16_t layer_input_local_cast_5[8];
  bfloat16_t w_shared_local_cast_6[8];
  bfloat16_t output_shared_local_cast_7[8];
  bfloat16_t layer_input_local_cast_8[8];
  bfloat16_t w_shared_local_cast_9[8];
  bfloat16_t output_shared_local_cast_10[8];
  bfloat16_t layer_input_local_cast_11[8];
  mixes[0] = 0x0p+0f/*0.000000e+00*/;
  rms[0] = 0x0p+0f/*0.000000e+00*/;
  cudaGridDependencySynchronize();
  for (int i_split = 0; i_split < 16; ++i_split) {
    rms[0] = (rms[0] + gemm_out_sqrsum[((((int64_t)i_split) * ((int64_t)num_tokens)) + ((int64_t)((int)blockIdx.x)))]);
  }
  rms[0] = rsqrtf(((rms[0] / 0x1.4p+14f/*2.048000e+04*/) + 0x1.79ca10c924223p-67f/*1.000000e-20*/));
  mixes[0] = 0x0p+0f/*0.000000e+00*/;
  for (int i_split_1 = 0; i_split_1 < 16; ++i_split_1) {
    mixes[0] = (mixes[0] + gemm_out_mul[(((((int64_t)((int)blockIdx.x)) * (int64_t)24) + ((((int64_t)i_split_1) * ((int64_t)num_tokens)) * (int64_t)24)) + (((int64_t)((int)threadIdx.x)) % (int64_t)24))]);
  }
  mixes[0] = (mixes[0] * rms[0]);
  if ((((int)threadIdx.x) / 24) == 0) {
    ((float*)mixes_shared)[(((int)threadIdx.x) % 24)] = mixes[0];
  }
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    if (((int)threadIdx.x) < 4) {
      pre_mix_out[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + ((int64_t)((int)threadIdx.x)))] = ((0x1p+0f/*1.000000e+00*/ / (0x1p+0f/*1.000000e+00*/ + expf((0x0p+0f/*0.000000e+00*/ - ((((float*)mixes_shared)[((int)threadIdx.x)] * hc_scale[0]) + hc_base[((int)threadIdx.x)]))))) + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/);
      post_mix[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + ((int64_t)((int)threadIdx.x)))] = ((0x1p+0f/*1.000000e+00*/ / (0x1p+0f/*1.000000e+00*/ + expf((0x0p+0f/*0.000000e+00*/ - ((((float*)mixes_shared)[(((int)threadIdx.x) + 4)] * hc_scale[1]) + hc_base[(((int)threadIdx.x) + 4)]))))) * 0x1p+1f/*2.000000e+00*/);
    }
    cm[0] = ((((float*)mixes_shared)[((((int)threadIdx.x) & 15) + 8)] * hc_scale[2]) + hc_base[((((int)threadIdx.x) & 15) + 8)]);
    row_max[0] = -CUDART_INF_F;
    row_max[0] = max(row_max[0], cm[0]);
    row_max[0] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<32>>::run(row_max[0]);
    cm[0] = expf((cm[0] - row_max[0]));
    row_sum[0] = 0x0p+0f/*0.000000e+00*/;
    row_sum[0] = (row_sum[0] + cm[0]);
    row_sum[0] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<32>>::run(row_sum[0]);
    cm[0] = ((cm[0] / row_sum[0]) + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/);
    col_sum[0] = 0x0p+0f/*0.000000e+00*/;
    col_sum[0] = (col_sum[0] + cm[0]);
    col_sum[0] = tl::AllReduce<tl::SumOp, 16, 4, 0, tl::NamedBarrier<32>>::run(col_sum[0]);
    cm[0] = (cm[0] / (col_sum[0] + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/));
    for (int __1 = 0; __1 < 19; ++__1) {
      row_sum[0] = 0x0p+0f/*0.000000e+00*/;
      row_sum[0] = (row_sum[0] + cm[0]);
      row_sum[0] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<32>>::run(row_sum[0]);
      cm[0] = (cm[0] / (row_sum[0] + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/));
      col_sum[0] = 0x0p+0f/*0.000000e+00*/;
      col_sum[0] = (col_sum[0] + cm[0]);
      col_sum[0] = tl::AllReduce<tl::SumOp, 16, 4, 0, tl::NamedBarrier<32>>::run(col_sum[0]);
      cm[0] = (cm[0] / (col_sum[0] + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/));
    }
    if ((((int)threadIdx.x) >> 4) == 0) {
      comb_mix[((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) & (int64_t)15))] = cm[0];
    }
  } else {
    if (((int)threadIdx.x) < 36) {
      float condval;
      if ((((int)threadIdx.x) == 32)) {
        condval = 0x1p+0f/*1.000000e+00*/;
      } else {
        condval = 0x0p+0f/*0.000000e+00*/;
      }
      ((float*)pre_mix_shared)[(((int)threadIdx.x) - 32)] = condval;
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(sumsq_per_pos + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    tl::__sync_thread_partial(3, 64);
    #pragma unroll
    for (int i_1 = 0; i_1 < 8; ++i_1) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)xs)[(((i_1 >> 1) * 1024) + (((((i_1 * 64) + ((int)threadIdx.x)) + 96) & 127) * 8))])), (&(residual[(((((int64_t)((int)blockIdx.x)) * (int64_t)20480) + ((((int64_t)i_1) >> (int64_t)1) * (int64_t)5120)) + (((((((int64_t)i_1) * (int64_t)64) + ((int64_t)((int)threadIdx.x))) + (int64_t)96) & (int64_t)127) * (int64_t)8))])));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_2 = 0; i_2 < 8; ++i_2) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)xs)[((((i_2 >> 1) * 1024) + (((((i_2 * 64) + ((int)threadIdx.x)) + 96) & 127) * 8)) + 4096)])), (&(residual[((((((int64_t)((int)blockIdx.x)) * (int64_t)20480) + ((((int64_t)i_2) >> (int64_t)1) * (int64_t)5120)) + (((((((int64_t)i_2) * (int64_t)64) + ((int64_t)((int)threadIdx.x))) + (int64_t)96) & (int64_t)127) * (int64_t)8)) + (int64_t)1024)])));
    }
    tl::cp_async_commit();
    for (int i0_h = 0; i0_h < 3; ++i0_h) {
      tl::cp_async_wait<1>();
      tl::__sync_thread_partial(3, 64);
      #pragma unroll
      for (int i_3 = 0; i_3 < 8; ++i_3) {
        *(uint4*)(xs_local_cast + 0) = *(uint4*)(((bfloat16_t*)xs) + (((((i0_h & 1) * 4096) + (i_3 * 512)) + (((int)threadIdx.x) * 8)) - 256));
        for (int vec = 0; vec < 2; ++vec) {
          float4 __2;
          uint2 v_ = *(uint2*)(xs_local_cast + (vec * 4));
          ((float2*)(&__2))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[0]);
          ((float2*)(&__2))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[1]);
          *(float4*)(xl + ((i_3 * 8) + (vec * 4))) = __2;
        }
      }
      tl::__sync_thread_partial(3, 64);
      #pragma unroll
      for (int i_4 = 0; i_4 < 8; ++i_4) {
        tl::cp_async_gs<16>((&(((bfloat16_t*)xs)[((((i0_h & 1) * 4096) + ((i_4 >> 1) * 1024)) + (((((i_4 * 64) + ((int)threadIdx.x)) + 96) & 127) * 8))])), (&(residual[(((((((int64_t)((int)blockIdx.x)) * (int64_t)20480) + ((((int64_t)i_4) >> (int64_t)1) * (int64_t)5120)) + (((int64_t)i0_h) * (int64_t)1024)) + (((((((int64_t)i_4) * (int64_t)64) + ((int64_t)((int)threadIdx.x))) + (int64_t)96) & (int64_t)127) * (int64_t)8)) + (int64_t)2048)])));
      }
      tl::cp_async_commit();
      #pragma unroll
      for (int i_5 = 0; i_5 < 4; ++i_5) {
        float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(ol + (i_5 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
      }
      tl::__sync_thread_partial(3, 64);
      for (int i_hc = 0; i_hc < 4; ++i_hc) {
        float pre = ((float*)pre_mix_shared)[i_hc];
        #pragma unroll
        for (int i_6 = 0; i_6 < 16; ++i_6) {
          ol[i_6] = (ol[i_6] + (pre * xl[((i_hc * 16) + i_6)]));
        }
      }
      #pragma unroll
      for (int i_7 = 0; i_7 < 4; ++i_7) {
        uint2 __3;
        float4 v__1 = *(float4*)(ol + (i_7 * 4));
        (reinterpret_cast<__nv_bfloat162*>(&__3))[0] = __float22bfloat162_rn(((float2*)(&v__1))[0]);
        (reinterpret_cast<__nv_bfloat162*>(&__3))[1] = __float22bfloat162_rn(((float2*)(&v__1))[1]);
        *(uint2*)(rounded + (i_7 * 4)) = __3;
      }
      #pragma unroll
      for (int i_8 = 0; i_8 < 4; ++i_8) {
        float4 __4;
        uint2 v__2 = *(uint2*)(rounded + (i_8 * 4));
        ((float2*)(&__4))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__2))[0]);
        ((float2*)(&__4))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__2))[1]);
        float4 value = __4;
        float4 __5;
          float4 v__3 = *(float4*)(sumsq_per_pos + (i_8 * 4));
          *(float2*)(&(__5.x)) = tl::fma2(*(float2*)(&(value.x)), *(float2*)(&(value.x)), *(float2*)(&(v__3.x)));
          *(float2*)(&(__5.z)) = tl::fma2(*(float2*)(&(value.z)), *(float2*)(&(value.z)), *(float2*)(&(v__3.z)));
        *(float4*)(sumsq_per_pos + (i_8 * 4)) = __5;
        *(uint2*)(((bfloat16_t*)output_shared) + (((((i0_h * 1024) + ((i_8 >> 1) * 512)) + (((int)threadIdx.x) * 8)) + ((i_8 & 1) * 4)) - 256)) = *(uint2*)(rounded + (i_8 * 4));
      }
    }
    tl::cp_async_wait<1>();
    tl::__sync_thread_partial(3, 64);
    #pragma unroll
    for (int i_9 = 0; i_9 < 8; ++i_9) {
      *(uint4*)(xs_local_cast_1 + 0) = *(uint4*)(((bfloat16_t*)xs) + (((i_9 * 512) + (((int)threadIdx.x) * 8)) + 3840));
      for (int vec_1 = 0; vec_1 < 2; ++vec_1) {
        float4 __6;
        uint2 v__4 = *(uint2*)(xs_local_cast_1 + (vec_1 * 4));
        ((float2*)(&__6))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__4))[0]);
        ((float2*)(&__6))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__4))[1]);
        *(float4*)(xl + ((i_9 * 8) + (vec_1 * 4))) = __6;
      }
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 4; ++i_10) {
      float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(ol + (i_10 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
    }
    for (int i_hc_1 = 0; i_hc_1 < 4; ++i_hc_1) {
      float pre_1 = ((float*)pre_mix_shared)[i_hc_1];
      #pragma unroll
      for (int i_11 = 0; i_11 < 16; ++i_11) {
        ol[i_11] = (ol[i_11] + (pre_1 * xl[((i_hc_1 * 16) + i_11)]));
      }
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 4; ++i_12) {
      uint2 __7;
      float4 v__5 = *(float4*)(ol + (i_12 * 4));
      (reinterpret_cast<__nv_bfloat162*>(&__7))[0] = __float22bfloat162_rn(((float2*)(&v__5))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__7))[1] = __float22bfloat162_rn(((float2*)(&v__5))[1]);
      *(uint2*)(rounded + (i_12 * 4)) = __7;
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 4; ++i_13) {
      float4 __8;
      uint2 v__6 = *(uint2*)(rounded + (i_13 * 4));
      ((float2*)(&__8))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__6))[0]);
      ((float2*)(&__8))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__6))[1]);
      float4 value_1 = __8;
      float4 __9;
        float4 v__7 = *(float4*)(sumsq_per_pos + (i_13 * 4));
        *(float2*)(&(__9.x)) = tl::fma2(*(float2*)(&(value_1.x)), *(float2*)(&(value_1.x)), *(float2*)(&(v__7.x)));
        *(float2*)(&(__9.z)) = tl::fma2(*(float2*)(&(value_1.z)), *(float2*)(&(value_1.z)), *(float2*)(&(v__7.z)));
      *(float4*)(sumsq_per_pos + (i_13 * 4)) = __9;
      *(uint2*)(((bfloat16_t*)output_shared) + (((((i_13 >> 1) * 512) + (((int)threadIdx.x) * 8)) + ((i_13 & 1) * 4)) + 2816)) = *(uint2*)(rounded + (i_13 * 4));
    }
    tl::cp_async_wait<0>();
    tl::__sync_thread_partial(3, 64);
    #pragma unroll
    for (int i_14 = 0; i_14 < 8; ++i_14) {
      *(uint4*)(xs_local_cast_2 + 0) = *(uint4*)(((bfloat16_t*)xs) + (((i_14 * 512) + (((int)threadIdx.x) * 8)) - 256));
      for (int vec_2 = 0; vec_2 < 2; ++vec_2) {
        float4 __10;
        uint2 v__8 = *(uint2*)(xs_local_cast_2 + (vec_2 * 4));
        ((float2*)(&__10))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__8))[0]);
        ((float2*)(&__10))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__8))[1]);
        *(float4*)(xl + ((i_14 * 8) + (vec_2 * 4))) = __10;
      }
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 4; ++i_15) {
      float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(ol + (i_15 * 4)) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
    }
    for (int i_hc_2 = 0; i_hc_2 < 4; ++i_hc_2) {
      float pre_2 = ((float*)pre_mix_shared)[i_hc_2];
      #pragma unroll
      for (int i_16 = 0; i_16 < 16; ++i_16) {
        ol[i_16] = (ol[i_16] + (pre_2 * xl[((i_hc_2 * 16) + i_16)]));
      }
    }
    #pragma unroll
    for (int i_17 = 0; i_17 < 4; ++i_17) {
      uint2 __11;
      float4 v__9 = *(float4*)(ol + (i_17 * 4));
      (reinterpret_cast<__nv_bfloat162*>(&__11))[0] = __float22bfloat162_rn(((float2*)(&v__9))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__11))[1] = __float22bfloat162_rn(((float2*)(&v__9))[1]);
      *(uint2*)(rounded + (i_17 * 4)) = __11;
    }
    #pragma unroll
    for (int i_18 = 0; i_18 < 4; ++i_18) {
      float4 __12;
      uint2 v__10 = *(uint2*)(rounded + (i_18 * 4));
      ((float2*)(&__12))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__10))[0]);
      ((float2*)(&__12))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__10))[1]);
      float4 value_2 = __12;
      float4 __13;
        float4 v__11 = *(float4*)(sumsq_per_pos + (i_18 * 4));
        *(float2*)(&(__13.x)) = tl::fma2(*(float2*)(&(value_2.x)), *(float2*)(&(value_2.x)), *(float2*)(&(v__11.x)));
        *(float2*)(&(__13.z)) = tl::fma2(*(float2*)(&(value_2.z)), *(float2*)(&(value_2.z)), *(float2*)(&(v__11.z)));
      *(float4*)(sumsq_per_pos + (i_18 * 4)) = __13;
      *(uint2*)(((bfloat16_t*)output_shared) + (((((i_18 >> 1) * 512) + (((int)threadIdx.x) * 8)) + ((i_18 & 1) * 4)) + 3840)) = *(uint2*)(rounded + (i_18 * 4));
    }
    sumsq[0] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv = 0; rv < 16; ++rv) {
      sumsq[0] = (sumsq[0] + sumsq_per_pos[(((rv & 1) * 8) + (rv >> 1))]);
    }
    tl::__sync_thread_partial(3, 64);
    sumsq[0] = tl::AllReduce<tl::SumOp, 64, 1, 32, tl::NamedBarrier<64>>::run(sumsq[0], (&(((float*)workspace)[0])));
    rsqrt_norm[0] = rsqrtf(((sumsq[0] / 0x1.4p+12f/*5.120000e+03*/) + 0x1.79ca10c924223p-67f/*1.000000e-20*/));
    #pragma unroll
    for (int i_19 = 0; i_19 < 2; ++i_19) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)w_shared)[(((i_19 * 512) + (((int)threadIdx.x) * 8)) - 256)])), (&(norm_weight[(((i_19 * 512) + (((int)threadIdx.x) * 8)) - 256)])));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_20 = 0; i_20 < 2; ++i_20) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)w_shared)[(((i_20 * 512) + (((int)threadIdx.x) * 8)) + 768)])), (&(norm_weight[(((i_20 * 512) + (((int)threadIdx.x) * 8)) + 768)])));
    }
    tl::cp_async_commit();
    for (int i0_h_1 = 0; i0_h_1 < 3; ++i0_h_1) {
      tl::cp_async_wait<1>();
      tl::__sync_thread_partial(3, 64);
      #pragma unroll
      for (int i_21 = 0; i_21 < 2; ++i_21) {
        *(uint4*)(w_shared_local_cast_3 + 0) = *(uint4*)(((bfloat16_t*)w_shared) + (((((i0_h_1 & 1) * 1024) + (i_21 * 512)) + (((int)threadIdx.x) * 8)) - 256));
        for (int vec_3 = 0; vec_3 < 2; ++vec_3) {
          float4 __14;
          uint2 v__12 = *(uint2*)(w_shared_local_cast_3 + (vec_3 * 4));
          ((float2*)(&__14))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__12))[0]);
          ((float2*)(&__14))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__12))[1]);
          *(float4*)(w_local + ((i_21 * 8) + (vec_3 * 4))) = __14;
        }
      }
      tl::__sync_thread_partial(3, 64);
      #pragma unroll
      for (int i_22 = 0; i_22 < 2; ++i_22) {
        tl::cp_async_gs<16>((&(((bfloat16_t*)w_shared)[(((((i0_h_1 & 1) * 1024) + (i_22 * 512)) + (((int)threadIdx.x) * 8)) - 256)])), (&(norm_weight[((((i0_h_1 * 1024) + (i_22 * 512)) + (((int)threadIdx.x) * 8)) + 1792)])));
      }
      tl::cp_async_commit();
      tl::__sync_thread_partial(3, 64);
      #pragma unroll
      for (int i_23 = 0; i_23 < 2; ++i_23) {
        *(uint4*)(output_shared_local_cast_4 + 0) = *(uint4*)(((bfloat16_t*)output_shared) + ((((i0_h_1 * 1024) + (i_23 * 512)) + (((int)threadIdx.x) * 8)) - 256));
        for (int vec_4 = 0; vec_4 < 2; ++vec_4) {
          float4 __15;
            float4 __16;
              float4 __17;
              uint2 v__13 = *(uint2*)(output_shared_local_cast_4 + (vec_4 * 4));
              ((float2*)(&__17))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__13))[0]);
              ((float2*)(&__17))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__13))[1]);
              float4 v__14 = make_float4(rsqrt_norm[0], rsqrt_norm[0], rsqrt_norm[0], rsqrt_norm[0]);
              *(float2*)(&(__16.x)) = tl::mul2(*(float2*)(&(__17.x)), *(float2*)(&(v__14.x)));
              *(float2*)(&(__16.z)) = tl::mul2(*(float2*)(&(__17.z)), *(float2*)(&(v__14.z)));
            float4 v__15 = *(float4*)(w_local + ((i_23 * 8) + (vec_4 * 4)));
            *(float2*)(&(__15.x)) = tl::mul2(*(float2*)(&(__16.x)), *(float2*)(&(v__15.x)));
            *(float2*)(&(__15.z)) = tl::mul2(*(float2*)(&(__16.z)), *(float2*)(&(v__15.z)));
          *(float4*)(ol_1 + ((i_23 * 8) + (vec_4 * 4))) = __15;
        }
      }
      #pragma unroll
      for (int i_24 = 0; i_24 < 2; ++i_24) {
        for (int vec_5 = 0; vec_5 < 2; ++vec_5) {
          uint2 __18;
          float4 v__16 = *(float4*)(ol_1 + ((i_24 * 8) + (vec_5 * 4)));
          (reinterpret_cast<__nv_bfloat162*>(&__18))[0] = __float22bfloat162_rn(((float2*)(&v__16))[0]);
          (reinterpret_cast<__nv_bfloat162*>(&__18))[1] = __float22bfloat162_rn(((float2*)(&v__16))[1]);
          *(uint2*)(layer_input_local_cast_5 + (vec_5 * 4)) = __18;
        }
        *(uint4*)(layer_input + (((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)i0_h_1) * (int64_t)1024)) + (((int64_t)i_24) * (int64_t)512)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)) - (int64_t)256)) = *(uint4*)(layer_input_local_cast_5 + 0);
      }
    }
    tl::cp_async_wait<1>();
    tl::__sync_thread_partial(3, 64);
    #pragma unroll
    for (int i_25 = 0; i_25 < 2; ++i_25) {
      *(uint4*)(w_shared_local_cast_6 + 0) = *(uint4*)(((bfloat16_t*)w_shared) + (((i_25 * 512) + (((int)threadIdx.x) * 8)) + 768));
      for (int vec_6 = 0; vec_6 < 2; ++vec_6) {
        float4 __19;
        uint2 v__17 = *(uint2*)(w_shared_local_cast_6 + (vec_6 * 4));
        ((float2*)(&__19))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__17))[0]);
        ((float2*)(&__19))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__17))[1]);
        *(float4*)(w_local + ((i_25 * 8) + (vec_6 * 4))) = __19;
      }
    }
    #pragma unroll
    for (int i_26 = 0; i_26 < 2; ++i_26) {
      *(uint4*)(output_shared_local_cast_7 + 0) = *(uint4*)(((bfloat16_t*)output_shared) + (((i_26 * 512) + (((int)threadIdx.x) * 8)) + 2816));
      for (int vec_7 = 0; vec_7 < 2; ++vec_7) {
        float4 __20;
          float4 __21;
            float4 __22;
            uint2 v__18 = *(uint2*)(output_shared_local_cast_7 + (vec_7 * 4));
            ((float2*)(&__22))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__18))[0]);
            ((float2*)(&__22))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__18))[1]);
            float4 v__19 = make_float4(rsqrt_norm[0], rsqrt_norm[0], rsqrt_norm[0], rsqrt_norm[0]);
            *(float2*)(&(__21.x)) = tl::mul2(*(float2*)(&(__22.x)), *(float2*)(&(v__19.x)));
            *(float2*)(&(__21.z)) = tl::mul2(*(float2*)(&(__22.z)), *(float2*)(&(v__19.z)));
          float4 v__20 = *(float4*)(w_local + ((i_26 * 8) + (vec_7 * 4)));
          *(float2*)(&(__20.x)) = tl::mul2(*(float2*)(&(__21.x)), *(float2*)(&(v__20.x)));
          *(float2*)(&(__20.z)) = tl::mul2(*(float2*)(&(__21.z)), *(float2*)(&(v__20.z)));
        *(float4*)(ol_1 + ((i_26 * 8) + (vec_7 * 4))) = __20;
      }
    }
    #pragma unroll
    for (int i_27 = 0; i_27 < 2; ++i_27) {
      for (int vec_8 = 0; vec_8 < 2; ++vec_8) {
        uint2 __23;
        float4 v__21 = *(float4*)(ol_1 + ((i_27 * 8) + (vec_8 * 4)));
        (reinterpret_cast<__nv_bfloat162*>(&__23))[0] = __float22bfloat162_rn(((float2*)(&v__21))[0]);
        (reinterpret_cast<__nv_bfloat162*>(&__23))[1] = __float22bfloat162_rn(((float2*)(&v__21))[1]);
        *(uint2*)(layer_input_local_cast_8 + (vec_8 * 4)) = __23;
      }
      *(uint4*)(layer_input + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)i_27) * (int64_t)512)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)) + (int64_t)2816)) = *(uint4*)(layer_input_local_cast_8 + 0);
    }
    tl::cp_async_wait<0>();
    tl::__sync_thread_partial(3, 64);
    #pragma unroll
    for (int i_28 = 0; i_28 < 2; ++i_28) {
      *(uint4*)(w_shared_local_cast_9 + 0) = *(uint4*)(((bfloat16_t*)w_shared) + (((i_28 * 512) + (((int)threadIdx.x) * 8)) - 256));
      for (int vec_9 = 0; vec_9 < 2; ++vec_9) {
        float4 __24;
        uint2 v__22 = *(uint2*)(w_shared_local_cast_9 + (vec_9 * 4));
        ((float2*)(&__24))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__22))[0]);
        ((float2*)(&__24))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__22))[1]);
        *(float4*)(w_local + ((i_28 * 8) + (vec_9 * 4))) = __24;
      }
    }
    #pragma unroll
    for (int i_29 = 0; i_29 < 2; ++i_29) {
      *(uint4*)(output_shared_local_cast_10 + 0) = *(uint4*)(((bfloat16_t*)output_shared) + (((i_29 * 512) + (((int)threadIdx.x) * 8)) + 3840));
      for (int vec_10 = 0; vec_10 < 2; ++vec_10) {
        float4 __25;
          float4 __26;
            float4 __27;
            uint2 v__23 = *(uint2*)(output_shared_local_cast_10 + (vec_10 * 4));
            ((float2*)(&__27))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__23))[0]);
            ((float2*)(&__27))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__23))[1]);
            float4 v__24 = make_float4(rsqrt_norm[0], rsqrt_norm[0], rsqrt_norm[0], rsqrt_norm[0]);
            *(float2*)(&(__26.x)) = tl::mul2(*(float2*)(&(__27.x)), *(float2*)(&(v__24.x)));
            *(float2*)(&(__26.z)) = tl::mul2(*(float2*)(&(__27.z)), *(float2*)(&(v__24.z)));
          float4 v__25 = *(float4*)(w_local + ((i_29 * 8) + (vec_10 * 4)));
          *(float2*)(&(__25.x)) = tl::mul2(*(float2*)(&(__26.x)), *(float2*)(&(v__25.x)));
          *(float2*)(&(__25.z)) = tl::mul2(*(float2*)(&(__26.z)), *(float2*)(&(v__25.z)));
        *(float4*)(ol_1 + ((i_29 * 8) + (vec_10 * 4))) = __25;
      }
    }
    #pragma unroll
    for (int i_30 = 0; i_30 < 2; ++i_30) {
      for (int vec_11 = 0; vec_11 < 2; ++vec_11) {
        uint2 __28;
        float4 v__26 = *(float4*)(ol_1 + ((i_30 * 8) + (vec_11 * 4)));
        (reinterpret_cast<__nv_bfloat162*>(&__28))[0] = __float22bfloat162_rn(((float2*)(&v__26))[0]);
        (reinterpret_cast<__nv_bfloat162*>(&__28))[1] = __float22bfloat162_rn(((float2*)(&v__26))[1]);
        *(uint2*)(layer_input_local_cast_11 + (vec_11 * 4)) = __28;
      }
      *(uint4*)(layer_input + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)i_30) * (int64_t)512)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)) + (int64_t)3840)) = *(uint4*)(layer_input_local_cast_11 + 0);
    }
  }
  cudaTriggerProgrammaticLaunchCompletion();
}

