#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/copy_sm90.h>
#include <tl_templates/cuda/copy_sm100.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void mhc_post_tilelang_kernel(const float* a, const bfloat16_t* b, const float* c, const bfloat16_t* d, bfloat16_t* x, int num_tokens);
extern "C" __global__ void __launch_bounds__(128, 1) mhc_post_tilelang_kernel(const float* a, const bfloat16_t* b, const float* c, const bfloat16_t* d, bfloat16_t* x, int num_tokens) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* b_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* d_shared = ((void*)((char*)buf_dyn_shmem + 8192));
  float a_local[16];
  float c_local[4];
  float b_local[32];
  bfloat16_t b_shared_local_cast[8];
  bfloat16_t d_shared_local_cast_1[8];
  float d_local[8];
  float x_local[32];
  bfloat16_t x_local_cast_2[8];
  cudaGridDependencySynchronize();
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    *(ulonglong4*)(a_local + (i * 8)) = tl::load_global_256(&(*(ulonglong4*)(a + ((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)i) * (int64_t)8)))));
  }
  *(float4*)(c_local + 0) = *(float4*)(c + (((int64_t)((int)blockIdx.x)) * (int64_t)4));
  for (int i0_h = 0; i0_h < 5; ++i0_h) {
    #pragma unroll
    for (int i_1 = 0; i_1 < 4; ++i_1) {
      *(uint4*)(((bfloat16_t*)b_shared) + ((i_1 * 1024) + (((int)threadIdx.x) * 8))) = *(uint4*)(b + ((((((int64_t)((int)blockIdx.x)) * (int64_t)20480) + (((int64_t)i_1) * (int64_t)5120)) + (((int64_t)i0_h) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
    }
    *(uint4*)(((bfloat16_t*)d_shared) + (((int)threadIdx.x) * 8)) = *(uint4*)(d + (((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)i0_h) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      *(uint4*)(b_shared_local_cast + 0) = *(uint4*)(((bfloat16_t*)b_shared) + ((i_2 * 1024) + (((int)threadIdx.x) * 8)));
      for (int vec = 0; vec < 2; ++vec) {
        float4 __1;
        uint2 v_ = *(uint2*)(b_shared_local_cast + (vec * 4));
        ((float2*)(&__1))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[0]);
        ((float2*)(&__1))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[1]);
        *(float4*)(b_local + ((i_2 * 8) + (vec * 4))) = __1;
      }
    }
    *(uint4*)(d_shared_local_cast_1 + 0) = *(uint4*)(((bfloat16_t*)d_shared) + (((int)threadIdx.x) * 8));
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      float4 __2;
      uint2 v__1 = *(uint2*)(d_shared_local_cast_1 + (i_3 * 4));
      ((float2*)(&__2))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__1))[0]);
      ((float2*)(&__2))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__1))[1]);
      *(float4*)(d_local + (i_3 * 4)) = __2;
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 32; ++i_4) {
      x_local[i_4] = (c_local[(i_4 >> 3)] * d_local[(i_4 & 7)]);
      for (int i_hci = 0; i_hci < 4; ++i_hci) {
        x_local[i_4] = (x_local[i_4] + (a_local[((i_hci * 4) + (i_4 >> 3))] * b_local[((i_hci * 8) + (i_4 & 7))]));
      }
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      for (int vec_1 = 0; vec_1 < 2; ++vec_1) {
        uint2 __3;
        float4 v__2 = *(float4*)(x_local + ((i_5 * 8) + (vec_1 * 4)));
        (reinterpret_cast<__nv_bfloat162*>(&__3))[0] = __float22bfloat162_rn(((float2*)(&v__2))[0]);
        (reinterpret_cast<__nv_bfloat162*>(&__3))[1] = __float22bfloat162_rn(((float2*)(&v__2))[1]);
        *(uint2*)(x_local_cast_2 + (vec_1 * 4)) = __3;
      }
      *(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)20480) + (((int64_t)i_5) * (int64_t)5120)) + (((int64_t)i0_h) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8))) = *(uint4*)(x_local_cast_2 + 0);
    }
  }
  cudaTriggerProgrammaticLaunchCompletion();
}

