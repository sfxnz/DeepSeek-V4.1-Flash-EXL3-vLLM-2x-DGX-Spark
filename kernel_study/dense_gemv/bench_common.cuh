// Benchmark helpers shared by the dense-gemv study extensions.
#pragma once
#include <cstdint>

namespace dgemv_bench {

__device__ __forceinline__ uint4 ld_na(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

// Hashed-order read flush (see benchutil.py for why it is hashed).
__global__ void flush_perm_kernel(const uint4* __restrict__ p, size_t nchunk_pow2, unsigned* out) {
  const int lane = threadIdx.x & 31;
  size_t warp = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5;
  size_t nwarps = ((size_t)gridDim.x * blockDim.x) >> 5;
  unsigned acc = 0;
  for (size_t c = warp; c < nchunk_pow2; c += nwarps) {
    size_t pc = (c * 0x9E3779B1ull) & (nchunk_pow2 - 1);
    const uint4* q = p + pc * 256 + lane;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      uint4 v = ld_na(q + 32 * k);
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
  }
  if (acc == 0x9E3779B9u) out[0] = acc;
}

__global__ void spin_kernel(long long cycles) {
  long long t0 = clock64();
  while (clock64() - t0 < cycles) {
  }
}

// Pure read of the same bytes a GEMV streams (flat grid-stride, the best
// calibrated pattern), as the in-process roofline arm.
template <int U>
__global__ void read_flat(const uint4* __restrict__ p, size_t n16, unsigned* out) {
  size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  unsigned acc = 0;
  size_t i = tid;
  for (; i + (size_t)(U - 1) * stride < n16; i += (size_t)U * stride) {
    uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) v[u] = ld_na(p + i + (size_t)u * stride);
#pragma unroll
    for (int u = 0; u < U; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  for (; i < n16; i += stride) {
    uint4 v = ld_na(p + i);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9E3779B9u) out[0] = acc;
}

}  // namespace dgemv_bench
