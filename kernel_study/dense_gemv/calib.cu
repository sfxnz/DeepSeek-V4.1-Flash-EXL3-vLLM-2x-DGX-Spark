// Streaming-read calibration for GB10 (sm_121a): what does a pure weight read
// of the dense decode shapes cost, cold L2, as a function of the access
// pattern, CTA count and bytes in flight? This sets the per-shape roofline the
// GEMV kernel is judged against (the chip's 250 GB/s is only reachable on
// long streams; a 6-22 MB kernel pays ramp and tail).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

__device__ __forceinline__ uint4 ld_na(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

__device__ __forceinline__ uint4 ld_na_pf256(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

template <bool PF>
__device__ __forceinline__ uint4 ld16(const void* p) {
  if constexpr (PF) return ld_na_pf256(p);
  else return ld_na(p);
}

// Read-reduce flush: evicts L2 with clean lines (a write flush would leave
// dirty lines whose write-back contends with the measured kernel).
__global__ void flush_kernel(const uint4* __restrict__ p, size_t n16, unsigned* out) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  unsigned acc = 0;
  for (; i < n16; i += stride) {
    uint4 v = ld_na(p + i);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9E3779B9u) out[0] = acc;
}

// Permuted read-flush: 4 KiB chunks visited in a multiplicative-hash order so no
// sequential stream is left running at the end of the flush (a stream
// prefetcher following a linear sweep was measured to warm the buffer that
// sits right after the flush buffer in memory: 75.8 vs 95.2 us for 21 MB).
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

__global__ void flush_write_kernel(uint4* __restrict__ p, size_t n16, unsigned seed) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n16; i += stride) p[i] = make_uint4(seed, (unsigned)i, seed ^ (unsigned)i, 0u);
}

// Busy-wait spacer that touches no memory (keeps the host ahead in warm runs).
__global__ void spin_kernel(long long cycles) {
  long long t0 = clock64();
  while (clock64() - t0 < cycles) {
  }
}

// (a) flat grid-stride, U loads in flight per thread.
template <int U, bool PF>
__global__ void read_flat(const uint4* __restrict__ p, size_t n16, unsigned* out) {
  size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  unsigned acc = 0;
  size_t i = tid;
  for (; i + (size_t)(U - 1) * stride < n16; i += (size_t)U * stride) {
    uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) v[u] = ld16<PF>(p + i + (size_t)u * stride);
#pragma unroll
    for (int u = 0; u < U; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  for (; i < n16; i += stride) {
    uint4 v = ld16<PF>(p + i);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9E3779B9u) out[0] = acc;
}

// (b) GEMV-tile pattern on a row-major [N, K] byte matrix: one warp per
// 16-row tile (tiles strided over warps), lane = 4*r + q reads row r and r+8
// at column 16*q + 64*j (8 rows x 64 B per instruction), D chunks in flight.
template <int D, bool PF>
__global__ void read_tiles(const uint8_t* __restrict__ w, int N, int K, unsigned* out) {
  const int lane = threadIdx.x & 31;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarps = (gridDim.x * blockDim.x) >> 5;
  const int r = lane >> 2, q = lane & 3;
  const int ntiles = N / 16;
  const int nchunks = K / 64;
  unsigned acc = 0;
  for (int t = warp; t < ntiles; t += nwarps) {
    const uint8_t* row0 = w + (size_t)(t * 16 + r) * K + 16 * q;
    const uint8_t* row1 = row0 + (size_t)8 * K;
    int j = 0;
    for (; j + D <= nchunks; j += D) {
      uint4 a[D], b[D];
#pragma unroll
      for (int d = 0; d < D; ++d) {
        a[d] = ld16<PF>(row0 + 64 * (j + d));
        b[d] = ld16<PF>(row1 + 64 * (j + d));
      }
#pragma unroll
      for (int d = 0; d < D; ++d) acc ^= a[d].x ^ a[d].y ^ a[d].z ^ a[d].w ^ b[d].x ^ b[d].y ^ b[d].z ^ b[d].w;
    }
    for (; j < nchunks; ++j) {
      uint4 a = ld16<PF>(row0 + 64 * j), b = ld16<PF>(row1 + 64 * j);
      acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w;
    }
  }
  if (acc == 0x9E3779B9u) out[0] = acc;
}

// (c) contiguous chunk per warp (the layout a fragment-native repack gives):
// warp w streams bytes [w*C, (w+1)*C), 512 B per instruction, D in flight.
template <int D, bool PF>
__global__ void read_chunks(const uint4* __restrict__ p, size_t n16, size_t chunk16, unsigned* out) {
  const int lane = threadIdx.x & 31;
  const size_t warp = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5;
  const size_t nwarps = ((size_t)gridDim.x * blockDim.x) >> 5;
  unsigned acc = 0;
  for (size_t c0 = warp * chunk16; c0 < n16; c0 += nwarps * chunk16) {
    size_t end = c0 + chunk16 < n16 ? c0 + chunk16 : n16;
    size_t i = c0 + lane;
    for (; i + 32 * (D - 1) < end; i += 32 * D) {
      uint4 v[D];
#pragma unroll
      for (int d = 0; d < D; ++d) v[d] = ld16<PF>(p + i + 32 * d);
#pragma unroll
      for (int d = 0; d < D; ++d) acc ^= v[d].x ^ v[d].y ^ v[d].z ^ v[d].w;
    }
    for (; i < end; i += 32) {
      uint4 v = ld16<PF>(p + i);
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
  }
  if (acc == 0x9E3779B9u) out[0] = acc;
}

// (d) the GEMV weight stream without the MMA: per-warp smem ring of S stages,
// each stage = 16 rows x KC contiguous bytes per row (cp.async 16 B). KC is the
// per-row burst: v2 used 128 B; longer bursts keep DRAM pages open.
template <int KC, int S>
__global__ void ring_read(const uint8_t* __restrict__ w, int N, int K, unsigned* out) {
  extern __shared__ __align__(128) uint8_t sm[];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, W = blockDim.x >> 5;
  const int gw = blockIdx.x * W + warp, GW = gridDim.x * W;
  const int T = N / 16, nspan = K / KC;
  const int ntile = gw < T ? (T - 1 - gw) / GW + 1 : 0;
  const int steps = ntile * nspan;
  uint8_t* ring = sm + warp * S * 16 * KC;
  const uint32_t ring_u32 = (uint32_t)__cvta_generic_to_shared(ring);
  constexpr int CPR = KC / 16;               // 16-B chunks per row
  constexpr int PER_LANE = 16 * CPR / 32;    // cp.async per lane per stage
  auto issue = [&](int s) {
    if (s < steps) {
      const int tile = gw + (s / nspan) * GW, k0 = (s % nspan) * KC;
      const uint32_t st = ring_u32 + (s % S) * 16 * KC;
#pragma unroll
      for (int i = 0; i < PER_LANE; ++i) {
        const int c = lane + 32 * i, row = c / CPR, ch = c % CPR;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(st + row * KC + ch * 16),
                     "l"(w + (size_t)(tile * 16 + row) * K + k0 + ch * 16));
      }
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };
  for (int s = 0; s < S - 1; ++s) issue(s);
  unsigned acc = 0;
  for (int s = 0; s < steps; ++s) {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(S - 2));
    __syncwarp();
    acc ^= *reinterpret_cast<const unsigned*>(ring + (s % S) * 16 * KC + lane * 4);
    __syncwarp();
    issue(s + S - 1);
  }
  asm volatile("cp.async.wait_group 0;\n" ::);
  if (acc == 0x9E3779B9u) out[0] = acc;
}

cudaStream_t cur() { return at::cuda::getCurrentCUDAStream(); }

}  // namespace

void ring(torch::Tensor w, torch::Tensor out, int64_t N, int64_t K, int64_t kc, int64_t stages, int64_t W,
          int64_t grid, int64_t extra_smem) {
  const int smem = W * stages * 16 * kc + extra_smem;  // extra: probe the L1/smem carve-out effect
#define RCASE(KC_, S_)                                                                                      \
  if (kc == KC_ && stages == S_) {                                                                          \
    static bool cfg = false;                                                                                \
    if (!cfg) {                                                                                             \
      cudaFuncSetAttribute(ring_read<KC_, S_>, cudaFuncAttributeMaxDynamicSharedMemorySize, 101376);       \
      cfg = true;                                                                                           \
    }                                                                                                       \
    ring_read<KC_, S_><<<grid, W * 32, smem, cur()>>>((const uint8_t*)w.data_ptr(), N, K, (unsigned*)out.data_ptr()); \
    return;                                                                                                 \
  }
  RCASE(128, 4) RCASE(128, 8) RCASE(256, 3) RCASE(256, 4) RCASE(256, 6) RCASE(512, 2) RCASE(512, 3) RCASE(512, 4)
  RCASE(1024, 2) RCASE(1024, 3) RCASE(2048, 2)
#undef RCASE
  TORCH_CHECK(false, "bad ring cfg");
}

void flush(torch::Tensor buf, torch::Tensor out) {
  size_t n16 = buf.numel() / 16;
  flush_kernel<<<48 * 8, 256, 0, cur()>>>((const uint4*)buf.data_ptr(), n16, (unsigned*)out.data_ptr());
}

void flush_perm(torch::Tensor buf, torch::Tensor out) {
  size_t nchunk = buf.numel() / 4096;
  TORCH_CHECK((nchunk & (nchunk - 1)) == 0, "flush_perm needs a power-of-two number of 4 KiB chunks");
  flush_perm_kernel<<<48 * 8, 256, 0, cur()>>>((const uint4*)buf.data_ptr(), nchunk, (unsigned*)out.data_ptr());
}

void flush_write(torch::Tensor buf, int64_t seed) {
  size_t n16 = buf.numel() / 16;
  flush_write_kernel<<<48 * 8, 256, 0, cur()>>>((uint4*)buf.data_ptr(), n16, (unsigned)seed);
}

void spin(int64_t cycles) { spin_kernel<<<1, 32, 0, cur()>>>(cycles); }

#define FLAT_CASE(U, PF)                                                                     \
  if (u == U && pf == PF) {                                                                  \
    read_flat<U, PF><<<grid, block, 0, cur()>>>((const uint4*)w.data_ptr(), n16, o);        \
    return;                                                                                  \
  }

void flat(torch::Tensor w, torch::Tensor out, int64_t grid, int64_t block, int64_t u, bool pf) {
  size_t n16 = w.numel() / 16;
  unsigned* o = (unsigned*)out.data_ptr();
  FLAT_CASE(1, false) FLAT_CASE(2, false) FLAT_CASE(4, false) FLAT_CASE(8, false)
  FLAT_CASE(1, true) FLAT_CASE(2, true) FLAT_CASE(4, true) FLAT_CASE(8, true)
  TORCH_CHECK(false, "bad u");
}

#define TILE_CASE(D, PF)                                                                    \
  if (d == D && pf == PF) {                                                                 \
    read_tiles<D, PF><<<grid, block, 0, cur()>>>((const uint8_t*)w.data_ptr(), N, K, o);   \
    return;                                                                                 \
  }

void tiles(torch::Tensor w, torch::Tensor out, int64_t N, int64_t K, int64_t grid, int64_t block, int64_t d, bool pf) {
  unsigned* o = (unsigned*)out.data_ptr();
  TILE_CASE(1, false) TILE_CASE(2, false) TILE_CASE(4, false) TILE_CASE(8, false)
  TILE_CASE(1, true) TILE_CASE(2, true) TILE_CASE(4, true) TILE_CASE(8, true)
  TORCH_CHECK(false, "bad d");
}

#define CHUNK_CASE(D, PF)                                                                           \
  if (d == D && pf == PF) {                                                                         \
    read_chunks<D, PF><<<grid, block, 0, cur()>>>((const uint4*)w.data_ptr(), n16, chunk16, o);    \
    return;                                                                                         \
  }

void chunks(torch::Tensor w, torch::Tensor out, int64_t chunk_bytes, int64_t grid, int64_t block, int64_t d, bool pf) {
  size_t n16 = w.numel() / 16;
  size_t chunk16 = chunk_bytes / 16;
  unsigned* o = (unsigned*)out.data_ptr();
  CHUNK_CASE(1, false) CHUNK_CASE(2, false) CHUNK_CASE(4, false) CHUNK_CASE(8, false)
  CHUNK_CASE(1, true) CHUNK_CASE(2, true) CHUNK_CASE(4, true) CHUNK_CASE(8, true)
  TORCH_CHECK(false, "bad d");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("flush", &flush);
  m.def("ring", &ring);
  m.def("flush_write", &flush_write);
  m.def("flush_perm", &flush_perm);
  m.def("spin", &spin);
  m.def("flat", &flat);
  m.def("tiles", &tiles);
  m.def("chunks", &chunks);
}
