#!/usr/bin/env python3
"""How fast can N CTAs stream a contiguous per-CTA chunk from cold DRAM on GB10?

Each CTA copies its own `chunk` bytes (from a buffer larger than L2 region, flushed before
every launch) into shared memory by one of several methods; the kernel records, per CTA,
globaltimer at entry and at completion. Reported: median over reps of (last CTA done - first
CTA entry) and the implied GB/s. Methods:
  bulk<B>   : one thread issues cp.async.bulk copies of B bytes, all up front, one mbarrier
  cpasync<W>: W warps issue 16 B cp.async.cg for the whole chunk, wait_group 0
  ldg<W>    : W warps load 16 B per lane with ld.global.nc (xor-reduced so nothing is dead)
"""
from __future__ import annotations

import ctypes
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/repo/docker/patch")
from cuda.bindings import driver as cu  # noqa: E402
from mhc_det_rt import Module, _ok  # noqa: E402

SRC = r"""
typedef unsigned int u32; typedef unsigned long long u64;
__device__ unsigned long long g_t[512 * 2];
__device__ __forceinline__ u32 sa(const void* p) { u64 a; asm("cvta.to.shared.u64 %0, %1;" : "=l"(a) : "l"(p)); return (u32)a; }
__device__ __forceinline__ u64 gt() { u64 c; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(c)::"memory"); return c; }
extern "C" __global__ void k_bulk(const unsigned char* src, int chunk, int piece, int* sink) {
  extern __shared__ __align__(128) unsigned char sm[];
  __shared__ __align__(8) u64 bar;
  u64 t0 = gt();
  const unsigned char* s = src + (size_t)blockIdx.x * chunk;
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(&bar)) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(sa(&bar)), "r"(chunk) : "memory");
    for (int o = 0; o < chunk; o += piece)
      asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                   :: "r"(sa(sm + o)), "l"(s + o), "r"(min(piece, chunk - o)), "r"(sa(&bar)) : "memory");
  }
  __syncthreads();
  u32 ok = 0;
  while (!ok) asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }" : "=r"(ok) : "r"(sa(&bar)) : "memory");
  u64 t1 = gt();
  if (threadIdx.x == 0) { g_t[blockIdx.x * 2] = t0; g_t[blockIdx.x * 2 + 1] = t1; if (sm[chunk - 1] == 0x5a && sm[0] == 0x5a) sink[0] = 1; }
}
extern "C" __global__ void k_bulksm(const unsigned char* src, int chunk, int piece, int* sink) {
  extern __shared__ __align__(128) unsigned char sm[];
  __shared__ __align__(8) u64 bar;
  u64 t0 = gt();
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(&bar)) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(sa(&bar)), "r"(chunk) : "memory");
    for (int o = 0, i = 0; o < chunk; o += piece, ++i)
      asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                   :: "r"(sa(sm + o)), "l"(src + ((size_t)i * gridDim.x + blockIdx.x) * piece), "r"(piece), "r"(sa(&bar)) : "memory");
  }
  __syncthreads();
  u32 ok = 0;
  while (!ok) asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }" : "=r"(ok) : "r"(sa(&bar)) : "memory");
  u64 t1 = gt();
  if (threadIdx.x == 0) { g_t[blockIdx.x * 2] = t0; g_t[blockIdx.x * 2 + 1] = t1; if (sm[chunk - 1] == 0x5a && sm[0] == 0x5a) sink[0] = 1; }
}
extern "C" __global__ void k_cpasync(const unsigned char* src, int chunk, int piece, int* sink) {
  extern __shared__ __align__(128) unsigned char sm[];
  u64 t0 = gt();
  const unsigned char* s = src + (size_t)blockIdx.x * chunk;
  for (int o = threadIdx.x * 16; o < chunk; o += blockDim.x * 16)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa(sm + o)), "l"(s + o) : "memory");
  asm volatile("cp.async.commit_group;" ::: "memory");
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
  u64 t1 = gt();
  if (threadIdx.x == 0) { g_t[blockIdx.x * 2] = t0; g_t[blockIdx.x * 2 + 1] = t1; if (sm[chunk - 1] == 0x5a && sm[0] == 0x5a) sink[0] = 1; }
}
extern "C" __global__ void k_ldg(const unsigned char* src, int chunk, int piece, int* sink) {
  u64 t0 = gt();
  const unsigned char* s = src + (size_t)blockIdx.x * chunk;
  uint4 acc = make_uint4(0, 0, 0, 0);
  #pragma unroll 8
  for (int o = threadIdx.x * 16; o < chunk; o += blockDim.x * 16) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(s + o));
    acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
  }
  __syncthreads();
  u64 t1 = gt();
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678u) sink[0] = 1;
  if (threadIdx.x == 0) { g_t[blockIdx.x * 2] = t0; g_t[blockIdx.x * 2 + 1] = t1; }
}
"""


def main() -> int:
    mod = Module(SRC, "stream.cu")
    gptr, _ = _ok(cu.cuModuleGetGlobal(mod.handle, b"g_t"), "g_t")
    total_src = torch.randint(0, 255, (256 * 2**20,), dtype=torch.uint8, device="cuda")  # 256 MB, > L2
    flush = torch.empty(64 * 2**20 // 4, device="cuda")
    sink = torch.zeros(4, dtype=torch.int32, device="cuda")
    out = {}
    configs = []
    for ctas, chunk in ((48, 25600), (48, 40960)):
        for piece in (1280, 2560, 5120):
            configs.append(("bulk", ctas, chunk, piece, 32))
            configs.append(("bulksm", ctas, chunk, piece, 32))
    rng = np.random.default_rng(0)
    for flush_mode in ("ro",):
        for kind, ctas, chunk, piece, threads in configs:
            f = mod.function(f"k_{kind}")
            smem = chunk if kind != "ldg" else 0
            spans = []
            for rep in range(40):
                if flush_mode == "rw":
                    flush.add_(1.0)
                else:
                    sink2 = flush.sum()  # read-only sweep: L2 ends up full of clean lines
                off = int(rng.integers(0, 200)) * 2**20  # a fresh region every rep
                base = total_src.data_ptr() + off
                f.launch((ctas,), (threads,), smem, [(base, ctypes.c_void_p), (chunk, ctypes.c_int),
                                                     (piece, ctypes.c_int), (sink.data_ptr(), ctypes.c_void_p)])
                torch.cuda.synchronize()
                host = np.zeros(512 * 2, dtype=np.uint64)
                _ok(cu.cuMemcpyDtoH(host.ctypes.data, gptr, host.nbytes), "dtoh")
                tt = host.reshape(512, 2)[:ctas].astype(np.int64)
                spans.append(((tt[:, 1].max() - tt[:, 0].min()) / 1000.0,
                              float(np.median(tt[:, 1] - tt[:, 0])) / 1000.0,
                              (tt[:, 0].max() - tt[:, 0].min()) / 1000.0))
            sp = np.array(spans[5:])
            span = float(np.median(sp[:, 0]))
            key = f"flush={flush_mode} {kind} ctas={ctas} chunk={chunk} piece={piece} threads={threads}"
            out[key] = {"span_us": round(span, 2), "per_cta_median_us": round(float(np.median(sp[:, 1])), 2),
                        "entry_skew_us": round(float(np.median(sp[:, 2])), 2),
                        "GBps": round(ctas * chunk / span / 1e3, 1)}
            print(key, out[key], flush=True)
    with open("/repo/results/2026-09-25-kernels/mhc-det/stream_probe.json", "w") as fh:
        json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
