"""L2 prefetch engines for the AR-window study (k3 comm fix pass), one CUDA extension.

The shipped lever (docker/patch/l2pf_kernel.py) issues every cp.async.bulk.prefetch.L2
request of its budget within ~1 us from one CTA: a burst that queues up to 5.5 MiB
(~25 us of DRAM time) at once. With a real NCCL LL all-reduce in flight (GDR off:
LL lines, flags and NIC DMA all in host memory) the AR waits behind that queue.
The engines here bound what the prefetcher puts in the DRAM queue:

  burst    bulk_prefetch: `ctas` CTAs x 128 lanes (default 1 = the lever's kernel), all
           requests at once. A CTA stays resident until the TMA unit has taken its requests,
           so 1 CTA runs ~as long as the transfer; more CTAs spread the issue over more SMs.
  pf       per-thread prefetch.global.L2 of 128 B lines from `ctas` CTAs x 256 threads (no
           TMA; the threads exit right after issuing)
  paced    1 thread issues one `chunk` prefetch every `gap` SM cycles (open loop:
           rate = chunk / gap, in-flight grows if DRAM latency grows)
  ring     1 thread keeps `depth` TMA bulk copies of `chunk` bytes in flight into a
           shared-memory ring (cp.async.bulk + mbarrier): at most depth x chunk bytes
           queued by the prefetcher at any time, closed loop (slows down when DRAM
           latency grows). The copies allocate the lines in L2 on the way to SMEM.
  demote   applypriority.global.L2::evict_normal over a range (undo an evict_last)

Every engine takes policy 0 (normal) or 1 (L2::evict_last via createpolicy).
Also: k_spin (clock), k_read (streaming read; mode 0 = normal loads, 1 = L2::evict_last,
2 = ld.global.cs = evict-first streaming, what p2b's __ldcs trellis loads do) for the harnesses,
and k_read_pf: the same streaming read whose CTAs first issue TMA L2 prefetches of up to four
ranges from thread 0 (the p2b stand-in of ar_window_nccl.py with p2b_pf_bench.py's prologue).
Host entry points are k_<name> (k_read: `read` would clash with POSIX read in the binding).
"""
from __future__ import annotations

import os

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

__device__ __forceinline__ uint64_t make_policy(int last) {
  uint64_t pol = 0;
  if (last) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}

__global__ void spin_kernel(long long cycles) {
  long long t0 = clock64();
  while (clock64() - t0 < cycles) { }
}

template <int MODE>  // 0 normal, 1 L2::evict_last, 2 .cs (evict-first streaming)
__global__ void read_kernel(const int4* p, long long n16, int* sink) {
  uint64_t pol = make_policy(MODE == 1);
  int acc = 0;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n16;
       i += (long long)gridDim.x * blockDim.x) {
    int a, b, c, d;
    if (MODE == 1)
      asm volatile("ld.global.L2::cache_hint.v4.s32 {%0,%1,%2,%3}, [%4], %5;"
                   : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(p + i), "l"(pol));
    else if (MODE == 2)
      asm volatile("ld.global.cs.v4.s32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(p + i));
    else
      asm volatile("ld.global.v4.s32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(p + i));
    acc ^= a ^ b ^ c ^ d;
  }
  if (acc == 0x7f7f7f7f) *sink = acc;
}

struct PfRanges { const char* p[4]; long long n[4]; };

__device__ __forceinline__ void prologue_prefetch(const PfRanges& r) {  // = p2b_pf_bench.py
  if (threadIdx.x != 0) return;
  long long first = 0;
  for (int i = 0; i < 4; i++) {
    const long long n = (r.n[i] + 16383) / 16384;
    long long c = blockIdx.x - first;
    if (c < 0) c += ((-c + gridDim.x - 1) / gridDim.x) * (long long)gridDim.x;
    for (; c < n; c += gridDim.x) {
      const long long off = c * 16384, rem = r.n[i] - off;
      const unsigned sz = (unsigned)(rem < 16384 ? rem : 16384) & ~15u;
      if (sz) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(r.p[i] + off), "r"(sz) : "memory");
    }
    first += n;
  }
}

__global__ void read_pf_kernel(const int4* p, long long n16, int* sink, PfRanges r) {
  prologue_prefetch(r);
  int acc = 0;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n16;
       i += (long long)gridDim.x * blockDim.x) {
    int a, b, c, d;
    asm volatile("ld.global.cs.v4.s32 {%0,%1,%2,%3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(p + i));
    acc ^= a ^ b ^ c ^ d;
  }
  if (acc == 0x7f7f7f7f) *sink = acc;
}

__global__ void pf_kernel(const char* p, long long bytes, int last) {
  long long n = (bytes + 127) / 128;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n;
       i += (long long)gridDim.x * blockDim.x) {
    if (last)
      asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(p + i * 128) : "memory");
    else
      asm volatile("prefetch.global.L2 [%0];" :: "l"(p + i * 128) : "memory");
  }
}

__global__ void burst_kernel(const char* p, long long bytes, int chunk, int last) {
  uint64_t pol = make_policy(last);
  long long n = (bytes + chunk - 1) / chunk;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n;
       i += (long long)gridDim.x * blockDim.x) {
    long long off = i * (long long)chunk;
    long long rem = bytes - off;
    unsigned sz = (unsigned)(rem < chunk ? rem : chunk) & ~15u;
    if (!sz) continue;
    if (last)
      asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;"
                   :: "l"(p + off), "r"(sz), "l"(pol) : "memory");
    else
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p + off), "r"(sz) : "memory");
  }
}

__global__ void paced_kernel(const char* p, long long bytes, int chunk, long long gap, int last) {
  if (threadIdx.x != 0) return;
  uint64_t pol = make_policy(last);
  long long t = clock64();
  for (long long off = 0; off < bytes; off += chunk) {
    long long rem = bytes - off;
    unsigned sz = (unsigned)(rem < chunk ? rem : chunk) & ~15u;
    if (sz) {
      if (last)
        asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;"
                     :: "l"(p + off), "r"(sz), "l"(pol) : "memory");
      else
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p + off), "r"(sz) : "memory");
    }
    t += gap;
    while (clock64() < t) { }
  }
}

__device__ __forceinline__ bool mbar_try_wait(unsigned bar, unsigned parity) {
  unsigned ok;
  asm volatile("{ .reg .pred P; mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; selp.u32 %0, 1, 0, P; }"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok != 0;
}

#define RING_MAX 32
__global__ void ring_kernel(const char* p, long long bytes, int chunk, int depth, int last) {
  extern __shared__ __align__(128) unsigned char ring_smem[];
  __shared__ __align__(8) unsigned long long bars[RING_MAX];
  if (threadIdx.x != 0) return;
  uint64_t pol = make_policy(last);
  for (int i = 0; i < depth; i++) {
    unsigned a = (unsigned)__cvta_generic_to_shared(&bars[i]);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(a) : "memory");
  }
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  long long n = (bytes + chunk - 1) / chunk;
  for (long long k = 0; k < n; k++) {
    int slot = (int)(k % depth);
    unsigned bar = (unsigned)__cvta_generic_to_shared(&bars[slot]);
    if (k >= depth) {
      unsigned parity = (unsigned)((k / depth - 1) & 1);
      while (!mbar_try_wait(bar, parity)) { }
    }
    long long off = k * (long long)chunk;
    long long rem = bytes - off;
    unsigned sz = (unsigned)(rem < chunk ? rem : chunk) & ~15u;
    unsigned dst = (unsigned)__cvta_generic_to_shared(ring_smem + (size_t)slot * chunk);
    unsigned long long st;
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;" : "=l"(st) : "r"(bar), "r"(sz) : "memory");
    if (last)
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
                   :: "r"(dst), "l"(p + off), "r"(sz), "r"(bar), "l"(pol) : "memory");
    else
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                   :: "r"(dst), "l"(p + off), "r"(sz), "r"(bar) : "memory");
  }
  long long k0 = n > depth ? n - depth : 0;
  for (long long k = k0; k < n; k++) {  // drain: the CTA must not exit with copies in flight
    unsigned bar = (unsigned)__cvta_generic_to_shared(&bars[k % depth]);
    unsigned parity = (unsigned)((k / depth) & 1);
    while (!mbar_try_wait(bar, parity)) { }
  }
}

__global__ void demote_kernel(const char* p, long long bytes) {
  long long n = bytes / 128;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n;
       i += (long long)gridDim.x * blockDim.x)
    asm volatile("applypriority.global.L2::evict_normal [%0], 128;" :: "l"(p + i * 128) : "memory");
}

static cudaStream_t cur() { return at::cuda::getCurrentCUDAStream(); }

// Every entry point checks its own launch, so an error names the kernel that caused it.
void k_spin(double cycles, int ctas) {
  spin_kernel<<<ctas, 32, 0, cur()>>>((long long)cycles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void k_read(torch::Tensor t, long long bytes, int ctas, int mode, torch::Tensor sink) {
  TORCH_CHECK(mode >= 0 && mode <= 2, "read: mode 0|1|2");
  const int4* p = (const int4*)t.data_ptr();
  if (mode == 1) read_kernel<1><<<ctas, 256, 0, cur()>>>(p, bytes / 16, (int*)sink.data_ptr());
  else if (mode == 2) read_kernel<2><<<ctas, 256, 0, cur()>>>(p, bytes / 16, (int*)sink.data_ptr());
  else read_kernel<0><<<ctas, 256, 0, cur()>>>(p, bytes / 16, (int*)sink.data_ptr());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void k_read_pf(torch::Tensor t, long long bytes, int ctas, torch::Tensor sink, std::vector<int64_t> ptrs,
               std::vector<int64_t> sizes) {
  TORCH_CHECK(ptrs.size() == sizes.size() && ptrs.size() <= 4, "read_pf: <= 4 ranges");
  PfRanges r{};
  for (size_t i = 0; i < ptrs.size(); i++) { r.p[i] = reinterpret_cast<const char*>(ptrs[i]); r.n[i] = sizes[i]; }
  read_pf_kernel<<<ctas, 256, 0, cur()>>>((const int4*)t.data_ptr(), bytes / 16, (int*)sink.data_ptr(), r);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void k_burst(torch::Tensor t, long long bytes, int chunk, int last, int ctas) {
  burst_kernel<<<ctas, 128, 0, cur()>>>((const char*)t.data_ptr(), bytes, chunk, last);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void k_pf(torch::Tensor t, long long bytes, int last, int ctas) {
  pf_kernel<<<ctas, 256, 0, cur()>>>((const char*)t.data_ptr(), bytes, last);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void k_paced(torch::Tensor t, long long bytes, int chunk, double gap_cycles, int last) {
  paced_kernel<<<1, 32, 0, cur()>>>((const char*)t.data_ptr(), bytes, chunk, (long long)gap_cycles, last);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void k_ring(torch::Tensor t, long long bytes, int chunk, int depth, int last) {
  TORCH_CHECK(depth >= 1 && depth <= RING_MAX && chunk % 16 == 0, "ring: bad depth/chunk");
  int smem = chunk * depth;
  int dev = 0, optin = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  TORCH_CHECK(smem + (int)(RING_MAX * sizeof(unsigned long long)) <= optin,
              "ring: depth x chunk = ", smem, " B + barriers exceeds the ", optin, " B shared-memory opt-in limit");
  static int set = 48 * 1024;  // no opt-in needed up to 48 KiB of dynamic shared memory
  if (smem > set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute((const void*)ring_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    set = smem;
  }
  ring_kernel<<<1, 32, smem, cur()>>>((const char*)t.data_ptr(), bytes, chunk, depth, last);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void k_demote(torch::Tensor t, long long bytes, int ctas) {
  demote_kernel<<<ctas, 256, 0, cur()>>>((const char*)t.data_ptr(), bytes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
CPP_SRC = """
void k_spin(double cycles, int ctas);
void k_read(torch::Tensor t, long long bytes, int ctas, int mode, torch::Tensor sink);
void k_read_pf(torch::Tensor t, long long bytes, int ctas, torch::Tensor sink, std::vector<int64_t> ptrs,
               std::vector<int64_t> sizes);
void k_burst(torch::Tensor t, long long bytes, int chunk, int last, int ctas);
void k_pf(torch::Tensor t, long long bytes, int last, int ctas);
void k_paced(torch::Tensor t, long long bytes, int chunk, double gap_cycles, int last);
void k_ring(torch::Tensor t, long long bytes, int chunk, int depth, int last);
void k_demote(torch::Tensor t, long long bytes, int ctas);
"""
FUNCS = ["k_spin", "k_read", "k_read_pf", "k_burst", "k_pf", "k_paced", "k_ring", "k_demote"]


def build(build_dir: str):
    from torch.utils.cpp_extension import load_inline

    os.makedirs(build_dir, exist_ok=True)
    return load_inline(
        name="l2pf_var3",
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        functions=FUNCS,
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_121a,code=sm_121a"],
        build_directory=build_dir,
        verbose=False,
    )


def _build_tload():
    """Triton closed-loop loader: one program, `TILE` int32 loads per iteration, each
    iteration XOR-folded before the next is issued (num_stages=1: no software
    pipelining), so at most TILE x 4 bytes are in flight. Loads are .cg (L2 only)."""
    import triton
    import triton.language as tl

    @triton.jit
    def _tload(base, n, sink, TILE: tl.constexpr):
        acc = tl.zeros([TILE], dtype=tl.int32)
        for start in range(0, n, TILE):
            offs = start + tl.arange(0, TILE)
            acc ^= tl.load(base + offs, mask=offs < n, other=0, cache_modifier=".cg")
        r = tl.xor_sum(acc, 0)
        tl.store(sink, r, mask=r == 2139062143)  # keeps the loads live; practically never stores

    return _tload


def parse_engine(spec: str) -> dict:
    """'tri' | 'burst[:c<CTAs>]' | 'pf:c<CTAs>' | 'paced:<GB/s>' | 'ring:<depth>x<KiB>' |
    'tload:<KiB>[w<warps>]' (+ '+last' for evict_last on the CUDA engines)."""
    last = spec.endswith("+last")
    base = spec[: -len("+last")] if last else spec
    kind, _, arg = base.partition(":")
    out = {"kind": kind, "last": int(last)}
    if kind in ("burst", "pf"):
        if arg and not arg.startswith("c"):
            raise ValueError(spec)
        out["ctas"] = int(arg[1:]) if arg else (1 if kind == "burst" else 48)
    elif kind == "paced":
        out["gbps"] = float(arg)
    elif kind == "ring":
        d, _, kib = arg.partition("x")
        out["depth"], out["chunk"] = int(d), int(kib) * 1024
    elif kind == "tload":
        kib, _, w = arg.partition("w")
        out["tile"], out["warps"] = int(kib) * 256, int(w or 16)
    elif kind != "tri":
        raise ValueError(spec)
    return out


def launcher(ext, tri_launch, sm_mhz: float, chunk: int = 16384):
    """launch(engine_dict, tensor, nbytes) on the current stream."""

    state: dict = {}

    def launch(e: dict, t, nbytes: int) -> None:
        if e["kind"] == "tload":
            import torch

            if "tload" not in state:
                state["tload"] = _build_tload()
                state["sink"] = torch.empty(1, dtype=torch.int32, device=t.device)
            state["tload"][(1,)](t.reshape(-1).view(torch.int32), nbytes // 4, state["sink"],
                                 TILE=e["tile"], num_warps=e["warps"], num_stages=1)
        elif e["kind"] == "tri":
            assert not e["last"], "the shipped Triton kernel has no cache hint"
            tri_launch(t, nbytes)
        elif e["kind"] == "burst":
            ext.k_burst(t, nbytes, chunk, e["last"], e["ctas"])
        elif e["kind"] == "pf":
            ext.k_pf(t, nbytes, e["last"], e["ctas"])
        elif e["kind"] == "paced":
            gap_ns = chunk / e["gbps"]  # bytes / (GB/s) = ns
            ext.k_paced(t, nbytes, chunk, gap_ns * sm_mhz / 1e3, e["last"])
        elif e["kind"] == "ring":
            ext.k_ring(t, nbytes, e["chunk"], e["depth"], e["last"])

    return launch
