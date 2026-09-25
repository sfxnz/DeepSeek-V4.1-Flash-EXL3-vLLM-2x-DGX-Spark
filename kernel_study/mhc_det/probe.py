#!/usr/bin/env python3
"""GB10 probe for the mHC det kernels (run in the serve image on spark2).

1. Device properties (SMs, L2, smem, clocks).
2. NVRTC (cuda.bindings) compile for sm_121a, cuModuleLoadData, cuLaunchKernelEx with the
   PDL attribute on torch's current stream, CUDA-graph capture + replay.
3. TF32 mma.sync m16n8k8 operand semantics: is B (fp32 bits) truncated to tf32, or rounded?
   D(B raw) is compared bitwise with D(B & ~0x1fff) and D(cvt.rna.tf32(B)) on random data.
4. Row independence: does D[row 0] change when other rows of A change?
"""
from __future__ import annotations

import ctypes
import json
import sys

import torch

sys.path.insert(0, "/repo/docker/patch")
from mhc_det_rt import Module  # noqa: E402

SRC = r"""
extern "C" __global__ void add_one(float* p, int n) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] += 1.0f;
}

__device__ __forceinline__ void mma_tf32(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// A: [ntrial][16][8] fp32, B: [ntrial][8(k)][8(n)] fp32, D: [ntrial][3][16][8]
// mode 0 raw B bits, 1 B & ~0x1fff, 2 cvt.rna.tf32.f32(B)
extern "C" __global__ void mma_modes(const float* A, const float* B, float* D, int ntrial) {
  int trial = blockIdx.x;
  if (trial >= ntrial) return;
  int lane = threadIdx.x;
  int g = lane / 4, t = lane % 4;
  const float* a = A + trial * 128;
  const float* b = B + trial * 64;
  unsigned af[4] = {__float_as_uint(a[g * 8 + t]), __float_as_uint(a[(g + 8) * 8 + t]),
                    __float_as_uint(a[g * 8 + t + 4]), __float_as_uint(a[(g + 8) * 8 + t + 4])};
  for (int mode = 0; mode < 3; ++mode) {
    unsigned b0 = __float_as_uint(b[t * 8 + g]);
    unsigned b1 = __float_as_uint(b[(t + 4) * 8 + g]);
    if (mode == 1) { b0 &= ~0x1fffu; b1 &= ~0x1fffu; }
    if (mode == 2) {
      asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(b0) : "f"(b[t * 8 + g]));
      asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(b1) : "f"(b[(t + 4) * 8 + g]));
    }
    unsigned bf[2] = {b0, b1};
    float d[4] = {0.f, 0.f, 0.f, 0.f};
    mma_tf32(d, af, bf);
    float* o = D + (trial * 3 + mode) * 128;
    o[g * 8 + t * 2] = d[0]; o[g * 8 + t * 2 + 1] = d[1];
    o[(g + 8) * 8 + t * 2] = d[2]; o[(g + 8) * 8 + t * 2 + 1] = d[3];
  }
}
"""


def main() -> int:
    out: dict = {}
    p = torch.cuda.get_device_properties(0)
    out["device"] = {
        "name": p.name, "sm_count": p.multi_processor_count, "cc": f"{p.major}.{p.minor}",
        "l2_bytes": getattr(p, "L2_cache_size", None), "total_mem_gb": round(p.total_memory / 2**30, 1),
        "smem_per_block_optin": getattr(p, "shared_memory_per_block_optin", None),
        "smem_per_sm": getattr(p, "shared_memory_per_multiprocessor", None),
        "regs_per_sm": getattr(p, "regs_per_multiprocessor", None),
        "max_threads_per_sm": getattr(p, "max_threads_per_multi_processor", None),
    }
    print(json.dumps(out["device"]), flush=True)

    mod = Module(SRC, "probe.cu")
    out["nvrtc_log"] = mod.log[-400:]
    # 2. launch + graph capture with PDL
    x = torch.zeros(1000, device="cuda")
    add_one = mod.function("add_one")
    add_one.launch((4,), (256,), 0, [(x.data_ptr(), ctypes.c_void_p), (1000, ctypes.c_int)], pdl=True)
    torch.cuda.synchronize()
    ok_eager = bool((x == 1).all())
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for _ in range(3):
                add_one.launch((4,), (256,), 0, [(x.data_ptr(), ctypes.c_void_p), (1000, ctypes.c_int)], pdl=True)
    torch.cuda.synchronize()
    x.zero_()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    ok_graph = bool((x == 15).all())
    out["launch"] = {"eager_ok": ok_eager, "graph_ok": ok_graph}
    print(json.dumps(out["launch"]), flush=True)

    # 3. tf32 semantics of the B operand
    ntrial = 4096
    gen = torch.Generator(device="cuda").manual_seed(1)
    A = torch.randn(ntrial, 16, 8, device="cuda", generator=gen).bfloat16().float()
    B = torch.randn(ntrial, 8, 8, device="cuda", generator=gen) * torch.exp(
        torch.randn(ntrial, 8, 8, device="cuda", generator=gen))
    D = torch.empty(ntrial, 3, 16, 8, device="cuda")
    f = mod.function("mma_modes")
    f.launch((ntrial,), (32,), 0, [(A.data_ptr(), ctypes.c_void_p), (B.data_ptr(), ctypes.c_void_p),
                                   (D.data_ptr(), ctypes.c_void_p), (ntrial, ctypes.c_int)])
    torch.cuda.synchronize()
    raw, trunc, rna = D[:, 0].view(torch.int32), D[:, 1].view(torch.int32), D[:, 2].view(torch.int32)
    out["tf32_b_operand"] = {
        "raw_eq_trunc_frac": float((raw == trunc).float().mean()),
        "raw_eq_rna_frac": float((raw == rna).float().mean()),
        "n": int(raw.numel()),
    }
    # fp64 reference distance, to see which is closer to exact products of truncated B
    print(json.dumps(out["tf32_b_operand"]), flush=True)

    # 4. row independence: vary rows 1..15 of A, row 0 output must be bitwise stable
    A2 = A.clone()
    A2[:, 1:, :] = torch.randn(ntrial, 15, 8, device="cuda", generator=gen).bfloat16().float()
    D2 = torch.empty_like(D)
    f.launch((ntrial,), (32,), 0, [(A2.data_ptr(), ctypes.c_void_p), (B.data_ptr(), ctypes.c_void_p),
                                   (D2.data_ptr(), ctypes.c_void_p), (ntrial, ctypes.c_int)])
    torch.cuda.synchronize()
    same_row0 = (D2[:, :, 0, :].view(torch.int32) == D[:, :, 0, :].view(torch.int32)).all().item()
    out["row_independence_row0_bitwise"] = bool(same_row0)
    print(json.dumps({"row_independence_row0_bitwise": bool(same_row0)}), flush=True)
    with open("/repo/results/2026-09-25-kernels/mhc-det/probe.json", "w") as fh:
        json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
