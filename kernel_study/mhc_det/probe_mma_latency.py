#!/usr/bin/env python3
"""Dependent-chain latency (SM cycles) of mma.sync m16n8k8 TF32 on GB10, 1 warp, register operands.

chainN: N dependent MMAs on one accumulator; chain3: 3 independent accumulators interleaved.
"""
from __future__ import annotations

import ctypes
import json
import sys

import torch

sys.path.insert(0, "/repo/docker/patch")
from mhc_det_rt import Module  # noqa: E402

SRC = r"""
__device__ __forceinline__ void mma(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
extern "C" __global__ void chain(float* out, long long* cyc, int n, int ilp) {
  unsigned a = __float_as_uint(1.0f + threadIdx.x), b = __float_as_uint(0.5f);
  float d0[4] = {0,0,0,0}, d1[4] = {0,0,0,0}, d2[4] = {0,0,0,0};
  long long t0 = clock64();
  if (ilp == 1) {
    for (int i = 0; i < n; ++i) mma(d0, a, a, a, a, b, b);
  } else {
    for (int i = 0; i < n; ++i) { mma(d0, a, a, a, a, b, b); mma(d1, a, a, a, a, b, b); mma(d2, a, a, a, a, b, b); }
  }
  long long t1 = clock64();
  out[threadIdx.x] = d0[0] + d1[1] + d2[2];
  if (threadIdx.x == 0) cyc[0] = t1 - t0;
}
"""


def main() -> int:
    f = Module(SRC, "chain.cu").function("chain")
    out = torch.zeros(32, device="cuda")
    cyc = torch.zeros(1, dtype=torch.int64, device="cuda")
    res = {}
    for ilp in (1, 3):
        for n in (160, 1000):
            vals = []
            for _ in range(20):
                f.launch((1,), (32,), 0, [(out.data_ptr(), ctypes.c_void_p), (cyc.data_ptr(), ctypes.c_void_p),
                                          (n, ctypes.c_int), (ilp, ctypes.c_int)])
                torch.cuda.synchronize()
                vals.append(int(cyc.item()))
            vals.sort()
            res[f"ilp{ilp}_n{n}"] = {"cycles": vals[len(vals) // 2], "cycles_per_step": round(vals[len(vals) // 2] / n, 2)}
    print(json.dumps(res), flush=True)
    with open("/repo/results/2026-09-25-kernels/mhc-det/probe_mma_latency.json", "w") as fh:
        json.dump(res, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
