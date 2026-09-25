#!/usr/bin/env python3
"""Phase timestamps (globaltimer) inside the det pk GEMM, cold and warm L2, T=4 and 8.

Prints per phase the median over CTAs of (stamp - earliest CTA entry), in us.
Stamps: 0 entry, 1 fn issued, 2 after griddepcontrol.wait, 3 x in smem, 4..9 stage-pair
waits, 10 loop end, 11 stores done.
"""
from __future__ import annotations

import ctypes
import json
import statistics
import sys

import numpy as np
import torch

sys.path.insert(0, "/repo/kernel_study/mhc_det")
import common as C  # noqa: E402
import mhc_det  # noqa: E402
from cuda.bindings import driver as cu  # noqa: E402
from mhc_det_rt import Module, _ok  # noqa: E402

RES = r"""
extern "C" __global__ void timer_res(unsigned long long* out, int n) {
  unsigned long long prev, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(prev));
  int k = 0;
  for (int i = 0; i < 2000000 && k < n; ++i) {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    if (t != prev) { out[k++] = t - prev; prev = t; }
  }
}
"""


def main() -> int:
    out: dict = {}
    m = Module(RES, "res.cu")
    buf = torch.zeros(64, dtype=torch.int64, device="cuda")
    m.function("timer_res").launch((1,), (1,), 0, [(buf.data_ptr(), ctypes.c_void_p), (64, ctypes.c_int)])
    torch.cuda.synchronize()
    steps = buf.tolist()
    out["globaltimer_step_ns"] = sorted(set(steps))[:8]
    print("globaltimer increments (ns):", out["globaltimer_step_ns"], flush=True)

    dk = mhc_det.DetKernels()
    src = dk.src
    pm = Module(src, "mhc_det_prof.cu", opts=["-DMHC_DET_PROF"])
    kern = {"pk": pm.function("mhc_det_gemm_pk_t8"), "f32": pm.function("mhc_det_gemm_f32_t8")}
    gptr, gsize = _ok(cu.cuModuleGetGlobal(pm.handle, b"g_prof"), "g_prof")
    fns = C.real_fns()
    emb = C.embeddings(64)
    name, fn = fns[41]
    fnp = mhc_det.pack_fn(fn)
    flush = torch.empty(64 * 2**20 // 4, device="cuda")
    for arm in ("pk", "f32"):
        for t in (4, 8):
            x_src = C.make_x("emb", t, C.K_FULL, seed=3, emb=emb)
            x = x_src.clone()
            mixes = torch.empty(16, t, 24, device="cuda")
            sqr = torch.empty(16, t, device="cuda")
            src = fnp if arm == "pk" else fn
            for mode in ("cold", "warm"):
                rows = []
                for rep in range(30):
                    if mode == "cold":
                        flush.sum()  # read-only sweep (clean L2), as common.Timer
                        x.copy_(x_src)
                    kern[arm].launch((3, 16), (32,), mhc_det.smem_bytes(t, C.K_FULL, arm == "pk"),
                                     [(x.data_ptr(), ctypes.c_void_p), (src.data_ptr(), ctypes.c_void_p),
                                      (mixes.data_ptr(), ctypes.c_void_p), (sqr.data_ptr(), ctypes.c_void_p),
                                      (t, ctypes.c_int), (C.K_FULL, ctypes.c_int)], pdl=True)
                    torch.cuda.synchronize()
                    host = np.zeros(64 * 16, dtype=np.uint64)
                    _ok(cu.cuMemcpyDtoH(host.ctypes.data, gptr, host.nbytes), "dtoh")
                    st = host.reshape(64, 16)[:48].astype(np.int64)
                    t0 = st[:, 0].min()
                    rows.append((st - t0) / 1000.0)
                arr = np.stack(rows[5:])  # [reps, 48, 16] us
                med = [round(float(np.median(arr[:, :, p])), 2) for p in range(12)]
                mx = [round(float(np.median(arr[:, :, p].max(axis=1))), 2) for p in range(12)]
                key = f"{arm}_T{t}_{mode}"
                out[key] = {"median_over_ctas": med, "max_over_ctas": mx}
                print(key, "med", med, "| max", mx, flush=True)
    with open("/repo/results/2026-09-25-kernels/mhc-det/prof_phases.json", "w") as fh:
        json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
