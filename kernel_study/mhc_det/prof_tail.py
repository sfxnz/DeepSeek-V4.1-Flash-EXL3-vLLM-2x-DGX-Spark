#!/usr/bin/env python3
"""globaltimer timeline of one det sublayer (post -> GEMM -> fused norm, PDL) at T=4/8, in a graph.

GEMM stamps (per CTA): 0 entry, 3 x in smem, 10 loop end, 11 done. Norm stamps (per token):
0 entry, 1 griddepcontrol.wait returned, 2 split sums done, 3 first sinkhorn step, 4 comb stored
(coefficient warp), 5 layer_input stored (layer_input warps). All relative to the first GEMM CTA
entry; medians over replays.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/repo/kernel_study/mhc_det")
import bench_path as BP  # noqa: E402
import common as C  # noqa: E402
import mhc_det  # noqa: E402
from cuda.bindings import driver as cu  # noqa: E402
from mhc_det_rt import _ok  # noqa: E402


def main() -> int:
    w = BP.load_weights()
    dk = mhc_det.DetKernels(opts=["-DMHC_DET_PROF"])
    gptr, _ = _ok(cu.cuModuleGetGlobal(dk.mod.handle, b"g_prof"), "g_prof")
    name = "layers.9.hc_ffn_fn"
    packed = mhc_det.pack_fn(w[name])
    path = BP.Path(w, dk, {name: packed}, True)
    out = {}
    flush = torch.empty(64 * 2**20 // 4, device="cuda")
    for t in (4, 8):
        g = torch.Generator(device="cuda").manual_seed(t)
        residual = (torch.randn(t, 4, 5120, device="cuda", generator=g) * 3).bfloat16()
        x = torch.randn(t, 5120, device="cuda", generator=g).bfloat16()
        post_mix = (2 * torch.sigmoid(torch.randn(t, 4, 1, device="cuda", generator=g))).contiguous()
        comb = torch.softmax(torch.randn(t, 4, 4, device="cuda", generator=g), -1).contiguous()
        pmix = torch.softmax(torch.randn(t, 4, device="cuda", generator=g), -1).contiguous()

        def body():
            r = path.post(x, residual, post_mix, comb)
            return path.pre(r, name, "layers.9", "ffn", pmix)

        gr = BP._graph(body)
        rows = []
        for rep in range(40):
            flush.sum()
            gr.replay()
            torch.cuda.synchronize()
            host = np.zeros(64 * 16, dtype=np.uint64)
            _ok(cu.cuMemcpyDtoH(host.ctypes.data, gptr, host.nbytes), "dtoh")
            st = host.reshape(64, 16).astype(np.int64)
            t0 = st[:48, 0].min()
            gemm_end = (st[:48, 11].max() - t0) / 1e3
            gemm_x = (np.median(st[:48, 3]) - t0) / 1e3
            nrm = (st[48:48 + t, :6] - t0) / 1e3
            rows.append([gemm_x, gemm_end] + [float(np.median(nrm[:, i])) for i in range(6)])
        a = np.median(np.array(rows[5:]), axis=0)
        keys = ["gemm_x_ready", "gemm_last_cta_done", "norm_entry", "norm_wait_returned", "norm_split_sums",
                "norm_sinkhorn_step1", "norm_comb_stored", "norm_layer_input_stored"]
        out[f"T{t}"] = {k: round(float(v), 2) for k, v in zip(keys, a)}
        print(t, json.dumps(out[f"T{t}"]), flush=True)
    with open("/repo/results/2026-09-25-kernels/mhc-det/prof_tail.json", "w") as fh:
        json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
