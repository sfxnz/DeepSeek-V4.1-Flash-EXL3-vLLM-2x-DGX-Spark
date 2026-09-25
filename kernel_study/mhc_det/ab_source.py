#!/usr/bin/env python3
"""Paired A/B of two mhc_det.cu sources in the serve-like path graph (serve image, spark2).

  ab_source.py <alt.cu> <tag> [--pdl-proxy] [--iters N]

Arms: "shipped" = docker/patch/mhc_det.cu, "alt" = <alt.cu> (each one NVRTC compile), plus the
stock kernels for the bitwise reference. Per T in {1, 3, 4, 6, 8}: every output of the whole
43-layer recurrence of both arms is compared bitwise with stock first; then one CUDA graph per
arm (86 sublayers, each preceded by the AR proxy: a plain copy, or with --pdl-proxy a
PDL-launched copy that waits on its primary) and a copies-only graph, replayed round-robin
(the shipped/alt order flips every rep), N reps. mHC us per pass = t(arm) - t(copies) of the
same rep; the paired delta alt - shipped is taken rep by rep.
Writes results/2026-09-25-kernels/mhc-det/ab_<tag>.json.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys

import torch

sys.path.insert(0, "/repo/kernel_study/mhc_det")
import bench_path as BP  # noqa: E402
import common as C  # noqa: E402
import mhc_det  # noqa: E402
from mhc_det_rt import Module  # noqa: E402

PDL_COPY = r"""
extern "C" __global__ void pdl_copy(const uint4* __restrict__ src, uint4* __restrict__ dst, int n) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) dst[i] = src[i];
}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("alt")
    ap.add_argument("tag")
    ap.add_argument("--pdl-proxy", action="store_true")
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    w = BP.load_weights()
    emb = C.embeddings(64)
    fn_names = [k for k in w if k.endswith("_fn") or k.endswith("_broadcast")]
    packed = {k: mhc_det.pack_fn(w[k]) for k in fn_names}
    subs = BP.sublayers()
    kern = {"shipped": mhc_det.DetKernels(), "alt": mhc_det.DetKernels(src_path=args.alt)}
    pcopy = Module(PDL_COPY, "pdl_copy.cu").function("pdl_copy")

    def proxy(dst, src):
        if not args.pdl_proxy:
            dst.copy_(src)
            return
        n = dst.numel() * dst.element_size() // 16
        pcopy.launch((min(48, max(1, n // 128)),), (128,), 0,
                     [(src.data_ptr(), ctypes.c_void_p), (dst.data_ptr(), ctypes.c_void_p), (n, ctypes.c_int)],
                     pdl=True)

    res = {"alt": args.alt, "pdl_proxy": args.pdl_proxy, "iters": args.iters, "device": torch.cuda.get_device_name(),
           "rows": []}
    for t in (1, 3, 4, 6, 8):
        g = torch.Generator(device="cuda").manual_seed(9 + t)
        src = [(torch.randn(t, C.HIDDEN, device="cuda", generator=g) * 2).bfloat16() for _ in subs]
        xouts = [s.clone() for s in src]
        ref = []
        BP.run_chain(BP.Path(w, None, packed, False), t, emb, xouts, ref)
        ok = {}
        for n, dk in kern.items():
            rec = []
            BP.run_chain(BP.Path(w, dk, packed, True), t, emb, xouts, rec)
            ok[n] = all(bool((BP.ints(a) == BP.ints(b)).all()) for ra, rb in zip(ref, rec) for a, b in zip(ra, rb))

        def make(path):
            def body():
                state = None
                for i, (prefix, sub, first) in enumerate(subs):
                    proxy(xouts[i], src[i])
                    if first and prefix == "layers.0":
                        e = emb[:t]
                        residual = e.unsqueeze(1).expand(-1, C.HC, -1).contiguous()
                        pm, cm, li, pr = path.pre(residual, "layers.0.hc_attn_fn_broadcast", prefix, sub, None, x=e)
                    elif first:
                        residual = emb[:t].unsqueeze(-2).repeat(1, C.HC, 1)
                        pm, cm, li, pr = path.pre(residual, f"{prefix}.hc_{sub}_fn", prefix, sub, None)
                    else:
                        rp, pp, cp, prp = state
                        residual = path.post(xouts[i], rp, pp, cp)
                        pm, cm, li, pr = path.pre(residual, f"{prefix}.hc_{sub}_fn", prefix, sub, prp)
                    state = (residual, pm, cm, pr)
            return body

        def copies():
            for i in range(len(subs)):
                proxy(xouts[i], src[i])

        graphs = {"copies": BP._graph(copies)}
        graphs.update({n: BP._graph(make(BP.Path(w, dk, packed, True))) for n, dk in kern.items()})
        for _ in range(3):
            for gr in graphs.values():
                gr.replay()
        torch.cuda.synchronize()
        samples = {k: [] for k in graphs}
        names = list(graphs)
        for it in range(args.iters):
            order = names if it % 2 == 0 else [names[0], names[2], names[1]]
            for k in order:
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
                graphs[k].replay()
                b.record()
                b.synchronize()
                samples[k].append(a.elapsed_time(b) * 1000.0)
        row = {"T": t, "copies_only_us": C.summarize(samples["copies"])["median_us"]}
        for n in kern:
            st = C.summarize([x - y for x, y in zip(samples[n], samples["copies"])])
            row[n] = {"bitwise_vs_stock": ok[n], "median_us": st["median_us"], "p10_us": st["p10_us"],
                      "p90_us": st["p90_us"]}
        d = C.summarize([x - y for x, y in zip(samples["alt"], samples["shipped"])])
        row["alt_minus_shipped_paired"] = {"median_us": d["median_us"], "p10_us": d["p10_us"], "p90_us": d["p90_us"]}
        res["rows"].append(row)
        print(json.dumps(row), flush=True)
    with open(f"{BP.OUT}/ab_{args.tag}.json", "w") as fh:
        json.dump(res, fh, indent=1)
    ok = all(r[n]["bitwise_vs_stock"] for r in res["rows"] for n in kern)
    print("bitwise PASS" if ok else "bitwise FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
