#!/usr/bin/env python3
"""A/B of GEMM compile variants in the serve-like path graph (det post + det GEMM + det norm).

Usage: sweep_variants.py name=-DFLAG=1,-DOTHER=2 name2=... (each arm is one NVRTC compile).
Also runs a bitwise check of every arm against stock on the whole recurrence at T=4 and 8.
"""
from __future__ import annotations

import json
import sys

import torch

sys.path.insert(0, "/repo/kernel_study/mhc_det")
import bench_path as BP  # noqa: E402
import common as C  # noqa: E402
import mhc_det  # noqa: E402


def main() -> int:
    arms = {}
    for a in sys.argv[1:]:
        name, _, opts = a.partition("=")
        arms[name] = [o for o in opts.split(",") if o]
    w = BP.load_weights()
    emb = C.embeddings(64)
    fn_names = [k for k in w if k.endswith("_fn") or k.endswith("_broadcast")]
    packed = {k: mhc_det.pack_fn(w[k]) for k in fn_names}
    subs = BP.sublayers()
    kern = {n: mhc_det.DetKernels(opts=o) for n, o in arms.items()}
    out = {"arms": arms}
    stock = BP.Path(w, None, packed, False)
    for t in (4, 8):
        g = torch.Generator(device="cuda").manual_seed(9 + t)
        src = [(torch.randn(t, C.HIDDEN, device="cuda", generator=g) * 2).bfloat16() for _ in subs]
        xouts = [s.clone() for s in src]
        ref = []
        BP.run_chain(stock, t, emb, xouts, ref)
        ok = {}
        for n, dk in kern.items():
            rec = []
            BP.run_chain(BP.Path(w, dk, packed, True), t, emb, xouts, rec)
            ok[n] = all(bool((BP.ints(a) == BP.ints(b)).all()) for ra, rb in zip(ref, rec) for a, b in zip(ra, rb))

        def make(path):
            def body():
                state = None
                for i, (prefix, sub, first) in enumerate(subs):
                    xouts[i].copy_(src[i])
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
                xouts[i].copy_(src[i])

        graphs = {"copies": BP._graph(copies)}
        graphs.update({n: BP._graph(make(BP.Path(w, dk, packed, True))) for n, dk in kern.items()})
        for _ in range(3):
            for gr in graphs.values():
                gr.replay()
        torch.cuda.synchronize()
        samples = {k: [] for k in graphs}
        for _ in range(200):
            for k, gr in graphs.items():
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
                gr.replay()
                b.record()
                b.synchronize()
                samples[k].append(a.elapsed_time(b) * 1000.0)
        res = {}
        for n in kern:
            per = [x - y for x, y in zip(samples[n], samples["copies"])]
            st = C.summarize(per)
            res[n] = {"bitwise_vs_stock": ok[n], "median_us": st["median_us"], "p10_us": st["p10_us"], "p90_us": st["p90_us"]}
        out[f"T{t}"] = res
        print(t, json.dumps(res), flush=True)
    with open("/repo/results/2026-09-25-kernels/mhc-det/sweep_variants.json", "w") as fh:
        json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
