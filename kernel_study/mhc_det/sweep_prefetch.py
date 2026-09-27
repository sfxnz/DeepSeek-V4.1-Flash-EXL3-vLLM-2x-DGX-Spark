#!/usr/bin/env python3
"""Sweep PREFETCH_STAGES (fn stages issued before griddepcontrol.wait) in the serve-like path graph."""
from __future__ import annotations

import json
import sys

import torch

sys.path.insert(0, "/repo/kernel_study/mhc_det")
import bench_path as BP  # noqa: E402
import common as C  # noqa: E402
import mhc_det  # noqa: E402


def main() -> int:
    w = BP.load_weights()
    emb = C.embeddings(64)
    fn_names = [k for k in w if k.endswith("_fn") or k.endswith("_broadcast")]
    packed = {k: mhc_det.pack_fn(w[k]) for k in fn_names}
    subs = BP.sublayers()
    variants = {p: mhc_det.DetKernels(opts=[f"-DPREFETCH_STAGES={p}"]) for p in (1, 2, 4, 6, 10)}
    out = {}
    for t in (4, 8):
        g = torch.Generator(device="cuda").manual_seed(9 + t)
        src = [(torch.randn(t, C.HIDDEN, device="cuda", generator=g) * 2).bfloat16() for _ in subs]
        xouts = [s.clone() for s in src]

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

        graphs = {f"pf{p}": BP._graph(make(BP.Path(w, dk, packed, True))) for p, dk in variants.items()}
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
        out[f"T{t}"] = {k: C.summarize(v) for k, v in samples.items()}
        print(t, json.dumps({k: (v["median_us"], v["p10_us"], v["p90_us"]) for k, v in out[f"T{t}"].items()}), flush=True)
    with open("/repo/results/2026-09-25-kernels/mhc-det/sweep_prefetch.json", "w") as fh:
        json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
