#!/usr/bin/env python3
"""torch.profiler breakdown of the serve-like mHC path graph (stock vs det), T=4 and 8.

Per kernel name: calls per pass, median duration, and the median gap to the previous kernel
(device idle between consecutive kernels inside the graph).
"""
from __future__ import annotations

import collections
import json
import statistics
import sys

import torch

sys.path.insert(0, "/repo/kernel_study/mhc_det")
import bench_path as BP  # noqa: E402
import common as C  # noqa: E402
import mhc_det  # noqa: E402


def main() -> int:
    w = BP.load_weights()
    emb = C.embeddings(64)
    dk = mhc_det.DetKernels()
    fn_names = [k for k in w if k.endswith("_fn") or k.endswith("_broadcast")]
    packed = {k: mhc_det.pack_fn(w[k]) for k in fn_names}
    subs = BP.sublayers()
    out = {}
    for t in (4, 8):
        g = torch.Generator(device="cuda").manual_seed(9 + t)
        src = [(torch.randn(t, C.HIDDEN, device="cuda", generator=g) * 2).bfloat16() for _ in subs]
        xouts = [s.clone() for s in src]
        for det in (False, True):
            path = BP.Path(w, dk, packed, det)

            def with_copies(path=path):
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

            gr = BP._graph(with_copies)
            for _ in range(5):
                gr.replay()
            torch.cuda.synchronize()
            reps = 20
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(reps):
                    gr.replay()
                torch.cuda.synchronize()
            evs = sorted([e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA],
                         key=lambda e: e.time_range.start)
            dur = collections.defaultdict(list)
            gap = collections.defaultdict(list)
            prev_end = None
            for e in evs:
                name = e.name[:60]
                dur[name].append(e.time_range.elapsed_us())
                if prev_end is not None:
                    gap[name].append(e.time_range.start - prev_end)
                prev_end = e.time_range.end
            key = f"T{t}_{'det' if det else 'stock'}"
            out[key] = {n: {"calls_per_pass": len(v) / reps, "median_us": round(statistics.median(v), 2),
                            "median_gap_before_us": round(statistics.median(gap[n]), 2) if gap[n] else None,
                            "sum_per_pass_us": round(sum(v) / reps, 1)}
                        for n, v in sorted(dur.items(), key=lambda kv: -sum(kv[1]))}
            print(key, json.dumps(out[key], indent=None), flush=True)
    with open("/repo/results/2026-09-25-kernels/mhc-det/prof_path.json", "w") as fh:
        json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
