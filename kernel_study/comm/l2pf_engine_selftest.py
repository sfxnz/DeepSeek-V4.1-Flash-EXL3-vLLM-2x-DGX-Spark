#!/usr/bin/env python3
"""Engine selftest for l2pf_variants (single GPU, seconds): every engine launches eagerly and
inside a CUDA graph without an error, and each one actually warms L2 (qkv_a-sized buffer
read time after a 2x-L2 streaming flush: cold vs after the engine ran to completion).
Also reports the device's shared-memory opt-in limit and the persisting-L2 limits."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "docker" / "patch"))

ENGINES = ("tri", "burst", "burst:c8", "burst:c48", "pf:c48", "pf:c192", "paced:60", "paced:150", "ring:2x16",
           "ring:4x16", "ring:6x16", "tload:32", "tload:64", "burst+last", "burst:c48+last", "pf:c48+last",
           "ring:6x16+last")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="/repo/kernel_study/comm/.l2pf_var_build")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--json")
    args = ap.parse_args()
    import torch

    import l2pf_kernel
    import l2pf_variants

    ext = l2pf_variants.build(args.build_dir)
    launch = l2pf_variants.launcher(ext, l2pf_kernel.launcher(torch), 2190.0)
    props = torch.cuda.get_device_properties(0)
    l2 = int(props.L2_cache_size)
    info = {"device": props.name, "l2_bytes": l2}
    try:
        from cuda.bindings import runtime as rt

        def attr(name):
            return rt.cudaDeviceGetAttribute(getattr(rt.cudaDeviceAttr, name), 0)[1]

        info.update(smem_optin=attr("cudaDevAttrMaxSharedMemoryPerBlockOptin"),
                    max_persisting_l2=attr("cudaDevAttrMaxPersistingL2CacheSize"),
                    persisting_l2_limit=rt.cudaDeviceGetLimit(rt.cudaLimit.cudaLimitPersistingL2CacheSize)[1])
    except Exception as exc:  # noqa: BLE001 - informational only
        info["attr_error"] = repr(exc)
    print(json.dumps(info), flush=True)
    buf = torch.randint(0, 255, (9461760,), dtype=torch.uint8, device="cuda")
    flush = torch.empty(2 * l2, dtype=torch.uint8, device="cuda")
    sink = torch.zeros(1, dtype=torch.int32, device="cuda")
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def read_us(prep):
        ts = []
        for _ in range(args.iters):
            ext.k_read(flush, flush.numel(), 48, 0, sink)
            prep()
            torch.cuda.synchronize()
            e0.record()
            ext.k_read(buf, buf.numel(), 48, 0, sink)
            e1.record()
            e1.synchronize()
            ts.append(e0.elapsed_time(e1) * 1e3)
        return round(statistics.median(ts), 2)

    res = {"cold_read_us": read_us(lambda: None), "warm_read_us": read_us(lambda: ext.k_read(buf, buf.numel(), 48, 0, sink))}
    ok = True
    for spec in ENGINES:
        e = l2pf_variants.parse_engine(spec)
        row = {}
        try:
            launch(e, buf, buf.numel() & ~15)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            with torch.cuda.graph(g, stream=s):
                launch(e, buf, buf.numel() & ~15)
            g.replay()
            torch.cuda.synchronize()
            row["read_after_us"] = read_us(lambda: launch(e, buf, buf.numel() & ~15))
            ts = []
            for _ in range(args.iters):
                ext.k_read(flush, flush.numel(), 48, 0, sink)
                torch.cuda.synchronize()
                e0.record()
                launch(e, buf, buf.numel() & ~15)
                e1.record()
                e1.synchronize()
                ts.append(e0.elapsed_time(e1) * 1e3)
            row["engine_us"] = round(statistics.median(ts), 2)
            row["engine_gbps"] = round(buf.numel() / (row["engine_us"] * 1e3), 1)
            if e["last"]:
                ext.k_demote(buf, buf.numel() & ~127, 48)
                torch.cuda.synchronize()
                row["read_after_demote_flush_us"] = read_us(lambda: None)
            row["ok"] = True
        except Exception as exc:  # noqa: BLE001 - report every engine
            row = {"ok": False, "error": repr(exc)[:300]}
            ok = False
            torch.cuda.synchronize()
        res[spec] = row
        print(spec, json.dumps(row), flush=True)
    out = {**info, **res, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
