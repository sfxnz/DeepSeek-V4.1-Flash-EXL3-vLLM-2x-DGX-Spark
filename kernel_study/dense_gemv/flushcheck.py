#!/usr/bin/env python3
"""Is the read-flush really cold? Time one read arm after flushes of growing
size (read vs write flush) and dump raw event samples to see the timer quantum."""
import json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchutil import load_ext, summarize, time_arms

ext = load_ext("dgemv_calib", ["calib.cu"])
o = torch.zeros(4, dtype=torch.int32, device="cuda")
res = {}
for name, N, K in (("wo_b", 5120, 4096), ("qkv_a", 1792, 5120), ("shared_down", 5120, 1152)):
    w = torch.randint(0, 255, (N * K,), dtype=torch.uint8, device="cuda")
    arm = lambda: ext.chunks(w, o, ((N * K + N // 16 - 1) // (N // 16) + 511) // 512 * 512, (N // 16 + 3) // 4, 128, 4, False)
    for mib in (64, 128, 256, 512, 1024):
        buf = torch.empty(mib << 20, dtype=torch.uint8, device="cuda")
        buf.random_(0, 255)
        for mode in ("read", "write"):
            pre = (lambda: ext.flush(buf, o)) if mode == "read" else (lambda: ext.flush_write(buf, 7))
            t = time_arms({"a": arm}, iters=200, pre=pre)
            s = summarize(t, N * K)["a"]
            key = f"{name}_{mode}{mib}"
            res[key] = {"stats": s, "raw_head": sorted(t["a"])[:5] + sorted(t["a"])[-5:]}
            print(f"{key:24s} med {s['median']:8.3f} mean {s['mean']:8.3f} p10 {s['p10']:8.3f} p90 {s['p90']:8.3f} "
                  f"{s['GBps_median']:6.1f} GB/s  distinct={len(set(round(x,3) for x in t['a']))} "
                  f"vals={sorted(set(round(x,3) for x in t['a']))[:6]}", flush=True)
        del buf
        torch.cuda.empty_cache()
json.dump(res, open("/repo/results/2026-09-25-kernels/dense-gemv/flushcheck.json", "w"), indent=1)
