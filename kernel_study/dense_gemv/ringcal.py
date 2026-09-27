#!/usr/bin/env python3
"""Does the per-row burst length (KC) of a per-warp cp.async ring decide the
GEMV weight-stream rate? Pure stream (no MMA), real shapes, cold protocol."""
import json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchutil import Flusher, Rotation, copies_for, load_ext, stats, time_arms

ext = load_ext("dgemv_calib", ["calib.cu"])
fl = Flusher(ext)
o = torch.zeros(4, dtype=torch.int32, device="cuda")
res = {}
for name, N, K in (("qkv_a", 1792, 5120), ("wo_b", 5120, 4096), ("wq_b", 16384, 1280), ("shared_down", 5120, 1152)):
    nb = N * K
    R = max(copies_for(nb), 64)
    copies = [torch.randint(0, 255, (nb,), dtype=torch.uint8, device="cuda") for _ in range(R)]
    T = N // 16
    fns = {"flat_g384_u8": lambda c: ext.flat(c, o, 384, 256, 8, False),
           "flat_g48_u2": lambda c: ext.flat(c, o, 48, 256, 2, False)}
    for kc, S in ((128, 4), (128, 8), (256, 3), (256, 4), (256, 6), (512, 2), (512, 3), (512, 4), (1024, 2), (1024, 3), (2048, 2)):
        if K % kc:
            continue
        for W in (1, 2, 4):
            smem = W * S * 16 * kc
            if smem > 101376:
                continue
            cps = min(max(1, 102400 // (smem + 1024)), 1536 // (32 * W))
            grid = min((T + W - 1) // W, 48 * cps)
            fns[f"ring_kc{kc}_s{S}_w{W}_g{grid}"] = (lambda c, kc=kc, S=S, W=W, grid=grid:
                                                      ext.ring(c, o, N, K, kc, S, W, grid))
    names = list(fns)
    rot = Rotation(copies, narms=len(names))
    arms = {n: (lambda n=n, k=k: fns[n](rot.get(k))) for k, n in enumerate(names)}
    t = time_arms(arms, iters=200, pre=fl)
    res[name] = {n: {**stats(v), "GBps": nb / stats(v)["median"] / 1e3} for n, v in t.items()}
    print(f"== {name} {nb/1e6:.2f} MB x{R}", flush=True)
    for n in sorted(names, key=lambda n: res[name][n]["median"])[:14]:
        v = res[name][n]
        print(f"   {n:28s} med {v['median']:8.2f} p10 {v['p10']:8.2f} p90 {v['p90']:8.2f} {v['GBps']:6.1f} GB/s", flush=True)
    del copies
    torch.cuda.empty_cache()
json.dump(res, open("/repo/results/2026-09-25-kernels/dense-gemv/ringcal.json", "w"), indent=1)
