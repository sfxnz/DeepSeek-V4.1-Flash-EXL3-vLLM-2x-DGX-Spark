#!/usr/bin/env python3
"""Activation staging cost in v4: queued behind the weight prologue (default)
vs issued first (diag 64); hot activation (touched after the flush) vs cold."""
import json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchutil import Flusher, Rotation, load_ext, stats, time_arms
from bench4 import v4_scales
from timing import build_copies

ext = load_ext("dgemv_calib", ["calib.cu"])
ext1 = load_ext("dgemv_v1", ["gemv_ext.cu"])
ext4 = load_ext("dgemv_v4", ["gemv4_ext.cu"])
fl = Flusher(ext)
o = torch.zeros(4, dtype=torch.int32, device="cuda")
res = {}
for name, N, K, M, cfg in (("qkv_a", 1792, 5120, 4, (4, 2, 512, 4, 28)), ("qkv_a", 1792, 5120, 8, (2, 3, 512, 8, 48)),
                           ("wo_b", 5120, 4096, 4, (4, 2, 512, 4, 48))):
    copies = build_copies(name, 64)
    scs = {id(c[0]): v4_scales(c, N, K, cfg[2])[1] for c in copies}
    x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
    y = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    fns = {}
    for d in (0, 2, 64):
        fns[f"diag{d}"] = (lambda c, d=d: ext4.gemv4(x, None, None, c[0], scs[id(c[0])], 0, y, *cfg, False, d))
    names = list(fns)
    for hot in (False, True):
        rot = Rotation(copies, narms=len(names))
        arms = {n: (lambda n=n, k=k: fns[n](rot.get(k))) for k, n in enumerate(names)}
        pre = (lambda: (fl(), ext1.read_flat(x.view(torch.uint8), o, 8, 2))) if hot else fl
        t = time_arms(arms, iters=200, pre=pre)
        for n in names:
            st = stats(t[n])
            key = f"{name}_M{M}_{'hot' if hot else 'cold'}act_{n}"
            res[key] = st
            print(f"{key:36s} med {st['median']:8.2f} p10 {st['p10']:8.2f} p90 {st['p90']:8.2f}", flush=True)
    del copies
json.dump(res, open("/repo/results/2026-09-25-kernels/dense-gemv/ringcal4.json", "w"), indent=1)
