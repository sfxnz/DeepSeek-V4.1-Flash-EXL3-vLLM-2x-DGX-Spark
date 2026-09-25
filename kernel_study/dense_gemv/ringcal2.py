#!/usr/bin/env python3
"""Why is the v4 GEMV stream (no MMA, no staging) slower than ring_read with the
same ring? Diag bits of gemv4: 1 no MMA, 2 no staging, 4 no scale loads,
8 ring_read's shallow schedule (prologue S-1, wait S-2). Cold protocol."""
import json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchutil import Flusher, Rotation, load_ext, stats, time_arms
from bench4 import v4_scales
from timing import build_copies

ext = load_ext("dgemv_calib", ["calib.cu"])
ext4 = load_ext("dgemv_v4", ["gemv4_ext.cu"])
fl = Flusher(ext)
o = torch.zeros(4, dtype=torch.int32, device="cuda")
res = {}
N, K = 1792, 5120
copies = build_copies("qkv_a", 64)
scs = {id(c[0]): v4_scales(c, N, K, 512)[1] for c in copies}
x = torch.randn(4, K, device="cuda").to(torch.bfloat16)
y = torch.empty(4, N, dtype=torch.bfloat16, device="cuda")
fns = {"ring_k512_s2_w4_g28": lambda c: ext.ring(c[0], o, N, K, 512, 2, 4, 28, 0),
       "ring_k512_s3_w4_g28": lambda c: ext.ring(c[0], o, N, K, 512, 3, 4, 28, 0)}
for d in (0, 3, 15, 19, 31):
    fns[f"gemv4_w4s2k512m4_diag{d}"] = (lambda c, d=d: ext4.gemv4(x, None, None, c[0], scs[id(c[0])], 0, y,
                                                                  4, 2, 512, 4, 28, False, d))
    fns[f"gemv4_w2s3k512m8_diag{d}"] = (lambda c, d=d: ext4.gemv4(x, None, None, c[0], scs[id(c[0])], 0, y,
                                                                  2, 3, 512, 8, 48, False, d))
names = list(fns)
rot = Rotation(copies, narms=len(names))
arms = {n: (lambda n=n, k=k: fns[n](rot.get(k))) for k, n in enumerate(names)}
t = time_arms(arms, iters=200, pre=fl)
for n in names:
    st = stats(t[n])
    res[n] = st
    print(f"{n:34s} med {st['median']:8.2f} p10 {st['p10']:8.2f} p90 {st['p90']:8.2f}", flush=True)
json.dump(res, open("/repo/results/2026-09-25-kernels/dense-gemv/ringcal2.json", "w"), indent=1)
