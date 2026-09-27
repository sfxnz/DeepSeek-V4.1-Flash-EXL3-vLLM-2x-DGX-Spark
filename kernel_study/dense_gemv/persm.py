#!/usr/bin/env python3
"""Per-SM streaming capacity: read a 64 MiB weight-like buffer with X CTAs
(one per SM) of T threads, U 16-B loads in flight per thread (LDG) - how many
GB/s can one SM pull, and how many SMs does 250 GB/s need? Cold protocol."""
import json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchutil import Flusher, Rotation, load_ext, stats, time_arms

ext = load_ext("dgemv_calib", ["calib.cu"])
fl = Flusher(ext)
o = torch.zeros(4, dtype=torch.int32, device="cuda")
nb = 32 << 20
copies = [torch.randint(0, 255, (nb,), dtype=torch.uint8, device="cuda") for _ in range(24)]
fns = {}
for X in (4, 8, 16, 28, 36, 48):
    for T, U in ((256, 8), (1024, 4), (1024, 8)):
        fns[f"X{X}_T{T}_U{U}"] = (lambda c, X=X, T=T, U=U: ext.flat(c, o, X, T, U, False))
names = list(fns)
rot = Rotation(copies, narms=len(names))
arms = {n: (lambda n=n, k=k: fns[n](rot.get(k))) for k, n in enumerate(names)}
t = time_arms(arms, iters=100, pre=fl)
res = {}
for n in names:
    st = stats(t[n])
    X = int(n.split("_")[0][1:])
    gbps = nb / st["median"] / 1e3
    res[n] = {**st, "GBps": gbps, "GBps_per_SM": gbps / X}
    print(f"{n:16s} med {st['median']:8.1f} us  {gbps:6.1f} GB/s  {gbps / X:5.2f} GB/s/SM", flush=True)
json.dump(res, open("/repo/results/2026-09-25-kernels/dense-gemv/persm.json", "w"), indent=1)
