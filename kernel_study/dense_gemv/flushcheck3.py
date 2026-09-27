#!/usr/bin/env python3
"""calib.py's many-arm interleave reads 'cold' 21 MB in 75.8 us, flushcheck2's
single arm in 95.2 us with the same kernel and flush. Find the variable."""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchutil import Flusher, load_ext, summarize, time_arms

ext = load_ext("dgemv_calib", ["calib.cu"])
fl = Flusher(ext)
o = torch.zeros(4, dtype=torch.int32, device="cuda")
N, K = 5120, 4096
nb = N * K
w = torch.randint(0, 255, (nb,), dtype=torch.uint8, device="cuda")
w_alt = torch.randint(0, 255, (nb,), dtype=torch.uint8, device="cuda")
chunk = 65536
single = lambda: ext.chunks(w, o, chunk, 80, 128, 4, False)
alt = lambda: ext.chunks(w_alt, o, chunk, 80, 128, 4, False)
def rep(tag, arms, pre=fl):
    s = summarize(time_arms(arms, iters=200, pre=pre), nb)
    for k in ("single", "alt"):
        if k in s:
            v = s[k]
            print(f"{tag:40s} {k:7s} med {v['median']:8.3f} mean {v['mean']:8.3f} p10 {v['p10']:8.3f} p90 {v['p90']:8.3f} {v['GBps_median']:6.1f} GB/s", flush=True)

rep("1 single alone", {"single": single})
rep("2 single + alt", {"single": single, "alt": alt})
flat = {f"flat{g}_{u}": (lambda g=g, u=u: ext.flat(w, o, g, 256, u, False)) for g in (48, 192, 384) for u in (2, 4, 8)}
rep("3 single + 9 flat(w)", {"single": single, **flat})
rep("4 single + alt + 9 flat(w)", {"single": single, "alt": alt, **flat})
tiles = {f"tiles{d}": (lambda d=d: ext.tiles(w, o, N, K, 80, 128, d, False)) for d in (2, 4, 8)}
rep("5 single + 3 tiles(w)", {"single": single, **tiles})
rep("6 single + alt + 3 tiles(w)", {"single": single, "alt": alt, **tiles})
