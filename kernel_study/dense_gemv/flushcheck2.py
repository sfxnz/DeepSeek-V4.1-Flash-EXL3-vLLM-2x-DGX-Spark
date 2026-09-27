#!/usr/bin/env python3
"""Why does calib.py read 'cold' faster than flushcheck.py? Controlled variants."""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchutil import load_ext, summarize, time_arms

ext = load_ext("dgemv_calib", ["calib.cu"])
o = torch.zeros(4, dtype=torch.int32, device="cuda")
N, K = 5120, 4096
nb = N * K
chunk = ((nb + 319) // 320 + 511) // 512 * 512
def mk(w):
    return lambda: ext.chunks(w, o, chunk, 80, 128, 4, False)
def rep(tag, arms, pre):
    t = time_arms(arms, iters=200, pre=pre)
    s = summarize(t, nb)
    for k, v in s.items():
        print(f"{tag:34s} {k:10s} med {v['median']:8.3f} p10 {v['p10']:8.3f} p90 {v['p90']:8.3f} {v['GBps_median']:6.1f} GB/s", flush=True)

# A: flush allocated first, then w (calib order)
fb = torch.randint(0, 255, (64 << 20,), dtype=torch.uint8, device="cuda")
w = torch.randint(0, 255, (nb,), dtype=torch.uint8, device="cuda")
pre = lambda: ext.flush(fb, o)
rep("A flush-first single", {"a": mk(w)}, pre)
rep("A flush-first 2 arms same w", {"a": mk(w), "b": mk(w)}, pre)
rep("A flush-first 8 arms same w", {str(i): mk(w) for i in range(8)}, pre)
# B: w allocated first, then flush (flushcheck order)
w2 = torch.randint(0, 255, (nb,), dtype=torch.uint8, device="cuda")
fb2 = torch.empty(64 << 20, dtype=torch.uint8, device="cuda"); fb2.random_(0, 255)
pre2 = lambda: ext.flush(fb2, o)
rep("B w-first single", {"a": mk(w2)}, pre2)
rep("B w-first 8 arms", {str(i): mk(w2) for i in range(8)}, pre2)
# C: cross: w with fb2, w2 with fb
rep("C w/fb2", {"a": mk(w)}, pre2)
rep("C w2/fb", {"a": mk(w2)}, pre)
# D: flush 2x
rep("D w double flush", {"a": mk(w)}, lambda: (ext.flush(fb, o), ext.flush(fb2, o)))
# E: permuted flush, adjacency case A and B
rep("E perm-flush fb->w (adjacent)", {"a": mk(w)}, lambda: ext.flush_perm(fb, o))
rep("E perm-flush fb2->w2", {"a": mk(w2)}, lambda: ext.flush_perm(fb2, o))
rep("E perm-flush fb->w2", {"a": mk(w2)}, lambda: ext.flush_perm(fb, o))
rep("E perm fb + perm fb2 -> w", {"a": mk(w)}, lambda: (ext.flush_perm(fb, o), ext.flush_perm(fb2, o)))
big = torch.empty(256 << 20, dtype=torch.uint8, device="cuda"); big.random_(0, 255)
rep("E perm 256MiB -> w", {"a": mk(w)}, lambda: ext.flush_perm(big, o))
rep("E perm 256MiB -> w2", {"a": mk(w2)}, lambda: ext.flush_perm(big, o))
print("ptrs", hex(fb.data_ptr()), hex(w.data_ptr()), hex(w2.data_ptr()), hex(fb2.data_ptr()))
