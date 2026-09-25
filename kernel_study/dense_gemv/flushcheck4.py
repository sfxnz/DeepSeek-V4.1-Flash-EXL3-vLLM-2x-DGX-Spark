#!/usr/bin/env python3
"""Placement-dependent retention: w right after the flush buffer reads warm even
with the hashed flush. Does rotating over R distinct copies make every read cold?"""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchutil import Flusher, load_ext, summarize, time_arms

ext = load_ext("dgemv_calib", ["calib.cu"])
fl = Flusher(ext)
o = torch.zeros(4, dtype=torch.int32, device="cuda")
N, K = 5120, 4096
nb = N * K
copies = [torch.randint(0, 255, (nb,), dtype=torch.uint8, device="cuda") for _ in range(8)]
print("flush", hex(fl.buf.data_ptr()), "copies", [hex(c.data_ptr()) for c in copies])
arms = {f"c{i}": (lambda c=c: ext.chunks(c, o, 65536, 80, 128, 4, False)) for i, c in enumerate(copies)}
s = summarize(time_arms(arms, iters=200, pre=fl), nb)
for k, v in s.items():
    print(f"interleaved  {k:4s} med {v['median']:8.3f} mean {v['mean']:8.3f} p10 {v['p10']:8.3f} p90 {v['p90']:8.3f} {v['GBps_median']:6.1f} GB/s", flush=True)
# rotation: one arm that reads the next copy each call
state = {"i": 0}
def rot():
    c = copies[state["i"] % len(copies)]
    state["i"] += 1
    ext.chunks(c, o, 65536, 80, 128, 4, False)
v = summarize(time_arms({"rot": rot}, iters=400, pre=fl), nb)["rot"]
print(f"rotating 8   med {v['median']:8.3f} mean {v['mean']:8.3f} p10 {v['p10']:8.3f} p90 {v['p90']:8.3f} {v['GBps_median']:6.1f} GB/s")
# a second, later-allocated flush buffer
fl2 = Flusher(ext)
print("flush2", hex(fl2.buf.data_ptr()))
s = summarize(time_arms(arms, iters=200, pre=fl2), nb)
for k, v in s.items():
    print(f"flush2       {k:4s} med {v['median']:8.3f} mean {v['mean']:8.3f} p10 {v['p10']:8.3f} p90 {v['p90']:8.3f} {v['GBps_median']:6.1f} GB/s", flush=True)
s = summarize(time_arms(arms, iters=200, pre=lambda: (fl(), fl2())), nb)
for k, v in s.items():
    print(f"flush1+2     {k:4s} med {v['median']:8.3f} mean {v['mean']:8.3f} p10 {v['p10']:8.3f} p90 {v['p90']:8.3f} {v['GBps_median']:6.1f} GB/s", flush=True)
