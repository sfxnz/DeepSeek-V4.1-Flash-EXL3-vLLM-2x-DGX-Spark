#!/usr/bin/env python3
"""Compile docker/patch/mhc_det.cu with NVRTC for this GPU and save the cubin (SASS via host cuobjdump)."""
import os
import sys

sys.path.insert(0, "/repo/docker/patch")
from mhc_det_rt import Module  # noqa: E402

src_path = sys.argv[1] if len(sys.argv) > 1 else "/repo/docker/patch/mhc_det.cu"
out = sys.argv[2] if len(sys.argv) > 2 else "/repo/results/2026-09-25-kernels/mhc-det/sass/mhc_det.cubin"
opts = sys.argv[3:]
os.makedirs(os.path.dirname(out), exist_ok=True)
m = Module(open(src_path).read(), os.path.basename(src_path), opts=opts)
open(out, "wb").write(m.cubin)
print("log:", m.log[-2000:])
print("wrote", out, len(m.cubin))
