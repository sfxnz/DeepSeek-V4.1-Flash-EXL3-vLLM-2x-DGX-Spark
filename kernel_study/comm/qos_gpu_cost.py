#!/usr/bin/env python3
"""Does holding the PM QoS CPU wake-up request (DSV41_PM_QOS_US=20) cost the GPU anything?

k3 comm fix pass (review, C cost and scope). GB10's CPU and GPU share one SoC power budget;
with the request held, idle cores stay in WFI (LPI-0) instead of LPI-1/LPI-3. This runs
sustained GPU work in blocks with the request off / on, alternating (off, on, off, on, ...)
in one process, and compares per-kernel CUDA-event times plus nvidia-smi SM clock and power
sampled every 100 ms during each block (a sampler this script starts, same clock as its blocks):
  read  1 GiB streaming read (DRAM-bound, normal loads, 48 CTAs x 256 threads)
  gemm  bf16 4096^3 torch.mm (tensor-core bound)
The request is the same one pm_qos.py holds: open /dev/cpu_dma_latency, write the int32
latency bound, keep the fd open; closing it drops the request. Needs root and the device.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def stats(xs):
    s = sorted(xs)
    q = lambda f: s[min(len(s) - 1, int(round(f * (len(s) - 1))))]  # noqa: E731
    return {"median": round(statistics.median(s), 3), "p10": round(q(0.1), 3), "p90": round(q(0.9), 3),
            "n": len(s)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--us", type=int, default=20)
    ap.add_argument("--blocks", type=int, default=8, help="alternating off/on blocks (even)")
    ap.add_argument("--block-s", type=float, default=3.0)
    ap.add_argument("--smi-out", required=True, help="where the 100 ms nvidia-smi samples go")
    ap.add_argument("--build-dir", default="/repo/kernel_study/comm/.l2pf_var_build")
    ap.add_argument("--json")
    args = ap.parse_args()
    sys.path.insert(0, str(HERE))
    import torch

    import l2pf_variants

    ext = l2pf_variants.build(args.build_dir)
    dev = "cuda"
    buf = torch.empty(1 << 30, dtype=torch.uint8, device=dev)
    sink = torch.zeros(1, dtype=torch.int32, device=dev)
    a = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    c = torch.empty(4096, 4096, device=dev, dtype=torch.bfloat16)
    for _ in range(5):
        ext.k_read(buf, buf.numel(), 48, 0, sink)
        torch.mm(a, b, out=c)
    torch.cuda.synchronize()
    smi_f = open(args.smi_out, "w")
    smi = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,clocks.sm,clocks.mem,power.draw",
                            "--format=csv,noheader", "-lms", "100"], stdout=smi_f, stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    fd = None
    blocks = []
    for i in range(args.blocks):
        on = i % 2 == 1
        if on:
            fd = os.open("/dev/cpu_dma_latency", os.O_WRONLY)
            os.write(fd, struct.pack("i", args.us))
        t0 = time.time()
        reads, gemms = [], []
        while time.time() - t0 < args.block_s:
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            ev[0].record()
            ext.k_read(buf, buf.numel(), 48, 0, sink)
            ev[1].record()
            torch.mm(a, b, out=c)
            ev[2].record()
            ev[2].synchronize()
            reads.append(ev[0].elapsed_time(ev[1]) * 1e3)
            gemms.append(ev[1].elapsed_time(ev[2]) * 1e3)
        t1 = time.time()
        if on:
            os.close(fd)
            fd = None
        blocks.append({"qos": on, "t0": t0, "t1": t1, "read_us": reads[2:], "gemm_us": gemms[2:]})
        print(f"block {i} qos={'on' if on else 'off'} read {statistics.median(reads):.1f} us "
              f"gemm {statistics.median(gemms):.1f} us n={len(reads)}", flush=True)
        time.sleep(0.3)
    time.sleep(1.0)  # let the sampler flush
    smi.terminate()
    smi.wait()
    smi_f.close()

    def smi_rows():
        rows = []
        for line in Path(args.smi_out).read_text().splitlines():
            parts = [p.strip() for p in line.split(",")]
            try:
                ts = time.mktime(time.strptime(parts[0].split(".")[0], "%Y/%m/%d %H:%M:%S"))
                ts += float("0." + parts[0].split(".")[1]) if "." in parts[0] else 0.0
                rows.append((ts, float(parts[1].split()[0]), float(parts[3].split()[0])))
            except (ValueError, IndexError):
                continue
        return rows

    rows = smi_rows()
    out = {"what": "PM QoS request off/on, alternating blocks, sustained GPU work", "us": args.us,
           "read_bytes": buf.numel(), "gemm": "bf16 4096x4096x4096", "arms": {}}
    for arm in ("off", "on"):
        sel = [bk for bk in blocks if bk["qos"] == (arm == "on")]
        reads = [x for bk in sel for x in bk["read_us"]]
        gemms = [x for bk in sel for x in bk["gemm_us"]]
        samples = [r for r in rows for bk in sel if bk["t0"] + 0.5 <= r[0] <= bk["t1"]]
        r = stats(reads)
        g = stats(gemms)
        out["arms"][arm] = {"blocks": len(sel), "read_us": r, "read_gbps": round(buf.numel() / (r["median"] * 1e3), 1),
                            "gemm_us": g, "gemm_tflops": round(2 * 4096**3 / (g["median"] * 1e6), 1),
                            "sm_mhz": stats([x[1] for x in samples]) if samples else None,
                            "power_w": stats([x[2] for x in samples]) if samples else None}
        print(arm, json.dumps(out["arms"][arm]), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
