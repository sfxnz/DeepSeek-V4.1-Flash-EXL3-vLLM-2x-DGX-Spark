"""Shared timing / build helpers for the dense-gemv study (runs inside the image).

Timing protocol (fixed before any result was seen):
- CUDA events around each call, >= 200 timed iterations after warmup.
- cold: a 64 MiB read-flush (> 2x the 24 MiB L2, clean lines) before every
  timed call, visiting 4 KiB chunks in hashed order: a linear sweep leaves a
  stream prefetcher running into whatever follows the flush buffer in memory
  (measured: 75.8 vs 95.2 us on a 21 MB read), the hashed order does not
  (95.2 us whatever the buffer placement; 256 MiB flush gives the same).
  Even hashed, one flush buffer left one specific 21 MB buffer warm (75.8 us,
  277 GB/s > the 273 GB/s DRAM peak) in 1 of 3 placements; two separately
  allocated 64 MiB flush buffers read back to back gave cold reads (93-100 us)
  for all 8 test copies, so the flush is two buffers. That still left some
  single buffers reading at 274-281 GB/s (above DRAM peak) while others of
  the same size read at 218-220: the L2/SLC keeps part of a buffer that is
  re-read every iteration (scan-resistant replacement; the flush lines are
  single-use). The serve reads each weight once per ~63 ms step with ~9.4 GB
  of other traffic in between, so no line is ever re-hit there. Hence the
  cold protocol: every arm rotates over R >= 16 distinct copies (>= 512 MiB
  total, real layer weights where available), arms read different copies in
  the same iteration (offset R/2), and the double hashed flush still runs
  before every timed call.
  warm: a memory-free spin kernel instead, same weights each time.
- arms are interleaved in the same process: iteration i runs every arm, in
  forward order on even i and reverse order on odd i (ABBA), each call
  preceded by its own flush / spacer.
- reported: median, p10, p90 (nearest rank), mean, min; bandwidth = unique
  bytes the call must move (weights + scales + activation in + output out) /
  median time, and % of 250 GB/s.
"""

from __future__ import annotations

import math
import os
import statistics

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PEAK_GBPS = 250.0
FLUSH_BYTES = 64 << 20


def load_ext(name: str, sources: list[str], extra_cuda: tuple[str, ...] = ()):
    from torch.utils.cpp_extension import load

    build = os.path.join(HERE, "build", name)
    os.makedirs(build, exist_ok=True)
    return load(
        name=name,
        sources=[os.path.join(HERE, s) for s in sources],
        build_directory=build,
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "-gencode=arch=compute_121a,code=sm_121a",
            "-lineinfo",
            "-Xptxas=-v",
            *extra_cuda,
        ],
        verbose=bool(os.environ.get("EXT_VERBOSE")),
    )


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(math.ceil(q * len(s))) - 1))]


def stats(xs: list[float]) -> dict:
    return {
        "median": statistics.median(xs),
        "p10": pct(xs, 0.10),
        "p90": pct(xs, 0.90),
        "mean": statistics.fmean(xs),
        "min": min(xs),
        "n": len(xs),
    }


class Flusher:
    def __init__(self, ext, nbytes: int = FLUSH_BYTES):
        self.ext = ext
        self.bufs = [torch.randint(0, 255, (nbytes,), dtype=torch.uint8, device="cuda")
                     for _ in range(2)]
        self.out = torch.zeros(4, dtype=torch.int32, device="cuda")

    def __call__(self):
        for b in self.bufs:
            self.ext.flush_perm(b, self.out)


def time_arms(arms: dict, iters: int = 200, warmup: int = 20, pre=None) -> dict[str, list[float]]:
    """Per-call microseconds for each arm; `pre` (flush or spacer) runs before every call."""
    names = list(arms)
    evs = {n: [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
               for _ in range(iters)] for n in names}
    for _ in range(warmup):
        for n in names:
            if pre is not None:
                pre()
            arms[n]()
    torch.cuda.synchronize()
    for i in range(iters):
        order = names if i % 2 == 0 else names[::-1]
        for n in order:
            if pre is not None:
                pre()
            s, e = evs[n][i]
            s.record()
            arms[n]()
            e.record()
    torch.cuda.synchronize()
    return {n: [s.elapsed_time(e) * 1000.0 for s, e in evs[n]] for n in names}


def gbps(nbytes: float, us: float) -> float:
    return nbytes / us / 1e3


def summarize(times: dict[str, list[float]], nbytes: dict[str, float] | float) -> dict:
    out = {}
    for n, xs in times.items():
        st = stats(xs)
        b = nbytes[n] if isinstance(nbytes, dict) else nbytes
        st["GBps_median"] = gbps(b, st["median"])
        st["pct_of_250"] = 100.0 * st["GBps_median"] / PEAK_GBPS
        st["bytes"] = b
        out[n] = st
    return out


class Rotation:
    """Round-robin over R distinct buffers; arm k starts at offset k*R//narms."""

    def __init__(self, bufs: list, narms: int = 2):
        self.bufs = bufs
        self.narms = narms
        self.t = [0] * narms

    def get(self, arm: int):
        R = len(self.bufs)
        i = (self.t[arm] + arm * R // self.narms) % R
        self.t[arm] += 1
        return self.bufs[i]


def copies_for(nbytes: int, min_copies: int = 16, min_total: int = 512 << 20, max_total: int = 1536 << 20) -> int:
    r = max(min_copies, -(-min_total // nbytes))
    return max(2, min(r, max_total // nbytes))
