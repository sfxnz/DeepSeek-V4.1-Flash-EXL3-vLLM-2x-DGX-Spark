#!/usr/bin/env python3
"""CPU critical path of EngramDiskStager.stage(): stock gather vs native gather.

What the GPU waits on every decode step: after hashes_ready.synchronize()
returns, the stock stage submits one _fast_stage_one per engram table to the
16-thread stage pool (gather v2 preadv loop + torch dequant chain + copy into
the pinned staging buffer) and blocks on the futures; only then are the H2D
copies and the target graph launched. This bench times exactly that span with
the image's real classes (DiskEngramTable with gather v2 + census, the real
disk_file_rows_owned), on the real tables, for:

  stock    the engram_stage_fast body (pool, 2 futures, fut.exception() waits)
  native   eng_gather_bf16 per table, called inline on the main thread
  native2  eng_gather_bf16 per table on the stage pool (GIL released inside)

Arms alternate inside every iteration (ABC, BCA, CAB ...). Each call gathers a
fresh set of ids (hash rows are new every step). Page-cache scenarios:
  hot     every row was read once before timing (prefetch v3 hit, pf_hit 100%)
  pf      rows dropped (DONTNEED), WILLNEED issued --lead-us before the call
          (the prefetch worker's head start), then timed
  cold    rows dropped, no WILLNEED (a prefetch miss)
Optional --noise: a Python thread spinning pure-Python work during each timed
call, the GIL pressure of a still-running prefetch worker.
Prints one JSON line: per scenario and arm, median/p10/p90/mean us, n.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "docker" / "patch" / "engram_native.c"


def build(tmp: Path) -> ctypes.CDLL:
    so = tmp / "engram_native.so"
    subprocess.run(["gcc", "-O3", "-shared", "-fPIC", "-o", str(so), str(SRC)], check=True)
    lib = ctypes.CDLL(str(so))
    lib.eng_gather_bf16.restype = ctypes.c_int
    lib.eng_gather_bf16.argtypes = [
        ctypes.c_int, ctypes.c_int64, ctypes.c_int, ctypes.c_int64,
        ctypes.c_int, ctypes.c_int, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_void_p,
    ]
    return lib


def load_text_config(snapshot: str):
    with open(os.path.join(snapshot, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    return SimpleNamespace(**cfg.get("text_config", cfg))


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def summarize(v):
    return dict(
        n=len(v), median=statistics.median(v), p10=pct(v, 0.10), p90=pct(v, 0.90),
        mean=statistics.fmean(v), min=min(v), max=max(v),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--tokens", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--scenarios", nargs="+", default=["hot", "pf", "cold"])
    ap.add_argument("--lead-us", type=float, default=500.0)
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--pin", action="store_true", help="pinned staging buffers (needs CUDA)")
    args = ap.parse_args()

    from vllm.models.deepseek_v4_1.common import engram as eng
    from vllm.models.deepseek_v4_1.common.engram_disk import DiskEngramTable

    layout = eng.EngramLayout(load_text_config(args.snapshot))
    heads = layout.n_hash_cols
    part = -(-heads // 2)
    h0 = args.rank * part
    h1 = min(h0 + part, heads)
    tables = []
    for li, layer_id in enumerate(layout.layer_ids):
        disk = DiskEngramTable(args.snapshot, layer_id, layout.head_dim, 32)
        sizes = [p for order in layout.primes[li] for p in order]
        v0, v1 = sum(sizes[:h0]), sum(sizes[:h0 + part])
        emb = SimpleNamespace(part_n_hash_cols=part, vocab_start_idx=v0, vocab_end_idx=v1, disk=disk)
        emb.disk_file_rows_owned = MethodType(eng.ParallelEngramEmbedding.disk_file_rows_owned, emb)
        offs = [sum(sizes[:c]) for c in range(heads)]
        tables.append(SimpleNamespace(layer_hash_index=li, embed_tokens=emb, sizes=sizes, offs=offs, v0=v0, v1=v1))
    nl = len(tables)
    dim, sb = layout.head_dim, layout.head_dim // 32
    max_n = max(args.tokens)
    pin = dict(pin_memory=True) if args.pin else {}
    hash_host = torch.empty((max_n, nl, h1 - h0), dtype=torch.int32, **pin)
    rows_host = [torch.empty((max_n, part, dim), dtype=torch.bfloat16, **pin) for _ in tables]
    sw = [torch.empty((max_n * part, dim), dtype=torch.uint8) for _ in tables]
    ss = [torch.empty((max_n * part, sb), dtype=torch.uint8) for _ in tables]
    stats = [(ctypes.c_int64 * 5)() for _ in tables]
    pool = ThreadPoolExecutor(max_workers=16)

    with tempfile.TemporaryDirectory() as tmpd:
        lib = build(Path(tmpd))
    lut = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).to(torch.float32).contiguous()
    rng = random.Random(1234 + args.rank)

    def make_ids(n):
        return [
            [[t.offs[c] + rng.randrange(t.sizes[c]) for c in range(h0, h1)] for _ in range(n)]
            for t in tables
        ]

    def load_host(ids, n):
        hash_host[:n] = torch.tensor(ids, dtype=torch.int32).permute(1, 0, 2)

    def stock(n):
        host = hash_host[:n]

        def _fast_stage_one(engram, buf):
            local = host[:, engram.layer_hash_index, :].to(torch.int64)
            file_rows, owned = engram.embed_tokens.disk_file_rows_owned(local)
            rows = engram.embed_tokens.disk.gather_dequant(file_rows, owned)
            staged = buf[:n]
            staged.copy_(rows.view(n, part, dim))

        futs = [pool.submit(_fast_stage_one, e, b) for e, b in zip(tables, rows_host)]
        for f in futs:
            exc = f.exception()
            if exc is not None:
                raise exc

    def native_one(i, n):
        t = tables[i]
        d = t.embed_tokens.disk
        rc = lib.eng_gather_bf16(
            d.w_fd, d.w_off, d.s_fd, d.s_off, dim, sb, t.v0, t.v1,
            hash_host.data_ptr() + i * (h1 - h0) * 4, nl * (h1 - h0), n, h1 - h0, part,
            lut.data_ptr(), rows_host[i].data_ptr(), sw[i].data_ptr(), ss[i].data_ptr(), 1, stats[i],
        )
        if rc:
            raise OSError(-rc, "eng_gather_bf16")

    def native(n):
        for i in range(nl):
            native_one(i, n)

    def native2(n):
        futs = [pool.submit(native_one, i, n) for i in range(nl)]
        for f in futs:
            exc = f.exception()
            if exc is not None:
                raise exc

    arms = {"stock": stock, "native": native, "native2": native2}

    def page_spans(ids):
        for ti, t in enumerate(tables):
            d = t.embed_tokens.disk
            for row in ids[ti]:
                for x in row:
                    if t.v0 <= x < t.v1:
                        for fd, off, ln in ((d.w_fd, d.w_off + x * d.dim, d.dim), (d.s_fd, d.s_off + x * d.sb, d.sb)):
                            a = off & ~4095
                            yield fd, a, ((off + ln + 4095) & ~4095) - a

    def drop(ids):
        for fd, a, ln in page_spans(ids):
            os.posix_fadvise(fd, a, ln, os.POSIX_FADV_DONTNEED)

    def willneed(ids):
        for fd, a, ln in page_spans(ids):
            os.posix_fadvise(fd, a, ln, os.POSIX_FADV_WILLNEED)

    # --noise: the spinner holds the GIL in ~0.1 ms chunks only while a timed stage call
    # runs (noisy set). Spinning through the untimed setup as well made every
    # GIL-releasing syscall there (hundreds of fadvise / pread per iteration) wait out a
    # 5 ms switch interval, and the first run did not finish in 3 minutes.
    stop = threading.Event()
    noisy = threading.Event()
    if args.noise:
        def spin():
            x = 0
            while not stop.is_set():
                if not noisy.wait(0.05):
                    continue
                for k in range(2000):
                    x ^= k * 2654435761
        threading.Thread(target=spin, daemon=True).start()

    # Correctness guard before timing: native == stock on one set per size.
    for n in args.tokens:
        ids = make_ids(n)
        load_host(ids, n)
        stock(n)
        ref = [b[:n].clone() for b in rows_host]
        native(n)
        for r, b in zip(ref, rows_host):
            if not torch.equal(r.view(torch.int16), b[:n].view(torch.int16)):
                print(json.dumps({"error": "native != stock", "n": n}))
                return 1

    out = {"args": vars(args), "torch": torch.__version__, "results": {}}
    names = list(arms)
    for scen in args.scenarios:
        for n in args.tokens:
            times = {a: [] for a in names}
            misses = []
            total = args.warmup + args.iters
            sets = [[make_ids(n) for _ in names] for _ in range(total)]
            if scen == "hot":
                for per in sets:
                    for ids in per:
                        load_host(ids, n)
                        native(n)  # reads every row once -> page cache
            for it in range(total):
                order = names[it % len(names):] + names[: it % len(names)]
                for k, a in enumerate(order):
                    ids = sets[it][k]
                    if scen in ("pf", "cold"):
                        drop(ids)
                        if scen == "pf":
                            willneed(ids)
                            t_end = time.perf_counter() + args.lead_us * 1e-6
                            while time.perf_counter() < t_end:
                                pass
                    load_host(ids, n)
                    if args.noise:
                        noisy.set()
                    t0 = time.perf_counter_ns()
                    arms[a](n)
                    t1 = time.perf_counter_ns()
                    noisy.clear()
                    if it >= args.warmup:
                        times[a].append((t1 - t0) / 1e3)
                        if a == "native":
                            misses.append(sum(st[2] for st in stats))
            out["results"][f"{scen}_n{n}"] = {a: summarize(v) for a, v in times.items()}
            out["results"][f"{scen}_n{n}"]["native_miss_rows_mean"] = statistics.fmean(misses) if misses else 0
    stop.set()
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
