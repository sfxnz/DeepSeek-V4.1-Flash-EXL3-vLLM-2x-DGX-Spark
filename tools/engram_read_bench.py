#!/usr/bin/env python3
"""CPU/NVMe micro-bench: Engram row gather strategies on the real table.

Read-only. No GPU, no torch, no page-cache drop. Loads the recipe's
engram_disk.py with the census + gather v2 patches applied (temp copy), opens
the real safetensors table like the serve does (POSIX_FADV_RANDOM), and times
one gather call per arm over fresh uniform-random rows, w file and s file:

  serial           gv2 _gv2_read_runs, WILLNEED off (the R23-R33 path)
  willneed+serial  gv2 _gv2_willneed pre-pass, then _gv2_read_runs (new)
  pool             stock _read_rows, 32 threads x chunk 16 (MAX_ROWS arm)
  willneed+pool    pre-pass, then the stock pool
  hot serial       re-read of the willneed+serial rows (page-cache ceiling)
  hot willneed     same rows again with the pre-pass (its cost on hot calls)

Cold caveat: nothing is dropped. Each cold arm samples new rows from a
~98 GB table while the page cache holds ~20 GB, so most rows are cold but
some may be resident. The serve reading the same file can warm pages too.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "docker/patch"
DEFAULT_SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-EXL3/snapshots/2.0bpw-mcg"


def load_disk_module(tmp: Path):
    sys.path.insert(0, str(PATCH))
    import engram_gather_v2
    import engram_stage_census

    vllm = tmp / "vllm"
    dst = vllm / "models/deepseek_v4_1/common/engram_disk.py"
    dst.parent.mkdir(parents=True)
    shutil.copy(PATCH / "engram_disk.py", dst)
    with contextlib.redirect_stdout(io.StringIO()):
        engram_stage_census.apply(vllm)
        engram_gather_v2.apply(vllm)
    spec = importlib.util.spec_from_file_location("engram_disk_bench", dst)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._ENG_GV2_WILLNEED_MIN = 1
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--rows", type=int, nargs="+", default=[2000, 20000], help="rows per call; serial runs only the first size")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=924)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        mod = load_disk_module(Path(tmp))
    idx = json.loads((args.snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    wname = f"layers.{args.layer}.engram.embed.weight"
    tbl = mod.DiskEngramTable.__new__(mod.DiskEngramTable)
    tbl.w_fd, tbl.w_off, w_shape = tbl._open(str(args.snapshot), idx[wname], wname)
    sname = f"layers.{args.layer}.engram.embed.scale"
    tbl.s_fd, tbl.s_off, s_shape = tbl._open(str(args.snapshot), idx[sname], sname)
    tbl.dim, tbl.sb = w_shape[1], s_shape[1]
    tbl.threads, tbl.chunk = 32, 16
    from concurrent.futures import ThreadPoolExecutor

    tbl.pool = ThreadPoolExecutor(max_workers=tbl.threads)
    n_rows = w_shape[0]
    rng = random.Random(args.seed)

    def bufs(n):
        return (
            memoryview(np.zeros(n * tbl.dim, np.uint8)).cast("B"),
            memoryview(np.zeros(n * tbl.sb, np.uint8)).cast("B"),
        )

    def serial(rel, willneed):
        mod._ENG_GV2_WILLNEED = willneed
        bw, bs = bufs(len(rel))
        t0 = time.perf_counter()
        fadv = tbl._gv2_willneed(rel)
        t1 = time.perf_counter()
        tbl._gv2_read_runs(tbl.w_fd, tbl.w_off, rel, tbl.dim, bw)
        tbl._gv2_read_runs(tbl.s_fd, tbl.s_off, rel, tbl.sb, bs)
        return time.perf_counter() - t0, t1 - t0, fadv, bytes(bw[: tbl.dim])

    def pool(rel, willneed):
        mod._ENG_GV2_WILLNEED = willneed
        bw, bs = bufs(len(rel))
        t0 = time.perf_counter()
        fadv = tbl._gv2_willneed(rel)
        t1 = time.perf_counter()
        tbl._read_rows(tbl.w_fd, tbl.w_off, rel, tbl.dim, bw)
        tbl._read_rows(tbl.s_fd, tbl.s_off, rel, tbl.sb, bs)
        return time.perf_counter() - t0, t1 - t0, fadv, bytes(bw[: tbl.dim])

    arms = []
    for size_i, n in enumerate(args.rows):
        plan = [("willneed+serial", serial, True), ("pool", pool, False), ("willneed+pool", pool, True)]
        if size_i == 0:
            plan.insert(0, ("serial", serial, False))
        for rep in range(args.reps):
            last = None
            for name, fn, wn in plan:
                rel = [rng.randrange(n_rows) for _ in range(n)]
                dt, pre, fadv, head = fn(rel, wn)
                ref = os.pread(tbl.w_fd, tbl.dim, tbl.w_off + rel[0] * tbl.dim)
                assert head == ref, "byte mismatch"
                arms.append(dict(arm=name, rows=n, rep=rep, ms=dt * 1e3, prepass_ms=pre * 1e3, fadvise=fadv))
                if name == "willneed+serial":
                    last = rel
            for name, wn in (("hot serial", False), ("hot willneed+serial", True)):
                dt, pre, fadv, _ = serial(last, wn)
                arms.append(dict(arm=name, rows=n, rep=rep, ms=dt * 1e3, prepass_ms=pre * 1e3, fadvise=fadv))

    summary = {}
    for a in arms:
        summary.setdefault((a["arm"], a["rows"]), []).append(a)
    table = []
    print(f"{'arm':22s} {'rows':>6s} {'ms/call (median)':>17s} {'us/row':>8s} {'prepass ms':>10s} {'fadvise':>8s}")
    for (name, n), runs in summary.items():
        ms = statistics.median(r["ms"] for r in runs)
        pre = statistics.median(r["prepass_ms"] for r in runs)
        fadv = statistics.median(r["fadvise"] for r in runs)
        table.append(dict(arm=name, rows=n, ms_median=round(ms, 2), us_per_row=round(ms * 1e3 / n, 2), prepass_ms=round(pre, 2), fadvise=fadv, ms_all=[round(r["ms"], 2) for r in runs]))
        print(f"{name:22s} {n:6d} {ms:17.1f} {ms * 1e3 / n:8.1f} {pre:10.1f} {fadv:8.0f}")

    if args.out:
        free = subprocess.run(["free", "-h"], capture_output=True, text=True).stdout
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                dict(
                    table=f"{args.snapshot}/{idx[wname]} layers.{args.layer} w={list(w_shape)} s={list(s_shape)}",
                    seed=args.seed,
                    reps=args.reps,
                    note="no cache drop; fresh uniform rows per cold arm; hot = re-read of willneed+serial rows",
                    summary=table,
                    runs=arms,
                    free_h_after=free,
                ),
                indent=1,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
