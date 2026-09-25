#!/usr/bin/env python3
"""Bit-exactness of docker/patch/engram_native.c against the stock Engram path.

Runs inside the serving image (torch + the patched vLLM), CPU only:

  docker run --rm --network none -v <repo>:/repo -v <hf cache>:/hf:ro \
      -e DSV41_ENGRAM_GATHER_V2=1 --entrypoint python3 <image> \
      /repo/kernel_study/fusion_host/engram_native_check.py --snapshot /hf/...

Checks (every one compares raw bf16 bits, never float equality):
  1. exhaustive dequant: all 256 x 256 (fp8 byte, ue8m0 byte) pairs through
     the stock torch chain vs eng_dequant_one and vs eng_gather_bf16 reading a
     synthetic table file (both row orders, so every byte hits every lane);
  2. ownership/padding: unowned ids, -1 padding heads, heads_valid <
     local_heads, rows straddling a 4 KiB page, n_tok = 0;
  3. real tables: random owned rows of the two real engram tables for both TP
     ranks, stock DiskEngramTable.gather_dequant (gather v2 path) vs native,
     token counts 1..8 plus 64, with hot and dropped page cache.
Prints one JSON summary line; exit 1 on any mismatch.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "docker" / "patch" / "engram_native.c"


def build(tmp: Path) -> ctypes.CDLL:
    so = tmp / "engram_native.so"
    subprocess.run(
        ["gcc", "-O3", "-shared", "-fPIC", "-o", str(so), str(SRC)], check=True
    )
    lib = ctypes.CDLL(str(so))
    lib.eng_dequant_one.restype = ctypes.c_uint16
    lib.eng_dequant_one.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint8]
    lib.eng_gather_bf16.restype = ctypes.c_int
    lib.eng_gather_bf16.argtypes = [
        ctypes.c_int, ctypes.c_int64, ctypes.c_int, ctypes.c_int64,
        ctypes.c_int, ctypes.c_int, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_void_p,
    ]
    return lib


def fp8_lut() -> torch.Tensor:
    return torch.arange(256, dtype=torch.int32).to(torch.uint8).view(
        torch.float8_e4m3fn
    ).to(torch.float32).contiguous()


def stock_dequant(w: torch.Tensor, s: torch.Tensor, owned: torch.Tensor) -> torch.Tensor:
    """The stock chain, verbatim from DiskEngramTable.gather_dequant."""
    r, dim = w.shape
    sb = s.shape[1]
    vals = w.view(torch.float8_e4m3fn).to(torch.float32).view(r, sb, -1)
    scale = (s.to(torch.int32) << 23).view(torch.float32)
    out = (vals * scale[:, :, None]).reshape(r, dim)
    out[~owned] = 0
    return out.to(torch.bfloat16)


def native_gather(lib, lut, w_fd, w_off, s_fd, s_off, dim, sb, v0, v1, ids, local_heads, flags=1):
    """ids: [n, heads_valid] int32 contiguous. Returns ([n*local_heads, dim] bf16, rc, stats)."""
    n, heads_valid = ids.shape
    rows = n * local_heads
    out = torch.empty((max(rows, 1), dim), dtype=torch.bfloat16)
    sw = torch.empty((max(rows, 1), dim), dtype=torch.uint8)
    ss = torch.empty((max(rows, 1), sb), dtype=torch.uint8)
    stats = (ctypes.c_int64 * 5)()
    rc = lib.eng_gather_bf16(
        w_fd, w_off, s_fd, s_off, dim, sb, v0, v1,
        ids.data_ptr(), ids.stride(0), n, heads_valid, local_heads,
        lut.data_ptr(), out.data_ptr(), sw.data_ptr(), ss.data_ptr(), flags, stats,
    )
    return out[:rows], rc, list(stats)


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.view(torch.int16), b.view(torch.int16))


def check_exhaustive(lib, lut, tmp: Path) -> dict:
    dim, sb = 256, 8
    res = {}
    # eng_dequant_one over every pair.
    w = torch.arange(256, dtype=torch.int32).to(torch.uint8).repeat(256).view(256, 256)
    s = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(256, 1).repeat(1, sb)
    ref = stock_dequant(w, s, torch.ones(256, dtype=torch.bool))
    one = torch.tensor(
        [[lib.eng_dequant_one(lut.data_ptr(), wb, sv) for wb in range(256)] for sv in range(256)],
        dtype=torch.int32,
    ).to(torch.int16).view(torch.bfloat16)
    res["dequant_one_mismatch"] = int((one.view(torch.int16) != ref.view(torch.int16)).sum())
    # Whole gather over a synthetic table: row r = scale byte r, all 256 fp8
    # bytes; a second copy with the bytes shuffled per row.
    g = torch.Generator().manual_seed(0)
    perm_rows = torch.stack([torch.randperm(256, generator=g) for _ in range(256)])
    w2 = torch.cat([w, torch.gather(w, 1, perm_rows)])  # 512 rows
    s2 = torch.cat([s, s])
    # per-block scales differ within a row as well
    s3 = torch.randint(0, 256, (512, sb), generator=g, dtype=torch.int32).to(torch.uint8)
    for tag, s_use in (("uniform_scale", s2), ("mixed_scale", s3)):
        path = tmp / f"table_{tag}.bin"
        w_off = 664  # like the real shard: rows are not 256-aligned
        s_off = w_off + w2.numel() + 40
        with open(path, "wb") as fh:
            fh.write(b"\0" * w_off)
            fh.write(w2.numpy().tobytes())
            fh.write(b"\0" * 40)
            fh.write(s_use.numpy().tobytes())
        fd = os.open(path, os.O_RDONLY)
        try:
            ids = torch.arange(512, dtype=torch.int32).view(64, 8)  # 64 tokens x 8 heads
            for flags in (0, 1):
                got, rc, st = native_gather(lib, lut, fd, w_off, fd, s_off, dim, sb, 0, 512, ids, 8, flags)
                ref2 = stock_dequant(w2, s_use, torch.ones(512, dtype=torch.bool))
                res[f"gather_{tag}_flags{flags}"] = dict(
                    rc=rc, equal=bits_equal(got, ref2), stats=st
                )
        finally:
            os.close(fd)
    return res


def check_ownership(lib, lut, tmp: Path) -> dict:
    dim, sb = 256, 8
    g = torch.Generator().manual_seed(1)
    nrows = 4096
    w = torch.randint(0, 256, (nrows, dim), generator=g, dtype=torch.int32).to(torch.uint8)
    s = torch.randint(100, 140, (nrows, sb), generator=g, dtype=torch.int32).to(torch.uint8)
    path = tmp / "own.bin"
    w_off, s_off = 664, 664 + nrows * dim
    with open(path, "wb") as fh:
        fh.write(b"\0" * w_off)
        fh.write(w.numpy().tobytes())
        fh.write(s.numpy().tobytes())
    fd = os.open(path, os.O_RDONLY)
    out = {}
    try:
        v0, v1 = 1000, 3000
        for n in (0, 1, 3, 4, 6, 8, 64):
            for heads_valid, local_heads in ((12, 12), (11, 12), (12, 16)):
                ids = torch.randint(-1, nrows, (max(n, 0), heads_valid), generator=g, dtype=torch.int32)
                if n:
                    ids[0, 0] = -1
                    ids[-1, -1] = v1  # just outside
                    ids[n // 2, 0] = v0  # first owned
                    ids[n // 2, -1] = 4096 - 1 if v1 > 4095 else v1 - 1
                got, rc, st = native_gather(lib, lut, fd, w_off, fd, s_off, dim, sb, v0, v1, ids.contiguous(), local_heads)
                # stock: pad heads with -1, owned mask, file row 0 for unowned
                if n:
                    pad = torch.full((n, local_heads - heads_valid), -1, dtype=torch.int64)
                    local = torch.cat([ids.to(torch.int64), pad], dim=1).reshape(-1)
                else:
                    local = torch.empty(0, dtype=torch.int64)
                owned = (local >= v0) & (local < v1)
                file_rows = torch.where(owned, local, torch.zeros_like(local))
                if n == 0:  # the stage returns before any gather; stock cannot view 0 rows
                    ref = torch.empty((0, dim), dtype=torch.bfloat16)
                else:
                    ref = stock_dequant(w[file_rows], s[file_rows], owned)
                out[f"n{n}_hv{heads_valid}_lh{local_heads}"] = dict(
                    rc=rc, equal=bits_equal(got, ref), owned=int(owned.sum()), stats=st
                )
    finally:
        os.close(fd)
    return out


def load_text_config(snapshot: str):
    from types import SimpleNamespace

    with open(os.path.join(snapshot, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    return SimpleNamespace(**cfg.get("text_config", cfg))


def check_real(lib, lut, snapshot: str, reps: int) -> dict:
    from types import MethodType, SimpleNamespace

    from vllm.models.deepseek_v4_1.common import engram as eng
    from vllm.models.deepseek_v4_1.common.engram_disk import DiskEngramTable

    layout = eng.EngramLayout(load_text_config(snapshot))
    heads = layout.n_hash_cols
    out = {}
    g = random.Random(2)
    for li, layer_id in enumerate(layout.layer_ids):
        disk = DiskEngramTable(snapshot, layer_id, layout.head_dim, 32)
        sizes = [p for order in layout.primes[li] for p in order]
        for rank in range(2):
            part = -(-heads // 2)
            h0 = rank * part
            h1 = min(h0 + part, heads)
            v0, v1 = sum(sizes[:h0]), sum(sizes[:h0 + part])
            emb = SimpleNamespace(part_n_hash_cols=part, vocab_start_idx=v0, vocab_end_idx=v1)
            rows_owned = MethodType(eng.ParallelEngramEmbedding.disk_file_rows_owned, emb)
            offs = [sum(sizes[:c]) for c in range(heads)]
            for n in (1, 2, 3, 4, 5, 6, 7, 8, 64):
                for rep in range(reps):
                    ids = torch.tensor(
                        [[offs[c] + g.randrange(sizes[c]) for c in range(h0, h1)] for _ in range(n)],
                        dtype=torch.int32,
                    )
                    if rep == 1:
                        ids[0, 0] = -1  # a dead id
                    if rep == 2:
                        # drop the rows from the page cache first
                        for x in ids.reshape(-1).tolist():
                            if v0 <= x < v1:
                                for fd, off, ln in ((disk.w_fd, disk.w_off + x * disk.dim, disk.dim), (disk.s_fd, disk.s_off + x * disk.sb, disk.sb)):
                                    a = off & ~4095
                                    os.posix_fadvise(fd, a, ((off + ln + 4095) & ~4095) - a, os.POSIX_FADV_DONTNEED)
                    got, rc, st = native_gather(lib, lut, disk.w_fd, disk.w_off, disk.s_fd, disk.s_off, disk.dim, disk.sb, v0, v1, ids, part)
                    file_rows, owned = rows_owned(ids.to(torch.int64))
                    ref = disk.gather_dequant(file_rows, owned)
                    key = f"layer{layer_id}_rank{rank}_n{n}"
                    ok = rc == 0 and bits_equal(got, ref)
                    d = out.setdefault(key, dict(calls=0, equal=0, miss=0))
                    d["calls"] += 1
                    d["equal"] += int(ok)
                    d["miss"] += st[2]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        lib = build(tmp)
        lut = fp8_lut()
        res = {
            "exhaustive": check_exhaustive(lib, lut, tmp),
            "ownership": check_ownership(lib, lut, tmp),
        }
        if args.snapshot:
            res["real"] = check_real(lib, lut, args.snapshot, args.reps)
    bad = []
    ex = res["exhaustive"]
    if ex["dequant_one_mismatch"]:
        bad.append("dequant_one")
    for k, v in ex.items():
        if isinstance(v, dict) and not (v["rc"] == 0 and v["equal"]):
            bad.append(k)
    for k, v in res["ownership"].items():
        if not (v["rc"] == 0 and v["equal"]):
            bad.append(k)
    for k, v in res.get("real", {}).items():
        if v["equal"] != v["calls"]:
            bad.append(k)
    res["torch"] = torch.__version__
    res["mismatches"] = bad
    print(json.dumps(res))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
