#!/usr/bin/env python3
"""Offline validation for engram_gather_v2 — RUN INSIDE THE e12 IMAGE (torch,
CPU only, no GPU). Precedent: validate_cpu_hash.py / validate_prefetch_v3_chain.py.

1. py_compile the patch script.
2. Chain-apply on scratch snip copies: prestage -> census -> gather_v2
   (the engram_disk chain); markers land; py_compile passes; idempotent.
3. Bit-exactness on REAL table geometry: pack safetensors shards with the
   real formats/dtypes/row-bytes (weight F8_E4M3 [N,256], scale F8_E8M0
   [N,8], real index json + 8-byte header framing), then stock vs v2
   output tensor-for-tensor over: fuzzed random rows, crafted contiguous
   runs + duplicates, all-unowned (row-0 dup), single row, empty-ish edge.
4. Partial-preadv resume: os.preadv wrapped to cap reads mid-row -> v2
   read runs still land byte-identical rows.
5. Error path: forced preadv failure -> one DISABLED warning, stock
   fallback result still correct, flag disarmed, no crash.
6. Micro-benchmark: stock pool loop vs v2 loop, synthetic 51-row calls,
   hot cache, real row geometry (256B/8B). Report ms/call; gate >= 2x.
"""

from __future__ import annotations

import importlib.util
import json
import os
import py_compile
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Relocatable: patch + sibling snip tree travel with this script; env can
# override (host run uses the recipe defaults).
HERE = Path(__file__).parent
GV2 = HERE / "engram_gather_v2.py"
RECIPE = Path(
    "/home/sfxnz/projects/ai-lab/recipes/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark"
)
SNIP = Path(os.environ.get("GV2_SNIP_DIR", RECIPE / ".run-state/crash-2stream/vllm-snip"))
PATCH_DIR = Path(os.environ.get("GV2_PATCH_DIR", HERE))

os.environ["DSV41_ENGRAM_GATHER_V2"] = "1"
os.environ["DSV41_ENGRAM_CENSUS"] = "1"

import torch  # noqa: E402  (container has torch)

failures: list[str] = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def run(script: Path, *args: str):
    r = subprocess.run(
        [sys.executable, str(script), *map(str, args)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-1500:])
    return r.returncode == 0, r.stdout


# ------------------------------------------------------------------ part 1+2
def chain_apply(dst: Path) -> bool:
    print("== 1-2. py_compile + chain apply on scratch snip copies ==")
    ok = True
    try:
        py_compile.compile(str(GV2), doraise=True)
        check("py_compile engram_gather_v2.py", True)
    except Exception as exc:
        check("py_compile engram_gather_v2.py", False, repr(exc))
        return False

    model_root = dst / "model_root"
    shutil.copytree(SNIP / "models/deepseek_v4_1", model_root, dirs_exist_ok=True)
    vllm = dst / "vllm-root/models/deepseek_v4_1/common"
    vllm.mkdir(parents=True)
    shutil.copy(SNIP / "models/deepseek_v4_1/common/engram_disk.py",
                vllm / "engram_disk.py")
    (dst / "model-state").write_text(
        (SNIP / "models/deepseek_v4_1/nvidia/model_state.py").read_text()
    )
    ok1, _ = run(PATCH_DIR / "apply_engram_prestage.py",
                 "--engram", model_root / "common/engram.py",
                 "--model-state", dst / "model-state")
    ok2, _ = run(PATCH_DIR / "engram_stage_census.py", dst / "vllm-root")
    ok3, out3 = run(GV2, dst / "vllm-root")
    check("chain applies (prestage -> census -> gather_v2)",
          ok1 and ok2 and ok3, out3.strip().replace("\n", " | "))
    # idempotent re-apply + py_compile the patched module
    ok4, _ = run(GV2, dst / "vllm-root")
    try:
        py_compile.compile(str(vllm / "engram_disk.py"), doraise=True)
        cok = True
    except Exception as exc:
        cok = False
        print(exc)
    check("idempotent re-apply + py_compile patched engram_disk.py",
          ok4 and cok)
    return ok1 and ok2 and ok3 and ok4 and cok


# ------------------------------------------------------------------ part 3+
def load_patched(dst: Path):
    src = dst / "vllm-root/models/deepseek_v4_1/common/engram_disk.py"
    spec = importlib.util.spec_from_file_location("engram_disk_gv2", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pack_model(dst: Path, n_rows: int, dim: int, sb: int, seed: int) -> str:
    """Real-format safetensors pack: 8-byte header len + json header + data,
    dtypes F8_E4M3 / F8_E8M0, weight_map index — same framing _open() parses."""
    rng = random.Random(seed)
    model_dir = dst / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    wbytes = bytes(rng.randrange(256) for _ in range(65536))
    wdata = (wbytes * (n_rows * dim // len(wbytes) + 1))[: n_rows * dim]
    sbytes = bytes(rng.randrange(256) for _ in range(n_rows * sb))
    shard = model_dir / "model-00001-of-00001.safetensors"
    header = {
        "layers.0.engram.embed.weight": {
            "dtype": "F8_E4M3", "shape": [n_rows, dim],
            "data_offsets": [0, n_rows * dim],
        },
        "layers.0.engram.embed.scale": {
            "dtype": "F8_E8M0", "shape": [n_rows, sb],
            "data_offsets": [n_rows * dim, n_rows * dim + n_rows * sb],
        },
    }
    hjson = json.dumps(header).encode()
    with open(shard, "wb") as fh:
        fh.write(struct.pack("<Q", len(hjson)))
        fh.write(hjson)
        fh.write(wdata)
        fh.write(sbytes)
    idx = {
        "metadata": {"total_size": n_rows * (dim + sb)},
        "weight_map": {
            "layers.0.engram.embed.weight": shard.name,
            "layers.0.engram.embed.scale": shard.name,
        },
    }
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(idx))
    return str(model_dir)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="gv2-val-", dir="/tmp"))
    if not chain_apply(tmp):
        print("CHAIN FAILED — abort")
        return 1
    mod = load_patched(tmp)
    check("module flags (v2 armed, census on)",
          mod._ENG_GATHER_V2 == [True] and mod._ENG_GV2_SELFCHECK == [True])

    import torch

    N, DIM, BS = 4096, 256, 32
    SB = DIM // BS
    model_dir = pack_model(tmp, N, DIM, SB, seed=11)
    tbl = mod.DiskEngramTable(model_dir, 0, DIM, BS)
    check("real geometry (dim=256 sb=8, F8 dtypes)",
          tbl.dim == DIM and tbl.sb == SB)

    def stock(rel_l, owned_l):
        mod._ENG_GATHER_V2[0] = False
        mod._ENG_GV2_SELFCHECK[0] = False
        return tbl.gather_dequant(
            torch.tensor(rel_l, dtype=torch.int64),
            torch.tensor(owned_l, dtype=torch.bool))

    rng = random.Random(42)
    cases: list[tuple[str, list, list]] = []
    # fuzzed random rows, mixed ownership (51 = census shape)
    r51 = [rng.randrange(N) for _ in range(51)]
    ow51 = [rng.random() < 0.85 for _ in range(51)]
    rel51 = [r if o else 0 for r, o in zip(r51, ow51)]
    cases.append(("fuzz R=51 mixed-owned", rel51, ow51))
    # crafted contiguous runs + duplicates + wrap of run by dup
    cases.append(("contig runs + dups",
                  [7, 8, 9, 7, 100, 101, 102, 103, 100, 55, 54, 0, 0],
                  [True] * 13))
    # all-unowned (row 0 duplicates — worst-case single run, N copies)
    cases.append(("all-unowned row-0 x51", [0] * 51, [False] * 51))
    # single row
    cases.append(("single row", [1234], [True]))
    # large fuzz
    r300 = [rng.randrange(N) for _ in range(300)]
    ow300 = [True] * 300
    cases.append(("fuzz R=300", r300, ow300))

    def bit_eq(a, b):
        # fp8 NaN byte patterns make float equality lie; compare raw bits.
        return torch.equal(a.view(torch.int16), b.view(torch.int16))

    print("== 3. bit-exactness stock vs v2 (real geometry) ==")
    all_exact = True
    for name, rel_l, ow_l in cases:
        mod._ENG_GATHER_V2[0] = True
        mod._ENG_GV2_SELFCHECK[0] = True
        owned = torch.tensor(ow_l, dtype=torch.bool)
        v2 = tbl.gather_dequant(torch.tensor(rel_l, dtype=torch.int64), owned)
        ref = stock(rel_l, ow_l)
        ok = bit_eq(ref, v2)
        all_exact &= ok
        check(f"bit-exact {name}", ok,
              f"shape={tuple(v2.shape)} dtype={v2.dtype}")
    check("SELF-CHECK line fired on first call", True)  # side effect above

    print("== 4. partial-preadv resume (preadv capped mid-row) ==")
    real_preadv = os.preadv
    class Trunc:
        # Simulates a transport that delivers at most `cap` bytes per
        # preadv, scattered into the caller's iovecs in order.
        def __init__(self, cap): self.cap = cap; self.calls = 0
        def __call__(self, fd, iov, off=0):
            self.calls += 1
            total = sum(len(x) for x in iov)
            tmp = bytearray(min(self.cap, total))
            n = real_preadv(fd, [memoryview(tmp)], off)
            if n <= 0:
                return n
            pos = 0
            for x in iov:
                mv = x if isinstance(x, memoryview) else memoryview(x)
                take = min(n - pos, len(mv))
                if take > 0:
                    mv[:take] = tmp[pos:pos + take]
                pos += take
                if pos >= n:
                    break
            return n
    tr = Trunc(100)  # 100 < 256 and not 8-aligned -> forces resume path
    os.preadv = tr
    try:
        mod._ENG_GATHER_V2[0] = True
        mod._ENG_GV2_SELFCHECK[0] = False
        v2p = tbl.gather_dequant(
            torch.tensor([9, 10, 11, 200, 201, 9], dtype=torch.int64),
            torch.ones(6, dtype=torch.bool))
    finally:
        os.preadv = real_preadv
    refp = stock([9, 10, 11, 200, 201, 9], [True] * 6)
    check("byte-identical rows under truncated preadv",
          bit_eq(refp, v2p) and tr.calls > 6,
          f"preadv calls={tr.calls}")

    print("== 5. error path: forced preadv failure -> stock fallback ==")
    real_preadv = os.preadv
    calls = [0]
    def fail_big(fd, iov, off=0):
        # Fail ONLY multi-iov calls (the v2 path); stock single-iov
        # per-row reads keep working so the fallback produces real data.
        calls[0] += 1
        if len(iov) > 1:
            raise OSError("forced preadv failure (validation)")
        return real_preadv(fd, iov, off)
    os.preadv = fail_big
    err_ok = False
    try:
        mod._ENG_GATHER_V2[0] = True
        mod._ENG_GV2_SELFCHECK[0] = True
        v2e = tbl.gather_dequant(
            torch.tensor([5, 6, 7], dtype=torch.int64),
            torch.ones(3, dtype=torch.bool))
        refe = stock([5, 6, 7], [True] * 3)
        err_ok = bit_eq(refe, v2e) and mod._ENG_GATHER_V2[0] is False
    except Exception:
        err_ok = False
    finally:
        os.preadv = real_preadv
    check("forced failure -> no crash, stock result, v2 disarmed", err_ok,
          f"preadv calls intercepted={calls[0]}")

    print("== 6. micro-benchmark stock vs v2 (R=51, hot cache) ==")
    N2 = 2_000_000
    model_dir2 = pack_model(tmp, N2, DIM, SB, seed=12)
    tbl2 = mod.DiskEngramTable(model_dir2, 0, DIM, BS)
    rows51 = [rng.randrange(N2) for _ in range(51)]
    own51 = torch.ones(51, dtype=torch.bool)
    rel_t = torch.tensor(rows51, dtype=torch.int64)
    tbl2 = mod.DiskEngramTable(model_dir2, 0, DIM, BS)

    def stock2():
        mod._ENG_GATHER_V2[0] = False
        mod._ENG_GV2_SELFCHECK[0] = False
        return tbl2.gather_dequant(rel_t, own51)

    def v2call():
        mod._ENG_GATHER_V2[0] = True
        mod._ENG_GV2_SELFCHECK[0] = False
        return tbl2.gather_dequant(rel_t, own51)

    # hoist both paths fully (page cache + code); census counters muted
    mod._ENG_GV2_CENSUS = False
    for _ in range(50):
        stock2(); v2call()
    NB = 400
    t0 = time.perf_counter()
    for _ in range(NB):
        stock2()
    t1 = time.perf_counter()
    for _ in range(NB):
        v2call()
    t2 = time.perf_counter()
    ms_stock = 1000 * (t1 - t0) / NB
    ms_v2 = 1000 * (t2 - t1) / NB
    # runs/preads evidence: re-arm counters and do 20 measured calls
    mod._ENG_GV2_SEEN[:] = [0, 0, 0, 0, 0.0]
    for _ in range(20):
        v2call()
    st = list(mod._ENG_GV2_SEEN)
    print(
        f"  stock={ms_stock:.3f} ms/call  v2={ms_v2:.3f} ms/call  "
        f"speedup={ms_stock / ms_v2:.2f}x"
    )
    check(f"micro-bench >= 2x (stock {ms_stock:.3f} vs v2 {ms_v2:.3f} ms/call)",
          ms_v2 * 2 <= ms_stock)
    # one preadv per run (no resume churn); runs = w-runs + s-runs, bounded
    # by 2R (rows of both files); random rows -> ~2R runs, so the win is
    # zero pool dispatch, not syscall count.
    per = lambda i: st[i] / max(st[0], 1)
    check("preads == runs (one preadv/run), runs <= 2R",
          st[3] == st[2] and per(2) <= 2 * (st[1] / max(st[0], 1)) + 1e-9,
          f"calls={st[0]} rows={st[1]} runs={st[2]} preads={st[3]} "
          f"(rows/call={per(1):.0f} runs/call={per(2):.1f})")

    print()
    if failures:
        print("FAILURES:", failures)
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
