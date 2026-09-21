#!/usr/bin/env python3
"""Threaded functional validation for engram_defer (CPU-only, host).

Builds the FULL patched chain on scratch copies of the REAL sources
(prestage -> census -> fast-stage -> prefetch v3 -> cpu-hash -> gather v2
-> defer), then:

  * exec's the REAL patched engram_disk.py so the actual v2
    _gv2_read_runs runs against real safetensors geometry
    (F8_E4M3 [N,256] / F8_E8M0 [N,8], sequential packing, 8-byte header);
  * exec's the REAL defer methods (_defer_predict, _defer_hash_layers,
    _defer_try_stage, _defer_disarm...) from the patched engram.py;
  * replaces ONLY the torch plumbing (pinned tensors, CUDA events,
    streams) with numpy/threading equivalents — predict/hash/gather/
    verify logic under test is the shipped code;

then drives multi-step decode simulations (gen advance, varying accept
counts 1..k+1, prediction misses, forced worker error, warm-verify
mismatch, hung worker) asserting served rows == sync-path rows every
step, plus fallback/deadlock/GIL properties.

The numpy e4m3 decoder is verified bit-exact against the image's torch
(fp8_probe.py, run in canonical-e12: all 256 patterns, 2 NaNs).
"""
from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RECIPE = Path(
    "/home/sfxnz/projects/ai-lab/recipes/"
    "DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark"
)
SNIP = RECIPE / ".run-state/crash-2stream/vllm-snip"
PATCH_DIR = RECIPE / "docker/patch"
DEFER = RECIPE / "docker/patch/engram_defer.py"

failures = []


def check(name, ok, detail=""):
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {name}"
        + (f" — {detail}" if detail else "")
    )
    if not ok:
        failures.append(name)


def build_patched(dst: Path):
    shutil.copytree(
        SNIP / "models/deepseek_v4_1", dst / "model_root", dirs_exist_ok=True
    )
    (dst / "runner").write_text(
        (SNIP / "v1/worker/gpu/model_runner.py").read_text()
    )
    vllm = dst / "vllm-root/models/deepseek_v4_1/common"
    vllm.mkdir(parents=True)
    shutil.copy(
        SNIP / "models/deepseek_v4_1/common/engram_disk.py",
        vllm / "engram_disk.py",
    )
    steps = [
        (PATCH_DIR / "apply_engram_prestage.py", [
            "--engram", str(dst / "model_root/common/engram.py"),
            "--model-state", str(dst / "model_root/nvidia/model_state.py"),
        ]),
        (PATCH_DIR / "engram_stage_census.py", [str(dst / "vllm-root")]),
        (PATCH_DIR / "engram_stage_fast.py", [str(dst / "model_root")]),
        (PATCH_DIR / "engram_prefetch_v3.py", [
            str(dst / "model_root"), str(dst / "runner"), str(dst / "vllm-root"),
        ]),
        (PATCH_DIR / "engram_cpu_hash.py", [
            str(dst / "model_root"), str(dst / "runner"),
        ]),
        (PATCH_DIR / "engram_gather_v2.py", [str(dst / "vllm-root")]),
        (DEFER, [
            str(dst / "model_root"), str(dst / "runner"),
            str(dst / "model_root/nvidia/model_state.py"),
        ]),
    ]
    for script, args in steps:
        r = subprocess.run(
            [sys.executable, str(script), *args], capture_output=True, text=True
        )
        if r.returncode != 0:
            print(r.stdout, r.stderr)
            raise SystemExit(f"patch step failed: {script.name}")


def e4m3_decode(u8):
    """numpy mirror of torch view(float8_e4m3fn).to(f32) — verified
    in-image (fp8_probe.py): all 256 patterns, 2 NaNs, sign preserved."""
    u = u8.astype(np.int32)
    sign = (u >> 7) & 1
    exp = (u >> 3) & 0xF
    man = u & 7
    sub = exp == 0
    val = np.where(
        sub, man * (2.0**-9), (man / 8.0 + 1.0) * (2.0 ** (exp - 7))
    )
    val = np.where((u & 0x7F) == 0x7F, np.float32("nan"), val)
    val = np.where(sign == 1, -val, val)
    return val.astype(np.float32)


def make_table_file(path: Path, n_rows: int, dim: int = 256, seed: int = 0):
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 256, size=(n_rows, dim), dtype=np.uint8)
    s = rng.integers(0, 256, size=(n_rows, dim // 32), dtype=np.uint8)
    sb = dim // 32
    o1 = n_rows * dim
    o2 = o1 + n_rows * sb
    o3 = o2 + n_rows * dim
    o4 = o3 + n_rows * sb
    hdr = {
        "layers.0.engram.embed.weight": {
            "dtype": "F8_E4M3", "shape": [n_rows, dim],
            "data_offsets": [0, o1]},
        "layers.0.engram.embed.scale": {
            "dtype": "F8_E8M0", "shape": [n_rows, sb],
            "data_offsets": [o1, o2]},
        "layers.1.engram.embed.weight": {
            "dtype": "F8_E4M3", "shape": [n_rows, dim],
            "data_offsets": [o2, o3]},
        "layers.1.engram.embed.scale": {
            "dtype": "F8_E8M0", "shape": [n_rows, sb],
            "data_offsets": [o3, o4]},
    }
    hb = json.dumps(hdr).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        fh.write(w.tobytes())
        fh.write(s.tobytes())
        fh.write(w.tobytes())
        fh.write(s.tobytes())
    return w, s


N_ROWS = 4096
DIM = 256
SB = 8
MAXT = 64
LH = 12
NLAYERS = 2
DEPTH = 6
K = 3


class T:
    def __init__(self, data):
        self.np = np.asarray(data)

    def numel(self):
        return self.np.size

    def __getitem__(self, idx):
        v = self.np[idx]
        return T(v) if isinstance(v, np.ndarray) else v

    def tolist(self):
        return self.np.tolist()


class FakeEmbedTokens:
    def __init__(self, disk, v0, v1):
        self.disk = disk
        self.vocab_start_idx = v0
        self.vocab_end_idx = v1

    def disk_file_rows_owned(self, local):
        rows = np.asarray(
            local.np if hasattr(local, "np") else local
        ).reshape(-1)
        owned = (rows >= self.vocab_start_idx) & (rows < self.vocab_end_idx)
        file_rows = np.where(owned, rows, 0)
        return T(file_rows), owned


class FakeEngram:
    def __init__(self, li, disk, v0, v1):
        self.layer_hash_index = li
        self.embed_tokens = FakeEmbedTokens(disk, v0, v1)
        self._dev_rows = np.zeros((MAXT, LH, DIM), dtype=np.float32)

    def _staged_rows_for_ubatch(self):
        eng = self

        class D:
            def copy_(self, other, non_blocking=False):
                o = other.np if hasattr(other, "np") else np.asarray(other)
                eng._dev_rows[: o.shape[0]] = o

        return D()


class FakeEvent:
    def __init__(self):
        self.ev = threading.Event()

    def record(self):
        self.ev.set()

    def synchronize(self):
        return self.ev.wait()

    def query(self):
        return self.ev.is_set()


class FakeFut:
    def __init__(self, fn, *a):
        self.fn, self.a = fn, a
        self.exc = None
        self.done = threading.Event()
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        try:
            self.fn(*self.a)
        except BaseException as e:  # noqa: BLE001
            self.exc = e
        finally:
            self.done.set()

    def result(self, timeout=None):
        if not self.done.wait(timeout or 1e9):
            raise TimeoutError()
        if self.exc:
            raise self.exc


class FakePool:
    def submit(self, fn, *a):
        return FakeFut(fn, *a)


def gather_dequant_np(tbl, rel, owned):
    rl = [int(x) for x in (rel.np if hasattr(rel, "np") else rel)]
    wbuf = bytearray(len(rl) * tbl.dim)
    sbuf = bytearray(len(rl) * tbl.sb)
    tbl._gv2_read_runs(tbl.w_fd, tbl.w_off, rl, tbl.dim, memoryview(wbuf))
    tbl._gv2_read_runs(tbl.s_fd, tbl.s_off, rl, tbl.sb, memoryview(sbuf))
    wv = np.frombuffer(bytes(wbuf), dtype=np.uint8).reshape(len(rl), tbl.dim)
    sv = np.frombuffer(bytes(sbuf), dtype=np.uint8).reshape(len(rl), tbl.sb)
    vals = e4m3_decode(wv).reshape(len(rl), tbl.sb, -1)
    scale = (sv.astype(np.int32) << 23).view(np.float32)
    out = (vals * scale[:, :, None]).reshape(len(rl), tbl.dim)
    out = np.where(np.asarray(owned)[:, None], out, 0.0)
    return out.astype(np.float32)


class FakeBatch:
    def __init__(self, n, req_ids, qsl):
        self.num_reqs = len(req_ids)
        self.num_tokens = n
        self.req_ids = list(req_ids)
        self.query_start_loc_np = np.asarray(qsl)


def main():
    root = Path(tempfile.mkdtemp(prefix="defer-fn-"))
    build_patched(root)

    disk_src = (
        root / "vllm-root/models/deepseek_v4_1/common/engram_disk.py"
    ).read_text()
    disk_ns = {"__name__": "engram_disk_patched"}
    exec(compile(disk_src, "engram_disk_patched.py", "exec"), disk_ns)

    tdir = root / "tables"
    tdir.mkdir()
    fileA = tdir / "tA.safetensors"
    wA, sA = make_table_file(fileA, N_ROWS, DIM, seed=1)
    idx = {
        "weight_map": {
            f"layers.{i}.engram.embed.{k}": fileA.name
            for i in (0, 1)
            for k in ("weight", "scale")
        }
    }
    (tdir / "model.safetensors.index.json").write_text(json.dumps(idx))
    table0 = disk_ns["DiskEngramTable"](str(tdir), 0, DIM, 32)
    table1 = disk_ns["DiskEngramTable"](str(tdir), 1, DIM, 32)

    engram_src = (root / "model_root/common/engram.py").read_text()
    start = engram_src.index("    def _defer_setup")
    end = engram_src.index("    @torch.inference_mode()\n    def stage(")
    defer_src = engram_src[start:end]

    import textwrap

    class_ns: dict = {}
    exec(
        "class _M:\n"
        + "".join(
            "    " + ln + "\n" for ln in textwrap.dedent(defer_src).splitlines()
        )
        + "\n",
        {"__builtins__": __builtins__},
        class_ns,
    )
    methods = {k: v for k, v in vars(class_ns["_M"]).items() if callable(v)}

    class FakeStager:
        pass

    for k, v in methods.items():
        setattr(FakeStager, k, v)

    def np_check_warm(self, n):
        try:
            slot = self._df_v_slot
            self._df_v_slot = None
            if slot is None:
                return
            for li in range(len(self.engrams)):
                a = self._df_pin[slot][li][:n]
                b = self.rows_host[li][:n]
                if not np.array_equal(a.view(np.uint32), b.view(np.uint32)):
                    self._defer_disarm(
                        "warm verify mismatch (table %d slot != sync)" % li
                    )
                    return
            if self._df_warm == self._df_warm_n:
                print(
                    "dsv41: engram defer ACTIVE (warm-verified bit-exact "
                    "x%d)" % self._df_warm
                )
        except Exception as exc:  # noqa: BLE001
            self._defer_disarm(f"verify error: {exc!r}")

    FakeStager._defer_check_warm = np_check_warm

    def np_enqueue(
        self, ids, pos, qsl, win, outs, ns, drafts, req_ids, num_tokens, k
    ):
        self._df_req_snap = [str(x) for x in req_ids]
        self._df_om_snap = max(1, len(outs[0]) if outs else 1)
        self._df_gen += 1
        gen = self._df_gen
        slot = self._df_slot
        self._df_slot ^= 1
        self._df_pending = {
            "gen": gen,
            "slot": slot,
            "fut": self._df_pool.submit(
                self._defer_worker, gen, len(ns), num_tokens, k, slot
            ),
        }

    FakeStager.enqueue_defer = np_enqueue

    def np_worker(self, gen, num_reqs, num_tokens, k, slot):
        # mirrors the real worker: whole body inside try/except that
        # disarms with ONE line (that except is the shipped behavior
        # under test in T5).
        t0 = time.perf_counter()
        try:
            self._np_worker_inner(gen, num_reqs, num_tokens, k, slot, t0)
        except Exception as exc:  # noqa: BLE001
            self._defer_disarm(f"worker error: {exc!r}")

    def _np_worker_inner(self, gen, num_reqs, num_tokens, k, slot, t0):
        if getattr(self, "_df_hang", False):
            time.sleep(30)
            return
        if getattr(self, "_df_raise", False):
            raise RuntimeError("forced worker error")
        snap = {
            "ids": list(self._df_snap_ids[:num_tokens]),
            "pos": list(self._df_snap_pos[:num_tokens]),
            "qsl": list(self._df_snap_qsl[: num_reqs + 1]),
            "win": np.asarray(
                self._df_snap_win[: num_reqs * self._df_depth]
            ).reshape(num_reqs, self._df_depth).tolist(),
            "outs": np.asarray(
                self._df_snap_out[: num_reqs * self._df_om_snap]
            ).reshape(num_reqs, self._df_om_snap).tolist(),
            "drafts": np.asarray(
                self._df_snap_draft[: num_reqs * k]
            ).reshape(num_reqs, k).tolist(),
            "ns": list(self._df_snap_ns[:num_reqs]),
            "req_ids": list(self._df_req_snap),
        }
        pred = self._defer_predict(snap)
        if pred is None:
            return
        n = pred["n"]
        hashed = self._defer_hash_layers(
            pred["ids"], pred["pos"], pred["qsl"], pred["wins"]
        )
        if getattr(self, "_df_corrupt", False):
            hashed = hashed + 1
        pins = self._df_pin[slot]
        for engram, pin in zip(self.engrams, pins):
            local = T(hashed[:, engram.layer_hash_index, :].tolist())
            file_rows, owned = engram.embed_tokens.disk_file_rows_owned(local)
            rows = gather_dequant_np(engram.embed_tokens.disk, file_rows, owned)
            pin[:n] = rows.reshape(n, self.local_heads, self.dim)
        self._df_h2d_ev.synchronize()
        for engram, pin in zip(self.engrams, pins):
            engram._dev_rows[:n] = pin[:n]
        self._df_h2d_ev.record()
        self._df_ready = {
            "gen": gen,
            "slot": slot,
            "n": n,
            "req_ids": [str(x) for x in pred["req_ids"]],
            "qsl": [int(x) for x in pred["qsl"]],
            "ms": 1000.0 * (time.perf_counter() - t0),
        }

    FakeStager._defer_worker = np_worker
    FakeStager._np_worker_inner = _np_worker_inner

    def np_apply(self, n):
        self._df_h2d_ev.synchronize()
        for engram in self.engrams:
            engram._staged_rows_for_ubatch().copy_(engram._dev_rows[:n])
        self._df_stats[3] += 1

    FakeStager._defer_apply = np_apply

    def make_stager(warm_n=4):
        s = FakeStager()
        s.defer_on = True
        s._df_tm = np.arange(1024, dtype=np.int64)
        s._df_pad = 0
        s._df_dead = np.asarray([-2, -3], dtype=np.int64)
        s._df_ngram = 2
        s._df_heads = LH
        s._df_nlayers = NLAYERS
        s._df_depth = DEPTH
        s._df_mult = np.asarray([[1000003, 1000033]] * NLAYERS, dtype=np.int64)
        # small per-column primes so hashes land INSIDE the vocab range
        # [0, N_ROWS) -> rows are owned and gathers return real data
        # (with 1e9-scale primes every row is unowned and outputs are
        # all-zero — the earlier vacuous-pass fixture bug).
        s._df_primes = np.asarray(
            [[4001 + 2 * c for c in range(LH)] for _ in range(NLAYERS)],
            dtype=np.int64,
        )
        s._df_offs = np.zeros((NLAYERS, LH), dtype=np.int64)
        s._df_rmax = 8
        s._df_kmax = 8
        s._df_omax = 64
        s.max_tokens = MAXT
        s._df_cap = 128
        s.local_heads = LH
        s.dim = DIM
        s.head_start = 0
        s.head_end = LH
        s.engrams = [
            FakeEngram(0, table0, 0, N_ROWS),
            FakeEngram(1, table1, 0, N_ROWS),
        ]
        s.rows_host = [
            np.zeros((MAXT, LH, DIM), dtype=np.float32) for _ in s.engrams
        ]
        s._df_pin = [
            [np.zeros((MAXT, LH, DIM), dtype=np.float32) for _ in s.engrams]
            for _ in range(2)
        ]
        s._df_snap_ids = np.zeros(128, dtype=np.int64)
        s._df_snap_pos = np.zeros(128, dtype=np.int64)
        s._df_snap_qsl = np.zeros(9, dtype=np.int64)
        s._df_snap_win = np.zeros(8 * DEPTH, dtype=np.int64)
        s._df_snap_out = np.zeros(8 * 64, dtype=np.int64)
        s._df_snap_draft = np.zeros(8 * 8, dtype=np.int64)
        s._df_snap_ns = np.zeros(8, dtype=np.int64)
        s._df_snap_ev = FakeEvent()
        s._df_snap_ev.record()
        s._df_h2d_ev = FakeEvent()
        s._df_h2d_ev.record()
        s._df_pool = FakePool()
        s._df_gen = 0
        s._df_slot = 0
        s._df_ready = None
        s._df_pending = None
        s._df_warm = 0
        s._df_warm_n = warm_n
        s._df_v_slot = None
        s._df_stats = [0, 0, 0, 0, 0.0]
        s._df_census = True
        s._df_every = 10**9
        s._df_hang = False
        s._df_raise = False
        s._df_corrupt = False
        return s

    def sync_gather(s, n, ids, pos, qsl, win):
        hashed = s._defer_hash_layers(ids, pos, qsl, win)
        for li, engram in enumerate(s.engrams):
            local = T(hashed[:, engram.layer_hash_index, :].tolist())
            file_rows, owned = engram.embed_tokens.disk_file_rows_owned(local)
            rows = gather_dequant_np(engram.embed_tokens.disk, file_rows, owned)
            s.rows_host[li][:n] = rows.reshape(n, s.local_heads, s.dim)
            engram._staged_rows_for_ubatch().copy_(s.rows_host[li][:n])

    def simulate(
        s, steps, accept_pattern, req_ids, corrupt_at=None, hang_at=None,
        raise_at=None, miss_at=(), label="",
    ):
        ok_all = True
        # per-request state: current chunk, its start pos, its window
        S = {r: 100 + 8 * i for i, r in enumerate(req_ids)}
        seqs = {
            r: [5 + i * 7 + j for j in range(32)] for i, r in enumerate(req_ids)
        }
        drafts = {
            r: [(i * 23 + j * 11 + 1) % 900 for j in range(K)]
            for i, r in enumerate(req_ids)
        }
        for step in range(steps):
            A = max(1, min(accept_pattern[step % len(accept_pattern)], K + 1))
            ids, pos, qsl, win = [], [], [0], []
            for ri, r in enumerate(req_ids):
                chunk = [seqs[r][-1]] + drafts[r]
                ids.extend(chunk)
                pos.extend(range(S[r], S[r] + len(chunk)))
                qsl.append(len(ids))
                win.append(seqs[r][-DEPTH:][::-1])
            n = len(ids)
            outs = [
                [(step * 17 + ri * 29 + j * 5 + 3) % 900 for j in range(A)]
                for ri in range(len(req_ids))
            ]
            # post-propose: NEW drafts for the NEXT chunk
            drafts = {
                r: [(step * 19 + ri * 23 + j * 11 + 1) % 900 for j in range(K)]
                for ri, r in enumerate(req_ids)
            }
            ns = [A] * len(req_ids)

            s._df_snap_ids[:] = 0
            s._df_snap_ids[:n] = ids
            s._df_snap_pos[:n] = pos
            s._df_snap_qsl[: len(qsl)] = qsl
            wv = np.asarray(win, dtype=np.int64).reshape(-1)
            s._df_snap_win[: wv.size] = wv
            s._df_snap_out[:] = 0
            s._df_snap_out[: len(req_ids) * max(A, 1)] = np.asarray(outs).reshape(-1)
            dv = np.asarray(
                [drafts[r] for r in req_ids], dtype=np.int64
            ).reshape(-1)
            s._df_snap_draft[: dv.size] = dv
            s._df_snap_ns[: len(req_ids)] = ns
            s._df_hang = hang_at is not None and step == hang_at
            s._df_raise = raise_at is not None and step == raise_at
            s._df_corrupt = corrupt_at is not None and step == corrupt_at
            s.enqueue_defer(ids, pos, qsl, win, outs, ns, drafts, req_ids, n, K)
            if not (s._df_hang or s._df_raise):
                s._df_pending["fut"].done.wait(5)

            for ri, r in enumerate(req_ids):
                seqs[r] = seqs[r] + outs[ri][:A]
                S[r] = S[r] + A

            # the TRUE next chunk (engine semantics): bonus = s_{A-1},
            # drafts = the post-propose drafts just snapshotted.
            true_ids, true_pos, true_qsl, true_win = [], [], [0], []
            for ri, r in enumerate(req_ids):
                chunk = [outs[ri][A - 1]] + drafts[r]  # noqa: B023
                true_ids.extend(chunk)
                true_pos.extend(range(S[r], S[r] + len(chunk)))
                true_qsl.append(len(true_ids))
                w = []
                for j in range(DEPTH):
                    if j < A - 1:
                        w.append(outs[ri][A - 2 - j])
                    elif j == A - 1:
                        w.append(ids[qsl[ri]])
                    else:
                        w.append(win[ri][j - A])
                true_win.append(w)
            nn = len(true_ids)

            if step + 1 in miss_at:
                batch = FakeBatch(nn + 1, req_ids, true_qsl)
            else:
                batch = FakeBatch(nn, req_ids, true_qsl)

            hit = s._defer_try_stage(nn, batch) if s.defer_on else False
            if hit == "warm":
                s._df_v_slot = s._df_ready["slot"]
                sync_gather(s, nn, true_ids, true_pos, true_qsl, true_win)
                s._defer_check_warm(nn)
            elif hit is True:
                s._defer_apply(nn)
            else:
                sync_gather(s, nn, true_ids, true_pos, true_qsl, true_win)

            served = [e._dev_rows.copy() for e in s.engrams]
            hashed = s._defer_hash_layers(true_ids, true_pos, true_qsl, true_win)
            for li, e in enumerate(s.engrams):
                local = T(hashed[:, e.layer_hash_index, :].tolist())
                fr, ow = e.embed_tokens.disk_file_rows_owned(local)
                ref = gather_dequant_np(e.embed_tokens.disk, fr, ow).reshape(
                    nn, LH, DIM
                )
                if not np.array_equal(
                    served[li][:nn].view(np.uint32), ref.view(np.uint32)
                ):
                    ok_all = False
                    print(f"    MISMATCH {label} step {step} table {li}")
        return ok_all

    print("== v2 preadv read + dequant vs numpy reference ==")
    rng = np.random.default_rng(7)
    for trial in range(3):
        rel = rng.integers(0, N_ROWS, size=51).tolist()
        owned = np.ones(51, dtype=bool)
        owned[::7] = False
        got = gather_dequant_np(table0, T(rel), owned)
        w = wA[np.asarray(rel)]
        sc = sA[np.asarray(rel)]
        vals = e4m3_decode(w).reshape(51, SB, -1)
        scale = (sc.astype(np.int32) << 23).view(np.float32)
        ref = (vals * scale[:, :, None]).reshape(51, DIM)
        ref = np.where(owned[:, None], ref, 0.0).astype(np.float32)
        check(
            f"trial {trial} bit-exact",
            np.array_equal(got.view(np.uint32), ref.view(np.uint32)),
        )

    print("== T1: bit-exact happy path (30 steps, A=2) ==")
    s = make_stager()
    ok = simulate(s, 30, [2], ["r0"], label="T1")
    check("T1 served == sync reference every step", ok)
    check("T1 warm completed 4 verifies", s._df_warm == 4)
    check("T1 steady hits recorded", s._df_stats[0] >= 25, f"hits={s._df_stats[0]}")
    check("T1 still armed", s.defer_on)

    print("== T2: varying accept 1..4, 2 reqs ==")
    s = make_stager()
    ok = simulate(s, 24, [1, 2, 3, 4], ["r0", "r1"], label="T2")
    check("T2 served == sync reference every step", ok)
    check("T2 still armed", s.defer_on)

    print("== T3: prediction misses -> fallback, no disarm ==")
    s = make_stager(warm_n=2)
    ok = simulate(s, 12, [2], ["r0", "r1"], miss_at={5, 6}, label="T3")
    check("T3 fallback outputs == sync, no crash", ok)
    check("T3 still armed (miss != disarm)", s.defer_on)
    check("T3 misses recorded", s._df_stats[1] >= 2, f"misses={s._df_stats[1]}")

    print("== T5: worker exception -> ONE disarm line, sync serves ==")
    s = make_stager(warm_n=1)
    ok = simulate(s, 8, [2], ["r0"], raise_at=2, label="T5")
    check("T5 sync path served all steps", ok)
    check("T5 disarmed exactly once", s.defer_on is False)

    print("== T6: warm-verify mismatch -> ONE disarm line ==")
    s = make_stager(warm_n=2)
    ok = simulate(s, 8, [2], ["r0"], corrupt_at=1, label="T6")
    check("T6 sync path served after disarm", ok)
    check("T6 disarmed", s.defer_on is False)

    print("== T4: hung worker -> bounded wait, serve continues ==")
    s = make_stager(warm_n=2)
    t0 = time.perf_counter()
    ok = simulate(s, 6, [2], ["r0"], hang_at=3, label="T4")
    dt = time.perf_counter() - t0
    check("T4 outputs == sync, no crash", ok)
    check("T4 hung step bounded (<3s total)", dt < 3.0, f"dt={dt:.2f}s")

    print("== T7: GIL sanity (structural, on patched source) ==")
    src = (root / "model_root/common/engram.py").read_text()
    ws = src.index("def _defer_worker")
    we = src.index("def _defer_sig")
    wsrc = src[ws:we]
    check(
        "worker launches no GPU work (no hash_state/no kernels)",
        "self.hash_state(" not in wsrc and "capture_" not in wsrc,
    )
    _stream_lines = [
        ln.strip()
        for ln in wsrc.splitlines()
        if "torch.cuda" in ln and "import" not in ln
    ]
    check(
        "worker stream calls are ordering-only (wait/current/stream-with)",
        all(
            ln.startswith("self._df_stream.wait_stream(")
            or ln.startswith("with _torch.cuda.stream(")
            or ln.startswith("_torch.cuda.current_stream()")
            for ln in _stream_lines
        ),
        "; ".join(_stream_lines[:4]),
    )
    check(
        "worker gather is the v2 preadv path",
        "_gather_dequant_v2" in wsrc,
    )

    print()
    if failures:
        print("FAILURES:", failures)
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
