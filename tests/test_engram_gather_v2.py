"""engram_gather_v2 read path on a synthetic safetensors file (CPU, numpy).

Port of results/2026-09-21-gatherv2/validate_gather_v2.py minus the torch
dequant half: _gv2_read_runs vs a per-row os.pread reference, the WILLNEED
prefill pre-pass and the DSV41_ENGRAM_GATHER_V2_MAX_ROWS gate.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import random
import re
import struct
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_engram_chain import _build_tree, engram_gather_v2, engram_stage_census  # noqa: E402

GV2_ENVS = ("DSV41_ENGRAM_WILLNEED", "DSV41_ENGRAM_WILLNEED_MIN_ROWS", "DSV41_ENGRAM_GATHER_V2_MAX_ROWS")


def _load_patched_disk(tmp: Path, env: dict | None = None):
    """census + gather v2 on the recipe engram_disk.py, imported as a module."""
    vllm = _build_tree(tmp)
    with contextlib.redirect_stdout(io.StringIO()):
        engram_stage_census.apply(vllm)
        engram_gather_v2.apply(vllm)
    clean = {k: v for k, v in os.environ.items() if k not in GV2_ENVS}
    clean.update(env or {})
    spec = importlib.util.spec_from_file_location(
        "engram_disk_gv2_%d" % random.randrange(1 << 30),
        vllm / "models/deepseek_v4_1/common/engram_disk.py",
    )
    mod = importlib.util.module_from_spec(spec)
    with mock.patch.dict(os.environ, clean, clear=True):
        spec.loader.exec_module(mod)
    return mod


def _pack_model(dst: Path, n_rows: int, dim: int, sb: int, seed: int) -> str:
    """Safetensors framing _open() parses: <Q header len, json header, data."""
    rng = np.random.default_rng(seed)
    model = dst / "model"
    model.mkdir()
    shard = model / "model-00001-of-00001.safetensors"
    header = {
        "layers.0.engram.embed.weight": {"dtype": "F8_E4M3", "shape": [n_rows, dim], "data_offsets": [0, n_rows * dim]},
        "layers.0.engram.embed.scale": {
            "dtype": "F8_E8M0",
            "shape": [n_rows, sb],
            "data_offsets": [n_rows * dim, n_rows * (dim + sb)],
        },
    }
    hjson = json.dumps(header).encode()
    with open(shard, "wb") as fh:
        fh.write(struct.pack("<Q", len(hjson)))
        fh.write(hjson)
        fh.write(rng.integers(0, 256, n_rows * (dim + sb), dtype=np.uint8).tobytes())
    idx = {"weight_map": {k: shard.name for k in header}}
    (model / "model.safetensors.index.json").write_text(json.dumps(idx))
    return str(model)


class _Trunc:
    """os.preadv that delivers at most `cap` bytes per call (forces resume)."""

    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.calls = 0
        self.real = os.preadv

    def __call__(self, fd, iov, off=0):
        self.calls += 1
        tmp = bytearray(min(self.cap, sum(len(x) for x in iov)))
        n = self.real(fd, [memoryview(tmp)], off)
        pos = 0
        for x in iov:
            take = min(n - pos, len(x))
            if take <= 0:
                break
            x[:take] = tmp[pos : pos + take]
            pos += take
        return n


class GatherV2ReadTests(unittest.TestCase):
    N, DIM, BS = 4096, 256, 32
    SB = DIM // BS

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.mod = _load_patched_disk(tmp / "a")
        cls.tbl = cls.mod.DiskEngramTable(_pack_model(tmp, cls.N, cls.DIM, cls.SB, 11), 0, cls.DIM, cls.BS)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tbl.pool.shutdown()
        for fd in (cls.tbl.w_fd, cls.tbl.s_fd):
            os.close(fd)
        cls._tmp.cleanup()

    def _cases(self):
        rng = random.Random(42)
        owned = [rng.random() < 0.85 for _ in range(51)]
        yield "fuzz R=51 mixed-owned", [rng.randrange(self.N) if o else 0 for o in owned]
        yield "contig runs + dups", [7, 8, 9, 7, 100, 101, 102, 103, 100, 55, 54, 0, 0]
        yield "all-unowned row 0", [0] * 51
        yield "single row", [1234]
        yield "last row", [self.N - 1, self.N - 2]
        yield "fuzz R=600", [rng.randrange(self.N) for _ in range(600)]

    def _files(self):
        t = self.tbl
        return (("w", t.w_fd, t.w_off, t.dim), ("s", t.s_fd, t.s_off, t.sb))

    def _v2(self, fd, base, rel, rb):
        buf = memoryview(np.zeros(len(rel) * rb, np.uint8)).cast("B")
        preads, runs = self.tbl._gv2_read_runs(fd, base, rel, rb, buf)
        return bytes(buf), preads, runs

    @staticmethod
    def _ref(fd, base, rel, rb):
        return b"".join(os.pread(fd, rb, base + r * rb) for r in rel)

    def test_read_runs_match_per_row_reference(self) -> None:
        for name, rel in self._cases():
            for fname, fd, base, rb in self._files():
                got, preads, runs = self._v2(fd, base, rel, rb)
                self.assertEqual(got, self._ref(fd, base, rel, rb), f"{name} {fname}")
                self.assertEqual(preads, runs, f"{name} {fname}: no partial reads expected")
        _, preads, runs = self._v2(self.tbl.w_fd, self.tbl.w_off, [7, 8, 9, 7, 100, 101], self.DIM)
        self.assertEqual((preads, runs), (2, 2))

    def test_stock_read_rows_agrees(self) -> None:
        rel = next(r for n, r in self._cases() if n == "fuzz R=600")
        for fname, fd, base, rb in self._files():
            buf = memoryview(np.zeros(len(rel) * rb, np.uint8)).cast("B")
            self.tbl._read_rows(fd, base, rel, rb, buf)
            self.assertEqual(bytes(buf), self._v2(fd, base, rel, rb)[0], fname)

    def test_partial_preadv_resumes_mid_row(self) -> None:
        rel = [9, 10, 11, 200, 201, 9, 4095]
        for fname, fd, base, rb in self._files():
            tr = _Trunc(100 if rb > 100 else 5)
            with mock.patch.object(os, "preadv", tr):
                got, preads, runs = self._v2(fd, base, rel, rb)
            self.assertEqual(got, self._ref(fd, base, rel, rb), fname)
            self.assertGreater(preads, runs, fname)
            self.assertEqual(preads, tr.calls)

    def test_willneed_defaults(self) -> None:
        self.assertIs(self.mod._ENG_GV2_WILLNEED, False)  # off until the E1 serve arm
        self.assertEqual(self.mod._ENG_GV2_WILLNEED_MIN, 512)
        self.assertEqual(self.mod._ENG_GV2_MAX_ROWS, 1 << 62)
        with tempfile.TemporaryDirectory() as tmp:
            mod = _load_patched_disk(
                Path(tmp),
                {
                    "DSV41_ENGRAM_WILLNEED": "1",
                    "DSV41_ENGRAM_WILLNEED_MIN_ROWS": "64",
                    "DSV41_ENGRAM_GATHER_V2_MAX_ROWS": "256",
                },
            )
        self.assertIs(mod._ENG_GV2_WILLNEED, True)
        self.assertEqual(mod._ENG_GV2_WILLNEED_MIN, 64)
        self.assertEqual(mod._ENG_GV2_MAX_ROWS, 256)

    def _willneed(self, rel, on=True, min_rows=4):
        calls = []
        with mock.patch.object(self.mod, "_ENG_GV2_WILLNEED", on), mock.patch.object(
            self.mod, "_ENG_GV2_WILLNEED_MIN", min_rows
        ), mock.patch.object(os, "posix_fadvise", lambda fd, o, n, adv: calls.append((fd, o, n, adv))):
            n = self.tbl._gv2_willneed(rel)
        self.assertEqual(n, len(calls))
        return calls

    def test_willneed_gate(self) -> None:
        rel = list(range(0, 4000, 37))
        self.assertEqual(self._willneed(rel, on=False), [])
        self.assertEqual(self._willneed(rel[:3]), [])
        self.assertEqual(self._willneed(rel[:48], min_rows=512), [])
        self.assertTrue(self._willneed(rel[:4]))

    def test_willneed_spans_cover_every_row_once(self) -> None:
        rng = random.Random(7)
        rel = [rng.randrange(self.N) for _ in range(700)] + [5, 5, 6, 0]
        calls = self._willneed(rel)
        for fname, fd, base, rb in self._files():
            spans = [(o, o + n) for f, o, n, adv in calls if f == fd]
            self.assertTrue(all(adv == os.POSIX_FADV_WILLNEED for *_, adv in calls))
            self.assertTrue(all(s % 4096 == 0 and e % 4096 == 0 and e > s for s, e in spans), fname)
            # sorted, disjoint and not touching (touching spans are merged)
            self.assertTrue(all(a[1] < b[0] for a, b in zip(spans, spans[1:])), fname)
            for r in set(rel):
                lo, hi = base + r * rb, base + (r + 1) * rb
                self.assertTrue(any(s <= lo and hi <= e for s, e in spans), f"{fname} row {r}")
        n_w = sum(1 for f, *_ in calls if f == self.tbl.w_fd)
        self.assertLess(len(calls) - n_w, n_w, "8-byte scale rows share pages")

    def test_willneed_then_read_is_byte_identical(self) -> None:
        rel = next(r for n, r in self._cases() if n == "fuzz R=600")
        with mock.patch.object(self.mod, "_ENG_GV2_WILLNEED", True), mock.patch.object(
            self.mod, "_ENG_GV2_WILLNEED_MIN", 4
        ):
            self.assertGreater(self.tbl._gv2_willneed(rel), 0)
        for fname, fd, base, rb in self._files():
            self.assertEqual(self._v2(fd, base, rel, rb)[0], self._ref(fd, base, rel, rb), fname)

    def test_max_rows_gate_routes_large_calls_to_stock(self) -> None:
        class Rel:
            def __init__(self, n):
                self.n = n

            def numel(self):
                return self.n

        class Stock(Exception):
            pass

        def stock_torch():
            raise Stock

        with mock.patch.object(self.mod, "_ENG_GATHER_V2", [True]), mock.patch.object(
            self.mod, "_torch", stock_torch
        ), mock.patch.object(self.tbl, "_gv2_checked", lambda rel, owned: "v2", create=True):
            for limit, n, want in ((1 << 62, 100000, "v2"), (4, 4, "v2"), (4, 5, "stock")):
                with mock.patch.object(self.mod, "_ENG_GV2_MAX_ROWS", limit):
                    try:
                        got = self.tbl.gather_dequant(Rel(n), None)
                    except Stock:
                        got = "stock"
                self.assertEqual(got, want, (limit, n))

    def test_census_is_consistent_under_13_stage_threads(self) -> None:
        # The stage pool runs one gather per table concurrently. Unlocked,
        # a thread could test st[0] after another reset it (0 % 32 == 0,
        # then // 0): ZeroDivisionError disarmed gv2 for the process.
        threads, per_thread, every = 13, 400, 4
        tables = [types.SimpleNamespace(_pf_expected=None) for _ in range(threads)]
        rows = list(range(96))
        owned = types.SimpleNamespace(tolist=lambda: [True] * len(rows))
        errors, out = [], io.StringIO()
        census = self.mod.DiskEngramTable._gv2_census

        def stage(tbl):
            try:
                for _ in range(per_thread):
                    tbl._pf_expected = set(range(0, 96, 2))
                    census(tbl, len(rows), 96, 96, 1e-4, 0, rows, owned)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        old = sys.getswitchinterval()
        self.mod._ENG_GV2_SEEN[:] = [0, 0, 0, 0, 0.0, 0, 0, 0, 0]
        sys.setswitchinterval(1e-6)
        try:
            with mock.patch.object(self.mod, "_ENG_GV2_CENSUS", True), mock.patch.object(
                self.mod, "_ENG_GV2_EVERY", every
            ), contextlib.redirect_stdout(out):
                ts = [threading.Thread(target=stage, args=(tb,)) for tb in tables]
                for th in ts:
                    th.start()
                for th in ts:
                    th.join()
        finally:
            sys.setswitchinterval(old)
        self.assertEqual(errors, [])
        lines = out.getvalue().splitlines()
        self.assertEqual(len(lines), threads * per_thread // every)
        self.assertEqual(sum(int(re.search(r"calls=(\d+)", ln).group(1)) for ln in lines),
                         threads * per_thread, "no call lost or double-counted")
        self.assertTrue(all(" pf_hit=100%(" in ln for ln in lines), lines[:2])

    def test_v2_matches_stock_with_census_pf_hit(self) -> None:
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("torch not installed")
        rel = [r for _, r in self._cases()][0]
        owned = torch.tensor([r != 0 for r in rel])
        rel_t = torch.tensor(rel, dtype=torch.int64)
        out = io.StringIO()
        with mock.patch.object(self.mod, "_ENG_GATHER_V2", [False]):
            ref = self.tbl.gather_dequant(rel_t, owned)
        self.tbl._pf_expected = {r for r in rel[:10] if r} | {self.N + 5}
        with mock.patch.object(self.mod, "_ENG_GV2_CENSUS", True), mock.patch.object(
            self.mod, "_ENG_GV2_EVERY", 1
        ), mock.patch.object(self.mod, "_ENG_GV2_WILLNEED_MIN", 4), contextlib.redirect_stdout(out):
            got = self.tbl._gather_dequant_v2(rel_t, owned)
        self.assertTrue(torch.equal(got.view(torch.int16), ref.view(torch.int16)))
        self.assertIn("[gv2-census]", out.getvalue())
        self.assertIn(" pf_hit=", out.getvalue())
        self.assertIsNone(self.tbl._pf_expected)


if __name__ == "__main__":
    unittest.main()
