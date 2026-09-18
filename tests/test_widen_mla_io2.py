#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/decode_dsv4_kernel.pin.cuh"


def _load():
    path = ROOT / "docker/patch/widen_mla_io2.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class WidenMlaIo2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load()
        cls.pin = FIXTURE.read_text()

    def test_io2_linear_tid_single_expect_leader(self) -> None:
        patched = self.mod.patch_cuh(self.pin)
        self.assertIn("constexpr int DSV4_IO_WARPS = 2;", patched)
        self.assertNotIn("constexpr int DSV4_IO_WARPS = 1;", patched)
        self.assertIn("eo + io_tid", patched)
        self.assertNotIn("eo + lane;", patched)
        self.assertIn("threadIdx.x == DSV4_MATH_THREADS", patched)
        self.assertNotIn("if (lane == 0) {\n      mbarrier_arrive_expect_tx", patched)
        self.assertIn("constexpr int DSV4_N_WARPS = 8;", patched)
        self.assertIn("constexpr int DSV4_KV_BUF_COUNT = 2;", patched)
        self.assertIn("constexpr int DSV4_CAND_WINDOW = 64;", patched)
        self.assertEqual(patched.count("eo + io_tid"), 2)

    def test_second_apply_is_noop(self) -> None:
        patched = self.mod.patch_cuh(self.pin)
        self.assertEqual(self.mod.patch_cuh(patched), patched)

    def test_apply_writes_cuh(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cuh = root / "include/flashinfer/attention/sparse_mla_sm120/decode_dsv4_kernel.cuh"
            cuh.parent.mkdir(parents=True)
            cuh.write_text(self.pin)
            self.assertTrue(self.mod.apply(root))
            out = cuh.read_text()
            self.assertIn("DSV4_IO_WARPS = 2;", out)
            self.assertFalse(self.mod.apply(root))
