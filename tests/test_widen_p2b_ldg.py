#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN_CU = ROOT / "tests/fixtures/p2b_moe.pin.cu"


def _load(rel: str):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class WidenP2bLdgTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.shapes = _load("docker/patch/widen_p2b_shapes.py")
        cls.mrow = _load("docker/patch/widen_p2b_mrow.py")
        cls.cfg1 = _load("docker/patch/widen_p2b_cfg1.py")
        cls.ldg = _load("docker/patch/widen_p2b_ldg.py")
        cls.pin = PIN_CU.read_text()

    def _cfg1(self) -> str:
        return self.cfg1.patch_cu(self.mrow.patch_cu(self.shapes.patch_cu(self.pin)))

    def test_pin_uses_streaming_ldcs_for_trellis(self) -> None:
        self.assertIn("__ldcs(", self.pin)
        self.assertGreaterEqual(self.pin.count("__ldcs("), 2)
        self.assertIn("ld_b", self.pin)

    def test_ldg_after_cfg1_replaces_every_ldcs(self) -> None:
        base = self._cfg1()
        n_ldcs = base.count("__ldcs(")
        self.assertGreaterEqual(n_ldcs, 2)
        patched = self.ldg.patch_cu(base)
        self.assertEqual(self.ldg.patch_cu(patched), patched)
        self.assertNotIn("__ldcs(", patched)
        self.assertEqual(patched.count("__ldg("), n_ldcs)
        self.assertIn("run_gemv_tile<BITS, 1, 1>", patched)
        self.assertIn("m * experts * warps_per_exp", patched)
        self.assertIn("__launch_bounds__(256, 4)", patched)

    def test_second_apply_is_noop(self) -> None:
        patched = self.ldg.patch_cu(self._cfg1())
        self.assertEqual(self.ldg.patch_cu(patched), patched)

    def test_apply_writes_p2b_moe_cu(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csrc = root / "csrc"
            csrc.mkdir()
            cu = csrc / "p2b_moe.cu"
            cu.write_text(self._cfg1())
            self.ldg.apply(root)
            out = cu.read_text()
            self.assertNotIn("__ldcs(", out)
            self.assertIn("__ldg(", out)
