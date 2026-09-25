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


class WidenP2bPf4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.shapes = _load("docker/patch/widen_p2b_shapes.py")
        cls.mrow = _load("docker/patch/widen_p2b_mrow.py")
        cls.cfg1 = _load("docker/patch/widen_p2b_cfg1.py")
        cls.pf4 = _load("docker/patch/attic/widen_p2b_pf4.py")
        cls.pin = PIN_CU.read_text()

    def _cfg1(self) -> str:
        return self.cfg1.patch_cu(self.mrow.patch_cu(self.shapes.patch_cu(self.pin)))

    def test_cfg1_still_pf2(self) -> None:
        base = self._cfg1()
        self.assertIn("constexpr int PF = CFG == 0 ? 4 : 2;", base)
        self.assertIn("run_gemv_tile<BITS, 1, 1>", base)

    def test_pf4_after_cfg1(self) -> None:
        patched = self.pf4.patch_cu(self._cfg1())
        self.assertIn("constexpr int PF = 4;", patched)
        self.assertNotIn("constexpr int PF = CFG == 0 ? 4 : 2;", patched)
        self.assertIn("run_gemv_tile<BITS, 1, 1>", patched)
        self.assertIn("__launch_bounds__(256, 4)", patched)
        self.assertIn("m * experts * warps_per_exp", patched)
        self.assertIn("__ldcs(", patched)
        self.assertIn("mma_ab_h", patched)
        self.assertEqual(patched.count("grid.sync()"), self._cfg1().count("grid.sync()"))

    def test_refuses_stock_pin_without_cfg1(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            self.pf4.patch_cu(self.pin)
        self.assertIn("cfg1", str(ctx.exception))

    def test_second_apply_is_noop(self) -> None:
        patched = self.pf4.patch_cu(self._cfg1())
        self.assertEqual(self.pf4.patch_cu(patched), patched)

    def test_apply_writes_p2b_moe_cu(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cu = root / "csrc" / "p2b_moe.cu"
            cu.parent.mkdir()
            cu.write_text(self._cfg1())
            self.pf4.apply(root)
            out = cu.read_text()
            self.assertIn("constexpr int PF = 4;", out)
            self.pf4.apply(root)
            self.assertEqual(cu.read_text(), out)
