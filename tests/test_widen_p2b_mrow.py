#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN_CU = ROOT / "tests/fixtures/p2b_moe.pin.cu"
PIN_PY = ROOT / "tests/fixtures/exl3_native_moe.pin.py"


def _load(name: str, rel: str):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TrueMrowP2bTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.shapes = _load("widen_p2b_shapes", "docker/patch/widen_p2b_shapes.py")
        cls.mrow = _load("widen_p2b_mrow", "docker/patch/widen_p2b_mrow.py")
        cls.pin_cu = PIN_CU.read_text()
        cls.pin_py = PIN_PY.read_text()

    def test_fixture_is_the_pin_kernel(self) -> None:
        self.assertIn("p2b_moe_batched_kernel", self.pin_cu)
        self.assertIn("const size_t a_row0 = 0", self.pin_cu)
        self.assertIn("for row in range(int(x2d.shape[0])):", self.pin_py)

    def test_shapes_then_mrow_on_pin_cu(self) -> None:
        shaped = self.shapes.patch_cu(self.pin_cu)
        patched = self.mrow.patch_cu(shaped)
        self.assertEqual(self.mrow.patch_cu(patched), patched)
        self.assertIn("m * experts * warps_per_exp", patched)
        self.assertIn("row * experts + e", patched)
        self.assertIn("(size_t) e * m + row", patched)
        self.assertIn("x_row + w * 128", patched)
        self.assertIn("accum + row * hidden + col", patched)
        self.assertNotIn("for (int row = 0; row < m; ++row)", patched)
        self.assertNotIn("x.size(0) == 1 &&", patched)
        self.assertEqual(patched.count("grid.sync()"), shaped.count("grid.sync()"))
        # GEMV tile is still one row; A2 already points at that row.
        self.assertIn("const size_t a_row0 = 0", patched)

    def test_mrow_on_stock_cu_without_shapes(self) -> None:
        patched = self.mrow.patch_cu(self.pin_cu)
        self.assertIn("m * experts * warps_per_exp", patched)
        self.assertEqual(self.mrow.patch_cu(patched), patched)

    def test_python_one_launch_not_per_row(self) -> None:
        patched = self.mrow.patch_py(self.pin_py)
        self.assertNotIn("for row in range(int(x2d.shape[0])):", patched)
        self.assertNotIn("xh[row : row + 1]", patched)
        self.assertIn("safe_ids,", patched)
        self.assertIn("xh,", patched)
        self.assertEqual(self.mrow.patch_py(patched), patched)

    def test_identity_would_fail(self) -> None:
        shaped = self.shapes.patch_cu(self.pin_cu)
        patched = self.mrow.patch_cu(shaped)
        self.assertNotEqual(patched, shaped)
        self.assertNotEqual(self.mrow.patch_py(self.pin_py), self.pin_py)


if __name__ == "__main__":
    unittest.main()
