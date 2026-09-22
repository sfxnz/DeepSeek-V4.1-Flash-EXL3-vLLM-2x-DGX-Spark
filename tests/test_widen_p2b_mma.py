#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
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


class WidenP2bMmaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.shapes = _load("docker/patch/widen_p2b_shapes.py")
        cls.mrow = _load("docker/patch/widen_p2b_mrow.py")
        cls.mma = _load("docker/patch/widen_p2b_mma.py")
        cls.pin = PIN_CU.read_text()

    def _patched(self) -> str:
        return self.mma.patch_cu(self.mrow.patch_cu(self.shapes.patch_cu(self.pin)))

    def test_mma_on_shapes_then_mrow_pin(self) -> None:
        patched = self._patched()
        self.assertEqual(self.mma.patch_cu(patched), patched)
        self.assertIn("const int* __restrict__ a_off", patched)
        self.assertIn("FragB f0s[WNT], f1s[WNT]", patched)
        self.assertIn("for (int row = 0; row < MAX_M; ++row)", patched)
        self.assertIn("if (ids[p] == src)", patched)
        self.assertNotIn("const size_t a_row0 = 0", patched)
        self.assertNotIn("run_gemv_tile<BITS, 1, 0>(B32, A2, C,", patched)

    def test_b_decode_is_outside_row_loop(self) -> None:
        patched = self._patched()
        tile = patched.split("void run_gemv_tile(")[1].split("void p2b_moe_batched_kernel")[0]
        decode_at = tile.find("FragB f0s[WNT], f1s[WNT]")
        row_at = tile.find("for (int row = 0; row < MAX_M; ++row)")
        self.assertGreater(decode_at, 0)
        self.assertGreater(row_at, decode_at)

    def test_ld_b_is_outside_row_loop(self) -> None:
        patched = self._patched()
        tile = patched.split("void run_gemv_tile(")[1].split("void p2b_moe_batched_kernel")[0]
        row_at = tile.find("for (int row = 0; row < MAX_M; ++row)")
        last_ldb = tile.rfind("ld_b(")
        self.assertGreater(row_at, 0)
        self.assertLess(last_ldb, row_at)

    def test_not_the_reverted_serial_moe_loop(self) -> None:
        patched = self._patched()
        self.assertNotIn("for (int row = 0; row < m; ++row)", patched)
        self.assertEqual(
            patched.count("grid.sync()"),
            self.mrow.patch_cu(self.shapes.patch_cu(self.pin)).count("grid.sync()"),
        )

    def test_mma_on_stock_without_mrow(self) -> None:
        patched = self.mma.patch_cu(self.pin)
        self.assertIn("FragB f0s[WNT], f1s[WNT]", patched)
        self.assertIn("if (ids[p] == src)", patched)

    def test_identity_would_fail(self) -> None:
        shaped = self.mrow.patch_cu(self.shapes.patch_cu(self.pin))
        patched = self.mma.patch_cu(shaped)
        self.assertNotEqual(patched, shaped)


if __name__ == "__main__":
    unittest.main()
