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


class WidenP2bCfg1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.shapes = _load("docker/patch/widen_p2b_shapes.py")
        cls.mrow = _load("docker/patch/widen_p2b_mrow.py")
        cls.cfg1 = _load("docker/patch/widen_p2b_cfg1.py")
        cls.pin = PIN_CU.read_text()

    def _after_mrow(self) -> str:
        return self.mrow.patch_cu(self.shapes.patch_cu(self.pin))

    def _patched(self) -> str:
        return self.cfg1.patch_cu(self._after_mrow())

    def test_fixture_is_cfg0_512(self) -> None:
        self.assertIn("p2b_moe_batched_kernel", self.pin)
        self.assertIn("run_gemv_tile<BITS, 1, 0>", self.pin)
        self.assertIn("__launch_bounds__(512)", self.pin)
        self.assertIn("kernel, 512, 0", self.pin)
        self.assertIn("dim3(512)", self.pin)
        self.assertIn("inter / 32", self.pin)
        self.assertIn("sh_red[16][1][32]", self.pin)

    def test_cfg1_after_shapes_then_mrow(self) -> None:
        base = self._after_mrow()
        patched = self.cfg1.patch_cu(base)
        self.assertEqual(self.cfg1.patch_cu(patched), patched)
        self.assertIn("run_gemv_tile<BITS, 1, 1>", patched)
        self.assertNotIn("run_gemv_tile<BITS, 1, 0>", patched)
        self.assertIn("__launch_bounds__(256, 4)", patched)
        self.assertNotIn("__launch_bounds__(512)", patched)
        self.assertNotIn("__launch_bounds__(256)", patched.replace("__launch_bounds__(256, 4)", ""))
        self.assertIn(
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, 256, 0);",
            patched,
        )
        self.assertIn("dim3(256)", patched)
        self.assertIn("num_groups_gate = inter / 64", patched)
        self.assertIn("num_groups_down = hidden / 64", patched)
        self.assertIn("sh_red[8]", patched)
        self.assertIn("__shared__ float sh_red[8][1][64];", patched)
        self.assertIn("float (*sh_red)[1][64]", patched)
        self.assertEqual(patched.count("grid.sync()"), base.count("grid.sync()"))
        self.assertIn("m * experts * warps_per_exp", patched)
        self.assertIn("row * experts + e", patched)
        self.assertIn("(size_t) e * m + row", patched)
        self.assertIn("const size_t a_row0 = 0", patched)
        self.assertNotIn("for (int row = 0; row < m; ++row)", patched)
        self.assertNotIn("FragB f0s[WNT], f1s[WNT]", patched)
        self.assertIn("template <int BITS>", patched)

    def test_second_apply_is_noop(self) -> None:
        patched = self._patched()
        self.assertEqual(self.cfg1.patch_cu(patched), patched)

    def test_cfg1_on_stock_pin(self) -> None:
        patched = self.cfg1.patch_cu(self.pin)
        self.assertIn("run_gemv_tile<BITS, 1, 1>", patched)
        self.assertIn("__launch_bounds__(256, 4)", patched)
        self.assertIn("sh_red[8]", patched)
        self.assertEqual(self.cfg1.patch_cu(patched), patched)

    def test_identity_would_fail(self) -> None:
        base = self._after_mrow()
        patched = self.cfg1.patch_cu(base)
        self.assertNotEqual(patched, base)

    def test_apply_rewrites_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cu = root / "csrc" / "p2b_moe.cu"
            cu.parent.mkdir(parents=True)
            cu.write_text(self._after_mrow())
            self.cfg1.apply(root)
            once = cu.read_text()
            self.assertIn("run_gemv_tile<BITS, 1, 1>", once)
            self.assertIn("__launch_bounds__(256, 4)", once)
            self.assertIn("kernel, 256, 0", once)
            self.assertIn("inter / 64", once)
            self.assertIn("sh_red[8]", once)
            self.cfg1.apply(root)
            self.assertEqual(cu.read_text(), once)


if __name__ == "__main__":
    unittest.main()
