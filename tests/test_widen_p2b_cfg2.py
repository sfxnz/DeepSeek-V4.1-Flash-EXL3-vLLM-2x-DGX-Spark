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


class WidenP2bCfg2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.shapes = _load("docker/patch/widen_p2b_shapes.py")
        cls.mrow = _load("docker/patch/widen_p2b_mrow.py")
        cls.cfg1 = _load("docker/patch/widen_p2b_cfg1.py")
        cls.cfg2 = _load("docker/patch/widen_p2b_cfg2.py")
        cls.pin = PIN_CU.read_text()

    def _after_cfg1(self) -> str:
        return self.cfg1.patch_cu(self.mrow.patch_cu(self.shapes.patch_cu(self.pin)))

    def _patched(self) -> str:
        return self.cfg2.patch_cu(self._after_cfg1())

    def test_cfg1_fixture_is_cols64(self) -> None:
        base = self._after_cfg1()
        self.assertIn("run_gemv_tile<BITS, 1, 1>", base)
        self.assertIn("constexpr int WNT = CFG == 0 ? 2 : 4;", base)
        self.assertIn("inter / 64", base)
        self.assertIn("hidden / 64", base)
        self.assertIn("__shared__ float sh_red[8][1][64];", base)
        self.assertIn("__launch_bounds__(256, 4)", base)

    def test_cfg2_after_cfg1(self) -> None:
        base = self._after_cfg1()
        patched = self.cfg2.patch_cu(base)
        self.assertEqual(self.cfg2.patch_cu(patched), patched)
        self.assertIn("run_gemv_tile<BITS, 1, 2>", patched)
        self.assertNotIn("run_gemv_tile<BITS, 1, 1>", patched)
        self.assertNotIn("run_gemv_tile<BITS, 1, 0>", patched)
        self.assertNotIn("run_gemv_tile<BITS, 2,", patched)
        self.assertIn(
            "constexpr int WNT = CFG == 0 ? 2 : (CFG == 1 ? 4 : 8);",
            patched,
        )
        self.assertNotIn("constexpr int WNT = CFG == 0 ? 2 : 4;", patched)
        self.assertIn("__launch_bounds__(256, 4)", patched)
        self.assertNotIn("__launch_bounds__(512)", patched)
        self.assertNotIn(
            "__launch_bounds__(256)",
            patched.replace("__launch_bounds__(256, 4)", ""),
        )
        self.assertIn(
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, 256, 0);",
            patched,
        )
        self.assertIn("dim3(256)", patched)
        self.assertNotIn("dim3(512)", patched)
        self.assertIn("num_groups_gate = inter / 128", patched)
        self.assertIn("num_groups_down = hidden / 128", patched)
        self.assertNotIn("inter / 64", patched)
        self.assertNotIn("hidden / 64", patched)
        self.assertIn("__shared__ float sh_red[8][1][128];", patched)
        self.assertIn("float (*sh_red)[1][128]", patched)
        self.assertNotIn("sh_red[8][1][64]", patched)
        self.assertEqual(patched.count("grid.sync()"), base.count("grid.sync()"))
        self.assertIn("m * experts * warps_per_exp", patched)
        self.assertIn("row * experts + e", patched)
        self.assertIn("(size_t) e * m + row", patched)
        self.assertIn("const size_t a_row0 = 0", patched)
        self.assertNotIn("for (int row = 0; row < m; ++row)", patched)
        self.assertNotIn("FragB f0s[WNT], f1s[WNT]", patched)
        self.assertIn("template <int BITS>", patched)
        self.assertIn("constexpr int WK = CFG == 0 ? 16 : 8;", patched)
        self.assertEqual(5120 % 128, 0)
        self.assertEqual(1152 % 128, 0)
        self.assertEqual(5120 // 128, 40)
        self.assertEqual(1152 // 128, 9)

    def test_second_apply_is_noop(self) -> None:
        patched = self._patched()
        self.assertEqual(self.cfg2.patch_cu(patched), patched)

    def test_cfg2_refuses_stock_pin(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            self.cfg2.patch_cu(self.pin)
        self.assertIn("cfg1", str(ctx.exception))

    def test_cfg2_after_cfg1_on_stock_pin(self) -> None:
        patched = self.cfg2.patch_cu(self.cfg1.patch_cu(self.pin))
        self.assertIn("run_gemv_tile<BITS, 1, 2>", patched)
        self.assertIn("__launch_bounds__(256, 4)", patched)
        self.assertIn("sh_red[8][1][128]", patched)
        self.assertEqual(self.cfg2.patch_cu(patched), patched)

    def test_identity_would_fail(self) -> None:
        base = self._after_cfg1()
        patched = self.cfg2.patch_cu(base)
        self.assertNotEqual(patched, base)

    def test_apply_rewrites_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cu = root / "csrc" / "p2b_moe.cu"
            cu.parent.mkdir(parents=True)
            cu.write_text(self._after_cfg1())
            self.cfg2.apply(root)
            once = cu.read_text()
            self.assertIn("run_gemv_tile<BITS, 1, 2>", once)
            self.assertIn("__launch_bounds__(256, 4)", once)
            self.assertIn("kernel, 256, 0", once)
            self.assertIn("inter / 128", once)
            self.assertIn("sh_red[8][1][128]", once)
            self.assertIn("CFG == 1 ? 4 : 8", once)
            self.cfg2.apply(root)
            self.assertEqual(cu.read_text(), once)


if __name__ == "__main__":
    unittest.main()
