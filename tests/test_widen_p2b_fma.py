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


class WidenP2bFmaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.shapes = _load("docker/patch/widen_p2b_shapes.py")
        cls.mrow = _load("docker/patch/widen_p2b_mrow.py")
        cls.cfg1 = _load("docker/patch/widen_p2b_cfg1.py")
        cls.fma = _load("docker/patch/attic/widen_p2b_fma.py")
        cls.pin = PIN_CU.read_text()

    def _cfg1(self) -> str:
        return self.cfg1.patch_cu(self.mrow.patch_cu(self.shapes.patch_cu(self.pin)))

    def test_fma_row_is_one_row_product(self) -> None:
        a16 = [float(i + 1) for i in range(16)]
        b16x8 = [[float((k + 1) * (n + 1)) for n in range(8)] for k in range(16)]
        got = self.fma.fma_row(a16, b16x8)
        want = tuple(sum(a16[k] * b16x8[k][n] for k in range(16)) for n in range(8))
        self.assertEqual(got, want)

    def test_ptx_warp_simulate_matches_one_row_product(self) -> None:
        a16 = [float(i + 1) for i in range(16)]
        b16x8 = [[float((k + 1) * (n + 1)) for n in range(8)] for k in range(16)]
        self.assertEqual(
            self.fma.fma_warp_simulate(a16, b16x8),
            self.fma.fma_warp_pack(a16, b16x8),
        )

    def test_collapsed_mapping_does_not_match_one_row_product(self) -> None:
        a16 = [float(i + 1) for i in range(16)]
        b16x8 = [[float((k + 1) * (n + 1)) for n in range(8)] for k in range(16)]
        self.assertNotEqual(
            self.fma.fma_warp_simulate_collapsed(a16, b16x8),
            self.fma.fma_warp_pack(a16, b16x8),
        )

    def test_cfg1_still_uses_padded_mma(self) -> None:
        base = self._cfg1()
        tile = base.split("void run_gemv_tile(")[1].split("void p2b_moe_batched_kernel")[0]
        self.assertIn("mma_ab_h", tile)
        self.assertIn("FragC_h ch[WNT][2]", tile)
        self.assertIn("constexpr int PF = CFG == 0 ? 4 : 2;", base)

    def test_fma_after_cfg1_drops_mma(self) -> None:
        base = self._cfg1()
        out = self.fma.patch_cu(base)
        tile = out.split("void run_gemv_tile(")[1].split("void p2b_moe_batched_kernel")[0]
        self.assertIn("run_gemv_tile<BITS, 1, 1>", out)
        self.assertIn("__launch_bounds__(256, 4)", out)
        self.assertIn("constexpr int PF = CFG == 0 ? 4 : 2;", out)
        self.assertNotIn("constexpr int PF = 4;", out)
        self.assertIn("__ldcs(", tile)
        self.assertNotIn("mma_ab_h", tile)
        self.assertNotIn("FragC_h ch[WNT][2]", tile)
        self.assertIn("dq8_regs_2bits", tile)
        self.assertEqual(out.count("grid.sync()"), base.count("grid.sync()"))
        self.assertIn("m * experts * warps_per_exp", out)
        self.assertIn("const int a_src = lane & 3;", tile)
        self.assertNotIn("a_src = lane >> 3", tile)
        self.assertIn("__shfl_xor_sync(0xffffffffu, p0, 1)", tile)
        self.assertIn("__shfl_xor_sync(0xffffffffu, p0, 2)", tile)
        self.assertNotIn("mask = 4", tile)
        self.assertIn("(t & 1) << 4", tile)
        self.assertNotIn("(t & 1) << 16", tile)

    def test_refuses_stock_pin_without_cfg1(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            self.fma.patch_cu(self.pin)
        self.assertIn("cfg1", str(ctx.exception))

    def test_second_apply_is_noop(self) -> None:
        patched = self.fma.patch_cu(self._cfg1())
        self.assertEqual(self.fma.patch_cu(patched), patched)

    def test_apply_writes_p2b_moe_cu(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cu = root / "csrc" / "p2b_moe.cu"
            cu.parent.mkdir()
            cu.write_text(self._cfg1())
            self.fma.apply(root)
            out = cu.read_text()
            self.assertNotIn("mma_ab_h", out)
            self.assertIn("dq8_regs_2bits", out)
            self.fma.apply(root)
            self.assertEqual(cu.read_text(), out)
