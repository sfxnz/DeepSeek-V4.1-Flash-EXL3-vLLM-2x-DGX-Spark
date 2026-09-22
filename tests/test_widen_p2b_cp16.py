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


class WidenP2bCp16Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.shapes = _load("docker/patch/widen_p2b_shapes.py")
        cls.mrow = _load("docker/patch/widen_p2b_mrow.py")
        cls.cfg1 = _load("docker/patch/widen_p2b_cfg1.py")
        cls.cp16 = _load("docker/patch/widen_p2b_cp16.py")
        cls.pin = PIN_CU.read_text()

    def _cfg1(self) -> str:
        return self.cfg1.patch_cu(self.mrow.patch_cu(self.shapes.patch_cu(self.pin)))

    def _tile(self, src: str) -> str:
        return src.split("void run_gemv_tile(")[1].split("void p2b_moe_batched_kernel")[0]

    def test_cfg1_fixture_still_uses_ldcs(self) -> None:
        base = self._cfg1()
        self.assertIn("__ldcs(", base)
        self.assertNotIn("cp.async.cg.shared.global.L2::128B", base)

    def test_cp16_after_cfg1_uses_16b_cg_not_4b_ca(self) -> None:
        base = self._cfg1()
        patched = self.cp16.patch_cu(base)
        self.assertNotIn("__ldcs(", patched)
        self.assertNotIn("uint32_t pf[PF][LOADS]", patched)
        self.assertIn("cp.async.cg.shared.global.L2::128B", patched)
        self.assertNotIn("cp.async.ca.shared.global", patched)
        self.assertIn("(lane & 3) != 0", patched)
        self.assertIn("__cvta_generic_to_shared", patched)
        self.assertNotIn("cvta.to.shared.u32", patched)
        self.assertIn("__align__(16)", patched)
        self.assertIn("__syncwarp()", patched)
        self.assertIn("run_gemv_tile<BITS, 1, 1>", patched)
        self.assertIn("__launch_bounds__(256, 4)", patched)
        self.assertIn("m * experts * warps_per_exp", patched)
        self.assertEqual(patched.count("grid.sync()"), base.count("grid.sync()"))

    def test_issue_before_mma_wait_after(self) -> None:
        patched = self.cp16.patch_cu(self._cfg1())
        loop = self._tile(patched).split("for (int ib = 0; ib < myn; ib += PF)")[1]
        commit_at = loop.find("cp.async.commit_group")
        mma_at = loop.find("mma_ab_h")
        wait_at = loop.rfind("cp.async.wait_group")
        sync_at = loop.rfind("__syncwarp()")
        self.assertGreater(commit_at, 0)
        self.assertGreater(mma_at, commit_at)
        self.assertGreater(wait_at, mma_at)
        self.assertGreater(sync_at, wait_at)

    def test_refuses_stock_pin_without_cfg1(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            self.cp16.patch_cu(self.pin)
        self.assertIn("cfg1", str(ctx.exception))

    def test_second_apply_is_noop(self) -> None:
        patched = self.cp16.patch_cu(self._cfg1())
        self.assertEqual(self.cp16.patch_cu(patched), patched)

    def test_apply_writes_p2b_moe_cu(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cu = root / "csrc" / "p2b_moe.cu"
            cu.parent.mkdir()
            cu.write_text(self._cfg1())
            self.cp16.apply(root)
            out = cu.read_text()
            self.assertIn("cp.async.cg.shared.global.L2::128B", out)
            self.assertNotIn("__ldcs(", out)
            self.cp16.apply(root)
            self.assertEqual(cu.read_text(), out)
