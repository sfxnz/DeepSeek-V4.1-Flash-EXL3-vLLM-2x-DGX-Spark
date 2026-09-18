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


class WidenP2bCpasyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.shapes = _load("docker/patch/widen_p2b_shapes.py")
        cls.mrow = _load("docker/patch/widen_p2b_mrow.py")
        cls.cfg1 = _load("docker/patch/widen_p2b_cfg1.py")
        cls.cpasync = _load("docker/patch/widen_p2b_cpasync.py")
        cls.pin = PIN_CU.read_text()

    def _cfg1(self) -> str:
        return self.cfg1.patch_cu(self.mrow.patch_cu(self.shapes.patch_cu(self.pin)))

    def _tile(self, src: str) -> str:
        return src.split("void run_gemv_tile(")[1].split("void p2b_moe_batched_kernel")[0]

    def test_pin_uses_streaming_ldcs_for_trellis(self) -> None:
        self.assertIn("__ldcs(", self.pin)
        self.assertGreaterEqual(self.pin.count("__ldcs("), 2)
        self.assertIn("ld_b", self.pin)
        self.assertIn("pf[PF][LOADS]", self.pin)

    def test_cpasync_after_shapes_mrow_cfg1(self) -> None:
        base = self._cfg1()
        self.assertIn("__ldcs(", base)
        self.assertIn("run_gemv_tile<BITS, 1, 1>", base)
        patched = self.cpasync.patch_cu(base)
        self.assertEqual(self.cpasync.patch_cu(patched), patched)
        tile = self._tile(patched)
        ld_b_at = tile.find("ld_b")
        prologue_end = tile.find("FragC_h")
        self.assertGreater(ld_b_at, 0)
        self.assertGreater(prologue_end, ld_b_at)
        prefetch = tile[ld_b_at:prologue_end]
        self.assertIn("cp.async.ca.shared.global", prefetch)
        self.assertNotIn("cp.async.cg.shared.global", prefetch)
        self.assertIn("__cvta_generic_to_shared", prefetch)
        self.assertNotIn("cvta.to.shared.u32", prefetch)
        self.assertIn("__shared__", tile)
        self.assertNotIn("__ldcs(", patched)
        self.assertNotIn("__ldg(", patched)
        self.assertNotIn("uint32_t pf[PF][LOADS]", patched)
        self.assertIn("run_gemv_tile<BITS, 1, 1>", patched)
        self.assertIn("m * experts * warps_per_exp", patched)
        self.assertIn("row * experts + e", patched)
        self.assertIn("(size_t) e * m + row", patched)
        self.assertIn("__launch_bounds__(256, 4)", patched)
        self.assertIn("dim3(256)", patched)
        self.assertIn("kernel, 256, 0", patched)
        self.assertNotIn("for (int row = 0; row < m; ++row)", patched)
        self.assertNotIn("FragB f0s[WNT], f1s[WNT]", patched)
        self.assertIn("const size_t a_row0 = 0", patched)
        self.assertEqual(patched.count("grid.sync()"), base.count("grid.sync()"))

    def test_issue_before_mma_wait_after(self) -> None:
        patched = self.cpasync.patch_cu(self._cfg1())
        loop = self._tile(patched).split("for (int ib = 0; ib < myn; ib += PF)")[1]
        commit_at = loop.find("cp.async.commit_group")
        mma_at = loop.find("mma_ab_h")
        wait_at = loop.rfind("cp.async.wait_group")
        self.assertGreater(commit_at, 0)
        self.assertGreater(mma_at, commit_at)
        self.assertGreater(wait_at, mma_at)
        self.assertIn("bw[", loop)

    def test_second_apply_is_noop(self) -> None:
        patched = self.cpasync.patch_cu(self._cfg1())
        self.assertEqual(self.cpasync.patch_cu(patched), patched)

    def test_identity_would_fail(self) -> None:
        base = self._cfg1()
        patched = self.cpasync.patch_cu(base)
        self.assertNotEqual(patched, base)

    def test_apply_rewrites_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cu = root / "csrc" / "p2b_moe.cu"
            cu.parent.mkdir(parents=True)
            cu.write_text(self._cfg1())
            self.cpasync.apply(root)
            once = cu.read_text()
            self.assertIn("cp.async.ca.shared.global", once)
            self.assertNotIn("__ldcs(", once)
            self.assertIn("run_gemv_tile<BITS, 1, 1>", once)
            self.assertIn("m * experts * warps_per_exp", once)
            self.cpasync.apply(root)
            self.assertEqual(cu.read_text(), once)


if __name__ == "__main__":
    unittest.main()
