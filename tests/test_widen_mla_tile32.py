"""Drive widen_mla_tile32 on the live FlashInfer DSV4 decode header."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "docker/patch/widen_mla_tile32.py"
PIN_CUH = ROOT / "tests/fixtures/decode_dsv4_kernel.pin.cuh"
PIN_CU = ROOT / "tests/fixtures/sparse_mla_sm120_decode_dsv4.pin.cu"
PIN_PY = ROOT / "tests/fixtures/sparse_mla_sm120.py.pin"
PIN_CORE = ROOT / "tests/fixtures/sparse_mla_core.pin.py"


def _load():
    spec = importlib.util.spec_from_file_location("widen_mla_tile32", PATCHER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestWidenMlaTile32(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load()
        cls.cuh = PIN_CUH.read_text()
        cls.cu = PIN_CU.read_text()
        cls.py = PIN_PY.read_text()
        cls.core = PIN_CORE.read_text()

    def test_pin_is_stock_tile64(self) -> None:
        self.assertIn("constexpr int DSV4_N_WARPS = 8;", self.cuh)
        self.assertIn("constexpr int DSV4_CAND_WINDOW = 64;", self.cuh)
        self.assertIn("constexpr int DSV4_KV_BUF_COUNT = 2;", self.cuh)
        self.assertIn("sm.w_fp8(vc & 1)", self.cuh)
        self.assertIn("2 * HPB * (DSV4_BI + 16)", self.cu)
        self.assertIn("_BI = 64", self.py)
        self.assertIn("split_tile = 64", self.core)

    def test_patch_cuh_tile32_keeps_kv_double_buffer(self) -> None:
        out = self.mod.patch_cuh(self.cuh)
        self.assertIn("constexpr int DSV4_N_WARPS = 4;", out)
        self.assertNotIn("constexpr int DSV4_N_WARPS = 8;", out)
        self.assertIn("constexpr int DSV4_CAND_WINDOW = 32;", out)
        self.assertNotIn("constexpr int DSV4_CAND_WINDOW = 64;", out)
        self.assertIn("constexpr int DSV4_KV_BUF_COUNT = 2;", out)
        self.assertIn("sm.w_fp8(0)", out)
        self.assertNotIn("sm.w_fp8(vc & 1)", out)

    def test_patch_cu_dyn_smem_one_w_fp8(self) -> None:
        out = self.mod.patch_cu(self.cu)
        self.assertIn("1 * HPB * (DSV4_BI + 16)", out)
        self.assertNotIn("2 * HPB * (DSV4_BI + 16)", out)

    def test_patch_py_bi_matches_window(self) -> None:
        out = self.mod.patch_py(self.py)
        self.assertIn("_BI = 32", out)
        self.assertNotIn("_BI = 64", out)

    def test_patch_core_workspace_splits_match_window(self) -> None:
        out = self.mod.patch_core(self.core)
        self.assertIn("split_tile = 32", out)
        self.assertNotIn("split_tile = 64", out)
        self.assertIn("if num_tokens > 64:", out)

    def test_patch_cuh_idempotent(self) -> None:
        once = self.mod.patch_cuh(self.cuh)
        self.assertEqual(self.mod.patch_cuh(once), once)

    def test_apply_rewrites_flashinfer_tree(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cuh = (
                root
                / "include/flashinfer/attention/sparse_mla_sm120/decode_dsv4_kernel.cuh"
            )
            cu = root / "csrc/sparse_mla_sm120_decode_dsv4.cu"
            py = root / "mla/_sparse_mla_sm120.py"
            core = root / "mla/_core.py"
            cuh.parent.mkdir(parents=True)
            cu.parent.mkdir(parents=True)
            py.parent.mkdir(parents=True)
            cuh.write_text(self.cuh)
            cu.write_text(self.cu)
            py.write_text(self.py)
            core.write_text(self.core)
            self.assertTrue(self.mod.apply(root))
            self.assertIn("DSV4_CAND_WINDOW = 32", cuh.read_text())
            self.assertIn("1 * HPB * (DSV4_BI + 16)", cu.read_text())
            self.assertIn("_BI = 32", py.read_text())
            self.assertIn("split_tile = 32", core.read_text())
            self.assertFalse(self.mod.apply(root))


if __name__ == "__main__":
    unittest.main()
