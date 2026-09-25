"""Drive widen_mla_kv_buf on the live FlashInfer DSV4 decode header."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "docker/patch/attic/widen_mla_kv_buf.py"
PIN = ROOT / "tests/fixtures/decode_dsv4_kernel.pin.cuh"


def _load():
    spec = importlib.util.spec_from_file_location("widen_mla_kv_buf", PATCHER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestWidenMlaKvBuf(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load()
        cls.pin = PIN.read_text()

    def test_pin_is_double_buffered(self) -> None:
        self.assertIn("constexpr int DSV4_KV_BUF_COUNT = 2;", self.pin)
        self.assertIn("sparse_mla_decode_dsv4_kernel", self.pin)
        self.assertGreaterEqual(self.pin.count("(chunk_idx - chunk_lo) & 1"), 2)

    def test_patch_cuh_drops_to_one_buffer(self) -> None:
        out = self.mod.patch_cuh(self.pin)
        self.assertIn("constexpr int DSV4_KV_BUF_COUNT = 1;", out)
        self.assertNotIn("constexpr int DSV4_KV_BUF_COUNT = 2;", out)
        self.assertNotIn("(chunk_idx - chunk_lo) & 1", out)
        self.assertGreaterEqual(out.count("% DSV4_KV_BUF_COUNT"), 2)

    def test_patch_cuh_idempotent(self) -> None:
        once = self.mod.patch_cuh(self.pin)
        self.assertEqual(self.mod.patch_cuh(once), once)

    def test_apply_rewrites_tree(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dest = (
                Path(td)
                / "include/flashinfer/attention/sparse_mla_sm120/decode_dsv4_kernel.cuh"
            )
            dest.parent.mkdir(parents=True)
            dest.write_text(self.pin)
            self.assertTrue(self.mod.apply(Path(td)))
            patched = dest.read_text()
            self.assertIn("constexpr int DSV4_KV_BUF_COUNT = 1;", patched)
            self.assertNotIn("(chunk_idx - chunk_lo) & 1", patched)
            self.assertFalse(self.mod.apply(Path(td)))


if __name__ == "__main__":
    unittest.main()
