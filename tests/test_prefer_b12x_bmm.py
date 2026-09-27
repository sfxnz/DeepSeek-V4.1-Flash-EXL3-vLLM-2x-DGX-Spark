"""Drive prefer_b12x_bmm on the MXFP8 BMM dispatch snippet."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "docker/patch/attic/prefer_b12x_bmm.py"
PIN = ROOT / "tests/fixtures/mxfp8_bmm_init.pin.py"


def _load():
    spec = importlib.util.spec_from_file_location("prefer_b12x_bmm", PATCHER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestPreferB12xBmm(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load()
        cls.pin = PIN.read_text()

    def test_pin_is_deepgemm_then_emulation(self) -> None:
        self.assertIn(
            "[DeepGemmMxfp8BmmLinearKernel, EmulationMxfp8LinearKernel]",
            self.pin,
        )
        self.assertNotIn("B12xMxfp8LinearKernel", self.pin)

    def test_patch_inserts_b12x_before_emulation(self) -> None:
        out = self.mod.patch_py(self.pin)
        self.assertNotIn("B12xMxfp8LinearKernel", out)
        self.assertIn("FlashInferCutlassMxfp8LinearKernel", out)
        self.assertIn("EmulationMxfp8LinearKernel", out)
        fi = out.index("FlashInferCutlassMxfp8LinearKernel")
        emu = out.index("EmulationMxfp8LinearKernel")
        self.assertLess(fi, emu)
        self.assertNotIn(
            "[DeepGemmMxfp8BmmLinearKernel, EmulationMxfp8LinearKernel]",
            out,
        )

    def test_patch_idempotent(self) -> None:
        once = self.mod.patch_py(self.pin)
        self.assertEqual(self.mod.patch_py(once), once)

    def test_apply_rewrites_tree(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "model_executor/kernels/linear/__init__.py"
            dest.parent.mkdir(parents=True)
            dest.write_text(self.pin)
            self.assertTrue(self.mod.apply(Path(td)))
            self.assertIn("FlashInferCutlassMxfp8LinearKernel", dest.read_text())
            self.assertFalse(self.mod.apply(Path(td)))


if __name__ == "__main__":
    unittest.main()
