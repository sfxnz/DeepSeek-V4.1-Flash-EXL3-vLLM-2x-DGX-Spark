"""Drive prefer_b12x_mxfp8 on the Cutlass MXFP8 apply_weights snippet."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "docker/patch/prefer_b12x_mxfp8.py"
PIN = ROOT / "tests/fixtures/mxfp8_flashinfer.pin.py"


def _load():
    spec = importlib.util.spec_from_file_location("prefer_b12x_mxfp8", PATCHER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestPreferB12xMxfp8(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load()
        cls.pin = PIN.read_text()

    def test_pin_hardcodes_cutlass(self) -> None:
        self.assertIn('backend="cutlass"', self.pin)
        self.assertNotIn('backend="auto"', self.pin)

    def test_patch_switches_cutlass_to_auto(self) -> None:
        out = self.mod.patch_py(self.pin)
        self.assertIn('backend="auto"', out)
        self.assertNotIn('backend="cutlass"', out)
        self.assertIn("weight.t()", out)

    def test_patch_idempotent(self) -> None:
        once = self.mod.patch_py(self.pin)
        self.assertEqual(self.mod.patch_py(once), once)

    def test_apply_rewrites_tree(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dest = (
                Path(td)
                / "model_executor/kernels/linear/mxfp8/flashinfer.py"
            )
            dest.parent.mkdir(parents=True)
            dest.write_text(self.pin)
            self.assertTrue(self.mod.apply(Path(td)))
            self.assertIn('backend="auto"', dest.read_text())
            self.assertFalse(self.mod.apply(Path(td)))


if __name__ == "__main__":
    unittest.main()
