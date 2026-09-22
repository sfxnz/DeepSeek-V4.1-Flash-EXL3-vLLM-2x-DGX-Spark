#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN_CU = ROOT / "tests/fixtures/p2b_moe.pin.cu"
PIN_PY = ROOT / "tests/fixtures/exl3_native_moe.pin.py"


def _load(rel: str):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class WidenP2bCodebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.shapes = _load("docker/patch/widen_p2b_shapes.py")
        cls.mrow = _load("docker/patch/widen_p2b_mrow.py")
        cls.cfg1 = _load("docker/patch/widen_p2b_cfg1.py")
        cls.codebook = _load("docker/patch/widen_p2b_codebook.py")
        cls.pin_cu = PIN_CU.read_text()
        cls.pin_py = PIN_PY.read_text()

    def _after_cfg1_cu(self) -> str:
        return self.cfg1.patch_cu(self.mrow.patch_cu(self.shapes.patch_cu(self.pin_cu)))

    def _after_mrow_py(self) -> str:
        return self.mrow.patch_py(self.pin_py)

    def _patched_cu(self) -> str:
        return self.codebook.patch_cu(self._after_cfg1_cu())

    def _patched_py(self) -> str:
        return self.codebook.patch_py(self._after_mrow_py())

    def test_full_chain_templates_cb(self) -> None:
        patched = self._patched_cu()
        self.assertIn("run_gemv_tile<BITS, CB, 1>", patched)
        self.assertNotIn("run_gemv_tile<BITS, 1, 1>", patched)
        self.assertNotIn("run_gemv_tile<BITS, 1, 0>", patched)
        self.assertIn("template <int BITS, int CB>\n__global__", patched)
        self.assertIn("template <int BITS, int CB>\nstatic void launch_moe_batched", patched)
        self.assertIn("(void*) p2b_moe_batched_kernel<BITS, CB>", patched)
        self.assertIn("if (mcg && kg == 2) launch_moe_batched<2, 1>(", patched)
        self.assertIn("else if (!mcg && kg == 2) launch_moe_batched<2, 2>(", patched)
        self.assertIn("launch_moe_batched<3, 2>", patched)
        self.assertIn("launch_moe_batched<4, 2>", patched)
        self.assertNotIn(
            "TORCH_CHECK(mcg && kg == ku && ku == kd && (kg == 2 || kg == 3 || kg == 4)",
            patched,
        )
        self.assertIn(
            "TORCH_CHECK(kg == ku && ku == kd && (kg == 2 || kg == 3 || kg == 4)",
            patched,
        )
        self.assertEqual(self.codebook.patch_cu(patched), patched)

    def test_python_reads_flags_and_passes_mcg(self) -> None:
        patched = self._patched_py()
        self.assertIn('getattr(layer, "_exl3_codebook_flags"', patched)
        self.assertIn("if mcg == mul1:", patched)
        self.assertIn("        mcg,", patched)
        self.assertNotIn("        True,\n        *extra_args,", patched)
        self.assertEqual(self.codebook.patch_py(patched), patched)

    def test_second_apply_is_noop(self) -> None:
        cu = self._patched_cu()
        py = self._patched_py()
        self.assertEqual(self.codebook.patch_cu(cu), cu)
        self.assertEqual(self.codebook.patch_py(py), py)

    def test_apply_rewrites_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cu = root / "csrc" / "p2b_moe.cu"
            py = root / "vllm_exl3" / "exl3.py"
            cu.parent.mkdir(parents=True)
            py.parent.mkdir(parents=True)
            cu.write_text(self._after_cfg1_cu())
            py.write_text(self._after_mrow_py())
            self.codebook.apply(root)
            once_cu = cu.read_text()
            once_py = py.read_text()
            self.assertIn("run_gemv_tile<BITS, CB, 1>", once_cu)
            self.assertIn("launch_moe_batched<2, 2>", once_cu)
            self.assertIn("_exl3_codebook_flags", once_py)
            self.codebook.apply(root)
            self.assertEqual(cu.read_text(), once_cu)
            self.assertEqual(py.read_text(), once_py)


if __name__ == "__main__":
    unittest.main()
