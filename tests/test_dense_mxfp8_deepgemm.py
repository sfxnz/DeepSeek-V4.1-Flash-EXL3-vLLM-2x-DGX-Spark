"""dense_mxfp8_deepgemm: env parsing, source rewrite on the pinned image file,
sitecustomize gating and env forwarding. CPU-only; torch paths skip."""

from __future__ import annotations

import importlib.util
import re
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "docker/patch/dense_mxfp8_deepgemm.py"
# Verbatim vllm/model_executor/kernels/linear/mxfp8/flashinfer.py from
# dsv41-flash-exl3-sm121:canonical-e12 (backend already "auto" in the image).
PIN = ROOT / "tests/fixtures/mxfp8_flashinfer_e12.pin.py"


def _load():
    spec = importlib.util.spec_from_file_location("dense_mxfp8_deepgemm", PATCHER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class EnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load()

    def test_default_off(self) -> None:
        self.assertFalse(self.mod.enabled_from_env({}))
        self.assertFalse(self.mod.enabled_from_env({"DSV41_DENSE_DG_SMALLM": "0"}))
        self.assertTrue(self.mod.enabled_from_env({"DSV41_DENSE_DG_SMALLM": "1"}))

    def test_default_shapes_are_the_six_decode_shapes(self) -> None:
        self.assertEqual(
            self.mod.shapes_from_env({}),
            {
                (5120, 1792),  # qkv_a
                (1280, 16384),  # wq_b
                (4096, 5120),  # wo_b
                (5120, 2304),  # shared gate_up
                (1152, 5120),  # shared down
                (15360, 5120),  # draft main_proj
            },
        )

    def test_shape_list_parses_k_by_n(self) -> None:
        got = self.mod.shapes_from_env({"DSV41_DENSE_DG_SHAPES": " 5120x1792, 4096X5120 ,"})
        self.assertEqual(got, {(5120, 1792), (4096, 5120)})

    def test_bad_shape_entry_raises(self) -> None:
        for bad in ("5120", "5120x", "axb", "5120x1792x3"):
            with self.assertRaises(ValueError, msg=bad):
                self.mod.shapes_from_env({"DSV41_DENSE_DG_SHAPES": bad})

    def test_maybe_apply_is_inert_without_prepared_scales(self) -> None:
        class Layer:
            pass

        self.assertIsNone(self.mod.maybe_apply(Layer(), object(), None))
        layer = Layer()
        layer._dsv41_dg_sf = None
        self.assertIsNone(self.mod.maybe_apply(layer, object(), None))

    def test_prepare_is_inert_when_off(self) -> None:
        class Layer:
            pass

        layer = Layer()
        import os

        old = os.environ.pop("DSV41_DENSE_DG_SMALLM", None)
        try:
            self.mod.prepare(None, layer, None)
        finally:
            if old is not None:
                os.environ["DSV41_DENSE_DG_SMALLM"] = old
        self.assertIsNone(layer._dsv41_dg_sf)


class PatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load()
        cls.pin = PIN.read_text()

    def test_pin_is_stock(self) -> None:
        self.assertNotIn("_dsv41_dg_", self.pin)
        self.assertIn("class FlashInferCutlassMxfp8LinearKernel", self.pin)
        for anchor in (self.mod.IMPORT_OLD, self.mod.PWAL_OLD, self.mod.APPLY_OLD):
            self.assertEqual(self.pin.count(anchor), 1)

    def test_hooks_land_in_the_cutlass_kernel_only(self) -> None:
        out = self.mod.patch_py(self.pin)
        compile(out, "flashinfer.py", "exec")
        cls_src = out.split("class FlashInferCutlassMxfp8LinearKernel", 1)[1]
        cutlass, rest = cls_src.split("\nclass FlashInferCutedslMxfp8LinearKernel", 1)
        self.assertIn("_dsv41_dg_prepare(self, layer, weight_scale_2d)", cutlass)
        self.assertIn("_dsv41_dg_out = _dsv41_dg_apply(layer, x, bias)", cutlass)
        self.assertNotIn("_dsv41_dg_prepare(", rest)
        self.assertNotIn("_dsv41_dg_apply(", rest)
        # prepare runs after the stock params are set; apply runs before quant.
        pwal = cutlass.split("def process_weights_after_loading", 1)[1].split("def apply_weights")[0]
        self.assertLess(pwal.index("weight_scale_swizzled.contiguous()"), pwal.index("_dsv41_dg_prepare"))
        aw = cutlass.split("def apply_weights", 1)[1]
        self.assertLess(aw.index("_dsv41_dg_apply"), aw.index("as_quantized_activation"))
        self.assertLess(aw.index("_dsv41_dg_apply"), aw.index("vllm_flashinfer.mm_mxfp8"))

    def test_import_fallback_is_a_noop(self) -> None:
        out = self.mod.patch_py(self.pin)
        self.assertIn("except ImportError:", out)
        self.assertIn("_dsv41_dg_apply = _dsv41_dg_prepare = lambda *a, **k: None", out)

    def test_patch_idempotent(self) -> None:
        once = self.mod.patch_py(self.pin)
        self.assertEqual(self.mod.patch_py(once), once)

    def test_works_before_or_after_prefer_b12x(self) -> None:
        cutlass = self.pin.replace('backend="auto",\n        )\n\n        if bias', 'backend="cutlass",\n        )\n\n        if bias', 1)
        self.assertIn('backend="cutlass"', cutlass)
        out = self.mod.patch_py(cutlass)
        self.assertIn("_dsv41_dg_apply(layer, x, bias)", out)

    def test_anchor_drift_refuses(self) -> None:
        drifted = self.pin.replace("        N, K = weight.shape\n", "        n_, k_ = weight.shape\n")
        with self.assertRaises(SystemExit):
            self.mod.patch_py(drifted)

    def test_apply_rewrites_tree(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "model_executor/kernels/linear/mxfp8/flashinfer.py"
            dest.parent.mkdir(parents=True)
            dest.write_text(self.pin)
            self.assertTrue(self.mod.apply(Path(td)))
            self.assertIn("_dsv41_dg_prepare", dest.read_text())
            self.assertFalse(self.mod.apply(Path(td)))


class WiringTests(unittest.TestCase):
    def test_sitecustomize_applies_only_when_enabled(self) -> None:
        site = (ROOT / "docker/patch/sitecustomize.py").read_text()
        block = site.split("from dense_mxfp8_deepgemm import apply", 1)[1][:400]
        self.assertIn("if _dense_dg_enabled():", block)
        self.assertIn("_apply_dense_dg(", block)
        # After prefer_b12x so either backend string is accepted.
        self.assertLess(site.index("prefer_b12x_mxfp8 import"), site.index("dense_mxfp8_deepgemm import"))

    def test_both_envs_forwarded_default_off(self) -> None:
        run = (ROOT / "run.sh").read_text()
        fwd = run.split("FORWARD_ENVS=(", 1)[1].split(")", 1)[0]
        items = re.findall(r"^\s+(\S+)$", fwd, re.M)
        self.assertIn("DSV41_DENSE_DG_SMALLM=0", items)
        self.assertIn("DSV41_DENSE_DG_SHAPES=", items)


class TorchRuntimeTests(unittest.TestCase):
    """CPU parts of the runtime hook (inside the image: torch present)."""

    def setUp(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest(f"torch missing: {exc}")
        self.mod = _load()

    def test_maybe_apply_falls_through(self) -> None:
        import torch

        class Layer:
            pass

        layer = Layer()
        layer._dsv41_dg_sf = torch.zeros(1)
        # more than 8 rows -> stock path (prefill)
        self.assertIsNone(self.mod.maybe_apply(layer, torch.zeros(9, 128, dtype=torch.bfloat16), None))
        # 0 rows -> stock path
        self.assertIsNone(self.mod.maybe_apply(layer, torch.zeros(0, 128, dtype=torch.bfloat16), None))
        # non-bf16 -> stock path
        self.assertIsNone(self.mod.maybe_apply(layer, torch.zeros(4, 128), None))
        # pre-quantized activation (not a tensor) -> stock path
        self.assertIsNone(self.mod.maybe_apply(layer, object(), None))


if __name__ == "__main__":
    unittest.main()
