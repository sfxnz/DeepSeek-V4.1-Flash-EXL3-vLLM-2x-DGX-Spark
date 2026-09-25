"""dense_gemv: env parsing, config table vs the compiled kernel table, source
rewrite on the pinned image file, lm_head/sitecustomize wiring, audit markers.
CPU-only; torch paths skip on the host (the GPU checks live in
kernel_study/dense_gemv and the load-time self-test)."""

from __future__ import annotations

import importlib.util
import re
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "docker/patch/dense_gemv.py"
KERNEL = ROOT / "docker/patch/dense_gemv_kernel.cu"
DG_PATCHER = ROOT / "docker/patch/dense_mxfp8_deepgemm.py"
# Verbatim vllm/model_executor/kernels/linear/mxfp8/flashinfer.py (e12 == e13).
PIN = ROOT / "tests/fixtures/mxfp8_flashinfer_e12.pin.py"


def _load(path=PATCHER, name="dense_gemv"):
    spec = importlib.util.spec_from_file_location(name, path)
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
        self.assertFalse(self.mod.enabled_from_env({"DSV41_DENSE_GEMV": "0"}))
        self.assertTrue(self.mod.enabled_from_env({"DSV41_DENSE_GEMV": "1"}))

    def test_default_shapes_are_every_tuned_shape(self) -> None:
        self.assertEqual(self.mod.shapes_from_env({}), frozenset(self.mod.CONFIGS))

    def test_indexer_wq_b_is_not_tuned(self) -> None:
        # measured no gain on the 1280x4096 pre-quantized shape: it stays on b12x
        self.assertNotIn((1280, 4096), self.mod.CONFIGS)
        with self.assertRaises(ValueError):
            self.mod.shapes_from_env({"DSV41_DENSE_GEMV_SHAPES": "1280x4096"})

    def test_shape_list_parses_k_by_n(self) -> None:
        got = self.mod.shapes_from_env({"DSV41_DENSE_GEMV_SHAPES": " 5120x1792, 4096X5120 ,"})
        self.assertEqual(got, {(5120, 1792), (4096, 5120)})

    def test_bad_shape_entry_raises(self) -> None:
        for bad in ("5120", "5120x", "axb", "5120x1792x3", "5120*1792"):
            with self.assertRaises(ValueError, msg=bad):
                self.mod.shapes_from_env({"DSV41_DENSE_GEMV_SHAPES": bad})

    def test_pick_buckets(self) -> None:
        buckets = self.mod.CONFIGS[(5120, 1792)][3]
        self.assertEqual(self.mod.pick(buckets, 1), (4, 2, 4))
        self.assertEqual(self.mod.pick(buckets, 4), (4, 2, 4))
        self.assertEqual(self.mod.pick(buckets, 5), (3, 2, 8))
        self.assertEqual(self.mod.pick(buckets, 8), (3, 2, 8))
        self.assertIsNone(self.mod.pick(buckets, 9))
        self.assertIsNone(self.mod.pick(self.mod.CONFIGS[(15360, 5120)][3], 6))

    def test_maybe_apply_is_inert_without_armed_state(self) -> None:
        class Layer:
            pass

        self.assertIsNone(self.mod.maybe_apply(Layer(), object(), None))
        layer = Layer()
        layer._dsv41_gemv = None
        self.assertIsNone(self.mod.maybe_apply(layer, object(), None))

    def test_prepare_is_inert_when_off(self) -> None:
        import os

        class Layer:
            pass

        layer = Layer()
        old = os.environ.pop("DSV41_DENSE_GEMV", None)
        try:
            self.mod.prepare(None, layer, None)
            self.mod.prepare_lmhead(None, layer, None)
        finally:
            if old is not None:
                os.environ["DSV41_DENSE_GEMV"] = old
        self.assertIsNone(layer._dsv41_gemv)


class ConfigTableTests(unittest.TestCase):
    """Every bucket the Python table can pick is compiled into kTable."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load()
        src = KERNEL.read_text()
        table = src.split("static const Entry kTable[] = {", 1)[1].split("};", 1)[0]
        cls.compiled = {
            (int(w), int(s), int(kc), 0 if sm == "COMPACT32" else 1, int(mr))
            for w, s, kc, sm, mr in re.findall(
                r"DSV41_GEMV_BOTH_INPUTS\((\d+), (\d+), (\d+), (COMPACT32|TILE), (\d+)\)", table)
        }

    def test_table_parsed(self) -> None:
        self.assertGreaterEqual(len(self.compiled), 5)

    def test_every_config_is_compiled(self) -> None:
        for (k, n), (name, kc, smode, buckets) in self.mod.CONFIGS.items():
            self.assertEqual(k % kc, 0, name)
            self.assertEqual(kc % 128, 0, name)
            self.assertEqual(n % 32, 0, name)
            for max_m, w, s, mr in buckets:
                self.assertLessEqual(max_m, mr, name)
                self.assertIn((w, s, kc, smode, mr), self.compiled, f"{name} M<={max_m}")

    def test_buckets_cover_m_1_to_max(self) -> None:
        for name, kc, smode, buckets in self.mod.CONFIGS.values():
            maxes = [b[0] for b in buckets]
            self.assertEqual(maxes, sorted(maxes), name)
            self.assertLessEqual(maxes[-1], self.mod.MAX_M, name)

    def test_lm_head_uses_per_row_scales(self) -> None:
        self.assertEqual(self.mod.CONFIGS[(5120, 64640)][2], 1)
        others = [v[2] for key, v in self.mod.CONFIGS.items() if key != (5120, 64640)]
        self.assertEqual(set(others), {0})


class PatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load()
        cls.pin = PIN.read_text()

    def test_pin_is_stock(self) -> None:
        self.assertNotIn("_dsv41_gemv_", self.pin)
        for anchor in (self.mod.IMPORT_OLD, self.mod.PWAL_OLD, self.mod.APPLY_OLD):
            self.assertEqual(self.pin.count(anchor), 1)

    def test_hooks_land_in_the_cutlass_kernel_only(self) -> None:
        out = self.mod.patch_py(self.pin)
        compile(out, "flashinfer.py", "exec")
        cls_src = out.split("class FlashInferCutlassMxfp8LinearKernel", 1)[1]
        cutlass, rest = cls_src.split("\nclass FlashInferCutedslMxfp8LinearKernel", 1)
        self.assertIn("_dsv41_gemv_prepare(self, layer, weight_scale_2d)", cutlass)
        self.assertIn("_dsv41_gemv_out = _dsv41_gemv_apply(layer, x, bias)", cutlass)
        self.assertNotIn("_dsv41_gemv_prepare(", rest)
        self.assertNotIn("_dsv41_gemv_apply(", rest)
        pwal = cutlass.split("def process_weights_after_loading", 1)[1].split("def apply_weights")[0]
        self.assertLess(pwal.index("weight_scale_swizzled.contiguous()"), pwal.index("_dsv41_gemv_prepare"))
        aw = cutlass.split("def apply_weights", 1)[1]
        # before the stock quant and before b12x, so a QuantizedActivation reaches the hook too
        self.assertLess(aw.index("_dsv41_gemv_apply"), aw.index("as_quantized_activation"))
        self.assertLess(aw.index("_dsv41_gemv_apply"), aw.index("vllm_flashinfer.mm_mxfp8"))

    def test_import_fallback_is_a_noop(self) -> None:
        out = self.mod.patch_py(self.pin)
        self.assertIn("_dsv41_gemv_apply = _dsv41_gemv_prepare = lambda *a, **k: None", out)

    def test_patch_idempotent(self) -> None:
        once = self.mod.patch_py(self.pin)
        self.assertEqual(self.mod.patch_py(once), once)

    def test_anchor_drift_refuses(self) -> None:
        drifted = self.pin.replace("        N, K = weight.shape\n", "        n_, k_ = weight.shape\n")
        with self.assertRaises(SystemExit):
            self.mod.patch_py(drifted)

    def test_coexists_with_the_deep_gemm_patch_text(self) -> None:
        dg = _load(DG_PATCHER, "dense_mxfp8_deepgemm")
        both = self.mod.patch_py(dg.patch_py(self.pin))
        compile(both, "flashinfer.py", "exec")
        self.assertIn("_dsv41_dg_apply(", both)
        self.assertIn("_dsv41_gemv_apply(", both)

    def test_apply_rewrites_tree(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "model_executor/kernels/linear/mxfp8/flashinfer.py"
            dest.parent.mkdir(parents=True)
            dest.write_text(self.pin)
            self.assertTrue(self.mod.apply(Path(td)))
            self.assertIn("_dsv41_gemv_prepare", dest.read_text())
            self.assertFalse(self.mod.apply(Path(td)))


class WiringTests(unittest.TestCase):
    def test_sitecustomize_applies_only_when_enabled(self) -> None:
        site = (ROOT / "docker/patch/sitecustomize.py").read_text()
        block = site.split("from dense_gemv import apply", 1)[1][:400]
        self.assertIn("if _dense_gemv_enabled():", block)
        self.assertIn("_apply_dense_gemv(", block)
        self.assertLess(site.index("prefer_b12x_mxfp8 import"), site.index("dense_gemv import"))

    def test_lmhead_hooks(self) -> None:
        src = (ROOT / "docker/patch/lmhead_mxfp8.py").read_text()
        self.assertIn("from dense_gemv import maybe_apply as _gemv_apply", src)
        pwal = src.split("def process_weights_after_loading(self, layer)", 1)[1].split("def apply(", 1)[0]
        self.assertIn("_gemv_prepare(self, layer, scale_2d)", pwal)
        body = src.split("def apply(self, layer, x, bias=None):", 1)[1]
        self.assertLess(body.index("_gemv_apply(layer, x, bias)"), body.index("mxfp8_e4m3_quantize("))

    def test_envs_forwarded_default_off(self) -> None:
        run = (ROOT / "run.sh").read_text()
        fwd = run.split("FORWARD_ENVS=(", 1)[1].split(")", 1)[0]
        items = re.findall(r"^\s+(\S+)$", fwd, re.M)
        self.assertIn("DSV41_DENSE_GEMV=0", items)
        self.assertIn("DSV41_DENSE_GEMV_SHAPES=", items)

    def test_audit_expects_the_armed_line_only_when_on(self) -> None:
        audit = _load(ROOT / "tools/engagement_audit.py", "engagement_audit")
        mod = _load()
        on, disarm = audit.expectations({"DSV41_DENSE_GEMV": "1"})
        self.assertIn(("dense_gemv.py", mod.LOG_ENGAGED), on)
        off, _ = audit.expectations({"DSV41_DENSE_GEMV": "0"})
        self.assertNotIn(("dense_gemv.py", mod.LOG_ENGAGED), off)
        self.assertIn("; b12x stays", disarm)

    def test_kernel_source_sits_next_to_the_module(self) -> None:
        mod = _load()
        self.assertEqual(mod._KERNEL_SRC, KERNEL.resolve())
        self.assertTrue(KERNEL.is_file())


class TorchRuntimeTests(unittest.TestCase):
    """CPU parts of the runtime (inside the image: torch present)."""

    def setUp(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest(f"torch missing: {exc}")
        self.mod = _load()

    def test_build_scales_compact(self) -> None:
        import torch

        n, k, kc = 64, 1024, 512
        g = torch.randint(100, 140, (n // 32, k // 32), dtype=torch.uint8)
        s2d = g.repeat_interleave(32, dim=0)
        out = self.mod.build_scales(s2d, n, k, kc, 0)
        self.assertEqual(tuple(out.shape), (2, 2, 16))
        self.assertTrue(torch.equal(out[:, :, :16].reshape(2, 32), g))

    def test_build_scales_compact_refuses_per_row(self) -> None:
        import torch

        s2d = torch.randint(100, 140, (64, 32), dtype=torch.uint8)
        with self.assertRaises(RuntimeError):
            self.mod.build_scales(s2d, 64, 1024, 512, 0)

    def test_build_scales_tile(self) -> None:
        import torch

        n, k, kc = 32, 1024, 512
        s2d = torch.arange(n * (k // 32), dtype=torch.int32).remainder(251).to(torch.uint8).view(n, k // 32)
        out = self.mod.build_scales(s2d, n, k, kc, 1)
        self.assertEqual(tuple(out.shape), (2, 2, 16, 16))
        self.assertEqual(int(out[1, 1, 3, 5]), int(s2d[16 + 3, 16 + 5]))


if __name__ == "__main__":
    unittest.main()
