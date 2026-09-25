"""mhc_det (DSV41_MHC_DET_SPLITS): env parsing, install wiring, anchors in pinned sources.

The kernels themselves are checked on the GPU (kernel_study/mhc_det: bitwise vs stock on real
weights, determinism, graph replay) and again at load by prepare(). Here: host logic only.

Fixtures (read-only copies, vllm/ paths):
  dsv41_model_e13.pin.py  models/deepseek_v4_1/nvidia/model.py from image
                          dsv41-flash-exl3-sm121:canonical-e13 (sha256:c81762335a12...);
                          file sha256 530ed24c8fd2e9daeb5c3d340ef52246618786f19eace8ec217c8d3bdf271110
  mhc_tilelang.pin.py     model_executor/kernels/mhc/tilelang.py (canonical-e12; identical in e13)
  dsv41_dspark.pin.py     models/deepseek_v4_1/nvidia/dspark.py (canonical-e12; identical in e13)
"""

import ast
import re
import sys
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures"
PATCH = ROOT / "docker" / "patch"
sys.path.insert(0, str(PATCH))

import decode_levers as dl  # noqa: E402
import mhc_det  # noqa: E402


def _func(path: Path, name: str, cls: str | None = None) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    scope = tree.body
    if cls is not None:
        scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    return next(n for n in scope if isinstance(n, ast.FunctionDef) and n.name == name)


def _calls(fn: ast.FunctionDef, callee: str) -> list[ast.Call]:
    return [n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == callee]


class EnvTests(unittest.TestCase):
    def test_off_values(self) -> None:
        for env in ({}, {"DSV41_MHC_DET_SPLITS": ""}, {"DSV41_MHC_DET_SPLITS": "0"}):
            self.assertIsNone(mhc_det.splits_from_env(env), env)

    def test_on_only_at_the_stock_split_count(self) -> None:
        self.assertEqual(mhc_det.splits_from_env({"DSV41_MHC_DET_SPLITS": "16"}), 16)
        for bad in ("40", "1", "32", "many"):
            with self.assertRaises(ValueError, msg=bad):
                mhc_det.splits_from_env({"DSV41_MHC_DET_SPLITS": bad})

    def test_conflicting_levers_refuse(self) -> None:
        with self.assertRaises(ValueError):
            mhc_det.splits_from_env({"DSV41_MHC_DET_SPLITS": "16", "DSV41_MHC_DECODE_SPLITS": "40"})
        with self.assertRaises(ValueError):
            mhc_det.splits_from_env({"DSV41_MHC_DET_SPLITS": "16", "DSV41_MHC_NO_DEEPGEMM": "1"})
        self.assertEqual(
            mhc_det.splits_from_env({"DSV41_MHC_DET_SPLITS": "16", "DSV41_MHC_DECODE_SPLITS": "0",
                                     "DSV41_MHC_NO_DEEPGEMM": "0"}), 16)


def _fake_vllm(drift: bool = False):
    stock = types.ModuleType("vllm.model_executor.kernels.mhc.tilelang")
    stock.mhc_post_tilelang = lambda *a, **k: "stock-post"
    stock.mhc_pre_delayed_tilelang = lambda *a, **k: "stock-pre"
    model = types.ModuleType("vllm.models.deepseek_v4_1.nvidia.model")
    model.mhc_post_tilelang = stock.mhc_post_tilelang
    model.mhc_pre_delayed_tilelang = (lambda *a, **k: "other") if drift else stock.mhc_pre_delayed_tilelang
    model.DeepseekV4Model = type("DeepseekV4Model", (), {"finalize_mhc_broadcast_weights": lambda self: "fin"})
    dspark = types.ModuleType("vllm.models.deepseek_v4_1.nvidia.dspark")
    dspark.mhc_post_tilelang = stock.mhc_post_tilelang
    dspark.DSparkDeepseekV4ForCausalLM = type("DSparkDeepseekV4ForCausalLM", (), {"load_weights": lambda self, w: {"x"}})
    return {mhc_det.STOCK_MOD: stock, mhc_det.MODEL_MOD: model, mhc_det.DSPARK_MOD: dspark}


class InstallTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = (mhc_det._S.on, mhc_det._S.failed, mhc_det._S.stock_post, mhc_det._S.stock_pre)
        mhc_det._S.on = mhc_det._S.failed = False

    def tearDown(self) -> None:
        mhc_det._S.on, mhc_det._S.failed, mhc_det._S.stock_post, mhc_det._S.stock_pre = self._saved

    def test_off_imports_nothing(self) -> None:
        with mock.patch.dict(sys.modules, {}), redirect_stdout(StringIO()) as out:
            mhc_det.install({})
        self.assertEqual(out.getvalue(), "")

    def test_on_patches_call_sites_and_load_hooks(self) -> None:
        mods = _fake_vllm()
        with mock.patch.dict(sys.modules, mods), redirect_stdout(StringIO()) as out, \
                mock.patch.object(mhc_det, "prepare") as prep:
            mhc_det.install({"DSV41_MHC_DET_SPLITS": "16"})
            m, d = mods[mhc_det.MODEL_MOD], mods[mhc_det.DSPARK_MOD]
            self.assertIs(m.mhc_post_tilelang, mhc_det._post)
            self.assertIs(m.mhc_pre_delayed_tilelang, mhc_det._pre)
            self.assertIs(d.mhc_post_tilelang, mhc_det._post)
            self.assertEqual(m.DeepseekV4Model().finalize_mhc_broadcast_weights(), "fin")
            self.assertEqual(d.DSparkDeepseekV4ForCausalLM().load_weights([]), {"x"})
            self.assertEqual([c.args[1] for c in prep.call_args_list], ["target", "draft"])
        self.assertIn("dsv41: mhc det armed", out.getvalue())
        self.assertIs(mhc_det._S.stock_pre, mods[mhc_det.STOCK_MOD].mhc_pre_delayed_tilelang)

    def test_drift_disarms_without_patching(self) -> None:
        mods = _fake_vllm(drift=True)
        before = mods[mhc_det.MODEL_MOD].mhc_post_tilelang
        with mock.patch.dict(sys.modules, mods), redirect_stdout(StringIO()) as out:
            mhc_det.install({"DSV41_MHC_DET_SPLITS": "16"})
        self.assertIn(mhc_det.LOG_DISARMED, out.getvalue())
        self.assertIs(mods[mhc_det.MODEL_MOD].mhc_post_tilelang, before)
        self.assertTrue(mhc_det._S.failed)

    def test_bad_value_disarms_loudly(self) -> None:
        with redirect_stdout(StringIO()) as out:
            mhc_det.install({"DSV41_MHC_DET_SPLITS": "40"})
        self.assertIn(mhc_det.LOG_DISARMED, out.getvalue())
        self.assertIn("only 16", out.getvalue())

    def test_decode_levers_runs_the_install_step(self) -> None:
        with mock.patch.object(mhc_det, "install") as inst, redirect_stdout(StringIO()):
            dl.install({"DSV41_MHC_DET_SPLITS": "16"})
        inst.assert_called_once()

    def test_prepare_disarms_on_error_and_stays_off(self) -> None:
        class Broken:
            def modules(self):
                raise RuntimeError("boom")

        with redirect_stdout(StringIO()) as out:
            mhc_det.prepare(Broken(), "target")
            mhc_det.prepare(Broken(), "draft")  # failed: no second attempt
        self.assertEqual(out.getvalue().count(mhc_det.LOG_DISARMED), 1)
        self.assertFalse(mhc_det._S.on)


class PinnedAnchorTests(unittest.TestCase):
    """The det path mirrors the pinned stock functions; flag upstream drift here."""

    def test_model_calls_the_patched_names(self) -> None:
        src = (FIX / "dsv41_model_e13.pin.py").read_text()
        self.assertRegex(src, r"from vllm\.model_executor\.kernels\.mhc\.tilelang import \(\s*"
                              r"mhc_post_tilelang,\s*mhc_pre_delayed_tilelang,")
        fwd = _func(FIX / "dsv41_model_e13.pin.py", "forward", "DeepseekV4DecoderLayer")
        self.assertEqual(len(_calls(fwd, "mhc_pre_delayed_tilelang")), 4)
        self.assertEqual(len(_calls(fwd, "mhc_post_tilelang")), 2)
        fin = _func(FIX / "dsv41_model_e13.pin.py", "finalize_mhc_broadcast_weights", "DeepseekV4Model")
        self.assertIn("hc_attn_fn_broadcast", ast.unparse(fin))
        pwal = _func(FIX / "dsv41_model_e13.pin.py", "process_weights_after_loading", "DeepseekV41LLMForCausalLM")
        self.assertIn("finalize_mhc_broadcast_weights", ast.unparse(pwal))

    def test_draft_imports_the_post_by_name(self) -> None:
        src = (FIX / "dsv41_dspark.pin.py").read_text()
        self.assertRegex(src, r"from vllm\.model_executor\.kernels\.mhc\.tilelang import \(\s*mhc_post_tilelang,")
        self.assertIn("def load_weights(self, weights", src)

    def test_det_pre_mirrors_stock_pre(self) -> None:
        stock = _func(FIX / "mhc_tilelang.pin.py", "mhc_pre_delayed_tilelang")
        det = _func(PATCH / "mhc_det.py", "det_pre_delayed")
        s_call = _calls(stock, "MHC_PRE_NORM_KERNEL")
        d_call = _calls(det, "MHC_PRE_NORM_KERNEL")
        self.assertEqual((len(s_call), len(d_call)), (1, 1))
        self.assertEqual([ast.unparse(a) for a in s_call[0].args], [ast.unparse(a) for a in d_call[0].args])
        self.assertEqual({k.arg: ast.unparse(k.value) for k in s_call[0].keywords},
                         {k.arg: ast.unparse(k.value) for k in d_call[0].keywords})
        gemm = _calls(stock, "tf32_hc_prenorm_gemm")
        self.assertEqual([ast.unparse(a) for a in gemm[0].args], ["x", "fn", "mixes", "sqrsum", "n_splits"])
        # the stock signature is what _pre forwards
        self.assertEqual([a.arg for a in stock.args.args] + [a.arg for a in stock.args.kwonlyargs],
                         [a.arg for a in _func(PATCH / "mhc_det.py", "_pre").args.args])

    def test_post_signature(self) -> None:
        stock = _func(FIX / "mhc_tilelang.pin.py", "mhc_post_tilelang")
        self.assertEqual([a.arg for a in stock.args.args], ["x", "residual", "post_layer_mix", "comb_res_mix"])
        self.assertEqual([a.arg for a in _func(PATCH / "mhc_det.py", "_post").args.args],
                         ["x", "residual", "post_layer_mix", "comb_res_mix"])


class KernelSourceTests(unittest.TestCase):
    def test_kernels_the_host_launches_exist(self) -> None:
        cu = (PATCH / "mhc_det.cu").read_text()
        names = set(re.findall(r'GEMM_KERNEL\((\w+),', cu)) | set(re.findall(r"^\s+(mhc_det_\w+)\(", cu, re.M))
        for name in ("mhc_det_gemm_pk_t8", "mhc_det_gemm_pk_t16", "mhc_det_post"):
            self.assertIn(name, names)

    def test_no_atomics(self) -> None:
        code = re.sub(r"//[^\n]*", "", (PATCH / "mhc_det.cu").read_text())  # drop comments
        self.assertNotRegex(code, r"\batom\.|\bred\.|\batomic[A-Z]\w*\s*\(")
        self.assertIn("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32", code)

    def test_smem_fits_and_split_math(self) -> None:
        self.assertEqual(mhc_det.kb_per_split(20480), 20)
        self.assertEqual(mhc_det.kb_per_split(5120), 5)
        with self.assertRaises(ValueError):
            mhc_det.kb_per_split(4096 + 64)
        for k in (5120, 20480):
            for t in range(1, mhc_det.MAX_T + 1):
                self.assertLessEqual(mhc_det.smem_bytes(t, k, packed=True), 101376, (t, k))

    def test_audit_knows_the_lever(self) -> None:
        sys.path.insert(0, str(ROOT / "tools"))
        import engagement_audit as ea

        self.assertEqual(ea.PATCHES["mhc_det.py"], {"DSV41_MHC_DET_SPLITS": "16"})
        exp, _ = ea.expectations({"DSV41_MHC_DET_SPLITS": "16"})
        self.assertIn(("mhc_det.py", mhc_det.LOG_ENGAGED), exp)
        exp, _ = ea.expectations({"DSV41_MHC_DET_SPLITS": "0"})
        self.assertNotIn(("mhc_det.py", mhc_det.LOG_ENGAGED), exp)


if __name__ == "__main__":
    unittest.main()
