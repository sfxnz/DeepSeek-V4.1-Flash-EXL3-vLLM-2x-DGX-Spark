"""mhc_det (DSV41_MHC_DET_SPLITS): env parsing, install wiring, anchors in pinned sources.

The kernels themselves are checked on the GPU (kernel_study/mhc_det: bitwise vs stock on real
weights, determinism, graph replay) and again at load by prepare(). Here: host logic only.

Fixtures (read-only copies, vllm/ paths):
  dsv41_model_e13.pin.py  models/deepseek_v4_1/nvidia/model.py from image
                          dsv41-flash-exl3-sm121:canonical-e13 (sha256:c81762335a12...);
                          file sha256 530ed24c8fd2e9daeb5c3d340ef52246618786f19eace8ec217c8d3bdf271110
  mhc_tilelang.pin.py     model_executor/kernels/mhc/tilelang.py (canonical-e12; identical in e13)
  dsv41_dspark.pin.py     models/deepseek_v4_1/nvidia/dspark.py (canonical-e12; identical in e13)
  dsv41_ar_sites_e13.pin.txt  the TP all-reduce path before every mHC post (canonical-e13 excerpts)
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
import mhc_det_overlap as ovl  # noqa: E402


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
        names |= set(re.findall(r"__launch_bounds__\([\d, ]+\)\s+(mhc_det_\w+)\(", cu))
        for name in ("mhc_det_gemm_t8", "mhc_det_gemm_t16", "mhc_det_post", "mhc_det_norm", "mhc_det_norm_li",
                     "mhc_det_norm_coef"):
            self.assertIn(name, names)

    def test_split_norm_waits_before_reading_the_post_output(self) -> None:
        cu = re.sub(r"//[^\n]*", "", (PATCH / "mhc_det.cu").read_text())
        li = cu[cu.index("DEV void norm_li("):cu.index("#define NORM_ARGS")]
        wait = li.index("if (wait_first) pdl_wait();")
        self.assertLess(wait, li.index("pre_mix_in + tok * 4"))
        self.assertLess(wait, li.index("ldcg_u2(rb + hc * NORM_H + p)"))
        self.assertLess(li.index("ldcg_u2(norm_w"), wait)  # the weight is fetched before the wait
        entry = cu[cu.index("mhc_det_norm_li(NORM_ARGS)"):cu.index("mhc_det_norm_coef(NORM_ARGS)")]
        self.assertIn("threadIdx.x,\n          true);", entry)  # the standalone entry waits
        fused = cu[cu.index("mhc_det_norm(NORM_ARGS)"):cu.index("mhc_det_norm_li(NORM_ARGS)")]
        self.assertIn("tid - 32, false);", fused)  # the fused one reads during the GEMM (its primary)
        coef = cu[cu.index("DEV void norm_coef("):cu.index("DEV void norm_li(")]
        self.assertLess(coef.index("pdl_wait();"), coef.index("sqr_p[s * T + tok]"))

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
                self.assertLessEqual(mhc_det.smem_bytes(t, k), 101376, (t, k))

    def test_audit_knows_the_lever(self) -> None:
        sys.path.insert(0, str(ROOT / "tools"))
        import engagement_audit as ea

        self.assertEqual(ea.PATCHES["mhc_det.py"], {"DSV41_MHC_DET_SPLITS": "16"})
        exp, _ = ea.expectations({"DSV41_MHC_DET_SPLITS": "16"})
        self.assertIn(("mhc_det.py", mhc_det.LOG_ENGAGED), exp)
        exp, _ = ea.expectations({"DSV41_MHC_DET_SPLITS": "0"})
        self.assertNotIn(("mhc_det.py", mhc_det.LOG_ENGAGED), exp)
        self.assertEqual(ea.PATCHES["mhc_det_overlap.py"],
                         {"DSV41_MHC_DET_SPLITS": "16", "DSV41_MHC_DET_OVERLAP": "1"})
        exp, disarm = ea.expectations({"DSV41_MHC_DET_SPLITS": "16", "DSV41_MHC_DET_OVERLAP": "1"})
        self.assertIn(("mhc_det_overlap.py", ovl.LOG_ENGAGED), exp)
        self.assertIn(ovl.LOG_DISARMED, disarm)
        exp, _ = ea.expectations({"DSV41_MHC_DET_SPLITS": "16", "DSV41_MHC_DET_OVERLAP": "0"})
        self.assertNotIn(("mhc_det_overlap.py", ovl.LOG_ENGAGED), exp)


# ---------------------------------------------------------------------------------------------
# DSV41_MHC_DET_OVERLAP (mhc_det_overlap.py)
# ---------------------------------------------------------------------------------------------
class _FakeStream:
    def __init__(self, name: str, log: list) -> None:
        self.name, self.log, self.device = name, log, "cuda:0"

    def wait_stream(self, other) -> None:
        self.log.append(("wait", self.name, other.name))

    def __eq__(self, other) -> bool:
        return isinstance(other, _FakeStream) and other.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)


class _FakeCuda:
    def __init__(self, log: list) -> None:
        self.log = log
        self.cur = _FakeStream("main", log)
        self.capturing = False
        self.made = 0

    def current_stream(self):
        return self.cur

    def is_current_stream_capturing(self) -> bool:
        return self.capturing

    def Stream(self, device=None):  # noqa: N802 - torch.cuda.Stream
        self.made += 1
        return _FakeStream("side", self.log)

    def stream(self, s):
        cuda = self

        class _Ctx:
            def __enter__(self):
                self.prev, cuda.cur = cuda.cur, s

            def __exit__(self, *a):
                cuda.cur = self.prev

        return _Ctx()


class _Cuda:
    is_cuda = True


def _fake_torch(log: list):
    t = types.ModuleType("torch")
    t.cuda = _FakeCuda(log)
    return t


class OverlapStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = dict(vars(ovl._S))
        self.log: list = []
        ovl._S.__init__()
        ovl._S.torch = _fake_torch(self.log)
        ovl._S.armed = ovl._S.on = True

    def tearDown(self) -> None:
        vars(ovl._S).clear()
        vars(ovl._S).update(self._saved)

    def _launch(self, tag: str):
        return lambda: self.log.append(("launch", tag, ovl._S.torch.cuda.current_stream().name))

    def test_env(self) -> None:
        self.assertFalse(ovl.on_from_env({}))
        self.assertFalse(ovl.on_from_env({"DSV41_MHC_DET_OVERLAP": "0"}))
        self.assertTrue(ovl.on_from_env({"DSV41_MHC_DET_OVERLAP": "1"}))
        for bad in ("2", "yes", "on"):
            with self.assertRaises(ValueError, msg=bad):
                ovl.on_from_env({"DSV41_MHC_DET_OVERLAP": bad})

    def test_fork_at_the_all_reduce_then_join_at_settle(self) -> None:
        ovl.defer(self._launch("a"), ("keep",))
        self.assertEqual(self.log, [])  # nothing launched until the all-reduce
        ovl.fork_at_all_reduce(_Cuda())
        self.assertEqual(self.log, [("wait", "side", "main"), ("launch", "a", "side")])
        ovl.fork_at_all_reduce(_Cuda())  # a second all-reduce: already forked
        ovl.settle()
        self.assertEqual(self.log[-1], ("wait", "main", "side"))
        self.assertIsNone(ovl._S.pending)
        self.assertEqual((ovl._S.forks, ovl._S.in_place), (1, 0))
        ovl.settle()  # nothing pending: no-op
        self.assertEqual(len(self.log), 3)

    def test_no_all_reduce_launches_in_place(self) -> None:
        ovl.defer(self._launch("a"), ())
        ovl.settle()
        self.assertEqual(self.log, [("launch", "a", "main")])
        self.assertEqual((ovl._S.forks, ovl._S.in_place), (0, 1))

    def test_no_fork_when_not_engaged_off_stream_or_host_tensor(self) -> None:
        cuda = ovl._S.torch.cuda
        ovl.defer(self._launch("a"), ())
        ovl.fork_at_all_reduce(object())  # not a CUDA tensor
        ovl._S.on = False
        ovl.fork_at_all_reduce(_Cuda())
        ovl._S.on = True
        other = _FakeStream("aux", self.log)
        cuda.cur = other
        ovl.fork_at_all_reduce(_Cuda())  # all-reduce on another stream than the pre's
        self.assertEqual(self.log, [])
        cuda.cur = _FakeStream("main", self.log)
        ovl.settle()
        self.assertEqual(self.log, [("launch", "a", "main")])

    def test_a_new_defer_settles_the_previous_one_first(self) -> None:
        ovl.defer(self._launch("a"), ())
        ovl.defer(self._launch("b"), ())
        self.assertEqual(self.log, [("launch", "a", "main")])
        ovl.fork_at_all_reduce(_Cuda())
        self.assertEqual(self.log[-1], ("launch", "b", "side"))

    def test_capture_boundary_raises_loudly(self) -> None:
        ovl.defer(self._launch("a"), ())
        ovl._S.torch.cuda.capturing = True
        with redirect_stdout(StringIO()) as out, self.assertRaises(RuntimeError):
            ovl.settle()
        self.assertIn(ovl.LOG_DISARMED, out.getvalue())
        self.assertEqual(self.log, [])  # neither launched nor joined
        self.assertFalse(ovl.active())

    def test_fork_skipped_when_capture_state_changed(self) -> None:
        ovl.defer(self._launch("a"), ())
        ovl._S.torch.cuda.capturing = True
        ovl.fork_at_all_reduce(_Cuda())
        self.assertEqual(self.log, [])

    def test_side_stream_is_made_once(self) -> None:
        for tag in "ab":
            ovl.defer(self._launch(tag), ())
            ovl.fork_at_all_reduce(_Cuda())
            ovl.settle()
        self.assertEqual(ovl._S.torch.cuda.made, 1)

    def test_wrapped_all_reduce_forks_then_reduces(self) -> None:
        calls = []

        class GC:
            def all_reduce(self, input_):
                calls.append(("ar", len(self_log)))
                return input_

        self_log = self.log
        ovl.wrap_all_reduce(GC)
        self.assertTrue(GC.all_reduce._dsv41_mhc_ovl)
        x = _Cuda()
        self.assertIs(GC().all_reduce(x), x)  # nothing pending: plain all-reduce
        ovl.defer(self._launch("a"), ())
        self.assertIs(GC().all_reduce(x), x)
        self.assertEqual(calls, [("ar", 0), ("ar", 2)])  # forked (wait + launch) before reducing


def _co_tensor_model_parallel_all_reduce(input_):
    return get_tp_group().all_reduce(input_)  # noqa: F821 - source text for the anchor check


class _RowParallelLinear:
    def forward(self, input_):
        output_parallel = input_
        output = tensor_model_parallel_all_reduce(output_parallel)  # noqa: F821
        return output


class _RowParallelLinearDrift:
    def forward(self, input_):
        return fused_all_reduce_norm(input_)  # noqa: F821


class _MoERunner:
    def _maybe_reduce_final_output(self, states, trunc_size):
        states = tensor_model_parallel_all_reduce(states)  # noqa: F821
        return states


def _fake_ar_mods(drift: bool = False):
    gc = type("GroupCoordinator", (), {"all_reduce": lambda self, input_: input_})
    ps = types.ModuleType(ovl.GC_MOD)
    ps.GroupCoordinator = gc
    co = types.ModuleType(ovl.CO_MOD)
    co.tensor_model_parallel_all_reduce = _co_tensor_model_parallel_all_reduce
    lin = types.ModuleType(ovl.LINEAR_MOD)
    lin.RowParallelLinear = _RowParallelLinearDrift if drift else _RowParallelLinear
    run = types.ModuleType(ovl.RUNNER_MOD)
    run.MoERunner = _MoERunner
    return {ovl.GC_MOD: ps, ovl.CO_MOD: co, ovl.LINEAR_MOD: lin, ovl.RUNNER_MOD: run, "torch": _fake_torch([])}


class OverlapInstallTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = dict(vars(ovl._S))
        self._det = (mhc_det._S.on, mhc_det._S.failed, mhc_det._S.stock_post, mhc_det._S.stock_pre, mhc_det._S.ovl)
        ovl._S.__init__()

    def tearDown(self) -> None:
        vars(ovl._S).clear()
        vars(ovl._S).update(self._saved)
        (mhc_det._S.on, mhc_det._S.failed, mhc_det._S.stock_post, mhc_det._S.stock_pre, mhc_det._S.ovl) = self._det

    def test_off_imports_nothing(self) -> None:
        with mock.patch.dict(sys.modules, {}), redirect_stdout(StringIO()) as out:
            self.assertEqual(ovl.install({}), "off")
        self.assertEqual(out.getvalue(), "")

    def test_arms_and_wraps_group_coordinator_once(self) -> None:
        mods = _fake_ar_mods()
        with mock.patch.dict(sys.modules, mods), redirect_stdout(StringIO()) as out:
            self.assertEqual(ovl.install({"DSV41_MHC_DET_OVERLAP": "1"}), "armed")
            gc = mods[ovl.GC_MOD].GroupCoordinator
            first = gc.all_reduce
            self.assertTrue(first._dsv41_mhc_ovl)
            self.assertEqual(ovl.install({"DSV41_MHC_DET_OVERLAP": "1"}), "armed")
            self.assertIs(gc.all_reduce, first)  # not wrapped twice
        self.assertIn("dsv41: mhc det overlap armed", out.getvalue())
        self.assertFalse(ovl.active())  # engages only after the load-time self-test

    def test_anchor_drift_disarms_without_wrapping(self) -> None:
        mods = _fake_ar_mods(drift=True)
        before = mods[ovl.GC_MOD].GroupCoordinator.all_reduce
        with mock.patch.dict(sys.modules, mods), redirect_stdout(StringIO()) as out:
            self.assertEqual(ovl.install({"DSV41_MHC_DET_OVERLAP": "1"}), "disarmed")
        self.assertIn(ovl.LOG_DISARMED, out.getvalue())
        self.assertIs(mods[ovl.GC_MOD].GroupCoordinator.all_reduce, before)

    def test_bad_value_disarms(self) -> None:
        with redirect_stdout(StringIO()) as out:
            self.assertEqual(ovl.install({"DSV41_MHC_DET_OVERLAP": "yes"}), "disarmed")
        self.assertIn(ovl.LOG_DISARMED, out.getvalue())

    def test_overlap_without_the_det_lever_is_loud(self) -> None:
        mhc_det._S.failed = False
        with redirect_stdout(StringIO()) as out:
            mhc_det.install({"DSV41_MHC_DET_OVERLAP": "1"})
        self.assertIn(ovl.LOG_DISARMED, out.getvalue())
        self.assertIn("needs DSV41_MHC_DET_SPLITS=16", out.getvalue())

    def test_det_install_arms_the_overlap(self) -> None:
        mhc_det._S.failed = False
        mods = {**_fake_vllm(), **_fake_ar_mods()}
        with mock.patch.dict(sys.modules, mods), redirect_stdout(StringIO()), \
                mock.patch.object(mhc_det, "prepare"):
            mhc_det.install({"DSV41_MHC_DET_SPLITS": "16", "DSV41_MHC_DET_OVERLAP": "1"})
        self.assertIs(mhc_det._S.ovl, ovl)
        mhc_det._S.ovl = None
        with mock.patch.dict(sys.modules, _fake_vllm()), redirect_stdout(StringIO()), \
                mock.patch.object(mhc_det, "prepare"):
            mhc_det.install({"DSV41_MHC_DET_SPLITS": "16"})
        self.assertIsNone(mhc_det._S.ovl)


class OverlapDispatchTests(unittest.TestCase):
    """mhc_det._pre / _post with the overlap module attached (no GPU: kernels mocked)."""

    def setUp(self) -> None:
        self._det = (mhc_det._S.on, mhc_det._S.stock_post, mhc_det._S.stock_pre, mhc_det._S.ovl,
                     mhc_det._S.captured, dict(mhc_det._S.packed))
        self.calls: list = []
        fake = types.SimpleNamespace(settle=lambda: self.calls.append("settle"), defer="DEFER",
                                     active=lambda: self.active)
        self.active = True
        mhc_det._S.ovl = fake
        mhc_det._S.stock_pre = lambda *a, **k: self.calls.append("stock_pre") or "stock"
        mhc_det._S.stock_post = lambda *a, **k: self.calls.append("stock_post") or "stock"

    def tearDown(self) -> None:
        (mhc_det._S.on, mhc_det._S.stock_post, mhc_det._S.stock_pre, mhc_det._S.ovl, mhc_det._S.captured,
         packed) = self._det
        mhc_det._S.packed.clear()
        mhc_det._S.packed.update(packed)

    def _pre(self):
        args = ("res", "fn", "scale", "base", 1e-6, 1e-6, 1e-6, 2.0, 20)
        with mock.patch.object(mhc_det, "_pre_eligible", return_value=True), \
                mock.patch.object(mhc_det, "_lookup", return_value="packed"), \
                mock.patch.object(mhc_det, "det_pre_delayed",
                                  side_effect=lambda *a, **k: self.calls.append(("det", k["defer"])) or "det"):
            return mhc_det._pre(*args, pre_mix="pm", norm_weight="w", norm_eps=1e-6)

    def test_pre_settles_then_defers_when_engaged(self) -> None:
        self.assertEqual(self._pre(), "det")
        self.assertEqual(self.calls, ["settle", ("det", "DEFER")])

    def test_pre_keeps_the_fused_path_when_not_engaged(self) -> None:
        self.active = False
        self._pre()
        self.assertEqual(self.calls, ["settle", ("det", None)])

    def test_stock_pre_still_settles_first(self) -> None:
        with mock.patch.object(mhc_det, "_pre_eligible", return_value=False):
            mhc_det._pre("res", "fn", "scale", "base", 1e-6, 1e-6, 1e-6, 2.0, 20, pre_mix="pm")
        self.assertEqual(self.calls, ["settle", "stock_pre"])

    def test_post_settles_before_anything(self) -> None:
        mhc_det._S.on = False
        with mock.patch.dict(sys.modules, {"torch": _fake_torch([])}):
            mhc_det._post("x", "res", "post", "comb")
        self.assertEqual(self.calls, ["settle", "stock_post"])

    def test_no_repack_after_capture(self) -> None:
        fn = types.SimpleNamespace(shape=(24, 20480), _version=3, data_ptr=lambda: 1234)
        mhc_det._S.packed[1234] = ((24, 20480), 2, "old-packed")
        mhc_det._S.captured, mhc_det._S.failed = True, False
        with mock.patch.dict(sys.modules, {"torch": _fake_torch([])}), redirect_stdout(StringIO()) as out, \
                mock.patch.object(mhc_det, "_pack_one") as pack:
            self.assertIsNone(mhc_det._lookup(fn))
        pack.assert_not_called()
        self.assertIn(mhc_det.LOG_DISARMED, out.getvalue())
        self.assertEqual(mhc_det._S.packed[1234][2], "old-packed")  # the captured copy stays alive
        mhc_det._S.failed = False


class OverlapPinnedTests(unittest.TestCase):
    """Where the fork and the join land in the pinned model: every pre is followed by the
    sublayer call that ends in a TP all-reduce, and every forward ends with a post."""

    def test_all_reduce_path_matches_the_anchors(self) -> None:
        pin = (FIX / "dsv41_ar_sites_e13.pin.txt").read_text()
        sections = dict(re.findall(r"^# ---- (\S+)\n(.*?)(?=^# ---- |\Z)", pin, re.S | re.M))
        for mod, cls, fn, needle in ovl.ANCHORS:
            key = f"{mod}:{cls}.{fn}" if cls else f"{mod}.{fn}"
            self.assertIn(needle, sections[key], key)
        gc = sections[f"{ovl.GC_MOD}:GroupCoordinator.all_reduce"]
        self.assertIn("def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:", gc)

    def test_each_pre_is_followed_by_an_all_reduce_site_before_the_next_post(self) -> None:
        fwd = ast.unparse(_func(FIX / "dsv41_model_e13.pin.py", "forward", "DeepseekV4DecoderLayer"))
        seq = [m.group(0) for m in re.finditer(
            r"mhc_pre_delayed_tilelang\(|mhc_post_tilelang\(|self\.attn\(|self\.ffn\(", fwd)]
        tail = seq[seq.index("self.attn("):]  # the three first-layer / carried pre branches come first
        self.assertEqual(tail, ["self.attn(", "mhc_post_tilelang(", "mhc_pre_delayed_tilelang(", "self.ffn("])

    def test_every_forward_ends_with_a_post(self) -> None:
        tgt = ast.unparse(_func(FIX / "dsv41_model_e13.pin.py", "forward", "DeepseekV4Model"))
        loop = tgt.index("for idx, layer in enumerate(")
        self.assertIn("hidden_states = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)", tgt[loop:])
        self.assertIn("aux_recon = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)", tgt[loop:])
        self.assertLess(tgt.index("hidden_states = final_aux_recon"), tgt.index("hc_collapse_triton("))
        drf = ast.unparse(_func(FIX / "dsv41_dspark.pin.py", "forward", "DSparkDeepseekV4Model"))
        self.assertLess(drf.index("for layer in self.layers:"),
                        drf.index("hidden_states = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)"))
        self.assertLess(drf.index("mhc_post_tilelang(hidden_states"), drf.index("hc_collapse_triton("))


if __name__ == "__main__":
    unittest.main()
