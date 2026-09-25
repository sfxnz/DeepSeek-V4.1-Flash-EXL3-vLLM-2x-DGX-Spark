"""engram_wkv_tp (DSV41_ENGRAM_WKV_TP): when wkv shards, the wiring, the rank-symmetric self-check.

Host only: stand-in vLLM classes and a numpy-backed stand-in for the few torch
calls the check makes. The GPU proof with the image's own classes and the pack's
real weights (TP=2 emulated in one process: sharded == stock bit for bit at M
1..2048 on layers 1 and 14, the processed shards == the stock processed weight and
swizzled scale byte for byte, the self-check engages, sabotage disarms and the
fallback equals stock, graph replays) is kernel_study/fusion_host/
engram_wkv_tp_check.py; engram_wkv_shard_check.py is the kernel-level probe.
"""

from __future__ import annotations

import ast
import sys
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "docker" / "patch"
sys.path.insert(0, str(PATCH))
import decode_levers as dl  # noqa: E402
import engram_wkv_tp as wtp  # noqa: E402


class FT:
    """numpy-backed stand-in for the tensor calls the check makes."""

    def __init__(self, a, dtype="bf16", device="cuda:0"):
        self.a = np.asarray(a)
        self.dtype = dtype
        self.device = device

    @property
    def shape(self):
        return self.a.shape

    @property
    def data(self):
        return self

    def dim(self):
        return self.a.ndim

    def contiguous(self):
        return self

    def view(self, what):
        return FT(self.a.reshape(-1), self.dtype, self.device) if what == -1 else self

    def to(self, _):
        return self

    def __mul__(self, k):
        return FT(self.a * k, self.dtype, self.device)

    def item(self):
        return self.a.reshape(-1)[0].item()


def fake_torch():
    t = types.SimpleNamespace(float32="f32", bfloat16="bf16", uint8="u8", int16="i16", int32="i32")

    class Gen:
        def __init__(self, device="cpu"):
            self.rng = None

        def manual_seed(self, seed):
            self.rng = np.random.default_rng(seed)
            return self

    t.Generator = Gen
    # small integers: every product and sum below is exact in float64, so a split GEMM
    # and the full one agree exactly, as the real kernel's columns do
    t.randn = lambda m, k, generator, dtype: FT(generator.rng.integers(-4, 5, (m, k)).astype(np.float64))
    t.zeros = lambda shape, dtype, device: FT(np.zeros(shape), dtype, device)
    t.tensor = lambda vals, dtype, device: FT(np.array(vals), dtype, device)
    t.equal = lambda a, b: bool(np.array_equal(a.a, b.a))
    return t


class Method:
    """Stand-in quant method: y = x @ w.T (w the layer's rows)."""

    def __init__(self, fail=False, flip=False):
        self.fail, self.flip = fail, flip

    def apply(self, layer, x, bias=None):
        if self.fail:
            raise RuntimeError("kernel launch failed")
        y = x.a @ layer.w.T
        if self.flip:
            y = y.copy()
            y.reshape(-1)[0] += 1.0
        return FT(y)


def sharded_layer(rank, full_w, method=None, tp=2):
    n = full_w.shape[0] // tp
    lyr = types.SimpleNamespace(
        input_size=full_w.shape[1], output_size=full_w.shape[0], output_size_per_partition=n,
        prefix="model.layers.1.engram.wkv", quant_method=method or Method(),
    )
    lyr.w = full_w[rank * n:(rank + 1) * n]
    lyr.weight = FT(lyr.w, "f8")
    lyr.weight_scale = FT(np.arange(rank * 10, rank * 10 + 10), "u8")
    return lyr


class FullLayer:
    """What make_full_layer_class builds, over the gathered weight."""

    def __init__(self, sharded, weight, weight_scale, offset=0.0):
        self.w = weight.a
        self.offset = offset
        self.weight_scale = weight_scale

    def forward(self, x):
        return FT(x.a @ self.w.T + self.offset)


class Collectives:
    """Rank 0's view of a 2-rank group; the peer's gather inputs are given."""

    def __init__(self, peer, peer_bad=0):
        self.peer = list(peer)
        self.peer_bad = peer_bad
        self.log = []

    def all_gather(self, t, dim):
        self.log.append(("all_gather", dim))
        other = self.peer[len([c for c in self.log if c[0] == "all_gather"]) - 1]
        axis = dim if dim >= 0 else t.a.ndim + dim
        return FT(np.concatenate([t.a, other.a], axis=axis), t.dtype, t.device)

    def all_reduce_sum(self, t):
        self.log.append(("all_reduce", None))
        return FT(t.a + self.peer_bad, t.dtype, t.device)


class RefuseTests(unittest.TestCase):
    def test_rules(self):
        self.assertIsNone(wtp.refuse_reason(2, False, 1, 25600))
        self.assertIn("TP=1", wtp.refuse_reason(1, False, 1, 25600))
        self.assertIn("sequence parallelism", wtp.refuse_reason(2, True, 1, 25600))
        self.assertIn("Engram DP=2", wtp.refuse_reason(2, False, 2, 25600))
        self.assertIn("128-row tiles", wtp.refuse_reason(2, False, 1, 25600 + 64))
        self.assertIsNone(wtp.refuse_reason(4, False, 1, 25600 * 2))

    def test_env(self):
        self.assertFalse(wtp.enabled({}))
        self.assertFalse(wtp.enabled({"DSV41_ENGRAM_WKV_TP": "0"}))
        self.assertTrue(wtp.enabled({"DSV41_ENGRAM_WKV_TP": "1"}))


class SelfCheckTests(unittest.TestCase):
    def setUp(self):
        self.torch = fake_torch()
        rng = np.random.default_rng(0)
        self.full_w = rng.integers(-3, 4, (512, 64)).astype(np.float64)
        wtp._STATE.update(sharded=0, engaged=0, disarmed=0)

    def run_check(self, r0_method=None, peer_bad=0, full_offset=0.0):
        r0 = sharded_layer(0, self.full_w, r0_method)
        r1 = sharded_layer(1, self.full_w)
        xs = wtp.probe_inputs(self.torch, 64, "cuda:0")
        peer, _ = wtp.contributions(self.torch, r1, xs)
        coll = Collectives(peer, peer_bad)
        engram = types.SimpleNamespace(wkv=r0)
        with redirect_stdout(StringIO()) as out:
            ok = wtp.self_check(
                self.torch, engram, r0, coll.all_gather, coll.all_reduce_sum,
                lambda s, w, sc: FullLayer(s, w, sc, full_offset),
            )
        return ok, engram, r0, coll, out.getvalue()

    def assert_symmetric(self, coll):
        n = len(wtp.PROBE_M)
        self.assertEqual(coll.log, [("all_gather", 0)] * 2 + [("all_gather", -1)] * n + [("all_reduce", None)])

    def test_equal_engages_and_keeps_the_sharded_layer(self):
        ok, engram, r0, coll, log = self.run_check()
        self.assertTrue(ok)
        self.assertIs(engram.wkv, r0)
        self.assertIn(wtp.LOG_ENGAGED, log)
        self.assertNotIn(wtp.LOG_DISARMED, log)
        self.assert_symmetric(coll)
        self.assertEqual(wtp._STATE["engaged"], 1)

    def test_mismatch_disarms_to_the_full_layer(self):
        ok, engram, r0, coll, log = self.run_check(r0_method=Method(flip=True))
        self.assertFalse(ok)
        self.assertIsInstance(engram.wkv, FullLayer)
        self.assertTrue(np.array_equal(engram.wkv.w, self.full_w))  # the gathered weight, rank order
        self.assertIn(wtp.LOG_DISARMED, log)
        self.assertIn("M=1", log)
        self.assert_symmetric(coll)

    def test_reference_mismatch_disarms(self):
        ok, engram, _, _, log = self.run_check(full_offset=1e-3)
        self.assertFalse(ok)
        self.assertIsInstance(engram.wkv, FullLayer)

    def test_local_failure_still_runs_every_collective(self):
        ok, engram, _, coll, log = self.run_check(r0_method=Method(fail=True))
        self.assertFalse(ok)
        self.assert_symmetric(coll)
        self.assertIn("kernel launch failed", log)
        self.assertIsInstance(engram.wkv, FullLayer)

    def test_peer_failure_disarms_here_too(self):
        ok, engram, _, coll, log = self.run_check(peer_bad=1)
        self.assertFalse(ok)
        self.assertIn("1 rank(s) failed; here: ok", log)
        self.assertIsInstance(engram.wkv, FullLayer)

    def test_probe_rows_are_rank_independent(self):
        a = wtp.probe_inputs(self.torch, 64, "cuda:0")
        b = wtp.probe_inputs(self.torch, 64, "cuda:1")
        self.assertEqual([x.shape[0] for x in a], list(wtp.PROBE_M))
        self.assertTrue(all(np.array_equal(x.a, y.a) for x, y in zip(a, b)))


def fake_vllm(tp_size=2):
    """Stand-in modules for install(): distributed, linear, engram."""
    calls = {}

    class QuantMethod:
        def __init__(self):
            self.processed = []

        def process_weights_after_loading(self, layer):
            self.processed.append(layer)

    class ReplicatedLinear:
        def __init__(self, input_size, output_size, bias=True, quant_config=None, return_bias=True, prefix=""):
            self.input_size, self.output_size = input_size, output_size
            self.bias = object() if bias else None
            self.quant_config, self.return_bias, self.prefix = quant_config, return_bias, prefix

    class ColumnParallelLinear:
        def __init__(self, input_size, output_size, bias=True, gather_output=False, quant_config=None,
                     prefix="", *, return_bias=True):
            calls["cpl"] = dict(input_size=input_size, output_size=output_size, bias=bias,
                                gather_output=gather_output, quant_config=quant_config, prefix=prefix,
                                return_bias=return_bias)
            self.input_size, self.output_size, self.prefix = input_size, output_size, prefix
            self.output_size_per_partition = output_size // tp_size
            self.quant_method = QuantMethod()

    class Engram:
        def __init__(self, config, quant_config, layout, layer_hash_index, use_sequence_parallel, prefix):
            self.use_sequence_parallel = use_sequence_parallel
            self.embed_tokens = types.SimpleNamespace(dp_size=layout.get("dp", 1))
            self.wkv = ReplicatedLinear(6144, layout.get("n", 25600), bias=False, quant_config=quant_config,
                                        return_bias=False, prefix=f"{prefix}.wkv")

    dist = types.ModuleType("vllm.distributed")
    dist.get_tensor_model_parallel_world_size = lambda: tp_size
    dist.tensor_model_parallel_all_gather = lambda t, dim=-1: t
    dist.tensor_model_parallel_all_reduce = lambda t: t
    linear = types.ModuleType("vllm.model_executor.layers.linear")
    linear.ReplicatedLinear, linear.ColumnParallelLinear = ReplicatedLinear, ColumnParallelLinear
    eng = types.ModuleType("vllm.models.deepseek_v4_1.common.engram")
    eng.Engram = Engram
    torch = types.ModuleType("torch")
    torch.nn = types.SimpleNamespace(Module=object, Parameter=lambda t, requires_grad=False: t)
    names = ("vllm", "vllm.model_executor", "vllm.model_executor.layers", "vllm.models",
             "vllm.models.deepseek_v4_1", "vllm.models.deepseek_v4_1.common")
    mods = {n: types.ModuleType(n) for n in names}
    mods.update({dist.__name__: dist, linear.__name__: linear, eng.__name__: eng, "torch": torch})
    return mods, eng, calls


class InstallTests(unittest.TestCase):
    def build(self, tp_size=2, sp=False, **layout):
        mods, eng, calls = fake_vllm(tp_size)
        with mock.patch.dict(sys.modules, mods), redirect_stdout(StringIO()) as out:
            msg = wtp.install()
            module = eng.Engram({}, "qc", layout, 0, use_sequence_parallel=sp, prefix="model.layers.1.engram")
            again = wtp.install()
        return module, calls, msg, again, out.getvalue()

    def test_tp2_builds_a_gathered_column_parallel_wkv(self):
        module, calls, msg, again, log = self.build()
        self.assertIn("Engram.__init__ wrapped", msg)
        self.assertEqual(again, "already installed")
        self.assertEqual(calls["cpl"], dict(
            input_size=6144, output_size=25600, bias=False, gather_output=True, quant_config="qc",
            prefix="model.layers.1.engram.wkv", return_bias=False,
        ))
        self.assertEqual(type(module.wkv).__name__, "ColumnParallelLinear")
        self.assertIn("dsv41: engram wkv column-parallel armed (model.layers.1.engram.wkv: 12800 of 25600", log)

    def test_the_check_runs_once_right_after_processing(self):
        module, _, _, _, _ = self.build()
        layer = module.wkv
        with mock.patch.object(wtp, "self_check") as check:
            layer.quant_method.process_weights_after_loading(layer)
            layer.quant_method.process_weights_after_loading(layer)
            layer.quant_method.process_weights_after_loading(object())
        self.assertEqual(check.call_count, 1)
        args = check.call_args[0]
        self.assertIs(args[1], module)
        self.assertIs(args[2], layer)
        self.assertEqual(len(layer.quant_method.processed), 3)  # the stock processing always runs first

    def test_stays_replicated_where_it_cannot_shard(self):
        for kwargs, why in (
            (dict(tp_size=1), "TP=1"),
            (dict(sp=True), "sequence parallelism"),
            (dict(dp=2), "Engram DP=2"),
            (dict(n=25600 + 64), "128-row tiles"),
        ):
            module, calls, _, _, log = self.build(**kwargs)
            self.assertEqual(type(module.wkv).__name__, "ReplicatedLinear", kwargs)
            self.assertNotIn("cpl", calls, kwargs)
            self.assertIn("dsv41: engram wkv stays replicated (model.layers.1.engram.wkv)", log)
            self.assertIn(why, log)

    def test_off_by_default(self):
        mods, eng, calls = fake_vllm()
        stock = eng.Engram.__init__
        with mock.patch.dict(sys.modules, mods), redirect_stdout(StringIO()) as out:
            dl._install_engram_wkv_tp({})
            dl._install_engram_wkv_tp({"DSV41_ENGRAM_WKV_TP": "0"})
        self.assertIs(eng.Engram.__init__, stock)
        self.assertEqual(out.getvalue(), "")

    def test_listed_in_install_run_sh_and_audit(self):
        src = (PATCH / "decode_levers.py").read_text()
        self.assertIn('("engram-wkv-tp", _install_engram_wkv_tp)', src)
        self.assertIn("  DSV41_ENGRAM_WKV_TP=0\n", (ROOT / "run.sh").read_text())
        sys.path.insert(0, str(ROOT / "tools"))
        import engagement_audit as ea

        self.assertEqual(ea.PATCHES["engram_wkv_tp.py"], {"DSV41_ENGRAM_WKV_TP": "1"})
        expected, disarm = ea.expectations({"DSV41_ENGRAM_WKV_TP": "1"})
        self.assertIn(("engram_wkv_tp.py", wtp.LOG_ENGAGED), expected)
        self.assertIn(wtp.LOG_DISARMED, disarm)

    def test_top_level_imports_are_stdlib_only(self):
        tree = ast.parse((PATCH / "engram_wkv_tp.py").read_text())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        mods = {a.name for n in top if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module for n in top if isinstance(n, ast.ImportFrom)}
        self.assertEqual(mods, {"__future__", "os"})

    def test_the_fallback_layer_has_no_quant_method_attribute(self):
        # vLLM's post-load pass processes every module with a quant_method; the
        # fallback holds already-processed tensors and must not be processed again.
        src = (PATCH / "engram_wkv_tp.py").read_text()
        cls_src = src[src.index("class ReplicatedWkv"):src.index("return ReplicatedWkv")]
        self.assertNotIn("self.quant_method", cls_src)
        self.assertIn("self._method = sharded.quant_method", cls_src)


if __name__ == "__main__":
    unittest.main()
