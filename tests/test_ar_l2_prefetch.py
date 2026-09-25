"""ar_l2_prefetch: plans, fork/join order, gating, anchors on pinned e13 sources (CPU only).

Fixtures are read-only copies from image dsv41-flash-exl3-sm121:canonical-e13
(sha256:c81762335a12...):
  dsv41_model_e13.pin.py   vllm/models/deepseek_v4_1/nvidia/model.py
  moe_runner_e13.pin.py    vllm/model_executor/layers/fused_moe/runner/moe_runner.py
"""

from __future__ import annotations

import ast
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(ROOT / "docker" / "patch"))

import ar_l2_prefetch as ap  # noqa: E402
import l2pf_kernel  # noqa: E402


class FakeTensor:
    def __init__(self, nbytes, name="t", rows=4, cuda=True):
        self.nbytes, self.name, self.is_cuda, self.shape = nbytes, name, cuda, (rows, 5120)

    def numel(self):
        return self.nbytes

    def element_size(self):
        return 1

    def dim(self):
        return 2


class FakeStream:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def wait_stream(self, other):
        self.log.append(("wait", self.name, other.name))


class FakeTorch:
    """Just the torch.cuda surface Prefetcher and the wrappers use; records the order."""

    def __init__(self):
        self.log = []
        self.capturing = False
        outer = self
        self._cur = [FakeStream("main", self.log)]

        class Cuda:
            @staticmethod
            def current_stream():
                return outer._cur[-1]

            @staticmethod
            def Stream():
                return FakeStream("side", outer.log)

            @staticmethod
            @contextmanager
            def stream(s):
                outer._cur.append(s)
                try:
                    yield
                finally:
                    outer._cur.pop()

            @staticmethod
            def synchronize():
                outer.log.append(("sync",))

            @staticmethod
            def is_current_stream_capturing():
                return outer.capturing

        self.cuda = Cuda


def make_classes(log):
    class Layer:
        def __init__(self, i, engram=False, nbytes=9_175_040, scale=286_720):
            self.i = i
            self.engram = object() if engram else None
            self.attn = types.SimpleNamespace(fused_wqa_wkv=types.SimpleNamespace(
                weight=FakeTensor(nbytes, f"w{i}"), weight_scale=FakeTensor(scale, f"s{i}")))

        def forward(self, x):
            log.append(("layer", self.i))
            return self.runner._maybe_reduce_final_output(x)

    class Model:
        def __init__(self, layers):
            self.layers, self.start_layer, self.end_layer = layers, 0, len(layers)

        def forward(self, x):
            for layer in self.layers:
                x = layer.forward(x)
            return x

    class Runner:
        def _maybe_reduce_final_output(self, states, *a, **k):
            log.append(("ar",))
            return states

    return Layer, Model, Runner


def build(n=4, engram_at=(), rows=4, capturing_first=False, launch_fails=False):
    torch = FakeTorch()
    Layer, Model, Runner = make_classes(torch.log)

    def launch(t, nbytes):
        if launch_fails:
            raise RuntimeError("no triton")
        torch.log.append(("prefetch", torch.cuda.current_stream().name, t.name, nbytes))

    pf = ap.Prefetcher(launch, torch)
    msgs = []
    ap.wrap(Layer, Model, Runner, pf, 5_767_168, msgs.append)
    runner = Runner()
    layers = [Layer(i, engram=i in engram_at) for i in range(n)]
    for layer in layers:
        layer.runner = runner
    torch.capturing = capturing_first
    return torch, Model(layers), msgs, pf, FakeTensor(0, "x", rows=rows)


class PlanTests(unittest.TestCase):
    def test_split_budget(self):
        self.assertEqual(ap.split_budget([100, 0], 1000), [96, 0])
        self.assertEqual(ap.split_budget([1_000_000, 50_000], 525_000), [500_000, 24_992])  # 16 B aligned
        self.assertEqual(ap.split_budget([], 10), [])
        for n in ap.split_budget([9_175_040, 286_720], 5_767_168):
            self.assertEqual(n % 16, 0)

    def test_budget_env(self):
        self.assertEqual(ap.budget_bytes({}), int(5.5 * 2**20))
        self.assertEqual(ap.budget_bytes({ap.MIB_ENV: "4"}), 4 * 2**20)
        for bad in ("0", "-1", "21"):
            with self.assertRaises(ValueError):
                ap.budget_bytes({ap.MIB_ENV: bad})

    def test_plans_skip_last_layer_and_engram_windows(self):
        torch = FakeTorch()
        Layer, _, _ = make_classes(torch.log)
        layers = [Layer(i, engram=i in (1, 3)) for i in range(5)]
        plans = ap.layer_plans(layers, 5_767_168)
        self.assertEqual([p is not None for p in plans], [False, True, False, True, False])
        self.assertEqual([t.name for t, _ in plans[1]], ["w2", "s2"])
        self.assertLessEqual(sum(n for _, n in plans[1]), 5_767_168)


class HookTests(unittest.TestCase):
    def test_fork_before_the_ar_join_before_the_next_layer(self):
        torch, model, msgs, pf, x = build(n=3)
        model.forward(x)
        log = [e for e in torch.log if e[0] != "sync"]
        # trial launch at the first forward, then per layer: layer, fork (wait+prefetch), AR, join
        trial = log[:3]
        self.assertEqual(trial[0], ("wait", "side", "main"))
        self.assertEqual(trial[1][:3], ("prefetch", "side", "w1"))
        self.assertEqual(trial[2], ("wait", "main", "side"))
        rest = log[3:]
        self.assertEqual(rest[0], ("layer", 0))
        self.assertEqual(rest[1], ("wait", "side", "main"))
        self.assertEqual([e[:3] for e in rest[2:4]], [("prefetch", "side", "w1"), ("prefetch", "side", "s1")])
        self.assertEqual(rest[4], ("ar",))
        self.assertEqual(rest[5], ("wait", "main", "side"))  # join on the next layer's entry
        self.assertEqual(rest[6], ("layer", 1))
        self.assertEqual(rest[-1], ("ar",))  # last layer: no plan, nothing pending at the end
        self.assertEqual(pf.forks, 1 + 2)
        self.assertTrue(any(m.startswith(ap.LOG_ENGAGED) and "2/3 layers" in m for m in msgs))

    def test_prefill_batches_do_not_fork(self):
        torch, model, _, pf, x = build(n=3, rows=ap.MAX_TOKENS + 1)
        model.forward(x)
        self.assertEqual(pf.forks, 1)  # the trial only
        self.assertFalse(pf.pending)

    def test_plans_wait_for_an_eager_forward(self):
        torch, model, msgs, pf, x = build(n=3, capturing_first=True)
        model.forward(x)
        self.assertEqual(pf.forks, 0)
        self.assertFalse(hasattr(model, "_dsv41_l2pf_plans"))
        torch.capturing = False
        model.forward(x)
        self.assertTrue(hasattr(model, "_dsv41_l2pf_plans"))
        self.assertEqual(pf.forks, 1 + 2)

    def test_failed_trial_disarms(self):
        torch, model, msgs, pf, x = build(n=3, launch_fails=True)
        model.forward(x)
        self.assertEqual(pf.forks, 0)
        self.assertTrue(msgs[0].startswith(ap.LOG_DISARMED))
        self.assertFalse(any(ap.LOG_ENGAGED in m for m in msgs))
        self.assertTrue(all(p is None for p in model._dsv41_l2pf_plans))

    def test_disarmed_plan_is_consumed_once(self):
        torch, model, _, pf, x = build(n=2)
        model.forward(x)
        self.assertIsNone(pf.armed)
        self.assertFalse(pf.pending)


def _method_src(path: Path, cls: str, meth: str) -> str:
    src = path.read_text()
    c = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == cls)
    m = next(n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == meth)
    return ast.get_source_segment(src, m)


class PinnedAnchorTests(unittest.TestCase):
    def test_anchors_in_pinned_sources(self):
        layer = _method_src(FIX / "dsv41_model_e13.pin.py", "DeepseekV4DecoderLayer", "forward")
        for a in ap.LAYER_ANCHORS:
            self.assertEqual(layer.count(a), 1, a)
        # the MoE AR is the last thing a layer does: ffn is its final call
        self.assertLess(layer.index(ap.LAYER_ANCHORS[0]), layer.index(ap.LAYER_ANCHORS[1]))
        model = _method_src(FIX / "dsv41_model_e13.pin.py", "DeepseekV4Model", "forward")
        for a in ap.MODEL_ANCHORS:
            self.assertIn(a, model)
        runner = _method_src(FIX / "moe_runner_e13.pin.py", "MoERunner", "_maybe_reduce_final_output")
        for a in ap.RUNNER_ANCHORS:
            self.assertEqual(runner.count(a), 1, a)

    def test_qkv_a_is_fused_wqa_wkv_and_engram_is_a_layer_attribute(self):
        src = (FIX / "dsv41_model_e13.pin.py").read_text()
        self.assertIn("self.engram: Engram | None = None", src)
        self.assertIn('("attn.fused_wqa_wkv", "attn.wq_a", 0)', src)


class KernelSourceTests(unittest.TestCase):
    def test_kernel_ptx_matches_the_measured_one(self):
        # kernel_study/comm/l2_prefetch_window.py measured this exact instruction (bulk60, -21.7 us)
        bench = (ROOT / "kernel_study/comm/l2_prefetch_window.py").read_text()
        self.assertIn("cp.async.bulk.prefetch.L2.global [%0], %1;", bench)
        self.assertIn("cp.async.bulk.prefetch.L2.global [$1], $2;", l2pf_kernel.PTX)
        self.assertIn(l2pf_kernel.PTX, (ROOT / "docker/patch/l2pf_kernel.py").read_text().split("def _build")[1])
        self.assertEqual(l2pf_kernel.CHUNK, 16384)


class WiringTests(unittest.TestCase):
    def test_sitecustomize_and_run_sh(self):
        site = (ROOT / "docker/patch/sitecustomize.py").read_text()
        self.assertIn('_patch("ar_l2_prefetch", _p_ar_l2_prefetch)', site)
        run = (ROOT / "run.sh").read_text()
        self.assertIn("  DSV41_AR_L2_PREFETCH=0\n", run)
        self.assertIn("  DSV41_AR_L2_PREFETCH_MIB=\n", run)

    def test_off_does_nothing(self):
        msgs = []
        self.assertEqual(ap.install({}, msgs.append), "off")
        self.assertEqual(msgs, [])


if __name__ == "__main__":
    unittest.main()
