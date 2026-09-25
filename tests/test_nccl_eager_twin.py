"""nccl_eager_twin: router, install/disarm, anchors in pinned canonical-e13 sources (CPU only).

Fixtures are read-only copies from image dsv41-flash-exl3-sm121:canonical-e13
(sha256:c81762335a12...), vllm/distributed/device_communicators/:
  cuda_communicator_e13.pin.py   cuda_communicator.py
  pynccl_e13.pin.py              pynccl.py
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(ROOT / "docker" / "patch"))

import nccl_eager_twin as nt  # noqa: E402

PYNCCL_PIN = FIX / "pynccl_e13.pin.py"
CUDACOMM_PIN = FIX / "cuda_communicator_e13.pin.py"


def _class_node(path: Path, name: str) -> ast.ClassDef:
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == name)


def _pinned_pynccl_methods() -> dict[str, list[str]]:
    """Public PyNcclCommunicator methods in the pin: name -> parameter names (self first)."""
    out = {}
    for node in _class_node(PYNCCL_PIN, "PyNcclCommunicator").body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            out[node.name] = [a.arg for a in node.args.args]
    return out


class FakePyNccl:
    """Signatures copied from the pinned PyNcclCommunicator (checked below)."""

    made: list = []

    def __init__(self, group=None, device=None, tag="twin"):
        self.group, self.device, self.tag = group, device, tag
        self.disabled, self.world_size, self.rank = False, 2, 0
        self.calls: list = []
        FakePyNccl.made.append(self)

    def _rec(self, name, *a):
        self.calls.append((name, a))
        return (self.tag, name)

    def all_reduce(self, in_tensor, out_tensor=None, op=None, stream=None):
        return self._rec("all_reduce", in_tensor, stream)

    def all_gather(self, output_tensor, input_tensor, stream=None):
        return self._rec("all_gather", output_tensor, stream)

    def all_gatherv(self, output_tensor, input_tensor, sizes, stream=None):
        return self._rec("all_gatherv", stream)

    def reduce_scatter(self, output_tensor, input_tensor, op=None, stream=None):
        return self._rec("reduce_scatter", stream)

    def reduce_scatterv(self, output_tensor, input_tensor, sizes, op=None, stream=None):
        return self._rec("reduce_scatterv", stream)

    def reduce(self, output_tensor, input_tensor, root, op=None, stream=None):
        return self._rec("reduce", stream)

    def scatter(self, output_tensor, input_tensor, sizes, root, stream=None):
        return self._rec("scatter", stream)

    def send(self, tensor, dst, stream=None):
        return self._rec("send", stream)

    def recv(self, tensor, src, stream=None):
        return self._rec("recv", stream)

    def broadcast(self, tensor, src, stream=None):
        return self._rec("broadcast", stream)

    def batch_isend_irecv(self, p2p_ops, stream=None):
        return self._rec("batch_isend_irecv", stream)

    def destroy(self):
        return self._rec("destroy")

    def suspend(self):
        return self._rec("suspend")

    def resume(self):
        return self._rec("resume")

    def group_start(self):
        return self._rec("group_start")

    def group_end(self):
        return self._rec("group_end")

    def register_comm_window(self, tensor):
        return self._rec("register_comm_window")

    def register_comm_window_raw(self, ptr, size):
        return self._rec("register_comm_window_raw")

    def deregister_comm_window(self, window):
        return self._rec("deregister_comm_window")

    @classmethod
    def from_unique_id_bytes(cls, unique_id_bytes, rank, world_size, device, library_path=None):
        return cls()


class Capture:
    """capturing(stream) stand-in: 'cap' streams are capturing; records what it saw."""

    def __init__(self):
        self.default = False
        self.seen: list = []

    def __call__(self, stream):
        self.seen.append(stream)
        return self.default if stream is None else stream == "cap"


def _router(eager=True):
    graph = FakePyNccl(tag="graph")
    twin = FakePyNccl(tag="twin") if eager else None
    cap = Capture()
    logs: list = []
    r = nt.GraphEagerRouter(graph, twin, cap, "tp:0", nt.stream_positions(FakePyNccl), logs.append)
    return r, graph, twin, cap, logs


class PinnedApiTests(unittest.TestCase):
    """The router's method tables cover the pinned PyNcclCommunicator exactly."""

    def test_every_public_method_is_classified(self):
        pinned = set(_pinned_pynccl_methods())
        known = set(nt.ROUTED) | set(nt.FAN_OUT) | set(nt.PASS) | set(nt.REFUSED) | set(nt.IGNORED)
        self.assertEqual(sorted(pinned - known), [], "unrouted PyNccl method")
        self.assertEqual(sorted(known - pinned), [], "stale name in the router tables")

    def test_stream_positions_match_the_pin(self):
        pinned = _pinned_pynccl_methods()
        want = {m: pinned[m].index("stream") - 1 for m in nt.ROUTED}
        self.assertEqual(nt.stream_positions(FakePyNccl), want)
        fake = {n: [p for p in v.__code__.co_varnames[: v.__code__.co_argcount]] for n, v in vars(FakePyNccl).items()
                if callable(v) and not n.startswith("_")}
        for m in nt.ROUTED + nt.FAN_OUT + nt.PASS:
            self.assertEqual(fake[m], pinned[m], m)

    def test_init_anchors_are_in_the_pin(self):
        src = CUDACOMM_PIN.read_text()
        init = next(n for n in _class_node(CUDACOMM_PIN, "CudaCommunicator").body
                    if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        body = ast.get_source_segment(src, init)
        for anchor in nt.INIT_ANCHORS:
            self.assertEqual(body.count(anchor), 1, anchor)
        params = [a.arg for a in init.args.args + init.args.kwonlyargs]
        for p in nt.INIT_PARAMS:
            self.assertIn(p, params)

    def test_cuda_communicator_only_calls_routed_methods(self):
        # Every `pynccl_comm.<name>(` call site in the pinned CudaCommunicator uses a
        # name the router handles, and no other attribute of it is assigned.
        src = CUDACOMM_PIN.read_text()
        called = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                base = node.func.value
                name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
                if name == "pynccl_comm":
                    called.add(node.func.attr)
        handled = set(nt.ROUTED) | set(nt.FAN_OUT) | set(nt.PASS)
        self.assertEqual(sorted(called - handled), [])
        self.assertIn("all_reduce", called)
        self.assertIn("all_gather", called)

    def test_group_kind(self):
        self.assertEqual(nt.group_kind("tp:0"), "tp")
        self.assertEqual(nt.group_kind("ep:0"), "ep")
        self.assertEqual(nt.group_kind("dcp:0"), "dcp")
        self.assertEqual(nt.group_kind(""), "")


class RouterTests(unittest.TestCase):
    def test_captured_goes_to_graph_eager_goes_to_twin(self):
        r, graph, twin, cap, _ = _router()
        self.assertEqual(r.all_reduce("x", stream="cap"), ("graph", "all_reduce"))
        self.assertEqual(r.all_reduce("x"), ("twin", "all_reduce"))
        cap.default = True  # stream=None resolves to a capturing current stream
        self.assertEqual(r.all_gather("o", "i"), ("graph", "all_gather"))
        cap.default = False
        self.assertEqual(r.all_gather("o", "i"), ("twin", "all_gather"))
        self.assertEqual([c[0] for c in graph.calls], ["all_reduce", "all_gather"])
        self.assertEqual([c[0] for c in twin.calls], ["all_reduce", "all_gather"])

    def test_every_routed_method_routes_both_ways(self):
        r, graph, twin, cap, _ = _router()
        for m in nt.ROUTED:
            nargs = nt.stream_positions(FakePyNccl)[m]
            self.assertEqual(getattr(r, m)(*range(nargs), stream="cap")[0], "graph", m)
            self.assertEqual(getattr(r, m)(*range(nargs), stream="s1")[0], "twin", m)

    def test_positional_stream_is_the_one_checked(self):
        r, graph, twin, cap, _ = _router()
        self.assertEqual(r.send("t", 1, "cap"), ("graph", "send"))
        self.assertEqual(r.all_reduce("i", "o", None, "cap"), ("graph", "all_reduce"))
        self.assertEqual(r.recv("t", 0, "s2"), ("twin", "recv"))
        self.assertEqual(cap.seen, ["cap", "cap", "s2"])

    def test_no_twin_group_is_eager_only(self):
        r, graph, _, cap, _ = _router(eager=False)
        self.assertEqual(r.all_reduce("x"), ("graph", "all_reduce"))
        with self.assertRaisesRegex(RuntimeError, "REFUSED"):
            r.all_reduce("x", stream="cap")
        self.assertEqual([c[0] for c in graph.calls], ["all_reduce"])

    def test_fan_out_and_pass(self):
        r, graph, twin, _, _ = _router()
        r.suspend()
        r.resume()
        r.destroy()
        r.group_start()
        r.group_end()
        self.assertEqual([c[0] for c in graph.calls], ["suspend", "resume", "destroy", "group_start", "group_end"])
        self.assertEqual([c[0] for c in twin.calls], ["suspend", "resume", "destroy"])

    def test_attributes_read_through_and_router_is_read_only(self):
        r, graph, _, _, _ = _router()
        graph.world_size = 7
        self.assertEqual(r.world_size, 7)
        self.assertFalse(r.disabled)
        with self.assertRaises(AttributeError):
            r.disabled = True
        for m in nt.REFUSED:
            with self.assertRaisesRegex(RuntimeError, "not routed"):
                getattr(r, m)

    def test_routing_is_reported_once_after_capture(self):
        r, _, _, _, logs = _router()
        r.all_reduce("x")  # eager before any capture: no report yet
        self.assertEqual(logs, [])
        r.all_reduce("x", stream="cap")
        r.all_reduce("x", stream="cap")
        r.all_reduce("x")
        r.all_reduce("x")
        self.assertEqual(len(logs), 1)
        self.assertIn("2 captured calls on the graph comm", logs[0])


FAKE_CC = '''
class CudaCommunicator:
    def __init__(self, cpu_group, device=None, device_group=None, unique_name="",
                 global_ranks=None, global_world_size=None, tcp_store_group=None, use_all2all=False):
        self.cpu_group, self.device, self.unique_name = cpu_group, device, unique_name
        self.pynccl_comm = None
        if unique_name != "off:0":
            self.pynccl_comm = PyNcclCommunicator(
                group=self.cpu_group if tcp_store_group is None else tcp_store_group,
                device=self.device,
            )
'''


class InstallTests(unittest.TestCase):
    def setUp(self):
        FakePyNccl.made = []
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _modules(self, cc_source: str):
        path = Path(self.tmp.name) / "fake_cuda_communicator.py"
        path.write_text(textwrap.dedent(cc_source))
        spec = importlib.util.spec_from_file_location("fake_cuda_communicator", path)
        cc = importlib.util.module_from_spec(spec)
        cc.PyNcclCommunicator = lambda **kw: FakePyNccl(tag="graph", **kw)
        spec.loader.exec_module(cc)
        pyn = types.ModuleType("vllm.distributed.device_communicators.pynccl")
        pyn.PyNcclCommunicator = FakePyNccl
        dc = types.ModuleType("vllm.distributed.device_communicators")
        dc.cuda_communicator, dc.pynccl = cc, pyn
        mods = {
            "vllm": types.ModuleType("vllm"),
            "vllm.distributed": types.ModuleType("vllm.distributed"),
            "vllm.distributed.device_communicators": dc,
            "vllm.distributed.device_communicators.cuda_communicator": cc,
            "vllm.distributed.device_communicators.pynccl": pyn,
        }
        return cc, mods

    def test_off_changes_nothing(self):
        env, logs = {}, []
        self.assertEqual(nt.install(env, logs.append), "off")
        self.assertEqual(env, {})
        self.assertEqual(logs, [])

    def test_mixing_off_without_the_twin_is_reset(self):
        env, logs = {nt.MIXING_ENV: "0"}, []
        self.assertEqual(nt.install(env, logs.append), "off")
        self.assertEqual(env[nt.MIXING_ENV], "1")
        self.assertTrue(logs and logs[0].startswith(nt.LOG_DISARMED[1]))

    def test_armed_wraps_once_and_sets_mixing_off(self):
        cc, mods = self._modules(FAKE_CC)
        env, logs = {nt.ENV: "1"}, []
        with mock.patch.dict(sys.modules, mods):
            self.assertEqual(nt.install(env, logs.append), "armed")
            wrapped = cc.CudaCommunicator.__init__
            self.assertTrue(wrapped._dsv41_eager_twin)
            self.assertEqual(nt.install(env, logs.append), "armed")
            self.assertIs(cc.CudaCommunicator.__init__, wrapped)
        self.assertEqual(env[nt.MIXING_ENV], "0")

    def test_wrapped_init_builds_twin_for_tp_and_guards_others(self):
        cc, mods = self._modules(FAKE_CC)
        env, logs = {nt.ENV: "1"}, []
        with mock.patch.dict(sys.modules, mods):
            nt.install(env, logs.append)
            tp = cc.CudaCommunicator("cpu", device="cuda:0", unique_name="tp:0")
            ep = cc.CudaCommunicator("cpu", device="cuda:0", unique_name="ep:0")
            off = cc.CudaCommunicator("cpu", device="cuda:0", unique_name="off:0")
            store = cc.CudaCommunicator("cpu", "cuda:0", None, "tp:1", None, None, "store")
        self.assertIsInstance(tp.pynccl_comm, nt.GraphEagerRouter)
        self.assertEqual(tp.pynccl_comm._graph.tag, "graph")
        self.assertEqual(tp.pynccl_comm._eager.tag, "twin")
        self.assertEqual((tp.pynccl_comm._eager.group, tp.pynccl_comm._eager.device), ("cpu", "cuda:0"))
        self.assertIsNone(ep.pynccl_comm._eager)
        self.assertIsNone(off.pynccl_comm)
        self.assertEqual(store.pynccl_comm._eager.group, "store")
        self.assertEqual(sum(nt.LOG_ENGAGED in m for m in logs), 2)
        self.assertEqual(sum("guard on ep:0" in m for m in logs), 1)

    def test_anchor_miss_disarms_and_keeps_mixing_on(self):
        cc, mods = self._modules(FAKE_CC.replace("tcp_store_group is None", "tcp_store_group == None"))
        env, logs = {nt.ENV: "1"}, []
        with mock.patch.dict(sys.modules, mods):
            self.assertEqual(nt.install(env, logs.append), "disarmed")
            self.assertFalse(getattr(cc.CudaCommunicator.__init__, "_dsv41_eager_twin", False))
        self.assertNotIn(nt.MIXING_ENV, env)
        self.assertTrue(logs[0].startswith(nt.LOG_DISARMED[0]))

    def test_new_pynccl_method_disarms(self):
        cc, mods = self._modules(FAKE_CC)

        class Grown(FakePyNccl):
            pass

        Grown.all_to_all = lambda self, a, b, stream=None: None
        mods["vllm.distributed.device_communicators.pynccl"].PyNcclCommunicator = Grown
        env, logs = {nt.ENV: "1"}, []
        with mock.patch.dict(sys.modules, mods):
            # check_anchors reads the class's own namespace: a subclass adding a method
            # is exactly what a changed image would look like.
            self.assertEqual(nt.install(env, logs.append), "disarmed")
        self.assertIn("all_to_all", logs[0])
        self.assertNotIn(nt.MIXING_ENV, env)

    def test_missing_vllm_disarms(self):
        env, logs = {nt.ENV: "1"}, []
        with mock.patch.dict(sys.modules, {"vllm": None}):
            self.assertEqual(nt.install(env, logs.append), "disarmed")
        self.assertNotIn(nt.MIXING_ENV, env)


class WiringTests(unittest.TestCase):
    def test_sitecustomize_installs_it(self):
        site = (ROOT / "docker/patch/sitecustomize.py").read_text()
        self.assertIn('_patch("nccl_eager_twin", _p_nccl_eager_twin)', site)
        self.assertIn("nccl_eager_twin.install()", site)

    def test_default_off_in_run_sh(self):
        run = (ROOT / "run.sh").read_text()
        self.assertIn("  DSV41_NCCL_EAGER_TWIN=0\n", run)
        self.assertNotIn("NCCL_GRAPH_MIXING_SUPPORT=", run)


if __name__ == "__main__":
    unittest.main()
