"""DSV41_ATTN_T2R_DEDUP (attn_t2r_dedup): token -> request map shared across KV groups.

Host-only: numpy arrays stand in for tensors (views, weakrefs, data_ptr) and a
stand-in torch module for the verify path. The GPU proof with the image's real
CommonAttentionMetadata (3 + 18 groups a step, bit-exact buffers, 168 -> 35
kernels) is kernel_study/fusion_host/t2r_dedup_bench.py; the stock method it
wraps is pinned in tests/fixtures/attn_backend_t2r_e13.pin.py.
"""

from __future__ import annotations

import ast
import sys
import types
import unittest
import weakref
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docker" / "patch"))
import attn_t2r_dedup as t2r  # noqa: E402
import decode_levers as dl  # noqa: E402

FIX = ROOT / "tests" / "fixtures" / "attn_backend_t2r_e13.pin.py"


class T(np.ndarray):
    """The tensor surface the lever touches."""

    device = "cuda:0"

    def data_ptr(self) -> int:
        return self.__array_interface__["data"][0]

    def copy_(self, other):
        self[...] = other
        return self

    def zero_(self):
        self[...] = 0
        return self


def t(values, dtype=np.int32) -> T:
    return np.asarray(values, dtype=dtype).view(T)


class FakeCAM:
    """Stock semantics of the pinned method, numpy instead of torch."""

    stock_calls = 0

    def __init__(self, qsl, qsl_cpu, num_tokens):
        self.query_start_loc = qsl
        self.query_start_loc_cpu = qsl_cpu
        self.num_actual_tokens = num_tokens
        self._token_to_req_indices_cache = None

    def token_to_req_indices(self, buffer):
        FakeCAM.stock_calls += 1
        num_tokens = self.num_actual_tokens
        if self._token_to_req_indices_cache is not None:
            return self._token_to_req_indices_cache[:num_tokens]
        num_mapped = int(self.query_start_loc_cpu[-1])
        lens = np.diff(np.asarray(self.query_start_loc))
        vals = np.repeat(np.arange(len(lens), dtype=np.int32), lens)[:num_mapped]
        buffer[:num_mapped].copy_(vals)
        if num_mapped < num_tokens:
            buffer[num_mapped:num_tokens].zero_()
        self._token_to_req_indices_cache = buffer[: max(num_mapped, num_tokens)]
        return self._token_to_req_indices_cache[:num_tokens]


def fake_torch(capturing=None):
    mod = types.ModuleType("torch")
    mod.cuda = types.SimpleNamespace(is_current_stream_capturing=lambda: bool(capturing and capturing[0]))
    mod.equal = lambda a, b: bool(np.array_equal(a, b))
    mod.empty_like = lambda x: np.empty_like(x).view(T)
    return mod


def install_on_fake(env=None, capturing=None):
    """A fresh class, the lever installed on it via decode_levers; returns (cls, log)."""
    cls = type("CAM", (FakeCAM,), {})
    backend = types.ModuleType("vllm.v1.attention.backend")
    backend.CommonAttentionMetadata = cls
    mods = {n: types.ModuleType(n) for n in ("vllm", "vllm.v1", "vllm.v1.attention")}
    mods[backend.__name__] = backend
    mods["torch"] = fake_torch(capturing)
    t2r._STATE.update(armed=True, engaged=False, computes=0, reuses=0)
    with mock.patch.dict(sys.modules, mods), redirect_stdout(StringIO()) as out:
        dl._install_t2r_dedup(env or {"DSV41_ATTN_T2R_DEDUP": "1"})
    return cls, mods, out.getvalue()


class ReuseRuleTests(unittest.TestCase):
    def test_reuse_needs_same_objects_counts_and_room(self) -> None:
        qsl, cpu = t([0, 4]), t([0, 4])
        view = t(np.zeros(8))[:4]
        prev = (weakref.ref(qsl), weakref.ref(cpu), 4, 4, view)
        buf = t(np.zeros(16))
        self.assertTrue(t2r.t2r_reuse(prev, qsl, cpu, 4, 4, buf))
        self.assertFalse(t2r.t2r_reuse(None, qsl, cpu, 4, 4, buf))
        self.assertFalse(t2r.t2r_reuse(prev, t([0, 4]), cpu, 4, 4, buf))  # equal values, other object
        self.assertFalse(t2r.t2r_reuse(prev, qsl, t([0, 4]), 4, 4, buf))
        self.assertFalse(t2r.t2r_reuse(prev, qsl, cpu, 8, 4, buf))
        self.assertFalse(t2r.t2r_reuse(prev, qsl, cpu, 4, 3, buf))
        self.assertFalse(t2r.t2r_reuse(prev, qsl, cpu, 4, 4, t(np.zeros(3))))
        self.assertFalse(t2r.t2r_reuse(prev, qsl, cpu, 4, 4, t(np.zeros(16), np.int64)))

    def test_dead_reference_never_matches_a_new_object(self) -> None:
        qsl, cpu = t([0, 4]), t([0, 4])
        prev = (weakref.ref(qsl), weakref.ref(cpu), 4, 4, t(np.zeros(8))[:4])
        del qsl
        self.assertFalse(t2r.t2r_reuse(prev, t([0, 4]), cpu, 4, 4, t(np.zeros(16))))

    def test_census_cadence(self) -> None:
        due = [c for c in range(1, 70000) if t2r.census_due(c)]
        self.assertEqual(due, [1024, 2048, 4096, 8192, 16384, 32768, 65536])

    def test_env(self) -> None:
        self.assertFalse(t2r.enabled({}))
        self.assertTrue(t2r.enabled({"DSV41_ATTN_T2R_DEDUP": "1"}))
        self.assertEqual(t2r.verify_calls({}), 8)
        self.assertEqual(t2r.verify_calls({"DSV41_ATTN_T2R_VERIFY": "0"}), 1)
        self.assertEqual(t2r.verify_calls({"DSV41_ATTN_T2R_VERIFY": "3"}), 3)


class PatchedMethodTests(unittest.TestCase):
    def prep(self, cls, mods, qvals, num_tokens, bufs, corrupt=None):
        qsl = t(qvals)[:]  # a fresh view object per prep, like the runner
        cpu = t(qvals)
        views = []
        with mock.patch.dict(sys.modules, mods):
            for i, b in enumerate(bufs):
                views.append(cls(qsl, cpu, num_tokens).token_to_req_indices(b))
                if corrupt is not None and i == 0:
                    corrupt(b)
        return views

    @staticmethod
    def ref(qvals, num_tokens):
        n = max(qvals[-1], num_tokens)
        out = np.zeros(n, np.int32)
        lens = np.diff(qvals)
        out[: qvals[-1]] = np.repeat(np.arange(len(lens)), lens)
        return out

    def test_one_computation_per_prep_and_stock_values(self) -> None:
        cls, mods, out = install_on_fake()
        self.assertIn("dsv41: attention t2r dedup: CommonAttentionMetadata.token_to_req_indices wrapped", out)
        self.assertTrue(getattr(cls.token_to_req_indices, "_dsv41_t2r_dedup", False))
        t2r._STATE["verify_left"] = 0  # the verify path has its own tests
        with redirect_stdout(StringIO()):
            for qvals, num_tokens in (([0, 4], 4), ([0, 4, 8], 8), ([0, 3, 6, 6], 8), ([0, 1], 1)):
                bufs = [t(np.full(32, -7)) for _ in range(21)]
                FakeCAM.stock_calls = 0
                views = self.prep(cls, mods, qvals, num_tokens, bufs)
                self.assertEqual(FakeCAM.stock_calls, 1, qvals)  # verify window already closed
                n = max(qvals[-1], num_tokens)
                ref = self.ref(qvals, num_tokens)
                for b, v in zip(bufs, views):
                    self.assertTrue(np.array_equal(b[:n], ref), qvals)
                    self.assertTrue((b[n:] == -7).all())  # untouched beyond n, like stock
                    self.assertEqual(len(v), num_tokens)
                    self.assertEqual(v.data_ptr(), b.data_ptr())  # each group keeps its own buffer
        self.assertEqual((t2r._STATE["computes"], t2r._STATE["reuses"]), (4, 80))

    def test_verified_reuse_prints_engaged_once_then_a_census(self) -> None:
        cls, mods, _ = install_on_fake({"DSV41_ATTN_T2R_DEDUP": "1", "DSV41_ATTN_T2R_VERIFY": "3"})
        bufs = [t(np.zeros(32)) for _ in range(3)]
        with redirect_stdout(StringIO()) as out:
            FakeCAM.stock_calls = 0
            self.prep(cls, mods, [0, 4], 4, bufs)
            self.assertEqual(FakeCAM.stock_calls, 3)  # 1 compute + 2 verified reuses
            self.prep(cls, mods, [0, 2, 4], 4, bufs)
        log = out.getvalue()
        self.assertEqual(log.count(t2r.LOG_ENGAGED), 1)
        self.assertEqual(log.count("[t2r-census] verified computes=2 reuses=3 reuses/compute=1.50"), 1)
        self.assertNotIn(t2r.LOG_DISARMED, log)
        self.assertEqual(t2r._STATE["verify_left"], 0)
        self.assertTrue(all(np.array_equal(b[:4], [0, 0, 1, 1]) for b in bufs))

    def test_mismatch_restores_stock_values_and_disarms(self) -> None:
        cls, mods, _ = install_on_fake({"DSV41_ATTN_T2R_DEDUP": "1", "DSV41_ATTN_T2R_VERIFY": "8"})
        bufs = [t(np.zeros(32)) for _ in range(3)]

        def overwrite(b):  # something rewrites the first group's buffer before the copy
            b[0] = 99

        with redirect_stdout(StringIO()) as out:
            self.prep(cls, mods, [0, 4], 4, bufs, corrupt=overwrite)
            self.assertIn(t2r.LOG_DISARMED, out.getvalue())
            self.assertNotIn(t2r.LOG_ENGAGED, out.getvalue())
            self.assertFalse(t2r._STATE["armed"])
            for b in bufs[1:]:
                self.assertTrue(np.array_equal(b[:4], [0, 0, 0, 0]))  # the stock map, not the copy
            FakeCAM.stock_calls = 0
            self.prep(cls, mods, [0, 2, 4], 4, bufs)
        self.assertEqual(FakeCAM.stock_calls, 3)  # disarmed: every group computes
        self.assertEqual(out.getvalue().count(t2r.LOG_DISARMED), 1)

    def test_no_verify_while_capturing(self) -> None:
        capturing = [True]
        cls, mods, _ = install_on_fake({"DSV41_ATTN_T2R_DEDUP": "1", "DSV41_ATTN_T2R_VERIFY": "2"}, capturing)
        bufs = [t(np.zeros(32)) for _ in range(3)]
        with redirect_stdout(StringIO()) as out:
            FakeCAM.stock_calls = 0
            self.prep(cls, mods, [0, 4], 4, bufs)
            self.assertEqual(FakeCAM.stock_calls, 1)
            self.assertEqual(t2r._STATE["verify_left"], 2)
            capturing[0] = False
            self.prep(cls, mods, [0, 4], 4, bufs)
        self.assertEqual(t2r._STATE["verify_left"], 0)
        self.assertEqual(out.getvalue().count(t2r.LOG_ENGAGED), 1)

    def test_next_prep_recomputes(self) -> None:
        cls, mods, _ = install_on_fake()
        bufs = [t(np.zeros(32)) for _ in range(3)]
        with redirect_stdout(StringIO()):
            self.prep(cls, mods, [0, 4], 4, bufs)
            FakeCAM.stock_calls = 0
            t2r._STATE["verify_left"] = 0
            self.prep(cls, mods, [0, 2, 4], 4, bufs)  # same shapes, new objects, new values
        self.assertEqual(FakeCAM.stock_calls, 1)
        self.assertTrue(all(np.array_equal(b[:4], [0, 0, 1, 1]) for b in bufs))

    def test_instance_cache_still_wins(self) -> None:
        cls, mods, _ = install_on_fake()
        qsl, cpu = t([0, 4]), t([0, 4])
        a, b = t(np.zeros(8)), t(np.zeros(8))
        with mock.patch.dict(sys.modules, mods):
            cam = cls(qsl, cpu, 4)
            first = cam.token_to_req_indices(a)
            second = cam.token_to_req_indices(b)  # stock: same instance -> first buffer's view
        self.assertEqual(second.data_ptr(), first.data_ptr())

    def test_off_by_default_touches_nothing(self) -> None:
        cls = type("CAM", (FakeCAM,), {})
        backend = types.ModuleType("vllm.v1.attention.backend")
        backend.CommonAttentionMetadata = cls
        with mock.patch.dict(sys.modules, {backend.__name__: backend}), redirect_stdout(StringIO()) as out:
            dl._install_t2r_dedup({})
            dl._install_t2r_dedup({"DSV41_ATTN_T2R_DEDUP": "0"})
        self.assertEqual(out.getvalue(), "")
        self.assertIs(cls.token_to_req_indices, FakeCAM.token_to_req_indices)

    def test_second_install_is_a_no_op(self) -> None:
        cls, mods, _ = install_on_fake()
        wrapped = cls.token_to_req_indices
        backend = mods["vllm.v1.attention.backend"]
        with mock.patch.dict(sys.modules, mods), redirect_stdout(StringIO()):
            self.assertEqual(t2r.install({}), "already installed")
        self.assertIs(backend.CommonAttentionMetadata.token_to_req_indices, wrapped)

    def test_listed_in_install_and_audit(self) -> None:
        src = (ROOT / "docker/patch/decode_levers.py").read_text()
        self.assertIn('("attn-t2r-dedup", _install_t2r_dedup)', src)
        sys.path.insert(0, str(ROOT / "tools"))
        import engagement_audit as ea

        self.assertEqual(ea.PATCHES["attn_t2r_dedup.py"], {"DSV41_ATTN_T2R_DEDUP": "1"})
        expected, disarm = ea.expectations({"DSV41_ATTN_T2R_DEDUP": "1"})
        self.assertIn(("attn_t2r_dedup.py", t2r.LOG_ENGAGED), expected)
        self.assertIn(t2r.LOG_DISARMED, disarm)
        self.assertNotIn(("attn_t2r_dedup.py", t2r.LOG_ENGAGED), ea.expectations({})[0])

    def test_top_level_imports_are_stdlib_only(self) -> None:
        tree = ast.parse(Path(t2r.__file__).read_text())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        mods = {a.name for n in top if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module for n in top if isinstance(n, ast.ImportFrom)}
        self.assertEqual(mods, {"__future__", "copy", "os", "weakref"})


class StockContractTests(unittest.TestCase):
    """What the lever assumes about the image's method (pinned from e13)."""

    def test_pinned_method_contract(self) -> None:
        src = FIX.read_text()
        for line in (
            "if self._token_to_req_indices_cache is not None:",
            "return self._token_to_req_indices_cache[:num_tokens]",
            "num_mapped_tokens = int(self.query_start_loc_cpu[-1])",
            "query_lens = self.query_start_loc[1:] - self.query_start_loc[:-1]",
            "buffer[:num_mapped_tokens].copy_(token_to_req_indices)",
            "buffer[num_mapped_tokens:num_tokens].zero_()",
            "self._token_to_req_indices_cache = buffer[: max(num_mapped_tokens, num_tokens)]",
        ):
            self.assertIn(line, src, line)


if __name__ == "__main__":
    unittest.main()
