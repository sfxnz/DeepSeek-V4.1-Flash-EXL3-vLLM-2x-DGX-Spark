"""DSV41_ATTN_T2R_DEDUP (decode_levers): token -> request map shared across KV groups.

Host-only: numpy arrays stand in for tensors (views, weakrefs, data_ptr). The
GPU proof with the image's real CommonAttentionMetadata (21 groups per prep,
bit-exact buffers, 168 -> 28 kernels) is kernel_study/fusion_host/
t2r_dedup_bench.py; the stock method it wraps is pinned in
tests/fixtures/attn_backend_t2r_e13.pin.py.
"""

from __future__ import annotations

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


def install_on_fake():
    cls = type("CAM", (FakeCAM,), {})
    backend = types.ModuleType("vllm.v1.attention.backend")
    backend.CommonAttentionMetadata = cls
    mods = {n: types.ModuleType(n) for n in ("vllm", "vllm.v1", "vllm.v1.attention")}
    mods[backend.__name__] = backend
    with mock.patch.dict(sys.modules, mods), redirect_stdout(StringIO()) as out:
        dl._install_t2r_dedup({"DSV41_ATTN_T2R_DEDUP": "1"})
    return cls, out.getvalue()


class ReuseRuleTests(unittest.TestCase):
    def test_reuse_needs_same_objects_counts_and_room(self) -> None:
        qsl, cpu = t([0, 4]), t([0, 4])
        view = t(np.zeros(8))[:4]
        prev = (weakref.ref(qsl), weakref.ref(cpu), 4, 4, view)
        buf = t(np.zeros(16))
        self.assertTrue(dl.t2r_reuse(prev, qsl, cpu, 4, 4, buf))
        self.assertFalse(dl.t2r_reuse(None, qsl, cpu, 4, 4, buf))
        self.assertFalse(dl.t2r_reuse(prev, t([0, 4]), cpu, 4, 4, buf))  # equal values, other object
        self.assertFalse(dl.t2r_reuse(prev, qsl, t([0, 4]), 4, 4, buf))
        self.assertFalse(dl.t2r_reuse(prev, qsl, cpu, 8, 4, buf))
        self.assertFalse(dl.t2r_reuse(prev, qsl, cpu, 4, 3, buf))
        self.assertFalse(dl.t2r_reuse(prev, qsl, cpu, 4, 4, t(np.zeros(3))))
        self.assertFalse(dl.t2r_reuse(prev, qsl, cpu, 4, 4, t(np.zeros(16), np.int64)))

    def test_dead_reference_never_matches_a_new_object(self) -> None:
        qsl, cpu = t([0, 4]), t([0, 4])
        prev = (weakref.ref(qsl), weakref.ref(cpu), 4, 4, t(np.zeros(8))[:4])
        del qsl
        self.assertFalse(dl.t2r_reuse(prev, t([0, 4]), cpu, 4, 4, t(np.zeros(16))))


class PatchedMethodTests(unittest.TestCase):
    def prep(self, cls, qvals, num_tokens, bufs):
        qsl = t(qvals)[:]  # a fresh view object per prep, like the runner
        cpu = t(qvals)
        return [cls(qsl, cpu, num_tokens).token_to_req_indices(b) for b in bufs]

    def test_one_computation_per_prep_and_stock_values(self) -> None:
        cls, out = install_on_fake()
        self.assertIn("dedup", out)
        self.assertTrue(getattr(cls.token_to_req_indices, "_dsv41_t2r_dedup", False))
        for qvals, num_tokens in (([0, 4], 4), ([0, 4, 8], 8), ([0, 3, 6, 6], 8), ([0, 1], 1)):
            bufs = [t(np.full(32, -7)) for _ in range(21)]
            FakeCAM.stock_calls = 0
            views = self.prep(cls, qvals, num_tokens, bufs)
            self.assertEqual(FakeCAM.stock_calls, 1, qvals)
            n = max(qvals[-1], num_tokens)
            ref = np.zeros(n, np.int32)
            lens = np.diff(qvals)
            ref[: qvals[-1]] = np.repeat(np.arange(len(lens)), lens)
            for b, v in zip(bufs, views):
                self.assertTrue(np.array_equal(b[:n], ref), qvals)
                self.assertTrue((b[n:] == -7).all())  # untouched beyond n, like stock
                self.assertEqual(len(v), num_tokens)
                self.assertEqual(v.data_ptr(), b.data_ptr())  # each group keeps its own buffer

    def test_next_prep_recomputes(self) -> None:
        cls, _ = install_on_fake()
        bufs = [t(np.zeros(32)) for _ in range(3)]
        FakeCAM.stock_calls = 0
        self.prep(cls, [0, 4], 4, bufs)
        self.prep(cls, [0, 2, 4], 4, bufs)  # same shapes, new objects, new values
        self.assertEqual(FakeCAM.stock_calls, 2)
        self.assertTrue(all(np.array_equal(b[:4], [0, 0, 1, 1]) for b in bufs))

    def test_instance_cache_still_wins(self) -> None:
        cls, _ = install_on_fake()
        qsl, cpu = t([0, 4]), t([0, 4])
        a, b = t(np.zeros(8)), t(np.zeros(8))
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

    def test_listed_in_install(self) -> None:
        src = (ROOT / "docker/patch/decode_levers.py").read_text()
        self.assertIn('("attn-t2r-dedup", _install_t2r_dedup)', src)


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
