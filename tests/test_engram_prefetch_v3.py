"""engram_prefetch_v3 worker on CPU: numpy stands in for the pinned buffers."""

from __future__ import annotations

import contextlib
import io
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_engram_chain import engram_prefetch_v3  # noqa: E402


def _stager_cls():
    torch_stub = types.SimpleNamespace(inference_mode=lambda: (lambda f: f))
    ns = {"torch": torch_stub}
    exec("class Stager:\n" + engram_prefetch_v3.STAGER_METHODS, ns)
    return ns["Stager"]


class PrefetchV3WorkerTests(unittest.TestCase):
    """Drive the real _prefetch_worker on CPU with numpy snapshot buffers."""

    def _stager(self):
        st = _stager_cls()()
        cap, depth, k = 64, 4, 3
        st._pf_cap, st._pf_rmax, st._pf_kmax, st._pf_depth = cap, 8, 8, depth
        st._pf_ngram, st._pf_heads = 3, 4
        cols = (st._pf_ngram - 1) * st._pf_heads
        st._pf_mult = np.array([[1, 3, 5]], dtype=np.int64)
        st._pf_primes = [1_000_003 + 2 * i for i in range(cols)]
        st._pf_offsets = [i * 10_000_000 for i in range(cols)]
        st._pf_pad = 0
        st._pf_tm = None  # the worker must use the cached list, not re-convert
        st._pf_tm_list = list(range(1000))
        st._pf_layers = [0]
        st._pf_heads_span = [(0, cols)]
        st._pf_pin_ids = np.array([11, 12, 13, 14] + [0] * (cap - 4), dtype=np.int64)
        st._pf_pin_pos = np.array([20, 21, 22, 23] + [0] * (cap - 4), dtype=np.int64)
        st._pf_pin_qsl = np.array([0, 4] + [0] * 7, dtype=np.int64)
        st._pf_pin_win = np.arange(8 * depth, dtype=np.int32) + 100
        st._pf_pin_out = np.arange(8 * (cap + 1), dtype=np.int64) + 200
        st._pf_pin_draft = np.arange(8 * 8, dtype=np.int64) + 300
        st._pf_pin_ns = np.array([2] + [0] * 7, dtype=np.int64)
        st._pf_sync = lambda: None
        st._pf_pub_gen = 0
        st._pf_debug = False
        st._pf_log_pairs = 0
        disk = types.SimpleNamespace(w_fd=101, w_off=4096, dim=256, s_fd=102, s_off=8192, sb=8, _pf_expected=None)
        st.engrams = [types.SimpleNamespace(embed_tokens=types.SimpleNamespace(disk=disk, vocab_start_idx=0, vocab_end_idx=1 << 40))]
        return st, disk, k

    def _run(self, st, gen, k):
        calls = []
        out = io.StringIO()
        with mock.patch.object(os, "posix_fadvise", lambda fd, o, n, adv: calls.append(fd)), mock.patch.dict(
            os.environ, {"DSV41_ENGRAM_FADVISE_CAP": "24"}
        ), contextlib.redirect_stdout(out):
            st._prefetch_worker(gen, 1, 4, k)
        self.assertNotIn("worker error", out.getvalue())
        return calls

    def test_worker_fadvises_every_predicted_row(self) -> None:
        st, disk, k = self._stager()
        st._pf_gen = 1
        calls = self._run(st, 1, k)
        rows = disk._pf_expected
        self.assertGreater(len(rows), 24, "enough rows that the old cap would have cut")
        self.assertEqual(calls.count(101), len(rows))
        self.assertEqual(calls.count(102), len(rows))

    def test_superseded_generation_is_dropped(self) -> None:
        st, disk, k = self._stager()
        st._pf_gen = 2  # a newer enqueue already started overwriting the snapshot
        self.assertEqual(self._run(st, 1, k), [])
        self.assertIsNone(disk._pf_expected)

    def test_enqueue_bumps_gen_before_overwriting_snapshot(self) -> None:
        src = engram_prefetch_v3.STAGER_METHODS
        enq = src[src.index("def enqueue_prefetch") : src.index("def _fadvise_row")]
        self.assertEqual(enq.count("self._pf_gen += 1"), 1)
        self.assertLess(enq.index("self._pf_gen += 1"), enq.index("self._pf_pin_ids[:num_tokens].copy_("))
        self.assertNotIn("FADVISE_CAP", src)
        self.assertNotIn("self._pf_tm.tolist()", src[src.index("def _prefetch_worker") :])


if __name__ == "__main__":
    unittest.main()
