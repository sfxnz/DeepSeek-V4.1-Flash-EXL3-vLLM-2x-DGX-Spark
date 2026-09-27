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

    def _run(self, st, gen, k, num_reqs=1, num_tokens=4):
        calls = []
        out = io.StringIO()
        with mock.patch.object(os, "posix_fadvise", lambda fd, o, n, adv: calls.append(fd)), mock.patch.dict(
            os.environ, {"DSV41_ENGRAM_FADVISE_CAP": "24"}
        ), contextlib.redirect_stdout(out):
            st._prefetch_worker(gen, num_reqs, num_tokens, k, k + 1)
        self.assertNotIn("worker error", out.getvalue())
        self.stdout = out.getvalue()
        return calls

    def _two_requests(self, ns):
        """Snapshot of a c=2 verify step: 4 rows each, sampled rows at stride k + 1."""
        st, disk, k = self._stager()
        st._pf_pin_ids[:8] = [11, 12, 13, 14, 21, 22, 23, 24]
        st._pf_pin_pos[:8] = [20, 21, 22, 23, 50, 51, 52, 53]
        st._pf_pin_qsl[:3] = [0, 4, 8]
        st._pf_pin_ns[:2] = ns
        st._pf_gen = 1
        seen = []
        real = st._pf_next_chunk

        def spy(r, A, ids, q0, q1, outs_r, drafts_r, win_r, S):
            seen.append((r, A, list(outs_r), S))
            return real(r, A, ids, q0, q1, outs_r, drafts_r, win_r, S)

        st._pf_next_chunk = spy
        return st, disk, k, seen

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

    def test_second_request_reads_its_own_sampled_row(self) -> None:
        # sampled_token_ids is [num_reqs, k + 1] flattened: request 1's row
        # starts at k + 1 = 4, not at _pf_cap = 64 (never written).
        st, disk, k, seen = self._two_requests([2, 3])
        self._run(st, 1, k, num_reqs=2, num_tokens=8)
        self.assertEqual([(r, A, row, S) for r, A, row, S in seen],
                         [(0, 2, [200, 201, 202, 203], 20), (1, 3, [204, 205, 206, 207], 50)])
        self.assertTrue(disk._pf_expected)

    def test_bad_num_sampled_skips_the_request_without_an_error(self) -> None:
        # num_sampled > k + 1 (a bad snapshot) raised IndexError at
        # outs_r[A - 1]; 3 of those disabled prefetch for the process.
        st, disk, k, seen = self._two_requests([70, 2])
        for _ in range(4):
            self._run(st, 1, k, num_reqs=2, num_tokens=8)
        self.assertEqual({(r, A) for r, A, *_ in seen}, {(1, 2)})
        self.assertNotIn("prefetch disabled", self.stdout)
        self.assertEqual(st._pf_bad, 4)
        self.assertTrue(disk._pf_expected)
        self.assertFalse(hasattr(st, "_pf_err_count"))

    def test_enqueue_keeps_side_stream_sources_alive(self) -> None:
        src = engram_prefetch_v3.STAGER_METHODS
        enq = src[src.index("def enqueue_prefetch") : src.index("def _fadvise_row")]
        self.assertIn("_src.record_stream(self._pf_stream)", enq)
        rec = enq[enq.index("for _src in") : enq.index("_src.record_stream")]
        for name in ("input_ids", "positions", "query_start_loc", "window",
                     "sampled_token_ids", "num_sampled", "draft_tokens"):
            self.assertIn(name, rec)
        self.assertLess(enq.index("self._pf_event.record()"), enq.index("for _src in"))
        self.assertIn("sampled_token_ids[:num_reqs, :w]", enq)
        self.assertIn("num_tokens, k, w", enq)

    def test_enqueue_bumps_gen_before_overwriting_snapshot(self) -> None:
        src = engram_prefetch_v3.STAGER_METHODS
        enq = src[src.index("def enqueue_prefetch") : src.index("def _fadvise_row")]
        self.assertEqual(enq.count("self._pf_gen += 1"), 1)
        self.assertLess(enq.index("self._pf_gen += 1"), enq.index("self._pf_pin_ids[:num_tokens].copy_("))
        self.assertNotIn("FADVISE_CAP", src)
        self.assertNotIn("self._pf_tm.tolist()", src[src.index("def _prefetch_worker") :])


if __name__ == "__main__":
    unittest.main()
