#!/usr/bin/env python3
"""CPU tests for tools/draft_subvocab_coverage.py and the research results dir."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


@unittest.skipIf(np is None, "numpy not installed")
class SubvocabTests(unittest.TestCase):
    def setUp(self) -> None:
        import draft_subvocab_coverage as sv

        self.sv = sv

    def test_rank_breaks_ties_by_lower_id(self) -> None:
        counts = np.array([0, 5, 5, 1, 0])
        self.assertEqual(self.sv.rank_ids(counts).tolist(), [1, 2, 3, 0, 4])

    def test_tp_topn_stays_inside_each_shard(self) -> None:
        counts = np.array([9, 8, 7, 6, 0, 0, 0, 1])  # shard 1 is cold
        keep = self.sv.topn_tp(counts, 4)
        self.assertEqual(keep.tolist(), [0, 1, 7, 4])
        self.assertEqual(self.sv.topn_global(counts, 4).tolist(), [0, 1, 2, 3])

    def test_coverage(self) -> None:
        ids = np.array([0, 0, 1, 3])
        self.assertAlmostEqual(self.sv.coverage(ids, np.array([0]), vocab=4), 0.5)

    def test_alpha_roundtrip_and_loss(self) -> None:
        a = self.sv.alpha_for(2.3, 3)
        self.assertAlmostEqual(self.sv.accept_len(a, 3), 2.3, places=6)
        self.assertLess(self.sv.accept_len(a, 3, 0.9), self.sv.accept_len(a, 3, 0.99))


class ResultsTests(unittest.TestCase):
    def test_committed_research_results_are_small_json(self) -> None:
        d = ROOT / "results/2026-09-24-review/research"
        for p in d.rglob("*"):
            if p.is_file():
                self.assertLess(p.stat().st_size, 5 << 20, p)
                if p.suffix == ".json":
                    json.loads(p.read_text())


if __name__ == "__main__":
    unittest.main()
