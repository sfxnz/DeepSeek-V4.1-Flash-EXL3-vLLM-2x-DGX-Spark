#!/usr/bin/env python3
"""CPU tests for tools/requant_probe.py (the GPU part runs in the campaign)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import requant_probe as rq  # noqa: E402

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


class PackageVersionTests(unittest.TestCase):
    def test_reads_distribution_metadata_not_module_attr(self) -> None:
        # Round 34 s1: exllamav3 1.5.1 has no __version__, so the JSON said '?'.
        import importlib.metadata as md

        self.assertEqual(rq.package_version("pip"), md.version("pip"))
        self.assertEqual(rq.package_version("no-such-dist-dsv41"), "?")
        src = (ROOT / "tools/requant_probe.py").read_text()
        self.assertIn('"exllamav3": package_version("exllamav3")', src)
        self.assertNotIn("__version__", src.split("def main", 1)[1])


@unittest.skipIf(np is None, "numpy not installed")
class RequantProbeTests(unittest.TestCase):
    def test_refit_matches_closed_form_and_never_hurts(self) -> None:
        rng = np.random.default_rng(0)
        w = rng.standard_normal((64, 48))
        q = (w + 0.4 * rng.standard_normal(w.shape)) * rng.uniform(0.5, 2.0, (64, 1))
        q2, r, c = rq.refit_identity(w, q)
        np.testing.assert_allclose(q2, r[:, None] * q * c[None, :], rtol=1e-10)
        err = lambda x: np.linalg.norm(x - w)  # noqa: E731
        self.assertLess(err(q2), err(q))
        # One more column step barely moves it: the refit is near a fixed point.
        c3 = (q2 * w).sum(0) / (q2 * q2).sum(0)
        self.assertLess(abs(err(q2 * c3) - err(q2)) / err(q2), 1e-2)

    def test_refit_guards_zero_rows(self) -> None:
        w = np.ones((4, 3))
        q = np.ones((4, 3))
        q[1] = 0.0
        _, r, c = rq.refit_identity(w, q)
        self.assertTrue(np.all(np.isfinite(r)) and np.all(r > 0) and np.all(np.isfinite(c)))

    def test_expert_order_is_numeric(self) -> None:
        names = [f"layers.0.ffn.experts.{e}.w{k}.weight" for e in (10, 2, 1) for k in (1, 2, 3)]
        self.assertEqual(rq.expert_stems(names, 2), ["layers.0.ffn.experts.1", "layers.0.ffn.experts.2"])

    def test_full_pack_hours_matches_history(self) -> None:
        # 353eb28: Viterbi ~4.1 s/tensor -> ~26 h across both Sparks.
        self.assertAlmostEqual(rq.full_pack_hours(4.1, 2), 26.2, delta=0.2)
        # REBUILD-PLAN.md: greedy 16.4 min per 1152-tensor shard, 20 shards per node.
        self.assertAlmostEqual(rq.full_pack_hours(16.4 * 60 / 1152, 2), 5.47, delta=0.05)


if __name__ == "__main__":
    unittest.main()
