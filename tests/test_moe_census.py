#!/usr/bin/env python3
from __future__ import annotations

import base64
import io
import random
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import moe_census as mc  # noqa: E402


def _random_rows(rng: np.random.Generator, t: int, k: int = 6, n: int = 384) -> np.ndarray:
    return np.stack([rng.choice(n, size=k, replace=False) for _ in range(t)])


class WindowDupTests(unittest.TestCase):
    def test_identical_rows(self) -> None:
        ids = np.tile(np.arange(6), (4, 1))
        self.assertAlmostEqual(mc.window_dup(ids, 4)[0], 1 - 6 / 24)

    def test_disjoint_rows(self) -> None:
        ids = np.arange(24).reshape(4, 6)
        np.testing.assert_array_equal(mc.window_dup(ids, 4), [0.0])

    def test_sliding_windows(self) -> None:
        # Rows 0,1 share 3 experts; rows 2,3 are fresh. m=2 -> 3 windows.
        ids = np.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 6, 7, 8], [9, 10, 11, 12, 13, 14], [15, 16, 17, 18, 19, 20]])
        np.testing.assert_allclose(mc.window_dup(ids, 2), [3 / 12, 0.0, 0.0])
        self.assertEqual(mc.window_dup(ids[:1], 2).size, 0)

    def test_random_baseline_matches_monte_carlo(self) -> None:
        rng = np.random.default_rng(5)
        ids = _random_rows(rng, 4000)
        self.assertAlmostEqual(mc.window_dup(ids, 4).mean(), mc.random_dup(4), delta=0.004)
        # 384 experts, top-6, m=4: ~2.3% by chance (audit corrections).
        self.assertAlmostEqual(mc.random_dup(4), 0.023, delta=0.001)


class CensusTests(unittest.TestCase):
    def _arr(self, dup_layer: bool) -> np.ndarray:
        rng = np.random.default_rng(1)
        t = 64
        arr = np.zeros((t, 3, 6), dtype=np.uint16)
        arr[:, 0, :] = _random_rows(rng, t)
        # Layer 1: every token picks the same six experts.
        arr[:, 1, :] = np.arange(6) if dup_layer else _random_rows(rng, t)
        # Layer 2 stays zero: never captured (dense layer).
        return arr

    def test_routed_layers_drop_zero_layers(self) -> None:
        self.assertEqual(mc.routed_layers(self._arr(True)), [0, 1])

    def test_census_per_layer_and_verdict(self) -> None:
        res = mc.census({"lail_r0": self._arr(True)}, 4)
        self.assertEqual(sorted(res["per_layer_dup"]), [0, 1])
        self.assertAlmostEqual(res["per_layer_dup"][1], 0.75)
        self.assertLess(res["per_layer_dup"][0], 0.08)
        self.assertAlmostEqual(res["dup_mean_over_layers"], (res["per_layer_dup"][0] + 0.75) / 2)
        self.assertTrue(res["verdict"].startswith("PROCEED"))
        self.assertAlmostEqual(res["upper_ms_per_step"], res["dup_mean_over_layers"] * 22.4)

    def test_random_routing_says_stop(self) -> None:
        res = mc.census({"code0_r0": self._arr(False)}, 4)
        self.assertTrue(res["verdict"].startswith("STOP"), res["verdict"])

    def test_impact_numbers(self) -> None:
        imp = mc.impact(0.10)
        self.assertAlmostEqual(imp["upper_ms_per_step"], 2.24)
        self.assertAlmostEqual(imp["realistic_ms_per_step"], 2.24 * 0.78)
        self.assertTrue(mc.verdict(0.09).startswith("MARGINAL"))

    def test_decode_b64_roundtrip(self) -> None:
        arr = self._arr(True)
        buf = io.BytesIO()
        np.save(buf, arr, allow_pickle=False)
        out = mc.decode_b64(base64.b64encode(buf.getvalue()).decode())
        np.testing.assert_array_equal(out, arr)


class SynthRoutingTests(unittest.TestCase):
    def test_exact_dup_rates(self) -> None:
        rng = random.Random(0)
        for dup in (0.0, 0.25, 0.5):
            for _ in range(20):
                rows = mc.synth_routing(4, 6, 384, dup, rng)
                ids = np.array(rows)
                self.assertTrue(all(len(set(r)) == 6 for r in rows))
                self.assertAlmostEqual(mc.window_dup(ids, 4)[0], dup)

    def test_too_many_repeats(self) -> None:
        with self.assertRaises(ValueError):
            mc.synth_routing(2, 6, 384, 0.75, random.Random(0))


class CensusBootTests(unittest.TestCase):
    def test_routed_experts_flag_reaches_both_ranks(self) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import run_sh_harness as h

        res = h.dry_run(EXTRA_ARGS="--enable-return-routed-experts")
        self.assertEqual(res["returncode"], 0, res["stdout"] + res["stderr"])
        for role in ("head", "worker"):
            _, args = h.image_and_args(res[role])
            self.assertEqual(args[-1], "--enable-return-routed-experts", role)


if __name__ == "__main__":
    unittest.main()
