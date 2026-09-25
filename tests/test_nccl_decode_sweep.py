#!/usr/bin/env python3
"""CPU tests for tools/nccl_decode_sweep.py and tools/nccl_decode_sweep.sh."""
from __future__ import annotations

import json
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import nccl_decode_sweep as sw  # noqa: E402


class DecodeSizeTests(unittest.TestCase):
    def test_sizes_follow_model_config(self) -> None:
        cfg = {(s["op"], s["rows"]): s["bytes"] for s in sw.decode_specs()}
        self.assertEqual(cfg[("all_reduce", 4)], 4 * 5120 * 2)  # c=1 verify step
        self.assertEqual(cfg[("all_reduce", 8)], 81920)  # c=2 verify step
        self.assertEqual(cfg[("all_reduce", 3)], 30720)  # c=1 draft
        self.assertEqual(cfg[("all_gather", 4)], 4 * 64640 * 2)  # logits per rank
        self.assertEqual(sorted({m for op, m in cfg if op == "all_reduce"}), [1, 3, 4, 6, 8])


class ArmTests(unittest.TestCase):
    def test_keep_equals_recipe_nccl_set(self) -> None:
        env = yaml.load((ROOT / "recipe.yaml").read_text(), Loader=yaml.BaseLoader)["serve"]["env"]
        self.assertEqual(sw.KEEP_ENV, {k: env[k] for k in sw.KEEP_ENV})
        self.assertEqual(sw.BASE_ENV["NCCL_IB_HCA"], env["HCA"])

    def test_base_env_matches_run_sh_docker_args(self) -> None:
        sh = (ROOT / "run.sh").read_text()
        for k, v in sw.BASE_ENV.items():
            if k in ("NCCL_IB_HCA", "NCCL_CROSS_NIC"):
                continue  # $HCA and the FORWARD_ENVS default
            self.assertIn(f'"{k}={v}"', sh, k)
        self.assertIn("NCCL_CROSS_NIC=1", sh)

    def test_every_arm_carries_base_env_and_keep_arm_is_exact(self) -> None:
        for name in sw.ARMS:
            env = sw.arm_env(name)
            for k, v in sw.BASE_ENV.items():
                self.assertEqual(env[k], v, name)
        self.assertEqual(sw.arm_env("keep"), {**sw.BASE_ENV, **sw.KEEP_ENV})
        self.assertNotIn("NCCL_PROTO", sw.arm_env("bare"))
        with self.assertRaises(KeyError):
            sw.arm_env("nope")

    def test_driver_refuses_with_serve_up_and_uses_recipe_image(self) -> None:
        sh = (ROOT / "tools/nccl_decode_sweep.sh").read_text()
        self.assertIn("refusing: GPU holders up", sh)
        image = yaml.load((ROOT / "recipe.yaml").read_text(), Loader=yaml.BaseLoader)["serve"]["env"]["IMAGE"]
        self.assertIn(f'IMAGE="${{IMAGE:-{image}}}"', sh)


class StatsTests(unittest.TestCase):
    def test_quantile_linear(self) -> None:
        xs = list(range(11))
        self.assertEqual(sw.quantile(xs, 0.1), 1.0)
        self.assertEqual(sw.quantile(xs, 0.5), 5.0)
        self.assertAlmostEqual(sw.quantile([0.0, 10.0], 0.9), 9.0)

    def test_summarize_medians_reps_and_models_steps(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for arm, us in (("keep", (20.0, 22.0, 30.0)), ("proto_ll", (10.0, 11.0, 12.0))):
                (root / arm).mkdir()
                for i, u in enumerate(us, 1):
                    row = {"op": "all_reduce", "rows": 4, "bytes": 40960, "mode": "gapped",
                           "median_us": u, "p10_us": u - 1, "p90_us": u + 1, "startup_us": 500.0 + i,
                           "host_launch_with": {"median_us": 300.0}}
                    doc = {"arm": arm, "nccl_version": "2.30.7", "env": {"NCCL_PROTO": "x"}, "rows": [row]}
                    (root / arm / f"rep{i}.rank0.json").write_text(json.dumps(doc))
            res = sw.summarize(root, {"c1": {("all_reduce", 4): 81, ("startup", "all_reduce", 4): 2}})
            keep = res["arms"]["keep"]
            self.assertEqual(keep["reps"], 3)
            cell = keep["cells"][0]
            self.assertEqual((cell["median_us"], cell["startup_us"], cell["host_launch_us"]), (22.0, 502.0, 300.0))
            self.assertAlmostEqual(keep["modeled_ms_per_step"]["c1"], (81 * 22.0 + 2 * 502.0) / 1e3)
            ll = res["arms"]["proto_ll"]["cells"][0]
            self.assertEqual(ll["vs_keep_pct"], -50.0)


class GraphCostTests(unittest.TestCase):
    def _slots(self, n_rep: int, base: list[float]) -> list[list[float]]:
        return [list(base) for _ in range(n_rep)]

    def test_steady_startup_and_launch(self) -> None:
        g = 10
        ref = self._slots(4, [300.0] * g)
        with_ = self._slots(4, [300.0 + 30.0 + (600.0 if s == 0 else 0.0) + (20.0 if s == 1 else 0.0)
                                for s in range(g)])
        c = sw.graph_costs(with_, ref, [900.0] * 4, [100.0] * 4)
        self.assertEqual(statistics.median(c["steady"]), 30.0)  # neighbour slowdown counts
        self.assertEqual(c["first_slots_excess_us"], [600.0, 20.0, 0.0, 0.0, 0.0])
        self.assertEqual(c["startup_us"], 620.0)
        self.assertEqual(c["launch_excess_us"], 800.0)
        self.assertEqual(c["replay_cost"][0], 10 * 30.0 + 620.0)

    def test_rejects_ragged_or_short_graphs(self) -> None:
        with self.assertRaises(ValueError):
            sw.graph_costs([[1.0] * 10, [1.0] * 9], [[1.0] * 10], [0.0], [0.0])
        with self.assertRaises(ValueError):
            sw.graph_costs([[1.0] * 3], [[1.0] * 3], [0.0], [0.0])


if __name__ == "__main__":
    unittest.main()
