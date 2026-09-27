#!/usr/bin/env python3
"""CPU tests for tools/nccl_twin_check.py and tools/nccl_twin_check.sh."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import nccl_twin_check as tc  # noqa: E402


class PlanTests(unittest.TestCase):
    def test_counts_match_the_r3_profile(self) -> None:
        p = tc.plan(4, 3)
        ops = lambda ph, op: sum(1 for x in p[ph] if x[1] == op)  # noqa: E731
        self.assertEqual(ops("target", "all_reduce"), 80)
        self.assertEqual(ops("target", "all_gather"), 2)
        self.assertEqual(ops("draft", "all_reduce"), 7)
        self.assertEqual(ops("draft", "all_gather"), 1)
        self.assertEqual([x[1] for x in p["eager"]], ["all_gather", "all_reduce"])

    def test_sizes_follow_model_config(self) -> None:
        p = tc.plan(4, 3)
        self.assertEqual(p["target"][0][3], 4 * 5120 * 2)  # layer 0: attn AR, MoE AR at m=4
        self.assertEqual(p["target"][1][3], 40960)
        self.assertEqual(p["target"][2][1], "all_gather")  # layer 1 opens with the engram AG
        self.assertEqual(p["draft"][0][3], 30720)  # AR m=3
        self.assertEqual(p["eager"][0][3], 4 * 64640 * 2)  # logits AG per rank
        self.assertEqual(p["draft"][-1][3], 3 * 64640 * 2)
        engram = [x for x in p["target"] if x[1] == "all_gather"]
        self.assertEqual(engram[0][3], 4 * 3072 * 2)  # 24 KiB, the serve's grid-6 AG
        c2 = tc.plan(8, 6)
        self.assertEqual(c2["target"][1][3], 81920)

    def test_seeds_are_exact_in_bf16(self) -> None:
        # bf16 holds every integer up to 256 exactly
        for step in range(200):
            s = tc.seed_of(step)
            self.assertTrue(1 <= s <= 40)
            self.assertLessEqual(tc.expected_ar(s), 256)
            self.assertLessEqual(max(tc.expected_ag_halves(s)), 256)
        self.assertEqual(tc.expected_ar(5), 15)
        self.assertEqual(tc.expected_ag_halves(5), [5, 6])

    def test_expected_segments_cover_every_output(self) -> None:
        p = tc.plan(4, 3)
        items = p["target"] + p["draft"] + p["eager"]
        runs = tc.expected_segments(items, 2)
        n_ag = sum(1 for x in items if x[1] == "all_gather")
        self.assertEqual(len(runs), len(items) + n_ag)  # an all-gather output is one run per rank
        want = sum(x[3] // 2 * (2 if x[1] == "all_gather" else 1) for x in items)
        self.assertEqual(sum(n for n, _, _ in runs), want)
        self.assertEqual(runs[0], (4 * 5120, 3, 0))  # layer 0 attn AR: seed * (1 + 2)
        ag = items.index(next(x for x in items if x[1] == "all_gather"))
        self.assertEqual(runs[ag:ag + 2], [(4 * 3072, 1, 0), (4 * 3072, 1, 1)])  # engram AG halves
        for seed in (1, 17, 40):  # the pattern reproduces the scalar expectations
            self.assertEqual(runs[0][1] * seed + runs[0][2], tc.expected_ar(seed))
            self.assertEqual([c * seed + o for _, c, o in runs[ag:ag + 2]], tc.expected_ag_halves(seed))
        self.assertEqual(tc.expected_segments(items, 1)[0], (4 * 5120, 1, 0))  # one-rank selftest

    def test_checked_steps(self) -> None:
        self.assertTrue(tc.is_checked(0, 10, 330))
        self.assertTrue(tc.is_checked(329, 10, 330))
        self.assertFalse(tc.is_checked(5, 10, 330))
        self.assertFalse(tc.is_checked(-1, 10, 330))


class ArmTests(unittest.TestCase):
    def test_base_env_is_the_serve_nccl_env(self) -> None:
        env = yaml.load((ROOT / "recipe.yaml").read_text(), Loader=yaml.BaseLoader)["serve"]["env"]
        for k in ("NCCL_BUFFSIZE", "NCCL_LL128_BUFFSIZE", "NCCL_PROTO", "NCCL_MAX_NCHANNELS"):
            self.assertEqual(tc.BASE_ENV[k], env[k], k)
        self.assertEqual(tc.BASE_ENV["NCCL_IB_HCA"], env["HCA"])
        sh = (ROOT / "run.sh").read_text()
        for k in ("NCCL_NET", "NCCL_IB_DISABLE", "NCCL_NVLS_ENABLE", "NCCL_CUMEM_ENABLE"):
            self.assertIn(f'"{k}={tc.BASE_ENV[k]}"', sh, k)

    def test_arms(self) -> None:
        self.assertEqual(tc.DEFAULT_ARMS, ("keep", "twin"))
        self.assertNotIn("NCCL_GRAPH_MIXING_SUPPORT", tc.arm_env("keep"))
        self.assertEqual(tc.arm_env("twin")["NCCL_GRAPH_MIXING_SUPPORT"], "0")
        self.assertEqual(tc.arm_env("twin_so0")["NCCL_GRAPH_STREAM_ORDERING"], "0")
        self.assertNotIn("mix0_single", tc.DEFAULT_ARMS)
        self.assertEqual(tc.arm_env("keep_qos")["DSV41_PM_QOS_US"], "20")
        self.assertNotIn("NCCL_GRAPH_MIXING_SUPPORT", tc.arm_env("keep_qos"))
        self.assertEqual(tc.arm_env("twin_qos")["NCCL_GRAPH_MIXING_SUPPORT"], "0")
        with self.assertRaises(KeyError):
            tc.arm_env("nope")

    def test_driver_refuses_busy_gpus_and_waits_for_foreign_processes(self) -> None:
        sh = (ROOT / "tools/nccl_twin_check.sh").read_text()
        self.assertIn("refusing: GPU holders up", sh)
        self.assertIn("foreign GPU process", sh)
        self.assertIn("nvidia-smi pmon -c 1", sh)
        image = yaml.load((ROOT / "recipe.yaml").read_text(), Loader=yaml.BaseLoader)["serve"]["env"]["IMAGE"]
        self.assertIn(f'IMAGE="${{IMAGE:-{image}}}"', sh)
        self.assertIn("docker/patch/nccl_eager_twin.py", sh)
        self.assertIn('"$ROOT/docker/patch/pm_qos.py"', sh)
        # PM QoS arms need root and the device; every other arm runs as the caller
        self.assertIn("then who=(--device /dev/cpu_dma_latency); fi", sh)
        # hard stop on a hang: SIGKILL 10 s after the timeout, tini forwards the TERM to python
        self.assertEqual(sh.count('timeout -k 10 "$TIMEOUT_S"'), 2)
        self.assertIn("local a=(--rm --init ", sh)


class CompareTests(unittest.TestCase):
    def test_compare_medians_reps_and_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for arm, meds in (("keep", (60000.0, 61000.0, 62000.0)), ("twin", (57000.0, 58000.0, 59000.0))):
                (root / arm).mkdir()
                for i, m in enumerate(meds, 1):
                    doc = {"arm": arm, "step_us": {"median": m, "p10": m - 500, "p90": m + 900}, "n_mismatches": 0}
                    (root / arm / f"rep{i}.rank0.json").write_text(json.dumps(doc))
            res = tc.compare(root)
        self.assertEqual(res["keep"]["step_us_median_of_medians"], 61000.0)
        self.assertEqual(res["twin"]["delta_vs_keep_ms"], -3.0)
        self.assertEqual(res["twin"]["mismatches"], 0)
        self.assertEqual(res["twin"]["device_checked_steps"], 0)  # older JSONs: no device check


if __name__ == "__main__":
    unittest.main()
