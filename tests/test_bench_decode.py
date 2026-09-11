#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import statistics
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_bench():
    path = ROOT / "bench_decode.py"
    spec = importlib.util.spec_from_file_location("bench_decode", path)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load bench_decode.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class BenchDecodeArithmeticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bench = _load_bench()

    def test_decode_tokens_drops_first_completion_token(self) -> None:
        self.assertEqual(self.bench.decode_tokens(200), 199)
        self.assertEqual(self.bench.decode_tokens(1), 0)
        self.assertEqual(self.bench.decode_tokens(0), 0)

    def test_decode_rate_on_fixture_stream(self) -> None:
        # 200 completion tokens, first is TTFT, 19.9 s of decode wall → 10 tok/s.
        self.assertAlmostEqual(self.bench.decode_rate(200, 19.9), 10.0)
        self.assertEqual(self.bench.decode_rate(200, 0.0), 0.0)
        self.assertEqual(self.bench.decode_rate(1, 5.0), 0.0)

    def test_aggregate_rate_on_fixture_wave(self) -> None:
        # Two streams of 200 completion tokens, wall 21 s, median TTFT 1 s.
        # decode tokens = 2 * 199 = 398 over 20 s → 19.9 tok/s.
        rate = self.bench.aggregate_rate([200, 200], 21.0, 1.0)
        self.assertAlmostEqual(rate, 19.9)
        self.assertEqual(self.bench.aggregate_rate([200], 0.5, 1.0), 0.0)

    def test_median_key_matches_statistics_median(self) -> None:
        rows = [{"decode_tok_s": 8.0}, {"decode_tok_s": 10.0}, {"decode_tok_s": 12.0}]
        self.assertEqual(
            self.bench.median_key(rows, "decode_tok_s"),
            statistics.median([8.0, 10.0, 12.0]),
        )

    def test_thinking_kwargs_are_v41(self) -> None:
        src = (ROOT / "bench_decode.py").read_text()
        self.assertIn('"thinking": False', src)
        self.assertIn('"reasoning_effort": "low"', src)
        self.assertIn("deepseek-ai/DeepSeek-V4.1-Flash", src)
        self.assertIn('"ignore_eos": True', src)

    def test_acceptance_empty_when_counters_missing(self) -> None:
        self.assertEqual(self.bench.acceptance(None, None), {})
        before = {"num_drafts": 0.0, "num_draft_tokens": 0.0, "num_accepted_tokens": 0.0}
        after = {"num_drafts": 10.0, "num_draft_tokens": 50.0, "num_accepted_tokens": 20.0}
        got = self.bench.acceptance(before, after)
        self.assertAlmostEqual(got["acceptance_len"], 3.0)
        self.assertAlmostEqual(got["draft_acceptance_rate"], 0.4)


class BenchDecodeStreamParseTests(unittest.TestCase):
    def test_summary_json_shape_is_documented(self) -> None:
        sample = json.dumps(
            [
                {
                    "phase": "prose",
                    "concurrency": 1,
                    "median_decode_tok_s": 10.0,
                    "median_ttft_s": 0.5,
                    "median_agg_tok_s": 10.0,
                    "median_completion_tokens": 200,
                    "n": 3,
                }
            ]
        )
        rows = json.loads(sample)
        self.assertEqual(rows[0]["phase"], "prose")
        self.assertEqual(rows[0]["concurrency"], 1)
        self.assertIn("median_decode_tok_s", rows[0])


if __name__ == "__main__":
    unittest.main()
