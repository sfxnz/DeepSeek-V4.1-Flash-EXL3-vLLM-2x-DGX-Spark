#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import importlib.util
import inspect
import io
import json
import statistics
import sys
import unittest
from pathlib import Path
from unittest import mock

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
        self.assertIn('"ignore_eos": ignore_eos', src)
        # Frozen cells keep ignore_eos on by default.
        sig = inspect.signature(self.bench.stream_one)
        self.assertIs(sig.parameters["ignore_eos"].default, True)

    def test_acceptance_empty_when_counters_missing(self) -> None:
        self.assertEqual(self.bench.acceptance(None, None), {})
        before = {"num_drafts": 0.0, "num_draft_tokens": 0.0, "num_accepted_tokens": 0.0}
        after = {"num_drafts": 10.0, "num_draft_tokens": 50.0, "num_accepted_tokens": 20.0}
        got = self.bench.acceptance(before, after)
        self.assertAlmostEqual(got["acceptance_len"], 3.0)
        self.assertAlmostEqual(got["draft_acceptance_rate"], 0.4)


class BenchDecodeHonestyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bench = _load_bench()

    def test_percentile_is_linear_interpolation(self) -> None:
        xs = [5.0, 1.0, 3.0, 2.0, 4.0]
        self.assertEqual(self.bench.percentile(xs, 50), 3.0)
        self.assertAlmostEqual(self.bench.percentile(xs, 90), 4.6)
        self.assertAlmostEqual(self.bench.percentile(xs, 99), 4.96)
        self.assertEqual(self.bench.percentile([7.0], 99), 7.0)
        self.assertIsNone(self.bench.percentile([], 50))
        got = self.bench.pct_fields(xs, "ttft_s")
        self.assertEqual(sorted(got), ["ttft_s_p50", "ttft_s_p90", "ttft_s_p99"])

    def test_inter_chunk_gaps_in_ms(self) -> None:
        gaps = self.bench.inter_chunk_ms([1.0, 1.065, 1.2])
        self.assertEqual(len(gaps), 2)
        self.assertAlmostEqual(gaps[0], 65.0)
        self.assertAlmostEqual(gaps[1], 135.0)
        self.assertEqual(self.bench.inter_chunk_ms([1.0]), [])

    def test_post_eos_fraction_on_audit_probe(self) -> None:
        # Audit probe: prose stops at 78 tokens; the frozen cell runs to 200.
        self.assertAlmostEqual(self.bench.post_eos_fraction(200, 78, "stop"), 0.61)
        # Probe hit max_tokens: no EOS seen, nothing is post-EOS.
        self.assertEqual(self.bench.post_eos_fraction(200, 200, "length"), 0.0)
        self.assertEqual(self.bench.post_eos_fraction(200, 250, "stop"), 0.0)
        self.assertEqual(self.bench.post_eos_fraction(0, 78, "stop"), 0.0)

    def test_ms_per_step_matches_live_step_time(self) -> None:
        # 200 tokens at 39.6 tok/s with acceptance 2.61 -> ~65.9 ms per verify
        # step (the real DSpark-3 step is ~64-68 ms, not 25 ms).
        decode_s = 199 / 39.6
        got = self.bench.ms_per_step(200, decode_s, 2.61)
        self.assertAlmostEqual(got, 1000 * decode_s / (199 / 2.61))
        self.assertTrue(64.0 < got < 68.0)
        self.assertIsNone(self.bench.ms_per_step(200, decode_s, None))
        self.assertIsNone(self.bench.ms_per_step(1, decode_s, 2.0))
        self.assertIsNone(self.bench.ms_per_step(200, 0.0, 2.0))

    def test_cell_summary_per_run_acceptance_and_percentiles(self) -> None:
        rows = [
            {"decode_tok_s": 30.0, "ttft_s": 0.2, "completion_tokens": 200,
             "decode_s": 199 / 30.0, "inter_chunk_ms": [60.0, 70.0],
             "finish_reason": "length", "acceptance_len": 2.0},
            {"decode_tok_s": 40.0, "ttft_s": 0.4, "completion_tokens": 200,
             "decode_s": 199 / 40.0, "inter_chunk_ms": [80.0],
             "finish_reason": "length", "acceptance_len": 3.0},
        ]
        run_accs = [{"acceptance_len": 2.0}, {"acceptance_len": 3.0}]
        pooled = {"acceptance_len": 2.5, "draft_acceptance_rate": 0.5}
        got = self.bench.cell_summary("prose_long", 1, rows, [30.0, 40.0], run_accs, pooled)
        self.assertEqual(got["run_acceptance_len"], [2.0, 3.0])
        self.assertEqual(got["median_run_acceptance_len"], 2.5)
        self.assertEqual(got["acceptance_len"], 2.5)  # pooled field kept
        # ms/step: 1000/30*2 = 66.67 and 1000/40*3 = 75 -> median 70.83
        self.assertAlmostEqual(got["median_ms_per_step"], (2000 / 30 + 75.0) / 2)
        self.assertEqual(got["inter_chunk_ms_p50"], 70.0)
        self.assertAlmostEqual(got["ttft_s_p50"], 0.3)
        self.assertEqual(got["finish_reasons"], {"length": 2})
        self.assertEqual(got["n"], 2)

    def test_cell_summary_without_metrics(self) -> None:
        rows = [{"decode_tok_s": 30.0, "ttft_s": 0.2, "completion_tokens": 200,
                 "decode_s": 6.6, "inter_chunk_ms": [], "finish_reason": None}]
        got = self.bench.cell_summary("prose", 1, rows, [30.0], [{}], {})
        self.assertIsNone(got["median_run_acceptance_len"])
        self.assertIsNone(got["median_ms_per_step"])
        self.assertIsNone(got["inter_chunk_ms_p50"])
        self.assertNotIn("acceptance_len", got)

    def test_natural_fields(self) -> None:
        nat = self.bench.natural_fields({"completion_tokens": 78, "finish_reason": "stop"}, 200)
        self.assertEqual(nat["natural_completion_tokens"], 78)
        self.assertFalse(nat["natural_covers_max"])
        nat = self.bench.natural_fields({"completion_tokens": 200, "finish_reason": "length"}, 200)
        self.assertTrue(nat["natural_covers_max"])

    def test_phase_groups_keep_both_meaning(self) -> None:
        self.assertEqual(self.bench.PHASE_GROUPS["both"], ["prose", "structured"])
        self.assertIn("prose_long", self.bench.PHASE_GROUPS["all"])
        self.assertIn("prose_long", self.bench.PHASES)

    def test_stream_one_records_finish_reason_and_chunk_gaps(self) -> None:
        events = [
            {"choices": [{"delta": {"content": "A"}}]},
            {"choices": [{"delta": {"content": "B"}}]},
            {"choices": [{"delta": {"content": "C"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 3}},
        ]
        lines = [f"data: {json.dumps(e)}\n".encode() for e in events] + [b"data: [DONE]\n"]
        sent = {}

        @contextlib.contextmanager
        def fake_urlopen(req, timeout=None):
            sent.update(json.loads(req.data.decode()))
            yield iter(lines)

        with mock.patch.object(self.bench.urllib.request, "urlopen", fake_urlopen):
            row = self.bench.stream_one("http://x/v1/chat/completions", "m", "p", 200,
                                        ignore_eos=False)
        self.assertFalse(sent["ignore_eos"])
        self.assertEqual(row["finish_reason"], "stop")
        self.assertEqual(row["chunks"], 3)
        self.assertEqual(len(row["inter_chunk_ms"]), 2)
        self.assertEqual(row["completion_tokens"], 3)
        self.assertIn("decode_s", row)

    def test_main_probes_natural_length_after_measured_runs(self) -> None:
        calls = []
        counters = {"n": 0.0}

        def fake_stream(url, model, prompt, max_tokens, ignore_eos=True):
            calls.append(ignore_eos)
            n = 200 if ignore_eos else 78
            return {"ttft_s": 0.1, "total_s": 5.1, "prompt_tokens": 20,
                    "completion_tokens": n, "decode_s": 5.0,
                    "decode_tok_s": self.bench.decode_rate(n, 5.0), "chunks": 80,
                    "inter_chunk_ms": [60.0] * 79,
                    "finish_reason": "length" if ignore_eos else "stop"}

        def fake_counters(url):
            counters["n"] += 1
            k = counters["n"]
            return {"num_drafts": 10 * k, "num_draft_tokens": 30 * k,
                    "num_accepted_tokens": 16 * k}

        argv = ["bench_decode.py", "--phase", "prose", "--concurrency", "1", "--runs", "2"]
        out = io.StringIO()
        with mock.patch.object(self.bench, "stream_one", fake_stream), \
                mock.patch.object(self.bench, "spec_counters", fake_counters), \
                mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(out):
            self.assertEqual(self.bench.main(), 0)
        self.assertEqual(calls, [True, True, False])  # probe comes last
        text = out.getvalue()
        rows = json.loads(text[text.rindex("SUMMARY ") + 8:])
        row = rows[0]
        self.assertEqual(row["natural_completion_tokens"], 78)
        self.assertEqual(row["natural_finish_reason"], "stop")
        self.assertAlmostEqual(row["post_eos_fraction"], 0.61)
        self.assertEqual(len(row["run_acceptance_len"]), 2)
        self.assertIsNotNone(row["median_ms_per_step"])
        self.assertIn("ms_step=[", text)


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
