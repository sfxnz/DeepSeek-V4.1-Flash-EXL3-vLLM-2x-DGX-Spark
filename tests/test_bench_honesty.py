#!/usr/bin/env python3
"""CPU-only tests for the bench-honesty tooling: novel corpus, micro phases,
four-numbers provenance parsing, warm-prefix math, e2e exit status."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "benches"))


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _summary(text: str):
    return json.loads(text[text.rindex("SUMMARY ") + 8:])


class NovelCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = _load("tools/corpus.py", "corpus_t")

    def test_novel_doc_is_deterministic_and_seeded(self) -> None:
        a = self.corpus.build_doc(2000, 0.25, seed=11, novel=True)
        self.assertEqual(a, self.corpus.build_doc(2000, 0.25, seed=11, novel=True))
        self.assertNotEqual(a, self.corpus.build_doc(2000, 0.25, seed=12, novel=True))
        self.assertGreaterEqual(len(a), 2000 / 0.25)

    def test_novel_doc_is_not_repo_text(self) -> None:
        doc = self.corpus.build_doc(4000, 0.25, seed=3, novel=True)
        repo = set(self.corpus._repo_segments())
        self.assertFalse(set(doc.split("\n\n")) & repo)
        for marker in ("DSV41", "run.sh", "FORCE_UNSAFE", "```"):
            self.assertNotIn(marker, doc)

    def test_novel_seeds_share_few_word_trigrams(self) -> None:
        def trigrams(text):
            w = text.split()
            return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}

        a = trigrams(self.corpus.build_doc(4000, 0.25, seed=1, novel=True))
        b = trigrams(self.corpus.build_doc(4000, 0.25, seed=2, novel=True))
        self.assertLess(len(a & b) / len(a), 0.02)

    def test_default_mode_unchanged_repo_text(self) -> None:
        doc = self.corpus.build_doc(1000, 0.25, seed=7)
        segs = set(self.corpus._repo_segments())
        self.assertTrue(set(doc.split("\n\n")) & segs)

    def test_char_ratio_novel_uses_novel_sample(self) -> None:
        seen = []
        self.corpus.char_ratio(lambda t: seen.append(t) or len(t) // 4, novel=True)
        self.assertNotIn("DSV41", seen[0])


class MicroPhaseTests(unittest.TestCase):
    def test_micro_reports_pp_warm_and_pp_novel(self) -> None:
        micro = _load("benches/micro.py", "micro_t")
        docs = []

        def fake_stream(url, model, prompt, max_tokens, timeout):
            docs.append(prompt)
            return {"ttft_s": 2.0, "wall_s": 3.0, "prompt_tokens": 1000,
                    "completion_tokens": max_tokens, "decode_s": 1.0}

        argv = ["micro.py", "--contexts", "1024", "--runs", "1"]
        out = io.StringIO()
        with mock.patch.object(micro, "make_tokenizer", lambda *a: lambda t: len(t) // 4), \
                mock.patch.object(micro, "stream_once", fake_stream), \
                mock.patch.object(micro, "spec_counters", lambda u: None), \
                mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(out):
            self.assertEqual(micro.main(), 0)
        rows = _summary(out.getvalue())
        self.assertEqual([r["phase"] for r in rows], ["pp_warm", "pp_novel", "tg"])
        self.assertEqual(rows[0]["median_rate_tok_s"], 500.0)
        self.assertNotIn("DSV41", docs[1])  # pp_novel doc is not repo text


class FourNumbersParseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fn = _load("tools/four_numbers_parse.py", "four_numbers_parse_t")

    def test_filter_env_keeps_levers_and_drops_secrets(self) -> None:
        env = ["HF_TOKEN=hf_x", "PATH=/bin", "VLLM_API_KEY=s", "NCCL_IB_HCA=rocep1s0f1",
               "DSV41_LMHEAD_MXFP8=1", "VLLM_HOST_IP=10.0.0.1", "HUGGING_FACE_HUB_TOKEN=y",
               "DSV41_PREFILL_EMPTY_CACHE_TOKENS=8192", "DSV41_DSPARK_REFINE_PASS=0",
               "VLLM_X_SECRET=s", "NCCL_X_PASSWORD=s"]
        got = self.fn.filter_env(env)
        self.assertEqual(got, ["DSV41_DSPARK_REFINE_PASS=0", "DSV41_LMHEAD_MXFP8=1",
                               "DSV41_PREFILL_EMPTY_CACHE_TOKENS=8192",
                               "NCCL_IB_HCA=rocep1s0f1", "VLLM_HOST_IP=10.0.0.1"])
        self.assertEqual(self.fn.env_digest(got), self.fn.env_digest(list(reversed(got))))

    def test_filter_env_cli(self) -> None:
        out = io.StringIO()
        with mock.patch.object(sys, "argv", ["x", "filter-env"]), \
                mock.patch.object(sys, "stdin", io.StringIO('["HF_TOKEN=a","NCCL_X=1"]')), \
                contextlib.redirect_stdout(out):
            self.assertEqual(self.fn.main(), 0)
        self.assertEqual(json.loads(out.getvalue()), ["NCCL_X=1"])

    def test_env_rank_diff_ignores_host_ip(self) -> None:
        a = ["VLLM_HOST_IP=10.100.8.1", "DSV41_X=1", "NCCL_Y=2"]
        b = ["VLLM_HOST_IP=10.100.8.2", "DSV41_X=1", "NCCL_Y=2"]
        self.assertEqual(self.fn.env_rank_diff(a, b), [])
        self.assertEqual(self.fn.env_rank_diff(a, b[:2]), ["NCCL_Y"])
        self.assertEqual(self.fn.env_rank_diff(a, ["DSV41_X=0", "NCCL_Y=2"]), ["DSV41_X"])

    def test_parse_time_docker_ns(self) -> None:
        t = self.fn.parse_time("2026-09-22T11:55:54.283742277Z\n")
        self.assertEqual(t.isoformat(), "2026-09-22T11:55:54.283742+00:00")
        self.assertIsNotNone(self.fn.parse_time("2026-09-24T12:00:00Z"))
        self.assertIsNone(self.fn.parse_time(""))

    def test_parse_host_state(self) -> None:
        text = ("== spark1 ==\nuptime_s 422127.47\nMemAvailable:   22322360 kB\n"
                "Cached:         19599248 kB\nInactive(file): 19459948 kB\n"
                "== spark2 ==\nuptime_s 9.5\nCached: 1048576 kB\n")
        got = self.fn.parse_host_state(text)
        self.assertEqual(got["spark1"]["uptime_s"], 422127.47)
        self.assertEqual(got["spark1"]["Cached_gib"], 18.69)
        self.assertIn("Inactive_file_gib", got["spark1"])
        self.assertEqual(got["spark2"], {"uptime_s": 9.5, "Cached_gib": 1.0})

    def test_parse_full_capture_with_canned_logs(self) -> None:
        prose_row = {"phase": "prose", "concurrency": 1, "median_decode_tok_s": 39.6,
                     "acceptance_len": 2.61, "median_ms_per_step": 65.9,
                     "post_eos_fraction": 0.61, "natural_finish_reason": "stop"}
        long_rows = [{"phase": "prose_long", "concurrency": c, "median_decode_tok_s": 27.0,
                      "natural_finish_reason": "length"} for c in (1, 2)]
        micro = [{"ctx": 8192, "phase": "pp", "median_rate_tok_s": 690.0},
                 {"ctx": 8192, "phase": "pp_novel", "median_rate_tok_s": 250.0},
                 {"ctx": 32768, "phase": "pp_warm", "median_rate_tok_s": 600.0},
                 {"ctx": 32768, "phase": "pp_novel", "median_rate_tok_s": 240.0},
                 {"ctx": 8192, "phase": "tg", "median_rate_tok_s": 30.0}]
        warm = {"cell": "warm_prefix", "prompt_tokens": 2050, "hits_warm": 1920,
                "expected_hits": 1920, "hit_fraction_of_expected": 1.0}
        files = {
            "00-header.log": "arm=x ts=2026-09-24T12:00:00Z host=spark1 out=o\n",
            "serve_env_spark1.json": json.dumps(["DSV41_X=1", "VLLM_HOST_IP=10.100.8.1"]),
            "serve_env_spark2.json": json.dumps(["DSV41_X=1", "VLLM_HOST_IP=10.100.8.2"]),
            "serve_started_spark1.txt": "2026-09-24T11:00:00.5Z\n",
            "serve_started_spark2.txt": "2026-09-24T10:59:30Z\n",
            "00-host-state.log": "== spark1 ==\nuptime_s 100.0\nCached: 2097152 kB\n",
            "01-prose.log": ("phase=prose c=1 run=1 wall=5s agg=39 tok/s per_stream=[39.60] "
                             "ttft=[0.1] acc=2.610 ms_step=[65.9]\n"
                             f"SUMMARY {json.dumps([prose_row], indent=2)}\n"),
            "02-micro.log": f"SUMMARY {json.dumps(micro, indent=1)}\n",
            "04-lail.log": ('run=1 {"lail_tok_s": 33.2}\n'
                            "natural phase=lail_prose completion_tokens=512 finish_reason=length\n"
                            'SUMMARY {"median_lail_tok_s": 33.2, "post_eos_fraction": 0.0}\n'),
            "06-prose-long.log": f"natural phase=prose_long\nSUMMARY {json.dumps(long_rows)}\n",
            "07-warm-prefix.log": f"cold ttft=1\nSUMMARY {json.dumps(warm, indent=1)}\n",
        }
        with tempfile.TemporaryDirectory() as d:
            for name, text in files.items():
                (Path(d) / name).write_text(text)
            res = self.fn.parse("x", "2026-09-24T12:00:00Z", Path(d), "complete")
        self.assertEqual(res["prose_median_tok_s"], 39.6)
        self.assertEqual(res["prose_runs"], [39.6])
        self.assertEqual(res["prose_post_eos_fraction"], 0.61)
        self.assertEqual(res["prose_median_ms_per_step"], 65.9)
        self.assertEqual(res["prefill_8k_tok_s"], 690.0)   # legacy "pp" name
        self.assertEqual(res["prefill_32k_tok_s"], 600.0)  # pp_warm
        self.assertEqual(res["prefill_novel_8k_tok_s"], 250.0)
        self.assertEqual(res["prefill_novel_32k_tok_s"], 240.0)
        self.assertEqual(res["lail_runs"], [33.2])  # natural probe line is not a run
        self.assertEqual(res["lail_post_eos_fraction"], 0.0)
        self.assertEqual(len(res["prose_long_cells"]), 2)
        self.assertEqual(res["warm_prefix"]["hits_warm"], 1920)
        self.assertTrue(res["serve_env_ranks_match"])
        self.assertEqual(set(res["serve_env_digest"]), {"spark1", "spark2"})
        self.assertEqual(res["serve_uptime_s"], 3599.5)
        self.assertEqual(res["host_state"]["spark1"]["Cached_gib"], 2.0)
        self.assertFalse(res["partial"])

    def test_parse_empty_dir_is_partial_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            res = self.fn.parse("x", "2026-09-24T12:00:00Z", Path(d), "partial")
        self.assertIsNone(res["serve_env_ranks_match"])
        self.assertIsNone(res["warm_prefix"])
        self.assertEqual(res["host_state"], {})

    def test_four_numbers_sh_runs_new_cells_and_filters_env(self) -> None:
        sh = (ROOT / "tools/four_numbers.sh").read_text()
        self.assertIn("four_numbers_parse.py filter-env", sh)
        self.assertNotIn("serve_env_raw", sh)
        self.assertIn("--phase prose_long --concurrency 1 2", sh)
        self.assertIn("tools/warm_prefix.py", sh)
        self.assertIn("00-host-state.log", sh)
        # The frozen cell is unchanged.
        self.assertIn("bench_decode.py --phase prose --concurrency 1 --max-tokens 200", sh)


class WarmPrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.wp = _load("tools/warm_prefix.py", "warm_prefix_t")

    def test_expected_hits_whole_blocks_minus_last_token(self) -> None:
        self.assertEqual(self.wp.expected_hits(2050, 128), 2048)
        self.assertEqual(self.wp.expected_hits(2049, 128), 2048)
        self.assertEqual(self.wp.expected_hits(2048, 128), 1920)
        self.assertEqual(self.wp.expected_hits(100, 128), 0)

    def test_parse_prefix_counters_ignores_external(self) -> None:
        text = (
            '# HELP x\n'
            'vllm:prefix_cache_queries_total{engine="0",model_name="m"} 588514.0\n'
            'vllm:prefix_cache_queries_created{engine="0",model_name="m"} 1.79e+09\n'
            'vllm:prefix_cache_hits_total{engine="0",model_name="m"} 2688.0\n'
            'vllm:external_prefix_cache_hits_total{engine="0",model_name="m"} 5.0\n'
        )
        self.assertEqual(self.wp.parse_prefix_counters(text),
                         {"queries": 588514.0, "hits": 2688.0})
        self.assertIsNone(self.wp.parse_prefix_counters("vllm:other 1\n"))
        self.assertEqual(self.wp.hits_delta({"hits": 1.0}, {"hits": 1921.0}), 1920.0)
        self.assertIsNone(self.wp.hits_delta(None, {"hits": 1.0}))


class E2eExitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.e2e = _load("benches/e2e.py", "e2e_t")

    def _run(self, tool_ok: bool):
        seen = {}

        def fake_post(url, model, messages, *, max_tokens, tools=None, stream=True,
                      timeout=900, ignore_eos=True):
            text = messages[-1]["content"]
            if tools:
                seen["tool_ignore_eos"] = ignore_eos
                name = "get_weather" if tool_ok else "nope"
                calls = [{"id": "1", "function": {"name": name,
                                                  "arguments": '{"city": "Paris"}'}}]
                content = ""
            elif "passcode" in text:
                seen["recall_ignore_eos"] = ignore_eos
                calls, content = [], "VERDIGRIS-4200"
            else:
                seen["coding_ignore_eos"] = ignore_eos
                calls = []
                content = "```bash\nif UTIL -gt 0.90 FORCE_UNSAFE_UTIL\n```"
            return {"ttft_s": 0.1, "wall_s": 1.0, "decode_s": 0.9, "content": content,
                    "tool_calls": calls, "prompt_tokens": 10, "completion_tokens": 20}

        out = io.StringIO()
        with mock.patch.object(self.e2e, "post_chat", fake_post), \
                mock.patch.object(self.e2e, "make_tokenizer", lambda *a: lambda t: 1), \
                mock.patch.object(self.e2e, "needle_prompt", lambda *a, **k: ("doc", 1)), \
                mock.patch.object(sys, "argv", ["e2e.py"]), contextlib.redirect_stdout(out):
            rc = self.e2e.main()
        return rc, seen

    def test_all_pass_exits_zero(self) -> None:
        rc, seen = self._run(tool_ok=True)
        self.assertEqual(rc, 0)
        self.assertFalse(seen["tool_ignore_eos"])
        self.assertTrue(seen["recall_ignore_eos"])
        self.assertTrue(seen["coding_ignore_eos"])

    def test_any_failed_phase_exits_one(self) -> None:
        rc, _ = self._run(tool_ok=False)
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
