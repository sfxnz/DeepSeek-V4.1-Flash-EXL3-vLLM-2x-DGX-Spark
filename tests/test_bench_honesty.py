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


if __name__ == "__main__":
    unittest.main()
