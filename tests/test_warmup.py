#!/usr/bin/env python3
"""tools/warmup.py request set (CPU only, no server)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import warmup  # noqa: E402


class WarmupRequestTests(unittest.TestCase):
    def test_three_nonce_requests_greedy_sampled_and_long_prefill(self) -> None:
        reqs = warmup.requests("00c0ffee00c0ffee")
        self.assertEqual([r[0] for r in reqs], ["greedy", "t=0.7", "prefill"])
        self.assertEqual([r[2] for r in reqs], [0.0, 0.7, 0.0])
        for _, prompt, _, _ in reqs:
            self.assertTrue(prompt.startswith("[00c0ffee00c0ffee]"), "nonce first keeps it out of the prefix cache")
        self.assertIn("17*19", reqs[0][1])
        chars = len(reqs[2][1])
        self.assertTrue(9_000 < chars < 14_000, chars)  # ~3k tokens at ~0.27 tokens/char

    def test_each_nonce_gives_a_different_prefill(self) -> None:
        self.assertNotEqual(warmup.requests("1" * 16)[2][1][20:], warmup.requests("2" * 16)[2][1][20:])


if __name__ == "__main__":
    unittest.main()
