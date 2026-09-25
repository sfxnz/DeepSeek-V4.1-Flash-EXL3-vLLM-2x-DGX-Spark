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
    def test_nonce_requests_greedy_sampled_and_prefills(self) -> None:
        reqs = warmup.requests("00c0ffee00c0ffee", vision=False)
        self.assertEqual([r[0] for r in reqs], ["greedy", "t=0.7", "prefill", "prefill-300", "prefill-1000"])
        self.assertEqual([r[2] for r in reqs], [0.0, 0.7, 0.0, 0.0, 0.0])
        for _, prompt, _, _ in reqs:
            self.assertTrue(prompt.startswith("[00c0ffee00c0ffee]"), "nonce first keeps it out of the prefix cache")
        self.assertIn("17*19", reqs[0][1])
        chars = len(reqs[2][1])
        self.assertTrue(9_000 < chars < 14_000, chars)  # ~3k tokens at ~0.27 tokens/char

    def test_bucket_prefills_land_inside_the_mhc_split_buckets(self) -> None:
        # GB10 mhc_pre_big_fuse_with_norm n_splits: 16 for 65-576 batch tokens,
        # 4 for 577-1536. Repo text runs ~0.22-0.40 tokens/char on V4.1.
        reqs = {r[0]: r for r in warmup.requests("00c0ffee00c0ffee")}
        for label, lo, hi in (("prefill-300", 65, 576), ("prefill-1000", 577, 1536)):
            chars = len(reqs[label][1])
            self.assertGreater(0.22 * chars, lo, label)
            self.assertLess(0.40 * chars, hi, label)

    def test_prefills_share_no_prefix_past_the_nonce(self) -> None:
        # One doc seed per prefill: a shared prefix would be a prefix-cache hit
        # and shrink the batch out of its bucket.
        prompts = [r[1] for r in warmup.requests("00c0ffee00c0ffee") if r[0].startswith("prefill")]
        self.assertEqual(len(prompts), 3)
        heads = {p[len("[00c0ffee00c0ffee]\n"):][:200] for p in prompts}
        self.assertEqual(len(heads), 3)

    def test_vision_request_is_last_and_optional(self) -> None:
        reqs = warmup.requests("00c0ffee00c0ffee")
        label, content, _, _ = reqs[-1]
        self.assertEqual(label, "vision")
        self.assertEqual([c["type"] for c in content], ["text", "image_url"])
        self.assertTrue(content[0]["text"].startswith("[00c0ffee00c0ffee]"))
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertNotIn("vision", [r[0] for r in warmup.requests("00c0ffee00c0ffee", vision=False)])

    def test_run_sh_skips_vision_on_language_model_only(self) -> None:
        run = (ROOT / "run.sh").read_text()
        self.assertIn('[[ "$LANGUAGE_MODEL_ONLY" == 1 ]] && vision_args=(--no-vision)', run)
        self.assertIn('--model "$SERVED_NAME" "${vision_args[@]}"', run)

    def test_each_nonce_gives_a_different_prefill(self) -> None:
        self.assertNotEqual(warmup.requests("1" * 16)[2][1][20:], warmup.requests("2" * 16)[2][1][20:])


if __name__ == "__main__":
    unittest.main()
