#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "docker/patch/sm120_page.py"
    spec = importlib.util.spec_from_file_location("sm120_page", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Sm120PageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _load()

    def test_coerce_upstream_swa_32_to_flashinfer_64(self) -> None:
        self.assertEqual(
            self.mod.coerce_swa_block_size(self.mod.UPSTREAM_V41_SWA_PAGE_BLOCK_SIZE),
            self.mod.FLASHINFER_DSV4_PAGE_BLOCK_SIZE,
        )
        self.assertEqual(self.mod.coerce_swa_block_size(32), 64)

    def test_coerce_leaves_non_32_pages(self) -> None:
        self.assertEqual(self.mod.coerce_swa_block_size(64), 64)
        self.assertEqual(self.mod.coerce_swa_block_size(128), 128)

    def test_vision_widened_prefill_topk_misses_dsv4_dispatch(self) -> None:
        topk = self.mod.prefill_swa_topk(
            self.mod.V41_SLIDING_WINDOW, self.mod.V41_VISION_MAX_IMAGE_TOKENS
        )
        self.assertEqual(topk, 1152)
        self.assertNotIn(topk, self.mod.FLASHINFER_DSV4_DECODE_TOPK)

    def test_text_only_prefill_topk_hits_dsv4_dispatch(self) -> None:
        topk = self.mod.prefill_swa_topk(self.mod.V41_SLIDING_WINDOW, 0)
        self.assertEqual(topk, 128)
        self.assertIn(topk, self.mod.FLASHINFER_DSV4_DECODE_TOPK)

    def test_language_model_only_zeros_vision_layers(self) -> None:
        self.assertEqual(self.mod.text_only_vision_n_layers(24, True), 0)
        self.assertEqual(self.mod.text_only_vision_n_layers(24, False), 24)


if __name__ == "__main__":
    unittest.main()
