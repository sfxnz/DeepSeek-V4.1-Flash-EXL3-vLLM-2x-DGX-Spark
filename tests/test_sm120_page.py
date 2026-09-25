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

    def test_indexer_kernel_pages_cover_ratio1_and_ratio2(self) -> None:
        sizes = self.mod.indexer_kernel_block_sizes()
        self.assertEqual(sizes, (64, 128))
        self.assertIn(64, sizes)
        self.assertIn(128, sizes)

    def test_lail_81_token_prefill_uses_orchestrator(self) -> None:
        self.assertFalse(self.mod.uses_prefill_orchestrator(16))
        self.assertFalse(self.mod.uses_prefill_orchestrator(64))
        self.assertTrue(self.mod.uses_prefill_orchestrator(81))

    def test_block64_ratio2_extra_page_is_the_lail_crash(self) -> None:
        extra = self.mod.extra_page_block_size(64, 2)
        self.assertEqual(extra, 32)
        self.assertNotEqual(extra, self.mod.FLASHINFER_DSV4_PAGE_BLOCK_SIZE)

    def test_ratio2_manager_bump_keeps_extra_page_64(self) -> None:
        manager = self.mod.manager_block_for_flashinfer_extra(64, 2)
        self.assertEqual(manager, 128)
        self.assertEqual(
            self.mod.extra_page_block_size(manager, 2),
            self.mod.FLASHINFER_DSV4_PAGE_BLOCK_SIZE,
        )
        self.assertEqual(self.mod.manager_block_for_flashinfer_extra(64, 1), 64)
        self.assertEqual(self.mod.manager_block_for_flashinfer_extra(64, 0), 64)

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

    def test_language_model_only_zeros_max_image_tokens_not_layers(self) -> None:
        self.assertEqual(self.mod.text_only_max_image_tokens(1024, True), 0)
        self.assertEqual(self.mod.text_only_max_image_tokens(1024, False), 1024)
        topk = self.mod.prefill_swa_topk(
            self.mod.V41_SLIDING_WINDOW,
            self.mod.text_only_max_image_tokens(
                self.mod.V41_VISION_MAX_IMAGE_TOKENS, True
            ),
        )
        self.assertIn(topk, self.mod.FLASHINFER_DSV4_DECODE_TOPK)

    def test_sm120_disables_persistent_indexer_topk(self) -> None:
        self.assertFalse(self.mod.use_persistent_indexer_topk(True, 512))
        self.assertTrue(self.mod.use_persistent_indexer_topk(False, 512))

    def test_patch_persistent_topk_source_excludes_sm120(self) -> None:
        patched = self.mod.patch_persistent_topk_source(self.mod.PERSISTENT_TOPK_OLD)
        self.assertIn("is_device_capability_family(120)", patched)
        self.assertEqual(self.mod.patch_persistent_topk_source(patched), patched)

    def test_sitecustomize_does_not_zero_vision_n_layers(self) -> None:
        site = (ROOT / "docker/patch/sitecustomize.py").read_text()
        self.assertIn("text_only_max_image_tokens", site)
        self.assertIn("vision_max_n_token", site)
        self.assertNotIn("self.vision_n_layers =", site)
        self.assertIn("DeepseekV4IndexerBackend", site)
        self.assertIn("DeepseekV4FlashInferMLASparseBackend", site)
        self.assertIn("indexer_kernel_block_sizes", site)
        self.assertIn("manager_block_for_flashinfer_extra", site)
        self.assertIn("DeepseekV4Attention", site)
        self.assertIn("patch_persistent_topk_source", site)

    def test_language_model_only_from_env_reads_flag_or_env(self) -> None:
        fn = self.mod.language_model_only_from_env
        self.assertTrue(fn(["vllm", "serve", "--language-model-only"], {}))
        self.assertTrue(fn(["vllm", "serve"], {"LANGUAGE_MODEL_ONLY": "1"}))
        self.assertFalse(fn(["vllm", "serve"], {"LANGUAGE_MODEL_ONLY": "0"}))
        self.assertFalse(fn(["vllm", "serve"], {}))

    def test_vision_on_does_not_collapse_config_max_image_tokens(self) -> None:
        self.assertEqual(self.mod.text_only_max_image_tokens(1024, False), 1024)
        self.assertGreater(self.mod.text_only_max_image_tokens(1024, False), 0)
        self.assertEqual(self.mod.text_only_vision_n_layers(32, False), 32)

    def test_vision_prefill_1152_clamps_to_dispatch_legal_1024(self) -> None:
        raw = self.mod.prefill_swa_topk(
            self.mod.V41_SLIDING_WINDOW, self.mod.V41_VISION_MAX_IMAGE_TOKENS
        )
        self.assertEqual(raw, 1152)
        clamped = self.mod.clamp_prefill_swa_topk(
            self.mod.V41_SLIDING_WINDOW, self.mod.V41_VISION_MAX_IMAGE_TOKENS
        )
        self.assertEqual(clamped, 1024)
        self.assertIn(clamped, self.mod.FLASHINFER_DSV4_DECODE_TOPK)

    def test_swa_dispatch_keeps_dual_prefill_topk_128(self) -> None:
        img = self.mod.swa_image_tokens_for_dispatch(
            self.mod.V41_SLIDING_WINDOW, self.mod.V41_VISION_MAX_IMAGE_TOKENS
        )
        self.assertEqual(img, 0)
        width = self.mod.V41_SLIDING_WINDOW + img
        self.assertEqual(width, 128)
        self.assertIn(width, self.mod.FLASHINFER_DSV4_DECODE_TOPK)

    def test_text_prefill_keeps_window_topk_when_vision_weights_loaded(self) -> None:
        width = self.mod.flashinfer_prefill_swa_width(
            self.mod.V41_SLIDING_WINDOW,
            self.mod.V41_VISION_MAX_IMAGE_TOKENS,
            has_image=False,
        )
        self.assertEqual(width, self.mod.V41_SLIDING_WINDOW)
        self.assertIn(width, self.mod.FLASHINFER_DSV4_DECODE_TOPK)

    def test_swa_and_attention_source_patches_are_idempotent(self) -> None:
        swa = self.mod.patch_swa_prefill_image_width_source(
            self.mod.SWA_IMAGE_WIDTH_OLD
        )
        self.assertIn("swa_image_tokens_for_dispatch", swa)
        self.assertEqual(self.mod.patch_swa_prefill_image_width_source(swa), swa)
        attn = self.mod.patch_attention_image_width_source(
            self.mod.ATTN_IMAGE_WIDTH_OLD
        )
        self.assertIn("swa_image_tokens_for_dispatch", attn)
        self.assertEqual(self.mod.patch_attention_image_width_source(attn), attn)

    def test_shipped_essay_corpus_has_eight_sequences(self) -> None:
        seqs = self.mod.load_essay_corpus(ROOT / "docker/patch/essay_corpus.json")
        self.assertEqual(len(seqs), 8)
        self.assertGreater(len(seqs[0]), 100)
        # 8/8 T=0.2 samples shared the first 4 continuation tokens.
        self.assertTrue(all(seqs[0][57:61] == s[57:61] for s in seqs))

    def test_corpus_lookup_follows_longest_self_distilled_path(self) -> None:
        corpus = [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [9, 9, 9, 9, 9],
        ]
        looked = self.mod.corpus_lookup_draft([0, 1, 2, 3, 4], corpus, num_draft=3, min_n=4)
        self.assertEqual(looked, [5, 6, 7])
        self.assertEqual(
            self.mod.corpus_lookup_draft([9, 8, 7, 6], corpus, 3, min_n=4),
            [],
        )
        merged = self.mod.overlay_corpus_on_drafts(
            [[0, 0, 0, 0, 0]], [[1, 2, 3, 4]], corpus, min_n=4
        )
        self.assertEqual(merged[0][:3], [5, 6, 7])
        self.assertEqual(len(merged[0]), 5)

    def test_apply_markov_finetune_copies_matching_weights(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed on host")

        class _E:
            def __init__(self, shape):
                self.weight = torch.zeros(shape)

        class _Head:
            def __init__(self):
                self.markov_w1 = _E((8, 2))
                self.markov_w2 = _E((8, 2))

        blob = {
            "w1": torch.arange(16, dtype=torch.float32).reshape(8, 2),
            "w2": torch.ones(8, 2),
        }
        head = _Head()
        self.mod.apply_markov_finetune(head, blob)
        self.assertEqual(head.markov_w1.weight[0, 0].item(), 0.0)
        self.assertEqual(head.markov_w1.weight[7, 1].item(), 15.0)
        self.assertEqual(head.markov_w2.weight.sum().item(), 16.0)
        with self.assertRaises(ValueError):
            self.mod.apply_markov_finetune(
                head, {"w1": torch.zeros(3, 2), "w2": blob["w2"]}
            )

    def test_markov_table_next_prefers_trigram(self) -> None:
        table = {"bi": {"1": 9}, "tri": {"2,1": 7}}
        self.assertEqual(self.mod.markov_table_next(2, 1, table), 7)
        self.assertEqual(self.mod.markov_table_next(None, 1, table), 9)
        self.assertIsNone(self.mod.markov_table_next(3, 4, table))

    def test_overlay_markov_table_walks_left_to_right(self) -> None:
        table = {"bi": {"10": 11, "11": 12, "12": 13}, "tri": {}}
        merged = self.mod.overlay_markov_table_on_drafts(
            [[0, 0, 0]], [[10]], table
        )
        self.assertEqual(merged, [[11, 12, 13]])
        # miss keeps DSpark draft
        self.assertEqual(
            self.mod.overlay_markov_table_on_drafts([[5, 6]], [[99]], table),
            [[5, 6]],
        )

    def test_shipped_essay_markov_table_is_compact_and_nonempty(self) -> None:
        table = self.mod.load_markov_table(ROOT / "docker/patch/essay_markov.json")
        self.assertGreater(table.get("n_trigram", 0), 50)
        self.assertGreater(table.get("n_bigram", 0), 50)
        self.assertIn("tri", table)
        self.assertTrue(all("," in k for k in table["tri"]))

    def test_prefix_lookup_draft_returns_repeated_suffix_continuation(self) -> None:
        # prefix ... 10 11 12 13 ... 10 11 12  -> drafts 13, 99 from the first match
        ids = [1, 2, 10, 11, 12, 13, 99, 7, 10, 11, 12]
        looked = self.mod.prefix_lookup_draft(ids, num_draft=2, min_n=2, max_n=5)
        self.assertEqual(looked, [13, 99])
        self.assertEqual(
            self.mod.prefix_lookup_draft([1, 2, 3, 4], 5, min_n=2, max_n=5),
            [],
        )

    def test_overlay_ngram_on_drafts_replaces_only_the_lookup_prefix(self) -> None:
        drafts = [[9, 9, 9, 9, 9]]
        prefixes = [[5, 6, 7, 8, 1, 5, 6, 7]]
        merged = self.mod.overlay_ngram_on_drafts(drafts, prefixes, min_n=2, max_n=4)
        self.assertEqual(merged[0][0], 8)
        self.assertEqual(len(merged[0]), 5)
        self.assertEqual(
            self.mod.overlay_ngram_on_drafts([[1, 2, 3]], [[9, 8, 7]], min_n=2),
            [[1, 2, 3]],
        )

    def test_overlay_ngram_on_draft_tail_keeps_head_and_fills_from_prefix_plus_head(
        self,
    ) -> None:
        # prefix ... 10 11 12 13 14 15 ... then DSpark head 10 11 12.
        # Tail lookup of prefix+head should yield 13, 14.
        prefix = [1, 2, 10, 11, 12, 13, 14, 15, 7]
        drafts = [[10, 11, 12, 99, 98]]
        merged = self.mod.overlay_ngram_on_draft_tail(
            drafts, [prefix], start_pos=3, min_n=2, max_n=5
        )
        self.assertEqual(merged, [[10, 11, 12, 13, 14]])
        miss = self.mod.overlay_ngram_on_draft_tail(
            [[4, 5, 6, 7, 8]], [[1, 2, 3]], start_pos=3, min_n=2, max_n=5
        )
        self.assertEqual(miss, [[4, 5, 6, 7, 8]])
        past_end = self.mod.overlay_ngram_on_draft_tail(
            [[4, 5]], [prefix], start_pos=3, min_n=2, max_n=5
        )
        self.assertEqual(past_end, [[4, 5]])

    def test_cache_logits_then_argmax_writes_cache_and_picks_mode(self) -> None:
        logits = [[0.1, 0.9, 0.0], [0.5, 0.2, 0.3]]
        cache = [[[0.0] * 3 for _ in range(5)] for _ in range(2)]
        ids = self.mod.cache_logits_then_argmax(logits, cache, [0, 1], 2)
        self.assertEqual(ids, [1, 0])
        self.assertEqual(cache[0][2], [0.1, 0.9, 0.0])
        self.assertEqual(cache[1][2], [0.5, 0.2, 0.3])
        self.assertEqual(cache[0][0], [0.0, 0.0, 0.0])
        tied = self.mod.cache_logits_then_argmax(
            [[0.5, 0.5, 0.1]], [[[0.0] * 3 for _ in range(2)]], [0], 0
        )
        self.assertEqual(tied, [0])

    def test_draft_logits_flat_index_is_req_times_n_spec_plus_step(self) -> None:
        self.assertEqual(
            self.mod.draft_logits_flat_index([0, 1], 2, 5), [2, 7]
        )
        self.assertEqual(self.mod.draft_logits_flat_index([3], 0, 5), [15])

    def test_dspark_softmax_verify_from_env_is_opt_in(self) -> None:
        self.assertFalse(self.mod.dspark_softmax_verify_from_env(0))
        self.assertFalse(self.mod.dspark_softmax_verify_from_env(2))
        self.assertTrue(self.mod.dspark_softmax_verify_from_env(1))

    def test_dspark_tail_ngram_start_pos_is_opt_in(self) -> None:
        start = self.mod.dspark_tail_ngram_start_pos
        self.assertIsNone(start(0, 3))
        self.assertIsNone(start(2, 3))
        self.assertEqual(start(1, 3), 3)
        self.assertEqual(start(1, 0), 0)
        self.assertIsNone(start(1, -1))

    def test_indexer_adaptive_source_patch_is_idempotent(self) -> None:
        patched = self.mod.patch_indexer_adaptive_source(
            self.mod.INDEXER_MISMATCH_OLD + "\n" + self.mod.INDEXER_CG_OLD
        )
        self.assertIn("return True", patched)
        self.assertIn("return AttentionCGSupport.ALWAYS", patched)
        self.assertNotIn("_supports_varlen_paged_mqa_logits", patched)
        self.assertEqual(self.mod.patch_indexer_adaptive_source(patched), patched)

    def test_skip_indexer_for_short_lail_context(self) -> None:
        skip = self.mod.skip_indexer_for_short_context
        # L.A.I.L 61+512 tokens, compress 1 and 2, topk 512.
        self.assertTrue(skip(61, 1, 512))
        self.assertTrue(skip(573, 2, 512))
        self.assertFalse(skip(573, 1, 512))
        self.assertFalse(skip(1024, 1, 512))
        self.assertFalse(skip(128, 0, 512))
        patched = self.mod.patch_indexer_short_context_source(
            self.mod.INDEXER_SKIP_OLD
        )
        self.assertNotIn("is_current_stream_capturing", patched)
        self.assertIn("self.compress_ratio > 0", patched)
        self.assertEqual(
            self.mod.patch_indexer_short_context_source(patched), patched
        )

    def test_scale_markov_bias_identity_and_zero(self) -> None:
        self.assertEqual(self.mod.scale_markov_bias(10.0, 1.0), 10.0)
        self.assertEqual(self.mod.scale_markov_bias(10.0, 0.0), 0.0)
        self.assertEqual(self.mod.scale_markov_bias(10.0, 0.5), 5.0)

    def test_sm120_native_indexer_decode_allows_dspark_next_n(self) -> None:
        self.assertTrue(self.mod.sm120_native_indexer_decode(True, 6))
        self.assertTrue(self.mod.sm120_native_indexer_decode(True, 1))
        self.assertFalse(self.mod.sm120_native_indexer_decode(False, 6))
        self.assertTrue(self.mod.sm120_native_indexer_decode(False, 1))
        patched = self.mod.patch_native_indexer_decode_source(
            self.mod.NATIVE_DECODE_OLD
        )
        self.assertIn("is_device_capability_family(120)", patched)
        self.assertIn("native_next_n_supported(next_n)", patched)
        self.assertEqual(self.mod.patch_native_indexer_decode_source(patched), patched)

    def test_compressed_extra_topk_covers_lail_without_leaving_dispatch(self) -> None:
        width = self.mod.compressed_extra_topk_width
        legal = self.mod.FLASHINFER_DSV4_DECODE_TOPK
        self.assertEqual(width(61, 512), 128)
        self.assertEqual(width(128, 512), 128)
        self.assertEqual(width(129, 512), 192)
        self.assertEqual(width(192, 512), 192)
        self.assertEqual(width(193, 512), 256)
        self.assertEqual(width(400, 512), 512)
        self.assertEqual(width(573, 512), 512)
        for n in (61, 128, 192, 256, 512, 573):
            self.assertIn(width(n, 512), legal)
        kv = self.mod.compressed_kv_len
        self.assertEqual(kv(61, 1), 61)
        self.assertEqual(kv(61, 2), 31)
        self.assertEqual(kv(573, 2), 287)
        self.assertEqual(width(kv(61, 2), 512), 128)
        self.assertEqual(width(kv(573, 2), 512), 512)
        slice_w = self.mod.extra_index_slice_width
        self.assertEqual(slice_w(61, 2, 512), 128)
        self.assertEqual(slice_w(61, 1, 512), 128)
        self.assertEqual(slice_w(573, 1, 512), 512)
        self.assertEqual(slice_w(573, 2, 512), 512)
        clamp = self.mod.clamp_index_topk
        self.assertEqual(clamp(512, 0), 512)
        self.assertEqual(clamp(512, 128), 128)
        self.assertEqual(clamp(512, 192), 192)
        self.assertEqual(clamp(128, 512), 128)
        self.assertEqual(clamp(512, 1152), 512)
        self.assertEqual(clamp(1024, 640), 512)
        self.assertIn(clamp(512, 128), legal)

    def test_engram_row_cache_hits_misses_and_lru_evict(self) -> None:
        cache: dict = {}
        order: list = []
        hits, missing = self.mod.engram_row_cache_split(cache, [10, 11, 10])
        self.assertEqual(hits, [None, None, None])
        self.assertEqual(missing, [10, 11])
        self.mod.engram_row_cache_store(cache, order, 10, "a", max_rows=2)
        self.mod.engram_row_cache_store(cache, order, 11, "b", max_rows=2)
        hits, missing = self.mod.engram_row_cache_split(cache, [11, 10, 12])
        self.assertEqual(hits, ["b", "a", None])
        self.assertEqual(missing, [12])
        self.mod.engram_row_cache_store(cache, order, 12, "c", max_rows=2)
        self.assertNotIn(10, cache)
        self.assertEqual(cache[11], "b")
        self.assertEqual(cache[12], "c")
        self.mod.engram_row_cache_store(cache, order, 11, "b2", max_rows=2)
        self.mod.engram_row_cache_store(cache, order, 13, "d", max_rows=2)
        self.assertNotIn(12, cache)
        self.assertEqual(cache[11], "b2")
        self.assertEqual(cache[13], "d")

    def test_mask_logits_keep_topk_keeps_largest_k(self) -> None:
        masked = self.mod.mask_logits_keep_topk([0.1, 0.9, 0.3, 0.8], 2)
        self.assertEqual(masked[1], 0.9)
        self.assertEqual(masked[3], 0.8)
        self.assertEqual(masked[0], float("-inf"))
        self.assertEqual(masked[2], float("-inf"))
        self.assertEqual(self.mod.mask_logits_keep_topk([1.0, 2.0], 0), [1.0, 2.0])
        self.assertEqual(self.mod.mask_logits_keep_topk([1.0, 2.0], 5), [1.0, 2.0])

    def test_dspark_draft_topk_from_env_none_is_dense_markov(self) -> None:
        fn = self.mod.dspark_draft_topk_from_env
        self.assertIsNone(fn(0))
        self.assertIsNone(fn(-1))
        self.assertEqual(fn(32), 32)
        self.assertEqual(fn(64), 64)

    def test_mla_chunks_per_block_from_env_none_is_autotune(self) -> None:
        fn = self.mod.mla_chunks_per_block_from_env
        self.assertIsNone(fn(0))
        self.assertIsNone(fn(-1))
        self.assertEqual(fn(1), 1)
        self.assertEqual(fn(4), 4)

    def test_fill_engram_cache_hits_merges_fetched_misses(self) -> None:
        keys = [10, 11, 10]
        hits = [None, "b", None]
        fetched = {10: "a"}
        filled = self.mod.fill_engram_cache_hits(keys, hits, fetched)
        self.assertEqual(filled, ["a", "b", "a"])

    def test_decode_mhc_pre_splits_collapse_at_dspark_verify_width(self) -> None:
        fn = self.mod.decode_mhc_pre_num_splits
        self.assertEqual(fn(6, 16), 1)
        self.assertEqual(fn(1, 16), 1)
        self.assertEqual(fn(12, 16), 1)
        self.assertEqual(fn(16, 16), 1)
        self.assertEqual(fn(17, 16), 16)
        self.assertEqual(fn(81, 16), 16)
        self.assertEqual(fn(64, 4), 4)

    def test_census_skips_device_sync_while_capturing(self) -> None:
        self.assertTrue(self.mod.census_skip_during_capture(True))
        self.assertFalse(self.mod.census_skip_during_capture(False))

    def test_timed_call_records_name_and_elapsed(self) -> None:
        ticks = iter([1.0, 1.25, 2.0, 2.5])
        rec = []
        out = self.mod.timed_call(lambda: next(ticks), rec, "engram", lambda: 7)
        self.assertEqual(out, 7)
        self.assertEqual(rec, [("engram", 0.25)])
        self.mod.timed_call(lambda: next(ticks), rec, "draft", lambda: None)
        totals = self.mod.census_totals(rec)
        self.assertEqual(totals["engram"]["n"], 1.0)
        self.assertEqual(totals["engram"]["s"], 0.25)
        self.assertEqual(totals["draft"]["n"], 1.0)
        self.assertEqual(totals["draft"]["s"], 0.5)
        text = self.mod.format_census_totals(totals)
        self.assertIn("draft n=1", text)
        self.assertIn("engram n=1", text)
        self.assertIn("avg_ms=250.0", text)

    def test_step_census_records_pre_draft_between_target_and_draft(self) -> None:
        import sys
        import types

        class Spec:
            def _generate_draft(self):
                return "d"

        class Model:
            def forward(self):
                return "t"

        class TargetCG:
            def run_fullgraph(self, desc):
                return "tg"

        class DraftCG:
            def run_fullgraph(self, desc):
                return None

        leaves = {
            "vllm.v1.worker.gpu.spec_decode.dspark.speculator": ("DSparkSpeculator", Spec),
            "vllm.models.deepseek_v4_1.nvidia.model": ("DeepseekV4Model", Model),
            "vllm.v1.worker.gpu.cudagraph_utils": ("ModelCudaGraphManager", TargetCG),
            "vllm.v1.worker.gpu.spec_decode.dflash.cudagraph": ("DFlashCudaGraphManager", DraftCG),
        }
        names = {leaf.rsplit(".", k)[0] for leaf in leaves for k in range(leaf.count(".") + 1)}
        saved = {n: sys.modules.get(n) for n in names}
        try:
            for n in names:
                sys.modules[n] = types.ModuleType(n)
            for leaf, (attr, cls) in leaves.items():
                setattr(sys.modules[leaf], attr, cls)
            rec: list = []
            self.mod.install_step_census(rec)
            spec, model = Spec(), Model()
            model.forward()
            spec._generate_draft()
            spec._generate_draft()  # no target in between: no pre_draft
            model.forward()
            spec._generate_draft()
            order = [name for name, _ in rec]
            self.assertEqual(
                order, ["target", "pre_draft", "draft", "draft", "target", "pre_draft", "draft"]
            )
            self.assertTrue(all(dt >= 0 for _, dt in rec))
            self.assertIn("pre_draft n=2", self.mod.format_census_totals(self.mod.census_totals(rec)))
            # FULL-graph decode: replays, not forward/_generate_draft.
            rec.clear()
            self.assertEqual(TargetCG().run_fullgraph(0), "tg")
            DraftCG().run_fullgraph(0)
            self.assertEqual([name for name, _ in rec], ["target", "pre_draft", "draft"])
            rec.clear()  # atexit dump prints nothing
        finally:
            for n, m in saved.items():
                if m is None:
                    sys.modules.pop(n, None)
                else:
                    sys.modules[n] = m

    def test_sitecustomize_census_is_opt_in(self) -> None:
        site = (ROOT / "docker/patch/sitecustomize.py").read_text()
        self.assertIn("DSV41_STEP_CENSUS", site)
        self.assertIn("install_step_census", site)
        self.assertIn('== "1"', site)
        self.assertIn("DSV41_INDEX_TOPK", site)
        self.assertIn("clamp_index_topk", site)
        self.assertIn("DSV41_MHC_DECODE_SPLITS", site)
        self.assertIn("decode_mhc_pre_num_splits", site)
        self.assertIn("DSV41_ENGRAM_CACHE", site)
        self.assertIn("fill_engram_cache_hits", site)
        self.assertIn("DSV41_MHC_NO_DEEPGEMM", site)
        self.assertIn("DSV41_DSPARK_DRAFT_TOPK", site)
        self.assertIn("dspark_draft_topk_from_env", site)
        self.assertIn("DSV41_DSPARK_TAIL_NGRAM", site)
        self.assertIn("overlay_ngram_on_draft_tail", site)
        self.assertIn("dspark_tail_ngram_start_pos", site)
        self.assertIn("DSV41_DSPARK_SOFTMAX_VERIFY", site)
        self.assertIn("dspark_softmax_verify_from_env", site)
        self.assertIn("greedy propose, softmax verify", site)
        head = site.split("Restrict DSpark backbone")[0]
        self.assertIn("gumbel_sample", head)
        self.assertIn("_zero_temperature", head)
        self.assertIn("logits_cache=self.draft_logits", head)
        self.assertIn("compute_draft_logits", site)
        self.assertIn("DSparkDeepseekV4ForCausalLM", site)
        self.assertIn("logits.fill_(float(\"-inf\"))", site)
        self.assertIn("logits.scatter_(-1, idx, vals)", site)
        self.assertNotIn("new_full", site)
        self.assertNotIn("self._draft_topk = _draft_k", site)
        self.assertNotIn("self.dspark_draft_topk = draft_k", site)
        self.assertNotIn("apply_markov_bias_gathered(", site)
        self.assertIn("DSV41_MLA_CHUNKS_PER_BLOCK", site)
        self.assertIn("mla_chunks_per_block_from_env", site)
        self.assertIn("sparse_mla_sm120_decode_dsv4", site)
        self.assertIn("chunks_per_block", site)

    def test_sitecustomize_does_not_hardcode_language_model_only_true(self) -> None:
        site = (ROOT / "docker/patch/sitecustomize.py").read_text()
        self.assertIn("language_model_only_from_env", site)
        self.assertIn("patch_swa_prefill_image_width_source", site)
        self.assertIn("patch_attention_image_width_source", site)
        self.assertIn("patch_native_indexer_decode_source", site)
        self.assertIn("patch_indexer_short_context_source", site)
        self.assertIn("patch_indexer_adaptive_source", site)
        self.assertNotIn(", True\n        )", site)


if __name__ == "__main__":
    unittest.main()
