"""GB10 FlashInfer SM120 DeepSeek-V4 sparse-MLA page and top-k pins.

flashinfer.mla._sparse_mla_sm120 decode-dsv4 is instantiated only for
page_block_size=64 and (num_heads, topk) in _DECODE_DSV4_DISPATCH. Upstream
vLLM V4.1 hardcodes SWA pages to 32 (DeepGEMM paged-MQA) and still widens
prefill SWA index rows by vision_max_n_token even under --language-model-only,
which yields topk=1152 and misses the dispatch table.
"""

FLASHINFER_DSV4_PAGE_BLOCK_SIZE = 64
UPSTREAM_V41_SWA_PAGE_BLOCK_SIZE = 32
FLASHINFER_DSV4_DECODE_TOPK = frozenset({128, 192, 256, 512, 1024})
V41_SLIDING_WINDOW = 128
V41_VISION_MAX_IMAGE_TOKENS = 1024


def coerce_swa_block_size(block_size: int) -> int:
    """Map the V4.1 SWA page of 32 onto the SM120 DSV4 decode page of 64."""
    if int(block_size) == UPSTREAM_V41_SWA_PAGE_BLOCK_SIZE:
        return FLASHINFER_DSV4_PAGE_BLOCK_SIZE
    return int(block_size)


def prefill_swa_topk(window_size: int, max_image_tokens: int) -> int:
    """Width of FlashInfer sparse-MLA `indices` for a SWA-only prefill row."""
    return int(window_size) + int(max_image_tokens)


def text_only_vision_n_layers(vision_n_layers: int, language_model_only: bool) -> int:
    """`--language-model-only` must not leave vision_n_layers > 0 on hf_config."""
    if language_model_only:
        return 0
    return int(vision_n_layers)
