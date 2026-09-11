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


FLASHINFER_DSV4_DECODE_MAX_TOKENS = 64


def indexer_kernel_block_sizes() -> tuple[int, ...]:
    """64 for compress_ratio=1 (DeepGEMM). 128 for compress_ratio=2 so extra pages stay 64."""
    return (FLASHINFER_DSV4_PAGE_BLOCK_SIZE, FLASHINFER_DSV4_PAGE_BLOCK_SIZE * 2)


def extra_page_block_size(manager_block: int, compress_ratio: int) -> int:
    """Compressed-KV page width FlashInfer sees as extra_page_block_size."""
    ratio = max(int(compress_ratio), 1)
    return int(manager_block) // ratio


def manager_block_for_flashinfer_extra(
    manager_block: int, compress_ratio: int
) -> int:
    """Raise ratio-2 manager pages to 128 so extra_page_block_size stays 64.

    L.A.I.L 81-token prefill died with extra_page_block_size=32
    (manager 64 / compress_ratio 2). Decode kernels tolerate that. The
    SM120 prefill orchestrator does not.
    """
    extra = extra_page_block_size(manager_block, compress_ratio)
    if extra == FLASHINFER_DSV4_PAGE_BLOCK_SIZE:
        return int(manager_block)
    if int(compress_ratio) <= 1:
        return int(manager_block)
    return FLASHINFER_DSV4_PAGE_BLOCK_SIZE * int(compress_ratio)


PERSISTENT_TOPK_OLD = """        use_persistent_topk = current_platform.is_cuda() and topk_tokens in (
            512,
            1024,
            2048,
        )"""
PERSISTENT_TOPK_NEW = """        use_persistent_topk = (
            current_platform.is_cuda()
            and topk_tokens in (512, 1024, 2048)
            and not current_platform.is_device_capability_family(120)
        )"""
KPOOL_PERSISTENT_OLD = (
    "        if current_platform.is_cuda() and select_k in (512, 1024, 2048):"
)
KPOOL_PERSISTENT_NEW = (
    "        if current_platform.is_cuda() and select_k in (512, 1024, 2048) "
    "and not current_platform.is_device_capability_family(120):"
)


def use_persistent_indexer_topk(is_sm120: bool, topk_tokens: int) -> bool:
    """GB10 cannot launch persistent_topk at TopK=512 with 2 decode rows."""
    if is_sm120:
        return False
    return int(topk_tokens) in (512, 1024, 2048)


def patch_persistent_topk_source(src: str) -> str:
    """Disable SM120 persistent_topk the same way Qwen excludes cooperative."""
    if "use_persistent_topk" in src and "is_device_capability_family(120)" in src:
        if PERSISTENT_TOPK_OLD not in src:
            return src
    if PERSISTENT_TOPK_OLD not in src:
        raise ValueError("persistent_topk dispatch not found")
    return src.replace(PERSISTENT_TOPK_OLD, PERSISTENT_TOPK_NEW, 1)


def patch_kpool_persistent_topk_source(src: str) -> str:
    if KPOOL_PERSISTENT_NEW in src:
        return src
    if KPOOL_PERSISTENT_OLD not in src:
        raise ValueError("kpool persistent_topk dispatch not found")
    return src.replace(KPOOL_PERSISTENT_OLD, KPOOL_PERSISTENT_NEW, 1)


def uses_prefill_orchestrator(num_tokens: int) -> bool:
    """FlashInfer SM120 takes the C++ prefill path when num_tokens > 64."""
    return int(num_tokens) > FLASHINFER_DSV4_DECODE_MAX_TOKENS


def coerce_swa_block_size(block_size: int) -> int:
    """Map the V4.1 SWA page of 32 onto the SM120 DSV4 decode page of 64."""
    if int(block_size) == UPSTREAM_V41_SWA_PAGE_BLOCK_SIZE:
        return FLASHINFER_DSV4_PAGE_BLOCK_SIZE
    return int(block_size)


def prefill_swa_topk(window_size: int, max_image_tokens: int) -> int:
    """Width of FlashInfer sparse-MLA `indices` for a SWA-only prefill row."""
    return int(window_size) + int(max_image_tokens)


def text_only_max_image_tokens(max_image_tokens: int, language_model_only: bool) -> int:
    """Collapse SWA prefill index width without dropping VL gate params.

    Zeroing ``vision_n_layers`` skips ``gate.bias_vl`` and KeyErrors on load.
    ``vision_max_n_token=0`` is enough for FlashInfer DSV4 decode topk=128.
    """
    if language_model_only:
        return 0
    return int(max_image_tokens)


def text_only_vision_n_layers(vision_n_layers: int, language_model_only: bool) -> int:
    """Do not apply this to hf_config: VL checkpoints need vision_n_layers>0."""
    if language_model_only:
        return 0
    return int(vision_n_layers)
