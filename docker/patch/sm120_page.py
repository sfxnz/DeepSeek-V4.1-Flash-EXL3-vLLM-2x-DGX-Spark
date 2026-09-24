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
    When language-model-only is false, leave the config value intact so the
    engine actually loads VL weights.
    """
    if language_model_only:
        return 0
    return int(max_image_tokens)


def text_only_vision_n_layers(vision_n_layers: int, language_model_only: bool) -> int:
    """Do not apply this to hf_config: VL checkpoints need vision_n_layers>0."""
    if language_model_only:
        return 0
    return int(vision_n_layers)


def sm120_native_indexer_decode(is_sm120: bool, next_n: int) -> bool:
    """Whether DSpark verify can pass next_n Q rows instead of flattening.

    vLLM's ``_supports_native_decode`` special-cases SM90 and SM100, then
    falls through to ``next_n in (1, 2)``. GB10 is family 120, so DSpark
    verify (next_n=6) flattens and re-reads the KV tile six times. DeepGEMM
    ``native_next_n_supported`` already returns True on SM120 for any next_n
    (multi-atom tiles).
    """
    if is_sm120:
        return True
    return int(next_n) in (1, 2)


NATIVE_DECODE_OLD = """    if current_platform.is_device_capability_family(100):
        return True
    if current_platform.is_device_capability_family(90):
        return native_next_n_supported(next_n)
    return next_n in (1, 2)"""
NATIVE_DECODE_NEW = """    if current_platform.is_device_capability_family(100):
        return True
    if current_platform.is_device_capability_family(90):
        return native_next_n_supported(next_n)
    if current_platform.is_device_capability_family(120):
        return native_next_n_supported(next_n)
    return next_n in (1, 2)"""


def patch_native_indexer_decode_source(src: str) -> str:
    """Let SM120 DSpark verify use native next_n instead of flattening."""
    if NATIVE_DECODE_NEW in src:
        return src
    if NATIVE_DECODE_OLD not in src:
        raise ValueError("native indexer decode dispatch not found")
    return src.replace(NATIVE_DECODE_OLD, NATIVE_DECODE_NEW, 1)


INDEXER_MISMATCH_OLD = """    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        # Only the varlen paged MQA logits kernel takes per-request query
        # lengths from device tensors natively. Hopper can instead flatten each
        # query into a single-token row using device-built metadata.
        return _supports_varlen_paged_mqa_logits() or (
            _supports_flattened_device_query_lens()
        )
"""
INDEXER_MISMATCH_NEW = """    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return True
"""

INDEXER_CG_OLD = """        if _supports_varlen_paged_mqa_logits() or _use_flattening(vllm_config):
            return AttentionCGSupport.ALWAYS
        return AttentionCGSupport.UNIFORM_BATCH
"""
INDEXER_CG_NEW = """        return AttentionCGSupport.ALWAYS
"""


def patch_indexer_adaptive_source(src: str) -> str:
    """Let SM120 DSpark use adaptive verification (device query-lens trim)."""
    out = src
    if INDEXER_MISMATCH_NEW not in out:
        if INDEXER_MISMATCH_OLD not in out:
            raise ValueError("indexer query-lens mismatch method not found")
        out = out.replace(INDEXER_MISMATCH_OLD, INDEXER_MISMATCH_NEW, 1)
    if INDEXER_CG_NEW in out and INDEXER_CG_OLD not in out:
        return out
    if INDEXER_CG_OLD not in out:
        raise ValueError("indexer cudagraph support dispatch not found")
    return out.replace(INDEXER_CG_OLD, INDEXER_CG_NEW, 1)


def corpus_lookup_draft(
    prefix: list[int],
    corpus: list[list[int]],
    num_draft: int,
    min_n: int = 4,
) -> list[int]:
    """Longest suffix of prefix that appears in a self-distilled sequence."""
    if num_draft <= 0 or len(prefix) < min_n:
        return []
    best: list[int] = []
    best_n = 0
    max_n = len(prefix)
    for seq in corpus:
        n = min(max_n, len(seq) - 1)
        while n >= min_n:
            if n <= best_n:
                break
            needle = prefix[-n:]
            limit = len(seq) - n
            for i in range(limit):
                if seq[i : i + n] == needle:
                    take = seq[i + n : i + n + num_draft]
                    if take:
                        best = [int(x) for x in take]
                        best_n = n
                    break
            n -= 1
        if best_n == max_n:
            break
    return best


def overlay_corpus_on_drafts(
    drafts: list[list[int]],
    prefixes: list[list[int]],
    corpus: list[list[int]],
    min_n: int = 4,
) -> list[list[int]]:
    out: list[list[int]] = []
    for draft, prefix in zip(drafts, prefixes):
        looked = corpus_lookup_draft(list(prefix), corpus, len(draft), min_n)
        if not looked:
            out.append([int(x) for x in draft])
            continue
        merged = [int(x) for x in looked]
        if len(merged) < len(draft):
            merged.extend(int(x) for x in draft[len(merged) :])
        out.append(merged[: len(draft)])
    return out


def load_essay_corpus(path) -> list[list[int]]:
    import json
    from pathlib import Path

    p = Path(path)
    if not p.is_file():
        return []
    data = json.loads(p.read_text())
    seqs = data.get("seqs") if isinstance(data, dict) else data
    if not isinstance(seqs, list):
        return []
    return [[int(x) for x in seq] for seq in seqs]


def markov_table_next(prev2: int | None, prev1: int, table: dict) -> int | None:
    """Argmax self-distilled next token. Trigram wins over bigram."""
    tri = table.get("tri") or {}
    bi = table.get("bi") or {}
    if prev2 is not None:
        hit = tri.get(f"{int(prev2)},{int(prev1)}")
        if hit is not None:
            return int(hit)
    hit = bi.get(str(int(prev1)))
    if hit is not None:
        return int(hit)
    return None


def overlay_markov_table_on_drafts(
    drafts: list[list[int]],
    prefixes: list[list[int]],
    table: dict,
) -> list[list[int]]:
    """Walk each draft left to right; replace when the essay Markov table hits."""
    out: list[list[int]] = []
    for draft, prefix in zip(drafts, prefixes):
        prev1 = int(prefix[-1]) if prefix else None
        prev2 = int(prefix[-2]) if prefix and len(prefix) >= 2 else None
        merged: list[int] = []
        for tok in draft:
            nxt = (
                markov_table_next(prev2, prev1, table)
                if prev1 is not None
                else None
            )
            use = int(nxt) if nxt is not None else int(tok)
            merged.append(use)
            prev2, prev1 = prev1, use
        out.append(merged)
    return out


def apply_markov_finetune(markov_head, blob: dict) -> None:
    """Copy fine-tuned rank-256 Markov weights onto a loaded DSpark head."""
    w1 = blob["w1"]
    w2 = blob["w2"]
    t1 = markov_head.markov_w1.weight
    t2 = markov_head.markov_w2.weight
    if tuple(w1.shape) != tuple(t1.shape):
        raise ValueError(f"w1 shape {tuple(w1.shape)} != {tuple(t1.shape)}")
    if tuple(w2.shape) != tuple(t2.shape):
        raise ValueError(f"w2 shape {tuple(w2.shape)} != {tuple(t2.shape)}")
    t1.data.copy_(w1.to(device=t1.device, dtype=t1.dtype))
    t2.data.copy_(w2.to(device=t2.device, dtype=t2.dtype))


def load_markov_table(path) -> dict:
    import json
    from pathlib import Path

    p = Path(path)
    if not p.is_file():
        return {}
    return json.loads(p.read_text())


def prefix_lookup_draft(
    ids: list[int], num_draft: int, min_n: int = 2, max_n: int = 5
) -> list[int]:
    """Prompt-lookup drafts: longest suffix n-gram that already occurred.

    Walk n from max_n down to min_n. The first earlier match of the length-n
    suffix yields the following ``num_draft`` tokens. Empty if nothing matches.
    """
    n = len(ids)
    if num_draft <= 0 or n < min_n + 1:
        return []
    max_n = min(int(max_n), n - 1)
    min_n = max(int(min_n), 1)
    for ng in range(max_n, min_n - 1, -1):
        needle = ids[n - ng :]
        limit = n - ng
        for start in range(limit - 1, -1, -1):
            if ids[start : start + ng] == needle:
                take = ids[start + ng : start + ng + num_draft]
                if take:
                    return [int(x) for x in take]
    return []


def overlay_ngram_on_drafts(
    drafts: list[list[int]],
    prefixes: list[list[int]],
    min_n: int = 2,
    max_n: int = 5,
) -> list[list[int]]:
    """Keep DSpark drafts; replace a prefix with prompt-lookup when it hits."""
    out: list[list[int]] = []
    for draft, prefix in zip(drafts, prefixes):
        looked = prefix_lookup_draft(list(prefix), len(draft), min_n, max_n)
        if not looked:
            out.append([int(x) for x in draft])
            continue
        merged = [int(x) for x in looked]
        if len(merged) < len(draft):
            merged.extend(int(x) for x in draft[len(merged) :])
        out.append(merged[: len(draft)])
    return out


def overlay_ngram_on_draft_tail(
    drafts: list[list[int]],
    prefixes: list[list[int]],
    start_pos: int = 3,
    min_n: int = 2,
    max_n: int = 5,
) -> list[list[int]]:
    """Keep DSpark drafts[:start_pos]. Fill the tail from prompt-lookup.

    Lookup is conditioned on prefix + the kept DSpark head so a hit continues
    that head. A miss leaves the Markov tail unchanged. Essay accept never
    used draft positions 3-4, so a miss cannot clip the 2.4 mean.
    """
    pos = int(start_pos)
    out: list[list[int]] = []
    for draft, prefix in zip(drafts, prefixes):
        d = [int(x) for x in draft]
        if pos <= 0:
            out.append(
                overlay_ngram_on_drafts([d], [list(prefix)], min_n, max_n)[0]
            )
            continue
        if pos >= len(d):
            out.append(d)
            continue
        head = d[:pos]
        tail_n = len(d) - pos
        looked = prefix_lookup_draft(list(prefix) + head, tail_n, min_n, max_n)
        if not looked:
            out.append(d)
            continue
        merged = head + [int(x) for x in looked]
        if len(merged) < len(d):
            merged.extend(d[len(merged) :])
        out.append(merged[: len(d)])
    return out


def cache_logits_then_argmax(
    logits: list[list[float]],
    cache: list[list[list[float]]],
    idx_map: list[int],
    step: int,
) -> list[int]:
    """Write pre-temperature draft logits into cache[req, step] and argmax.

    Greedy propose, softmax verify. Leviathan q is softmax(U+Markov), not
    one-hot. Ties prefer the lowest index, matching torch.argmax.
    """
    ids: list[int] = []
    step_i = int(step)
    for i, row in enumerate(logits):
        req = int(idx_map[i])
        cache[req][step_i] = [float(x) for x in row]
        best = 0
        best_v = float(row[0])
        for j, v in enumerate(row):
            fv = float(v)
            if fv > best_v:
                best_v = fv
                best = j
        ids.append(best)
    return ids


def dspark_softmax_verify_from_env(env_flag: int) -> bool:
    """True allocates draft_logits and caches them under greedy argmax."""
    return int(env_flag) == 1


def draft_logits_flat_index(idx_map: list[int], step: int, n_spec: int) -> list[int]:
    """Row in a [max_req * n_spec, vocab] view of draft_logits[req, step]."""
    n = int(n_spec)
    s = int(step)
    return [int(i) * n + s for i in idx_map]


def dspark_tail_ngram_start_pos(env_flag: int, env_pos: int = 3) -> int | None:
    """None disables the overlay. A non-negative pos is the first replaced index."""
    if int(env_flag) != 1:
        return None
    pos = int(env_pos)
    if pos < 0:
        return None
    return pos


def skip_indexer_for_short_context(
    max_seq_len: int, compress_ratio: int, topk: int
) -> bool:
    """Skip Lightning Indexer GEMM when every compressed token fits in topk.

    vLLM also requires ``not is_current_stream_capturing()``, so CUDA graphs
    bake in the full 32-head indexer even for a 61-token L.A.I.L prefill.
    """
    if int(compress_ratio) <= 0:
        return False
    return int(max_seq_len) // int(compress_ratio) <= int(topk)


INDEXER_SKIP_OLD = """            if (
                indexer_metadata.max_seq_len // self.compress_ratio <= self.topk_tokens
                and not torch.cuda.is_current_stream_capturing()
            ):"""
INDEXER_SKIP_NEW = """            if (
                self.compress_ratio > 0
                and indexer_metadata.max_seq_len // self.compress_ratio
                <= self.topk_tokens
            ):"""


def patch_indexer_short_context_source(src: str) -> str:
    """Let the short-context indexer skip run during CUDA graph capture."""
    if INDEXER_SKIP_NEW in src:
        return src
    if INDEXER_SKIP_OLD not in src:
        raise ValueError("short-context indexer skip not found")
    return src.replace(INDEXER_SKIP_OLD, INDEXER_SKIP_NEW, 1)


def scale_markov_bias(bias, scale: float):
    """Scale DSpark sequential Markov logits. 0 is backbone-only drafts."""
    if float(scale) == 1.0:
        return bias
    return bias * float(scale)


def language_model_only_from_env(argv: list[str], env) -> bool:
    """Workers may not see the CLI flag; LANGUAGE_MODEL_ONLY is the durable signal."""
    if "--language-model-only" in argv:
        return True
    return env.get("LANGUAGE_MODEL_ONLY", "0") == "1"


def clamp_prefill_swa_topk(window_size: int, max_image_tokens: int) -> int:
    """Largest FlashInfer SM120 DSV4 decode topk that fits window+image.

    Vision-on prefill wants 128+1024=1152, which is not in
    ``FLASHINFER_DSV4_DECODE_TOPK``. Decode still uses ``window_size`` (128).
    """
    raw = int(window_size) + int(max_image_tokens)
    if raw in FLASHINFER_DSV4_DECODE_TOPK:
        return raw
    legal = [t for t in FLASHINFER_DSV4_DECODE_TOPK if t <= raw]
    if not legal:
        return min(FLASHINFER_DSV4_DECODE_TOPK)
    return max(legal)


def swa_image_tokens_for_dispatch(window_size: int, max_image_tokens: int) -> int:
    """Image columns added to the FlashInfer SWA index row.

    SM120 dual-cache prefill (SWA + compressed, 38/40 backbone layers) is
    instantiated only for SWA topk=128. Returning extra image columns of 896
    or 1024 makes ``dispatch_dsv4_dual`` return false on the L.A.I.L 81-token
    prefill. Keep the index width at ``window_size``. This does not write
    hf_config.vision_max_n_token, so VL weights still load.
    """
    del window_size, max_image_tokens
    return 0


def flashinfer_prefill_swa_width(
    window_size: int, max_image_tokens: int, has_image: bool
) -> int:
    """Width passed to FlashInfer for a prefill SWA index row."""
    if not has_image or int(max_image_tokens) <= 0:
        if int(window_size) in FLASHINFER_DSV4_DECODE_TOPK:
            return int(window_size)
        return clamp_prefill_swa_topk(window_size, 0)
    return clamp_prefill_swa_topk(window_size, max_image_tokens)


SWA_IMAGE_WIDTH_OLD = """        self.max_image_tokens = (
            getattr(hf_config, "vision_max_n_token", 0)
            if getattr(hf_config, "vision_n_layers", 0) > 0
            else 0
        )
        self.prefill_index_width = self.window_size + self.max_image_tokens"""
SWA_IMAGE_WIDTH_NEW = """        self.max_image_tokens = (
            getattr(hf_config, "vision_max_n_token", 0)
            if getattr(hf_config, "vision_n_layers", 0) > 0
            else 0
        )
        from sm120_page import swa_image_tokens_for_dispatch
        self.max_image_tokens = swa_image_tokens_for_dispatch(
            self.window_size, self.max_image_tokens
        )
        self.prefill_index_width = self.window_size + self.max_image_tokens"""

ATTN_IMAGE_WIDTH_OLD = """        self.max_image_tokens = (
            getattr(config, "vision_max_n_token", 0)
            if getattr(config, "vision_n_layers", 0) > 0
            else 0
        )"""
ATTN_IMAGE_WIDTH_NEW = """        self.max_image_tokens = (
            getattr(config, "vision_max_n_token", 0)
            if getattr(config, "vision_n_layers", 0) > 0
            else 0
        )
        from sm120_page import swa_image_tokens_for_dispatch
        self.max_image_tokens = swa_image_tokens_for_dispatch(
            self.window_size, self.max_image_tokens
        )"""


def patch_swa_prefill_image_width_source(src: str) -> str:
    """Clamp SWA prefill allocation to a FlashInfer SM120 DSV4 topk."""
    if SWA_IMAGE_WIDTH_NEW in src:
        return src
    if SWA_IMAGE_WIDTH_OLD not in src:
        raise ValueError("SWA prefill image width assignment not found")
    return src.replace(SWA_IMAGE_WIDTH_OLD, SWA_IMAGE_WIDTH_NEW, 1)


def patch_attention_image_width_source(src: str) -> str:
    """Clamp DeepseekV4Attention SWA image columns the same way as sparse_swa."""
    if ATTN_IMAGE_WIDTH_NEW in src:
        return src
    if ATTN_IMAGE_WIDTH_OLD not in src:
        raise ValueError("Attention max_image_tokens assignment not found")
    return src.replace(ATTN_IMAGE_WIDTH_OLD, ATTN_IMAGE_WIDTH_NEW, 1)


def timed_call(clock, records: list, name: str, fn, *args, **kwargs):
    """Run fn and append (name, seconds) to records. clock() is monotonic seconds."""
    t0 = clock()
    out = fn(*args, **kwargs)
    records.append((name, clock() - t0))
    return out


def census_totals(records: list) -> dict[str, dict[str, float]]:
    """Sum wall seconds and call counts per name."""
    out: dict[str, dict[str, float]] = {}
    for name, dt in records:
        row = out.setdefault(name, {"n": 0.0, "s": 0.0})
        row["n"] += 1.0
        row["s"] += float(dt)
    return out


def format_census_totals(totals: dict[str, dict[str, float]]) -> str:
    parts = []
    for name in sorted(totals):
        row = totals[name]
        n = row["n"]
        s = row["s"]
        avg_ms = (1000.0 * s / n) if n else 0.0
        parts.append(f"{name} n={int(n)} sum_s={s:.3f} avg_ms={avg_ms:.1f}")
    return " ".join(parts)


def compressed_kv_len(seq_len: int, compress_ratio: int) -> int:
    """Compressed-KV tokens for a layer. Ratio 0/1 is uncompressed."""
    ratio = int(compress_ratio)
    tokens = int(seq_len)
    if ratio <= 1:
        return tokens
    return (tokens + ratio - 1) // ratio


def compressed_extra_topk_width(
    active_len: int, trained_topk: int, legal: frozenset | None = None
) -> int:
    """Smallest FlashInfer SM120 extra-index width that covers active KV.

    ``active_len`` is compressed-KV length (see compressed_kv_len). Extra
    tensors are allocated at ``index_topk`` (512). Split-K follows that width
    even when the sequence is shorter. Pick the smallest legal DSV4 topk that
    is >= min(active_len, trained_topk).
    """
    allowed = FLASHINFER_DSV4_DECODE_TOPK if legal is None else frozenset(int(x) for x in legal)
    need = max(1, min(int(active_len), int(trained_topk)))
    cover = [t for t in allowed if t >= need]
    if cover:
        return min(cover)
    under = [t for t in allowed if t <= int(trained_topk)]
    if under:
        return max(under)
    return int(trained_topk)


def extra_index_slice_width(
    seq_len: int, compress_ratio: int, trained_topk: int
) -> int:
    """Decode extra-index width for one layer's current sequence."""
    return compressed_extra_topk_width(
        compressed_kv_len(seq_len, compress_ratio), trained_topk
    )


def fill_engram_cache_hits(keys, hits: list, fetched: dict) -> list:
    """Rebuild a row list from cache hits and a fetched-miss map."""
    out = []
    for key, hit in zip(keys, hits):
        if hit is not None:
            out.append(hit)
        else:
            out.append(fetched[int(key)])
    return out


def decode_mhc_pre_num_splits(
    num_tokens: int, heuristic_splits: int, decode_cap: int = 16
) -> int:
    """Collapse DeepGEMM MHC prenorm split-K at decode widths.

    ``compute_mhc_pre_num_splits(20480, 6)`` is 16 because the heuristic keys
    off hidden width, not token count. A 6x20480x24 mix does not need 16
    splits. Prefill (num_tokens > decode_cap) keeps the heuristic.
    """
    if int(num_tokens) <= int(decode_cap):
        return 1
    return max(1, int(heuristic_splits))


def mask_logits_keep_topk(values: list[float], k: int) -> list[float]:
    """Keep the k largest scores; the rest become -inf. Ties prefer lower idx."""
    n = len(values)
    kk = int(k)
    if kk <= 0 or kk >= n:
        return list(values)
    ranked = sorted(range(n), key=lambda i: (-float(values[i]), i))
    keep = set(ranked[:kk])
    neginf = float("-inf")
    return [float(values[i]) if i in keep else neginf for i in range(n)]


def dspark_draft_topk_from_env(env_k: int) -> int | None:
    """None keeps dense Markov. A positive k masks compute_draft_logits to
    the backbone's top-k before dense _sample_sequential / markov_bias.
    Do not set hf_config.dspark_draft_topk (Qwen-only). k<=0 is dense."""
    k = int(env_k)
    if k <= 0:
        return None
    return k


def mla_chunks_per_block_from_env(env_k: int) -> int | None:
    """None keeps FlashInfer AutoTuner/heuristic. A positive k is baked as
    sparse_mla_sm120_decode_dsv4 chunks_per_block (tactic 1..num_splits).
    k<=0 is heuristic."""
    k = int(env_k)
    if k <= 0:
        return None
    return k


def clamp_index_topk(configured: int, clamp_to: int, legal: frozenset | None = None) -> int:
    """Cap indexer extra width at a FlashInfer SM120 DSV4 decode topk.

    clamp_to<=0 leaves the pack value. A value already in the dispatch set
    is used as-is. Anything else snaps down to the largest legal topk that
    still fits, so a typo cannot miss FLASHINFER_DSV4_DISPATCH.
    """
    configured_i = int(configured)
    clamp_i = int(clamp_to)
    if clamp_i <= 0:
        return configured_i
    allowed = FLASHINFER_DSV4_DECODE_TOPK if legal is None else frozenset(int(x) for x in legal)
    if clamp_i in allowed:
        return min(configured_i, clamp_i)
    under = [t for t in allowed if t <= clamp_i]
    if not under:
        return configured_i
    return min(configured_i, max(under))


def engram_row_cache_split(cache: dict, keys: list[int]) -> tuple[list, list[int]]:
    """Split hash-row ids into cache hits and unique misses, preserving key order."""
    hits = []
    missing: list[int] = []
    seen: set[int] = set()
    for raw in keys:
        key = int(raw)
        if key in cache:
            hits.append(cache[key])
            continue
        hits.append(None)
        if key not in seen:
            missing.append(key)
            seen.add(key)
    return hits, missing


def engram_row_cache_store(
    cache: dict, order: list, key: int, value, max_rows: int
) -> None:
    """Insert one row. Evict the oldest key when the cache is full."""
    key = int(key)
    cap = int(max_rows)
    if cap <= 0:
        cache.clear()
        order.clear()
        return
    if key in cache:
        order.remove(key)
    while order and len(order) >= cap:
        old = order.pop(0)
        cache.pop(old, None)
    cache[key] = value
    order.append(key)


def census_skip_during_capture(capturing: bool) -> bool:
    """CUDA graph capture cannot host a device synchronize around the wrap."""
    return bool(capturing)


def _gpu_sync() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def install_step_census(records: list | None = None) -> list:
    """Wrap draft, target forward, and Engram staging. Off unless called.

    CUDA graphs bake inner kernels, so this times graph replay and host
    staging, not individual MHC/MoE ops. One device sync before and after
    each GPU wrap so leftover work is not billed to the next bucket.

    pre_draft: target end to draft start on this rank (sampler, draft prep,
    their GPU work). Compare ranks: the rank with the larger pre_draft is
    the one the other waits for at the draft-start all-reduce.
    """
    import time

    sink: list = records if records is not None else []
    ticks = {"draft": 0, "target_end": None}

    def wrap_gpu(fn, name: str):
        def inner(*args, **kwargs):
            try:
                import torch

                capturing = (
                    torch.cuda.is_available()
                    and torch.cuda.is_current_stream_capturing()
                )
            except Exception:
                capturing = False
            if census_skip_during_capture(capturing):
                return fn(*args, **kwargs)
            _gpu_sync()
            if name == "draft" and ticks["target_end"] is not None:
                sink.append(("pre_draft", time.perf_counter() - ticks["target_end"]))
                ticks["target_end"] = None

            def run():
                out = fn(*args, **kwargs)
                _gpu_sync()
                return out

            out = timed_call(time.perf_counter, sink, name, run)
            if name == "target":
                ticks["target_end"] = time.perf_counter()
            if name == "draft":
                ticks["draft"] += 1
                if ticks["draft"] % 8 == 0:
                    _dump()
            return out

        return inner

    def wrap_cpu(fn, name: str):
        def inner(*args, **kwargs):
            return timed_call(time.perf_counter, sink, name, fn, *args, **kwargs)

        return inner

    try:
        from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

        DSparkSpeculator._generate_draft = wrap_gpu(
            DSparkSpeculator._generate_draft, "draft"
        )
    except Exception:
        pass
    try:
        from vllm.models.deepseek_v4_1.nvidia.model import DeepseekV4Model

        DeepseekV4Model.forward = wrap_gpu(DeepseekV4Model.forward, "target")
    except Exception:
        pass
    try:
        from vllm.models.deepseek_v4_1.common.engram import EngramDiskStager

        EngramDiskStager.stage = wrap_cpu(EngramDiskStager.stage, "engram")
    except Exception:
        pass

    def _dump() -> None:
        text = format_census_totals(census_totals(sink))
        if not text:
            return
        try:
            from vllm.logger import init_logger

            init_logger("vllm.dsv41_census").info("step census %s", text)
        except Exception:
            print("step census", text, flush=True)
        try:
            from pathlib import Path

            Path("/tmp/dsv41-step-census.txt").write_text(text + "\n")
        except OSError:
            pass

    import atexit

    atexit.register(_dump)
    return sink
