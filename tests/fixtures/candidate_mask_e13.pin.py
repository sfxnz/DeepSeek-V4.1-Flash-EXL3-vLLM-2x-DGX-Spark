# Pinned from image dsv41-flash-exl3-sm121:canonical-e13 (sha256:c81762335a12).
# 1) vllm/model_executor/kernels/attention/dsa/candidate_blocks.py: the stock flags
#    and mask kernels and apply_candidate_mask;
# 2) vllm/model_executor/layers/sparse_attn_indexer.py: the decode call site and the
#    decode top-k call (sm_120 family takes top_k_per_row_decode).
# docker/patch/candidate_mask_bounded.py (DSV41_CANDIDATE_MASK_BOUNDED) mirrors (1).

# ---- (1) candidate_blocks.py ----
@triton.jit(do_not_specialize=["width", "nblocks"])
def _candidate_flags_kernel(
    candidates,
    starts,
    flags,
    stride_row,
    stride_col,
    stride_start,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    K: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, 1024)
    for tile in range(tl.cdiv(nblocks + 1, 1024)):
        slots = tile * 1024 + offsets
        tl.store(flags + row * (nblocks + 1) + slots, 0, slots <= nblocks)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    cols = tl.arange(0, triton.next_power_of_2(K))
    block = tl.load(
        candidates + row * stride_row + cols * stride_col, cols < K, other=-1
    ).to(tl.int64)
    # Preserve the packed-column clamp for candidates beyond the logits width.
    block = tl.where(start + block * BLOCK_SIZE >= width, nblocks, block)
    tl.debug_barrier()
    tl.store(flags + row * (nblocks + 1) + block, 1, (cols < K) & (block >= 0))


@triton.jit(do_not_specialize=["width", "nblocks"])
def _mask_candidates_kernel(
    logits,
    starts,
    ends,
    flags,
    stride_row,
    stride_col,
    stride_start,
    stride_end,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    valid = (cols >= start) & (cols < end) & (cols < width)
    block = (cols - start) // BLOCK_SIZE
    keep = tl.load(flags + row * (nblocks + 1) + block, valid, other=0)
    edge = tl.load(flags + row * (nblocks + 1) + nblocks)
    keep = (keep != 0) | ((cols == width - 1) & (edge != 0))
    tl.store(
        logits + row * stride_row + cols * stride_col,
        -float("inf"),
        (cols < width) & ~(valid & keep),
    )


def apply_candidate_mask(
    logits: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    candidate_blocks: torch.Tensor,
    block_size: int,
    row_repeat: int = 1,
) -> None:
    """Mask packed logits outside causal bounds and request-local candidates."""
    assert logits.is_cuda
    rows, width = logits.shape
    if not rows or not width:
        return
    nblocks = triton.cdiv(width, block_size)
    flags = torch.empty((rows, nblocks + 1), device=logits.device, dtype=torch.uint8)
    start_stride = row_ks.stride(0) if row_ks is not None else 0
    _candidate_flags_kernel[(rows,)](
        candidate_blocks,
        row_ks,
        flags,
        *candidate_blocks.stride(),
        start_stride,
        width,
        nblocks,
        block_size,
        candidate_blocks.shape[1],
        row_ks is not None,
        row_repeat,
    )
    _mask_candidates_kernel[(rows, triton.cdiv(width, 1024))](
        logits,
        row_ks,
        row_ke,
        flags,
        *logits.stride(),
        start_stride,
        row_ke.stride(0),
        width,
        nblocks,
        block_size,
        row_ks is not None,
        row_repeat,
        1024,
    )

# ---- (2) sparse_attn_indexer.py, decode path (excerpt, not importable) ----
#         num_rows = logits.shape[0]
#         if candidate_blocks is not None:
#             # Two-level selection (v4.1) on the decode logits; columns are
#             # request-local compressed positions. seq_lens is (B, next_n)
#             # for native spec decode (per-row effective lens) and (B, 1)
#             # otherwise.
#             vis = seq_lens.reshape(-1)
#             row_repeat = next_n if vis.numel() != num_rows else 1
#             vis = vis[:num_rows]
#             decode_candidates = candidate_blocks[:num_rows]
#             if candidate_write:
#                 _select_candidate_blocks(
#                     logits,
#                     None,
#                     vis,
#                     decode_candidates.shape[1],
#                     candidate_block_size,
#                     decode_candidates,
#                     row_repeat,
#                 )
#             else:
#                 _apply_candidate_mask(
#                     logits,
#                     None,
#                     vis,
#                     decode_candidates,
#                     candidate_block_size,
#                     row_repeat,
#                 )
#             ops.top_k_per_row_decode(
#                 logits,
#                 next_n,
#                 seq_lens,
#                 topk_indices,
#                 num_rows,
#                 logits.stride(0),
#                 logits.stride(1),
#                 topk_tokens,
#             )
# ---- (3) sparse_attn_indexer.py, the decode logits (excerpt) ----
#             logits = fp8_fp4_paged_mqa_logits(
#                 (padded_q_quant_cast, padded_q_scale),
#                 kv_cache,
#                 weights[:num_padded_tokens],
#                 seq_lens,
#                 decode_metadata.block_table,
#                 decode_metadata.schedule_metadata,
#                 max_model_len=max_model_len,
#                 clean_logits=False,
#                 indices=decode_metadata.indices,
#             )
