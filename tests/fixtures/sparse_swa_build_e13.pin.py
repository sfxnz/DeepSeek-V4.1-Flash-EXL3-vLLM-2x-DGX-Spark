# Pinned from image dsv41-flash-exl3-sm121:canonical-e13 (sha256:c81762335a12),
# vllm/v1/attention/backends/mla/sparse_swa.py:
# 1) the stock _compute_swa_indices_and_lens_kernel;
# 2) DeepseekSparseSWAMetadataBuilder.build (class line + the method, verbatim).
# docker/patch/swa_meta_fused.py (DSV41_SWA_META_FUSED) rewrites two blocks of (2)
# and mirrors the causal no-image path of (1).

# ---- (1) ----
@triton.jit(
    do_not_specialize=[
        "swa_indices_stride",
        "block_table_stride",
        "token_offset",
    ]
)
def _compute_swa_indices_and_lens_kernel(
    swa_indices_ptr,
    swa_indices_stride,
    swa_lens_ptr,
    window_size,
    index_width,
    left_visible_ptr,
    right_visible_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    token_to_req_indices_ptr,
    is_valid_token_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    token_offset,
    HAS_IMAGE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    token_idx = pid + token_offset
    is_valid = tl.load(is_valid_token_ptr + token_idx)
    if not is_valid:
        tl.store(swa_lens_ptr + pid, 0)
        # Clear the row so a padded token cannot gather through stale indices.
        for i in range(0, index_width, TRITON_BLOCK_SIZE):
            offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
            tl.store(
                swa_indices_ptr + pid * swa_indices_stride + offset,
                -1,
                mask=offset < index_width,
            )
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start

    seq_len = tl.load(seq_lens_ptr + req_idx)
    prefix_len = seq_len - query_len

    pos = prefix_len + token_idx - query_start
    if HAS_IMAGE:
        # In-image bidirectional visibility widens the window: the window
        # starts up to max(left - (window - 1), 0) positions earlier and
        # extends `right` positions past the query token.
        left = tl.load(left_visible_ptr + token_idx)
        right = tl.load(right_visible_ptr + token_idx)
    else:
        left = 0
        right = 0
    left_add = tl.maximum(left - (window_size - 1), 0)
    start_pos = tl.maximum(pos - (window_size - 1) - left_add, 0)
    end_pos = pos + right + 1

    swa_len = end_pos - start_pos
    tl.store(swa_lens_ptr + pid, swa_len)

    for i in range(0, index_width, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)

        pos_offset = start_pos + offset
        block_indices = pos_offset // block_size
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices,
            mask=pos_offset < end_pos,
        )
        block_offsets = pos_offset % block_size
        slot_ids = block_numbers * block_size + block_offsets

        slot_ids = tl.where(offset < swa_len, slot_ids, -1)
        tl.store(
            swa_indices_ptr + pid * swa_indices_stride + offset,
            slot_ids,
            mask=offset < index_width,
        )


# ---- (2) ----
class DeepseekSparseSWAMetadataBuilder(AttentionMetadataBuilder):
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekSparseSWAMetadata:
        """Build SWA metadata for mixed decode/prefill batches.

        The batch is assumed to be reordered with decodes first (by vLLM scheduler).
        We use split_decodes_and_prefills() to find the boundary, then build
        separate window_topk_idxs for each portion.

        For prefill, we use chunked prefill to align with the indexer's chunking.
        """
        seq_lens = common_attn_metadata.seq_lens
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        # Split into decode and prefill portions using configurable threshold
        (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens) = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=self.decode_threshold
            )
        )

        # NOTE: Ensure all metadata tensors maintain fixed memory addresses
        # for CUDA graph compatibility.
        token_to_req_indices = common_attn_metadata.token_to_req_indices(
            self.token_to_req_indices
        )

        is_valid_token = self.is_valid_token[: slot_mapping.shape[0]]
        is_valid_token.copy_(slot_mapping >= 0)

        non_causal = not common_attn_metadata.causal
        decode_swa_width = (
            self.noncausal_index_width if non_causal else self.window_size
        )
        decode_swa_indices = self.decode_swa_indices
        if num_decode_tokens > 0:
            self.decode_swa_lens[num_decode_tokens:] = 0
            if non_causal:
                assert self.is_dspark, (
                    "Non-causal DeepseekV4 SWA is only supported for the DSpark "
                    "speculation mode, but causal=False was set without DSpark."
                )
                if self.decode_swa_indices_noncausal is None:
                    self.decode_swa_indices_noncausal = torch.zeros(
                        self._max_tokens,
                        1,
                        self.noncausal_index_width,
                        dtype=torch.int32,
                        device=self.device,
                    )
                decode_swa_indices = self.decode_swa_indices_noncausal
                _COMPUTE_DSPARK_NONCAUSAL_SWA_INDICES_KERNEL(
                    decode_swa_indices,
                    self.decode_swa_lens,
                    self.window_size,
                    self.noncausal_index_width,
                    query_start_loc,
                    seq_lens,
                    token_to_req_indices,
                    is_valid_token,
                    block_table,
                    self.block_size,
                    num_tokens=num_decode_tokens,
                    token_offset=0,
                )
            else:
                _COMPUTE_SWA_INDICES_AND_LENS_KERNEL(
                    decode_swa_indices,
                    self.decode_swa_lens,
                    self.window_size,
                    decode_swa_indices.shape[-1],
                    self.decode_swa_lens,  # unused (HAS_IMAGE=False)
                    self.decode_swa_lens,  # unused (HAS_IMAGE=False)
                    query_start_loc,
                    seq_lens,
                    token_to_req_indices,
                    is_valid_token,
                    block_table,
                    self.block_size,
                    num_tokens=num_decode_tokens,
                    token_offset=0,
                )

        # Vision variant: per-token in-image visibility for prefill tokens.
        # Decode tokens are always past the image spans (spans are prefilled
        # atomically), so the decode path above never needs them.
        prefill_left_visible: torch.Tensor | None = None
        prefill_right_visible: torch.Tensor | None = None
        mm_ranges = common_attn_metadata.mm_req_doc_ranges
        if (
            self.max_image_tokens > 0
            and num_prefill_tokens > 0
            and mm_ranges
            and any(mm_ranges.values())
        ):
            prefill_left_visible, prefill_right_visible = self._build_image_visibility(
                common_attn_metadata.num_reqs,
                mm_ranges,
                num_decode_tokens,
                num_prefill_tokens,
                seq_lens,
                query_start_loc,
                token_to_req_indices,
            )

        # Prefill SWA indices live in paged coordinates. `token_offset` lets
        # the kernel read is_valid_token / token_to_req_indices at absolute
        # prefill positions while writing output starting at index 0.
        if num_prefill_tokens > 0:
            has_image = prefill_left_visible is not None
            prefill_swa_indices = self.prefill_swa_indices[:num_prefill_tokens]
            prefill_swa_lens = self.prefill_swa_lens[:num_prefill_tokens]
            _COMPUTE_SWA_INDICES_AND_LENS_KERNEL(
                prefill_swa_indices,
                prefill_swa_lens,
                self.window_size,
                self.prefill_index_width,
                prefill_left_visible if has_image else prefill_swa_lens,
                prefill_right_visible if has_image else prefill_swa_lens,
                query_start_loc,
                seq_lens,
                token_to_req_indices,
                is_valid_token,
                block_table,
                self.block_size,
                num_tokens=num_prefill_tokens,
                token_offset=num_decode_tokens,
                has_image=has_image,
            )

        # Pre-compute DeepseekV4 prefill metadata shared across all attention layers.
        deepseek_v4_fields = self._build_deepseek_v4_metadata(
            num_decodes,
            num_prefills,
            seq_lens,
            seq_lens_cpu,
            query_start_loc,
            query_start_loc_cpu,
        )

        # Per-layer-type tile-scheduler plan holders. Empty FlashMLASchedMeta
        # per present DeepseekV4 layer type; the first flash_mla_with_kvcache call of
        # each type triggers the planner and all same-type layers reuse the
        # resulting plan for the rest of the step.
        tile_sched = self.build_tile_scheduler(num_decode_tokens)

        return DeepseekSparseSWAMetadata(
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            block_table=block_table,
            slot_mapping=slot_mapping,
            is_valid_token=is_valid_token,
            token_to_req_indices=token_to_req_indices,
            decode_swa_indices=decode_swa_indices[:num_decode_tokens],
            decode_swa_lens=self.decode_swa_lens[:num_decode_tokens],
            decode_swa_width=decode_swa_width,
            prefill_swa_indices=(
                self.prefill_swa_indices[:num_prefill_tokens]
                if num_prefill_tokens > 0
                else None
            ),
            prefill_swa_lens=(
                self.prefill_swa_lens[:num_prefill_tokens]
                if num_prefill_tokens > 0
                else None
            ),
            prefill_left_visible=prefill_left_visible,
            prefill_right_visible=prefill_right_visible,
            block_size=self.block_size,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_prefill_tokens,
            # Upper bound on decode-split rows for the kernel's max_q_len
            # hint. common max_query_len bounds every row (scheduled max under
            # adaptive verification), clamped to what the split can admit so a
            # mixed batch's prefill max does not inflate decode scheduling.
            max_decode_query_len=min(
                common_attn_metadata.max_query_len, self.decode_threshold
            ),
            tile_sched_swaonly=tile_sched[_LAYER_TYPE_SWAONLY],
            tile_sched_c4a=tile_sched[_LAYER_TYPE_C4A],
            tile_sched_c128a=tile_sched[_LAYER_TYPE_C128A],
            tile_sched_c1a=tile_sched[_LAYER_TYPE_C1A],
            tile_sched_c2a=tile_sched[_LAYER_TYPE_C2A],
            **deepseek_v4_fields,  # type: ignore[arg-type]
        )
