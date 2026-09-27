# Pinned from image dsv41-flash-exl3-sm121:canonical-e13 (sha256:c81762335a12),
# vllm/v1/attention/backend.py, class CommonAttentionMetadata (method only).
# decode_levers DSV41_ATTN_T2R_DEDUP relies on its per-instance cache contract.


class CommonAttentionMetadata:
    _token_to_req_indices_cache = None

    def token_to_req_indices(self, buffer: torch.Tensor) -> torch.Tensor:
        """Build or reuse the per-token request index mapping."""
        num_tokens = self.num_actual_tokens
        if self._token_to_req_indices_cache is not None:
            assert self._token_to_req_indices_cache.device == buffer.device
            assert self._token_to_req_indices_cache.dtype == torch.int32
            assert self._token_to_req_indices_cache.shape[0] >= num_tokens
            return self._token_to_req_indices_cache[:num_tokens]

        # Built from the device query_start_loc: adaptive verification decides the
        # per-request draft split on device, so the CPU copy carries the right total
        # but not the right per-request boundaries. Padding requests have a query
        # length of zero and drop out of the repeat.
        num_mapped_tokens = int(self.query_start_loc_cpu[-1])
        query_lens = self.query_start_loc[1:] - self.query_start_loc[:-1]
        assert buffer.shape[0] >= max(num_mapped_tokens, num_tokens)
        token_to_req_indices = torch.repeat_interleave(
            torch.arange(query_lens.shape[0], dtype=torch.int32, device=buffer.device),
            query_lens,
            output_size=num_mapped_tokens,
        )
        buffer[:num_mapped_tokens].copy_(token_to_req_indices)
        if num_mapped_tokens < num_tokens:
            buffer[num_mapped_tokens:num_tokens].zero_()
        self._token_to_req_indices_cache = buffer[: max(num_mapped_tokens, num_tokens)]
        return self._token_to_req_indices_cache[:num_tokens]
