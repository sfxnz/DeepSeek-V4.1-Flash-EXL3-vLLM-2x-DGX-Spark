#!/usr/bin/env python3
"""Stage disk Engram rows in prepare_inputs so FULL CUDA graphs can capture.

Tony/Kai GB10 path: hash + pread + H2D happen before the forward. The captured
forward only reads persistent staged_rows. File rows stay global ids (this
pack's shards are unsharded tables). Idempotent.
"""
from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "DSV41_ENGRAM_PRESTAGE graph-safe disk staging"

PREPARE_OLD = '''    def prepare_embeddings(self, hash_ids: torch.Tensor) -> None:
        """Gather this layer's rows on the main stream before decoder layers.

        `hash_ids` covers every DP replica sharing the table (see
        `gather_engram_hashes`), so only `embed` narrows back to this one.
        """
        rows = self._staged_rows_for_ubatch()[: hash_ids.shape[0]]
        assert rows.shape[0] == hash_ids.shape[0], "engram staging buffer too small"
        self.embed_tokens.lookup(hash_ids, rows)
'''

PREPARE_NEW = '''    def prepare_embeddings(self, hash_ids: torch.Tensor) -> None:
        """Gather this layer's rows on the main stream before decoder layers.

        `hash_ids` covers every DP replica sharing the table (see
        `gather_engram_hashes`), so only `embed` narrows back to this one.
        """
        # --- ''' + MARKER + ''' ---
        if getattr(self, "prestage", False):
            return
        rows = self._staged_rows_for_ubatch()[: hash_ids.shape[0]]
        assert rows.shape[0] == hash_ids.shape[0], "engram staging buffer too small"
        self.embed_tokens.lookup(hash_ids, rows)
'''

STAGED_OLD = """        self.staged_rows = torch.empty(
            max_tokens * self.embed_tokens.dp_size,
            self.embed_tokens.part_n_hash_cols,
            layout.head_dim,
            dtype=torch.bfloat16,
        )
"""

STAGED_NEW = """        self.staged_rows = torch.zeros(
            max_tokens * self.embed_tokens.dp_size,
            self.embed_tokens.part_n_hash_cols,
            layout.head_dim,
            dtype=torch.bfloat16,
        )
        # --- """ + MARKER + """ ---
        self.prestage = self.embed_tokens.disk is not None and bool(
            getattr(vllm_config, "use_v2_model_runner", False)
        )
"""

DISK_OLD = '''    def _disk_lookup(self, indices: torch.Tensor, out: torch.Tensor) -> None:
        """Host gather + dequant, then H2D into `out` ([T, local_heads, dim] bf16)."""
        t = indices.shape[0]
        local_heads = self.part_n_hash_cols
        ids = indices.detach().to("cpu", dtype=torch.int64)
        head_end = min(self.head_start + local_heads, self.n_hash_cols)
        local = ids[:, self.head_start:head_end]
        if local.shape[1] < local_heads:
            pad = torch.full((t, local_heads - local.shape[1]), -1, dtype=torch.int64)
            local = torch.cat([local, pad], dim=1)
        rows = local.reshape(-1)
        owned = (rows >= self.vocab_start_idx) & (rows < self.vocab_end_idx)
        # HF shards store the full unsharded table; file rows are global ids.
        file_rows = torch.where(owned, rows, torch.zeros_like(rows))
        assert self.disk is not None
        deq = self.disk.gather_dequant(file_rows, owned)
        out[:t].copy_(deq.view(t, local_heads, self.dim))
'''

DISK_NEW = '''    def disk_file_rows_owned(self, local: torch.Tensor):
        """local: [T, heads] int64 CPU hash ids -> (file_rows, owned) flat.

        HF shards store the full unsharded table, so file rows are global ids.
        """
        t = local.shape[0]
        local_heads = self.part_n_hash_cols
        if local.shape[1] < local_heads:
            pad = torch.full((t, local_heads - local.shape[1]), -1, dtype=torch.int64)
            local = torch.cat([local, pad], dim=1)
        rows = local.reshape(-1)
        owned = (rows >= self.vocab_start_idx) & (rows < self.vocab_end_idx)
        file_rows = torch.where(owned, rows, torch.zeros_like(rows))
        return file_rows, owned

    def _disk_lookup(self, indices: torch.Tensor, out: torch.Tensor) -> None:
        """Host gather + dequant, then H2D into `out` ([T, local_heads, dim] bf16)."""
        # --- ''' + MARKER + ''' ---
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Engram DISK lookup reached a CUDA-graph capture. Stage rows "
                "in prepare_inputs (EngramDiskStager) or run --enforce-eager."
            )
        t = indices.shape[0]
        local_heads = self.part_n_hash_cols
        ids = indices.detach().to("cpu", dtype=torch.int64)
        head_end = min(self.head_start + local_heads, self.n_hash_cols)
        file_rows, owned = self.disk_file_rows_owned(ids[:, self.head_start:head_end])
        assert self.disk is not None
        deq = self.disk.gather_dequant(file_rows, owned)
        out[:t].copy_(deq.view(t, local_heads, self.dim))
'''

STAGER = '''


class EngramDiskStager:
    """Stage DISK Engram rows in prepare_inputs, outside a captured forward.

    One GPU hash, one host sync, pread+dequant on CPU, async H2D into each
    layer's persistent staged_rows. The forward only reads that buffer.
    File rows are global ids (unsharded HF shards).
    """

    def __init__(self, hash_state: NgramHashState, engrams: list) -> None:
        assert engrams and all(e.embed_tokens.disk is not None for e in engrams)
        self.hash_state = hash_state
        self.engrams = sorted(engrams, key=lambda e: e.layer_hash_index)
        emb = self.engrams[0].embed_tokens
        self.local_heads = emb.part_n_hash_cols
        self.head_start = emb.head_start
        self.head_end = min(emb.head_start + emb.part_n_hash_cols, emb.n_hash_cols)
        self.dim = emb.dim
        self.max_tokens = self.engrams[0].staged_rows.shape[0]
        num_layers = hash_state.multipliers.shape[0]
        self.hash_host = torch.empty(
            (self.max_tokens, num_layers, self.head_end - self.head_start),
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.rows_host = [
            torch.empty(
                (self.max_tokens, self.local_heads, self.dim),
                dtype=torch.bfloat16,
                device="cpu",
                pin_memory=True,
            )
            for _ in self.engrams
        ]
        self.hashes_ready = torch.cuda.Event()
        self.num_staged = 0

    @torch.inference_mode()
    def stage(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        lookback_token_ids: torch.Tensor,
        num_tokens: int,
    ) -> int:
        n = min(int(num_tokens), self.max_tokens)
        if n <= 0 or not self.hash_state.ensure_cache():
            return 0
        from .mm_preprocess import image_sentinel_mask

        ids = input_ids[:n]
        hashes = self.hash_state(
            ids,
            positions[:n],
            query_start_loc,
            image_sentinel_mask(ids),
            lookback_token_ids,
            image_sentinel_mask(lookback_token_ids),
            None,
            None,
        )
        host = self.hash_host[:n]
        host.copy_(hashes[:, :, self.head_start : self.head_end], non_blocking=True)
        self.hashes_ready.record()
        self.hashes_ready.synchronize()
        for engram, buf in zip(self.engrams, self.rows_host):
            local = host[:, engram.layer_hash_index, :].to(torch.int64)
            file_rows, owned = engram.embed_tokens.disk_file_rows_owned(local)
            rows = engram.embed_tokens.disk.gather_dequant(file_rows, owned)
            staged = buf[:n]
            staged.copy_(rows.view(n, self.local_heads, self.dim))
            dest = engram._staged_rows_for_ubatch()[:n]
            dest.copy_(staged, non_blocking=True)
        self.num_staged = n
        return n
'''

MS_IMPORT_OLD = """from vllm.config import VllmConfig
from vllm.triton_utils import tl, triton
"""

MS_IMPORT_NEW = """from vllm.config import VllmConfig
from vllm.models.deepseek_v4_1.common.engram import (
    Engram,
    EngramDiskStager,
    NgramHashState,
)
from vllm.triton_utils import tl, triton
"""

MS_INIT_OLD = """            self.lookback_token_ids = torch.full(
                (self.max_num_reqs, depth), -1, dtype=torch.int32, device=device
            )

    def prepare_inputs(
"""

MS_INIT_NEW = """            self.lookback_token_ids = torch.full(
                (self.max_num_reqs, depth), -1, dtype=torch.int32, device=device
            )

        # --- """ + MARKER + """ ---
        self.engram_stager: EngramDiskStager | None = None
        engrams = [m for m in model.modules() if isinstance(m, Engram) and m.prestage]
        if engrams:
            hash_states = [m for m in model.modules() if isinstance(m, NgramHashState)]
            assert len(hash_states) == 1, (
                f"expected one NgramHashState, found {len(hash_states)}"
            )
            self.engram_stager = EngramDiskStager(hash_states[0], engrams)

    def prepare_inputs(
"""

MS_PREP_OLD = """        model_inputs[\"lookback_token_ids\"] = window
        return model_inputs
"""

MS_PREP_NEW = """        model_inputs[\"lookback_token_ids\"] = window
        if self.engram_stager is not None and input_batch.input_ids is not None:
            positions = model_inputs.get(\"positions\")
            self.engram_stager.stage(
                input_batch.input_ids,
                positions if positions is not None else input_batch.positions,
                input_batch.query_start_loc[: input_batch.num_reqs + 1],
                window,
                input_batch.num_tokens,
            )
        return model_inputs
"""


def _once(src: str, old: str, new: str, label: str) -> str:
    if MARKER in src and old not in src:
        return src
    if old not in src:
        raise SystemExit(f"apply_engram_prestage: missing marker {label}")
    return src.replace(old, new, 1)


def patch_engram(src: str) -> str:
    if MARKER in src and "class EngramDiskStager" in src:
        return src
    src = _once(src, STAGED_OLD, STAGED_NEW, "staged_rows")
    src = _once(src, PREPARE_OLD, PREPARE_NEW, "prepare_embeddings")
    src = _once(src, DISK_OLD, DISK_NEW, "_disk_lookup")
    if "class EngramDiskStager" not in src:
        if not src.endswith("\n"):
            src += "\n"
        src += STAGER
    return src


def patch_model_state(src: str) -> str:
    if MARKER in src and "EngramDiskStager" in src:
        return src
    src = _once(src, MS_IMPORT_OLD, MS_IMPORT_NEW, "model_state import")
    src = _once(src, MS_INIT_OLD, MS_INIT_NEW, "model_state init")
    src = _once(src, MS_PREP_OLD, MS_PREP_NEW, "model_state prepare")
    return src


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engram", type=Path, required=True)
    ap.add_argument("--model-state", type=Path, required=True)
    args = ap.parse_args()
    args.engram.write_text(patch_engram(args.engram.read_text()))
    args.model_state.write_text(patch_model_state(args.model_state.read_text()))
    print(f"apply_engram_prestage: patched {args.engram} and {args.model_state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
