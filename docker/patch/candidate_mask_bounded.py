"""Decode candidate mask bounded by the row length (DSV41_CANDIDATE_MASK_BOUNDED=1).

V4.1's candidate-consumer indexer layers (4 per decode step) run
apply_candidate_mask on the decode logits before top_k_per_row_decode. The
decode logits are [rows, max_model_len] (paged_mqa_logits with
clean_logits=False: only [0, seq_len) is written), so the stock kernels zero a
(1M/8 + 1)-slot flag row and write -inf into every column past seq_len: the r3
trace has _mask_candidates_kernel at 70.4 us and _candidate_flags_kernel at
7.3 us per call, grid (4, 1024), 0.34 ms/step of in-graph time.

top_k_per_row_decode reads each row only up to its own end (seq_len - next_n +
j + 1 <= seq_len), so nothing past seq_len is ever read. With the lever the
decode call (no row starts) launches two variants: the flags kernel zeroes
only the slots the mask reads (blocks below the row end, plus the edge slot),
and the mask kernel's programs whose 1024-column tile starts at or past the
row end return at once. Every column below the row end gets exactly the stock
value. Prefill calls (row starts given) keep the stock function. The first
DSV41_CANDIDATE_MASK_VERIFY eager decode calls (default 4, minimum 1; never
during CUDA graph capture) also run the stock mask on a copy and compare
every column below each row's end bit for bit; a mismatch prints one
LOG_DISARMED line, keeps the stock result and disarms for good. Decode calls
replay CUDA graphs, so eager decode calls may never come: the first eager call
of any kind (a prefill's) also self-tests the bounded kernels on synthetic
decode logits (rows 1/4/8, ends odd, below one block and at the width) and
prints LOG_ENGAGED or disarms.

Top-level imports are stdlib only. decode_levers.install() calls install()
when the env is on.
"""

from __future__ import annotations

import os

# Boot-log marker for tools/engagement_audit.py.
LOG_ENGAGED = "dsv41: candidate mask bounded self-check bit-exact"
LOG_DISARMED = "dsv41: candidate mask bounded DISABLED ->"

TILE = 1024
DEFAULT_VERIFY = 4
_STATE = {"armed": True, "verify_left": DEFAULT_VERIFY, "verified": 0, "engaged": False}


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return env.get("DSV41_CANDIDATE_MASK_BOUNDED", "0") == "1"


def verify_calls(env=None) -> int:
    env = os.environ if env is None else env
    return max(1, int(env.get("DSV41_CANDIDATE_MASK_VERIFY", "") or DEFAULT_VERIFY))


def _disarm(exc: BaseException) -> None:
    if _STATE["armed"]:
        _STATE["armed"] = False
        print("dsv41: candidate mask bounded DISABLED -> stock mask: %r" % (exc,), flush=True)


def _build_kernels(tl, triton):
    @triton.jit(do_not_specialize=["width", "nblocks"])
    def _candidate_flags_bounded_kernel(
        candidates, ends, flags,
        stride_row, stride_col, stride_end,
        width, nblocks,
        BLOCK_SIZE: tl.constexpr, K: tl.constexpr, ROW_REPEAT: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        end = tl.load(ends + row // ROW_REPEAT * stride_end).to(tl.int64)
        # the mask reads flags[block] for block < cdiv(min(end, width), BS), and flags[nblocks]
        live = tl.minimum((tl.minimum(end, width) + BLOCK_SIZE - 1) // BLOCK_SIZE, nblocks)
        offsets = tl.arange(0, 1024)
        for tile in range(tl.cdiv(live, 1024)):
            slots = tile * 1024 + offsets
            tl.store(flags + row * (nblocks + 1) + slots, 0, slots < live)
        tl.store(flags + row * (nblocks + 1) + nblocks, 0)
        cols = tl.arange(0, triton.next_power_of_2(K))
        block = tl.load(
            candidates + row * stride_row + cols * stride_col, cols < K, other=-1
        ).to(tl.int64)
        block = tl.where(block * BLOCK_SIZE >= width, nblocks, block)
        tl.debug_barrier()
        tl.store(flags + row * (nblocks + 1) + block, 1, (cols < K) & (block >= 0))

    @triton.jit(do_not_specialize=["width", "nblocks"])
    def _mask_candidates_bounded_kernel(
        logits, ends, flags,
        stride_row, stride_col, stride_end,
        width, nblocks,
        BLOCK_SIZE: tl.constexpr, ROW_REPEAT: tl.constexpr, TILE: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        tile0 = tl.program_id(1) * TILE
        end = tl.load(ends + row // ROW_REPEAT * stride_end)
        if tile0 < end:
            cols = tile0 + tl.arange(0, TILE)
            valid = (cols >= 0) & (cols < end) & (cols < width)
            block = cols // BLOCK_SIZE
            keep = tl.load(flags + row * (nblocks + 1) + block, valid, other=0)
            edge = tl.load(flags + row * (nblocks + 1) + nblocks)
            keep = (keep != 0) | ((cols == width - 1) & (edge != 0))
            tl.store(
                logits + row * stride_row + cols * stride_col,
                -float("inf"),
                (cols < width) & ~(valid & keep),
            )

    return _candidate_flags_bounded_kernel, _mask_candidates_bounded_kernel


def make_apply(torch, triton, stock_apply, flags_kernel, mask_kernel):
    """apply_candidate_mask with the bounded kernels for decode calls."""

    def bounded(logits, row_ke, candidate_blocks, block_size, row_repeat):
        rows, width = logits.shape
        nblocks = triton.cdiv(width, block_size)
        flags = torch.empty((rows, nblocks + 1), device=logits.device, dtype=torch.uint8)
        flags_kernel[(rows,)](
            candidate_blocks, row_ke, flags,
            *candidate_blocks.stride(), row_ke.stride(0),
            width, nblocks,
            block_size, candidate_blocks.shape[1], row_repeat,
        )
        mask_kernel[(rows, triton.cdiv(width, TILE))](
            logits, row_ke, flags,
            *logits.stride(), row_ke.stride(0),
            width, nblocks,
            block_size, row_repeat, TILE,
        )

    def verify(logits, ref, row_ke, row_repeat) -> None:
        rows, width = logits.shape
        ends = row_ke.cpu()
        for r in range(rows):
            e = min(int(ends[r // row_repeat]), width)
            if e > 0 and not torch.equal(logits[r, :e].view(torch.int32), ref[r, :e].view(torch.int32)):
                raise RuntimeError(f"bounded != stock below the end (row {r}, end {e})")

    def selftest(device, block_size) -> None:
        g = torch.Generator(device=device).manual_seed(0)
        width = 65536
        for rows in (1, 4, 8):
            ends = torch.randint(1, 40000, (rows,), device=device, generator=g, dtype=torch.int32)
            ends[0] = 37 * block_size + 3
            if rows > 1:
                ends[1] = block_size - 3
            if rows > 2:
                ends[2] = width
            nb = triton.cdiv(width, block_size)
            cand = torch.randint(-1, nb + 3, (rows, 2048), device=device, generator=g, dtype=torch.int32)
            logits = torch.randn(rows, width, device=device, generator=g)
            ref = logits.clone()
            bounded(logits, ends, cand, block_size, 1)
            stock_apply(ref, None, ends, cand, block_size, 1)
            verify(logits, ref, ends, 1)

    def apply_candidate_mask(logits, row_ks, row_ke, candidate_blocks, block_size, row_repeat=1):
        if _STATE["armed"] and not _STATE["engaged"] and not torch.cuda.is_current_stream_capturing():
            try:
                selftest(logits.device, block_size)
                _STATE["engaged"] = True
                print(
                    "dsv41: candidate mask bounded self-check bit-exact (rows 1/4/8, width 65536)",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                _disarm(exc)
        if row_ks is not None or not _STATE["armed"]:
            return stock_apply(logits, row_ks, row_ke, candidate_blocks, block_size, row_repeat)
        assert logits.is_cuda
        rows, width = logits.shape
        if not rows or not width:
            return
        check = _STATE["verify_left"] > 0 and not torch.cuda.is_current_stream_capturing()
        ref = logits.clone() if check else None
        bounded(logits, row_ke, candidate_blocks, block_size, row_repeat)
        if check:
            stock_apply(ref, row_ks, row_ke, candidate_blocks, block_size, row_repeat)
            try:
                verify(logits, ref, row_ke, row_repeat)
            except RuntimeError as exc:
                _disarm(exc)
                logits.copy_(ref)
                return
            _STATE["verify_left"] -= 1
            _STATE["verified"] += 1

    apply_candidate_mask._dsv41_bounded = True
    return apply_candidate_mask


def install() -> str:
    import torch
    from vllm.model_executor.layers import sparse_attn_indexer as sai
    from vllm.triton_utils import tl, triton

    stock = sai._apply_candidate_mask
    if getattr(stock, "_dsv41_bounded", False):
        return "already installed"
    _STATE["verify_left"] = verify_calls()
    flags_k, mask_k = _build_kernels(tl, triton)
    sai._apply_candidate_mask = make_apply(torch, triton, stock, flags_k, mask_k)
    return f"sparse_attn_indexer._apply_candidate_mask wrapped (decode calls, verify={_STATE['verify_left']})"
