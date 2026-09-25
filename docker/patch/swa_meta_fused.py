"""SWA decode metadata as one kernel per KV-cache group (DSV41_SWA_META_FUSED=1).

DeepseekSparseSWAMetadataBuilder.build runs, for every KV-cache group and every
decode step, four eager GPU ops after the token -> request map: is_valid_token
= slot_mapping >= 0 (compare + DtoD copy), decode_swa_lens[n:] = 0 (fill) and
_compute_swa_indices_and_lens_kernel. The r3 trace has 15 such causal builds
per target step (plus 3 non-causal DSpark draft builds), queued back to back
at ~4.1 us per op.

With the lever, a causal pure-decode build (no prefill tokens, causal is the
bool True, decode tokens <= slot_mapping rows) launches one Triton kernel that
writes the same three buffers: programs below num_decode_tokens compute their
token's validity from slot_mapping and then the stock SWA indices/lens
arithmetic (integer, so bit-exact by construction); the other programs write
validity for the remaining slots and zero the lens tail up to the buffer end.
Every other build (prefill, mixed, non-causal DSpark draft groups, images) and
update_draft_decode_metadata are the stock code. The build method is the
image's own source with two blocks rewritten (verbatim anchors, else one
LOG_DISARMED line and the stock method stays). The first DSV41_SWA_META_VERIFY
fused builds (default 8, minimum 1) run the stock ops into the real buffers,
then the fused kernel over poisoned buffers, and compare every written
element; a mismatch or a launch error prints one LOG_DISARMED line, puts the
stock values back and disarms for good. The builder runs eagerly each step
(outside the CUDA graphs), so the check sees real batches.

Top-level imports are stdlib only. decode_levers.install() calls install()
when the env is on.
"""

from __future__ import annotations

import os
import re

# Boot-log markers for tools/engagement_audit.py.
LOG_ENGAGED = "dsv41: swa metadata fused self-check bit-exact"
LOG_DISARMED = "dsv41: swa metadata fused DISABLED ->"

DEFAULT_VERIFY = 8
CHUNK = 1024
_STATE = {"armed": True, "verify_left": DEFAULT_VERIFY, "engaged": False}

VALID_OLD = """        is_valid_token = self.is_valid_token[: slot_mapping.shape[0]]
        is_valid_token.copy_(slot_mapping >= 0)

        non_causal = not common_attn_metadata.causal
"""
VALID_NEW = """        is_valid_token = self.is_valid_token[: slot_mapping.shape[0]]
        # dsv41 swa_meta_fused: one kernel for validity, lens tail and indices.
        _dsv41_fused = _dsv41_swa.fused_ok(
            self, common_attn_metadata, num_decode_tokens, num_prefill_tokens
        )
        if not _dsv41_fused:
            is_valid_token.copy_(slot_mapping >= 0)

        non_causal = not common_attn_metadata.causal
"""
DECODE_OLD = """        if num_decode_tokens > 0:
            self.decode_swa_lens[num_decode_tokens:] = 0
            if non_causal:
"""
DECODE_NEW = """        if num_decode_tokens > 0 and _dsv41_fused:
            _dsv41_swa.decode(
                self,
                decode_swa_indices,
                query_start_loc,
                seq_lens,
                token_to_req_indices,
                slot_mapping,
                is_valid_token,
                block_table,
                num_decode_tokens,
            )
        elif num_decode_tokens > 0:
            self.decode_swa_lens[num_decode_tokens:] = 0
            if non_causal:
"""


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return env.get("DSV41_SWA_META_FUSED", "0") == "1"


def verify_calls(env=None) -> int:
    env = os.environ if env is None else env
    return max(1, int(env.get("DSV41_SWA_META_VERIFY", "") or DEFAULT_VERIFY))


def patch_source(src: str) -> str:
    # The method is re-compiled outside its class: no zero-arg super(), no
    # __class__ cell, no name-mangled attributes.
    if "super()" in src or "__class__" in src or re.search(r"\.__[A-Za-z]\w*(?<!__)\b", src):
        raise ValueError("DeepseekSparseSWAMetadataBuilder.build needs its class scope")
    for old, new, what in ((VALID_OLD, VALID_NEW, "validity block"), (DECODE_OLD, DECODE_NEW, "decode branch")):
        if src.count(old) != 1:
            raise ValueError(f"{what} of DeepseekSparseSWAMetadataBuilder.build not found verbatim")
        src = src.replace(old, new)
    return src


def _disarm(exc: BaseException) -> None:
    if _STATE["armed"]:
        _STATE["armed"] = False
        print("dsv41: swa metadata fused DISABLED -> stock ops: %r" % (exc,), flush=True)


def _build_kernel(tl, triton):
    @triton.jit(do_not_specialize=["num_decode", "num_slots", "max_tokens", "swa_stride", "bt_stride"])
    def _swa_decode_fused_kernel(
        swa_indices, swa_stride, swa_lens, window_size, index_width,
        query_start_loc, seq_lens, token_to_req, slot_mapping, is_valid,
        block_table, bt_stride, block_size,
        num_decode, num_slots, max_tokens,
        TRITON_BLOCK_SIZE: tl.constexpr, CHUNK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid < num_decode:
            valid = tl.load(slot_mapping + pid) >= 0
            tl.store(is_valid + pid, valid)
            if not valid:
                tl.store(swa_lens + pid, 0)
                for i in range(0, index_width, TRITON_BLOCK_SIZE):
                    offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
                    tl.store(swa_indices + pid * swa_stride + offset, -1, mask=offset < index_width)
            else:
                # _compute_swa_indices_and_lens_kernel, causal, no image, token_offset 0
                req_idx = tl.load(token_to_req + pid)
                query_start = tl.load(query_start_loc + req_idx)
                query_end = tl.load(query_start_loc + req_idx + 1)
                query_len = query_end - query_start
                seq_len = tl.load(seq_lens + req_idx)
                prefix_len = seq_len - query_len
                pos = prefix_len + pid - query_start
                start_pos = tl.maximum(pos - (window_size - 1), 0)
                end_pos = pos + 1
                swa_len = end_pos - start_pos
                tl.store(swa_lens + pid, swa_len)
                for i in range(0, index_width, TRITON_BLOCK_SIZE):
                    offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
                    pos_offset = start_pos + offset
                    block_indices = pos_offset // block_size
                    block_numbers = tl.load(
                        block_table + req_idx * bt_stride + block_indices, mask=pos_offset < end_pos
                    )
                    block_offsets = pos_offset % block_size
                    slot_ids = block_numbers * block_size + block_offsets
                    slot_ids = tl.where(offset < swa_len, slot_ids, -1)
                    tl.store(swa_indices + pid * swa_stride + offset, slot_ids, mask=offset < index_width)
        else:
            e = num_decode + (pid - num_decode) * CHUNK + tl.arange(0, CHUNK)
            in_slots = e < num_slots
            slot = tl.load(slot_mapping + e, mask=in_slots, other=0)
            tl.store(is_valid + e, slot >= 0, mask=in_slots)
            tl.store(swa_lens + e, 0, mask=e < max_tokens)

    return _swa_decode_fused_kernel


class SwaFused:
    def __init__(self, torch, triton, kernel, stock_kernel):
        self.torch = torch
        self.triton = triton
        self.kernel = kernel
        self.stock_kernel = stock_kernel

    def fused_ok(self, builder, cm, num_decode_tokens, num_prefill_tokens) -> bool:
        return (
            _STATE["armed"]
            and num_decode_tokens > 0
            and num_prefill_tokens == 0
            and cm.causal is True
            and num_decode_tokens <= cm.slot_mapping.shape[0]
            and builder.decode_swa_indices.dtype == self.torch.int32
        )

    def _launch(self, builder, indices, qsl, seq_lens, t2r, slots, valid, block_table, nd) -> None:
        lens = builder.decode_swa_lens
        num_slots, max_tokens = slots.shape[0], lens.shape[0]
        rest = max(num_slots, max_tokens) - nd
        grid = (nd + max(0, self.triton.cdiv(rest, CHUNK)),)
        self.kernel[grid](
            indices, indices.stride(0), lens, builder.window_size, indices.shape[-1],
            qsl, seq_lens, t2r, slots, valid,
            block_table, block_table.stride(0), builder.block_size,
            nd, num_slots, max_tokens,
            TRITON_BLOCK_SIZE=1024, CHUNK=CHUNK,
        )

    def _stock(self, builder, indices, qsl, seq_lens, t2r, slots, valid, block_table, nd) -> None:
        """The image's ops for the same build, verbatim."""
        valid.copy_(slots >= 0)
        builder.decode_swa_lens[nd:] = 0
        self.stock_kernel(
            indices, builder.decode_swa_lens, builder.window_size, indices.shape[-1],
            builder.decode_swa_lens, builder.decode_swa_lens,
            qsl, seq_lens, t2r, valid, block_table, builder.block_size,
            num_tokens=nd, token_offset=0,
        )

    def decode(self, builder, indices, qsl, seq_lens, t2r, slots, valid, block_table, nd) -> None:
        torch = self.torch
        args = (builder, indices, qsl, seq_lens, t2r, slots, valid, block_table, nd)
        if _STATE["verify_left"] <= 0 or torch.cuda.is_current_stream_capturing():
            try:
                return self._launch(*args)
            except Exception as exc:  # noqa: BLE001
                _disarm(exc)
                return self._stock(*args)
        # stock into the real buffers first (they stay right if we disarm),
        # snapshot, then the fused kernel over poisoned buffers; compare all
        self._stock(*args)
        ref = (indices[:nd].clone(), builder.decode_swa_lens.clone(), valid.clone())
        indices[:nd].fill_(-7)
        builder.decode_swa_lens.fill_(-7)
        valid.fill_(True)
        err = None
        try:
            self._launch(*args)
            got = (indices[:nd], builder.decode_swa_lens, valid)
            if not all(torch.equal(a, b) for a, b in zip(got, ref)):
                err = RuntimeError(f"fused != stock (decode tokens {nd}, slots {slots.shape[0]})")
        except Exception as exc:  # noqa: BLE001
            err = exc
        if err is not None:
            indices[:nd].copy_(ref[0])
            builder.decode_swa_lens.copy_(ref[1])
            valid.copy_(ref[2])
            _disarm(err)
            return
        _STATE["verify_left"] -= 1
        if not _STATE["engaged"]:
            _STATE["engaged"] = True
            print(
                "dsv41: swa metadata fused self-check bit-exact (decode tokens %d, width %d)"
                % (nd, indices.shape[-1]),
                flush=True,
            )


def install() -> str:
    import inspect
    import textwrap

    import torch
    from vllm.triton_utils import tl, triton
    from vllm.v1.attention.backends.mla import sparse_swa as mod

    cls = mod.DeepseekSparseSWAMetadataBuilder
    if getattr(cls.build, "_dsv41_swa_fused", False):
        return "already installed"
    try:
        src = textwrap.dedent(patch_source(inspect.getsource(cls.build)))
    except (OSError, TypeError, ValueError) as exc:
        _disarm(exc)
        return "not installed"
    _STATE["verify_left"] = verify_calls()
    ns = dict(mod.__dict__)
    ns["_dsv41_swa"] = SwaFused(torch, triton, _build_kernel(tl, triton), mod._COMPUTE_SWA_INDICES_AND_LENS_KERNEL)
    exec(compile(src, f"{mod.__file__} [dsv41 swa_meta_fused]", "exec", dont_inherit=True), ns)
    build = ns["build"]
    build._dsv41_swa_fused = True
    build.__qualname__ = cls.build.__qualname__
    cls.build = build
    return f"DeepseekSparseSWAMetadataBuilder.build fused decode path (verify={_STATE['verify_left']})"
