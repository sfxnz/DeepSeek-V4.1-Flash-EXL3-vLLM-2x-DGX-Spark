#!/usr/bin/env python3
"""Prefetch Engram disk rows for the next decode step (fadvise WILLNEED).

Even after parallel staging, the L.A.I.L decode step loses ~11 ms/step to
cold-row NVMe reads: each step's newly accepted tokens hash to first-touch
rows, the staging gather runs after the step's GPU work drains, and the GPU
idles while preads land.

Timeline per step: verify graph -> postprocess (num_sampled known) -> draft
graph (~15-25 ms GPU). The tokens the NEXT step will stage are exactly this
step's verify output (`sampled_token_ids`), resident on the GPU BEFORE the
draft graph launches. This patch snapshots those tokens (plus the chunk
ids/positions and lookback window) with a side-stream async D2H at that
moment, then a background thread computes the next chunk's n-gram hashes on
CPU — an exact port of `_hash_ids_kernel`'s rolling
`rolling ^= value * multiplier`, `rolling % prime + offset` — and issues
POSIX_FADV_WILLNEED for the covering rows of both tables. Kernel readahead
then overlaps the draft graph; the staging gather finds the pages hot.

The prefetch is advisory only: a wrong prediction wastes readahead, never
correctness (staging still hashes on the GPU and reads what it needs).
Effectiveness shows up directly as the DSV41_ENGRAM_CENSUS read_w average
dropping toward the warm floor.

DSV41_ENGRAM_PREFETCH=1 enables. Apply AFTER engram_stage_fast.py.
Idempotent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "# --- engram-prefetch ---"

STAGER_INIT_TAIL = """        self.hashes_ready = torch.cuda.Event()
        self.num_staged = 0
"""

STAGER_INIT_NEW = """        self.hashes_ready = torch.cuda.Event()
        self.num_staged = 0
""" + MARKER + """
        self._prefetch_setup()
"""

STAGER_METHODS = MARKER + '''
    def _prefetch_setup(self) -> None:
        """One-time CPU copies of the hash constants + pinned mirrors."""
        import os as _os
        from concurrent.futures import ThreadPoolExecutor as _TPE

        self.prefetch_on = _os.environ.get("DSV41_ENGRAM_PREFETCH", "0") == "1"
        if not self.prefetch_on:
            return
        hs = self.hash_state
        import torch as _torch

        self._pf_tm = hs.token_map.to("cpu").to(_torch.int64)
        self._pf_mult = hs.multipliers.to("cpu").to(_torch.int64)
        self._pf_primes = (
            hs.primes.to("cpu").to(_torch.int64).reshape(-1).tolist()
        )
        self._pf_offsets = (
            hs.offsets.to("cpu").to(_torch.int64).reshape(-1).tolist()
        )
        self._pf_pad = int(hs.pad_id)
        self._pf_ngram = int(hs.multipliers.shape[1])
        self._pf_heads = int(hs.primes.shape[-1])
        self._pf_depth = int(hs.lookback_depth)
        self._pf_layers = [e.layer_hash_index for e in self.engrams]
        self._pf_heads_span = [
            (
                e.embed_tokens.head_start,
                min(
                    e.embed_tokens.head_start + e.embed_tokens.part_n_hash_cols,
                    e.embed_tokens.n_hash_cols,
                ),
            )
            for e in self.engrams
        ]
        cap = 32
        self._pf_cap = cap
        self._pf_pin_ids = _torch.zeros(cap, dtype=_torch.int64, pin_memory=True)
        self._pf_pin_pos = _torch.zeros(cap, dtype=_torch.int64, pin_memory=True)
        self._pf_pin_win = _torch.zeros(
            8 * self._pf_depth, dtype=_torch.int32, pin_memory=True
        )
        self._pf_pin_out = _torch.zeros(
            8 * (cap + 1), dtype=_torch.int64, pin_memory=True
        )
        self._pf_pin_ns = _torch.zeros(8, dtype=_torch.int64, pin_memory=True)
        self._pf_event = _torch.cuda.Event()
        self._pf_stream = _torch.cuda.Stream()
        self._pf_pool = _TPE(max_workers=1)
        self._pf_gen = 0
        print(
            f"dsv41: engram prefetch armed (ngram={self._pf_ngram} "
            f"heads={self._pf_heads} depth={self._pf_depth} "
            f"layers={self._pf_layers} "
            f"mult={tuple(hs.multipliers.shape)} "
            f"primes={tuple(hs.primes.shape)} "
            f"offsets={tuple(hs.offsets.shape)}) "
            f"mult_rows={self._pf_mult.tolist()} "
            f"primes_flat={self._pf_primes} "
            f"offsets_flat={self._pf_offsets} "
            f"tm_head={self._pf_tm[:16].tolist()}",
            flush=True,
        )

    @torch.inference_mode()
    def enqueue_prefetch(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        window: torch.Tensor,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_reqs: int,
        num_tokens: int,
    ) -> None:
        """Side-stream snapshot of next-chunk tokens; never blocks the step."""
        if not getattr(self, "prefetch_on", False):
            return
        try:
            import torch as _torch

            if num_tokens > self._pf_cap or num_reqs > 8 or num_tokens <= 0:
                return
            depth = self._pf_depth
            with _torch.cuda.stream(self._pf_stream):
                # Order after everything already submitted to the main
                # stream (verify + postprocess); the draft graph is
                # submitted later, so the copies overlap it.
                self._pf_stream.wait_stream(_torch.cuda.current_stream())
                self._pf_pin_ids[:num_tokens].copy_(
                    input_ids[:num_tokens].to(_torch.int64), non_blocking=True
                )
                self._pf_pin_pos[:num_tokens].copy_(
                    positions[:num_tokens].to(_torch.int64), non_blocking=True
                )
                if window.numel():
                    win = window[:num_reqs, :].reshape(-1).to(_torch.int32)
                    self._pf_pin_win[: win.numel()].copy_(win, non_blocking=True)
                out = sampled_token_ids[:num_reqs, : self._pf_cap].reshape(-1).to(
                    _torch.int64
                )
                self._pf_pin_out[: out.numel()].copy_(out, non_blocking=True)
                self._pf_pin_ns[:num_reqs].copy_(
                    num_sampled[:num_reqs].to(_torch.int64), non_blocking=True
                )
                self._pf_event.record()
            self._pf_gen += 1
            self._pf_pool.submit(
                self._prefetch_worker, self._pf_gen, num_reqs, num_tokens
            )
        except Exception as exc:  # noqa: BLE001
            self.prefetch_on = False
            print(f"dsv41: engram prefetch disabled: {exc!r}", flush=True)

    def _prefetch_worker(self, gen: int, num_reqs: int, num_tokens: int) -> None:
        """Hash the predicted chunks on CPU; fadvise the covering rows."""
        import os as _os

        try:
            self._pf_event.synchronize()
            if gen != self._pf_gen:
                return  # a newer snapshot superseded this one
            ids = self._pf_pin_ids[:num_tokens].tolist()
            pos = self._pf_pin_pos[:num_tokens].tolist()
            win2 = self._pf_pin_win[
                : num_reqs * self._pf_depth
            ].reshape(num_reqs, self._pf_depth)
            outs = self._pf_pin_out[: num_reqs * self._pf_cap].reshape(
                num_reqs, self._pf_cap
            )
            ns = self._pf_pin_ns[:num_reqs].tolist()
            tm = self._pf_tm.tolist()
            ngram = self._pf_ngram
            pad = self._pf_pad
            depth = self._pf_depth
            tables = [
                (
                    e.embed_tokens.disk.w_fd,
                    e.embed_tokens.disk.w_off,
                    e.embed_tokens.disk.dim,
                    e.embed_tokens.disk.s_fd,
                    e.embed_tokens.disk.s_off,
                    e.embed_tokens.disk.sb,
                    e.embed_tokens.vocab_start_idx,
                    e.embed_tokens.vocab_end_idx,
                    layer,
                    h0,
                    h1,
                )
                for e, layer, (h0, h1) in zip(
                    self.engrams, self._pf_layers, self._pf_heads_span
                )
            ]
            last_pos = pos[-1]
            for r in range(num_reqs):
                k = max(int(ns[r]), 0)
                if k == 0:
                    continue
                chunk = [
                    int(t)
                    for t in outs[r].tolist()[:k]
                    if 0 <= int(t) < 200000
                ][: ngram + 1]
                if not chunk:
                    continue
                # Spec-decode commit: the verify chunk is [anchor, d1..d5]
                # at positions pos[0..5]; the next step's chunk is the
                # accepted tokens, at positions pos[0]+1 .. (they follow
                # the ANCHOR, not the end of the verify chunk).
                anchor_pos = pos[0]
                cpos = list(
                    range(anchor_pos + 1, anchor_pos + 1 + len(chunk))
                )
                # History before the next chunk start (anchor_pos+1): the
                # anchor itself, then the current window (col 0 = newest).
                hist_newest_first = [ids[0]] + win2[r].tolist()
                for (
                    w_fd,
                    w_off,
                    dim,
                    s_fd,
                    s_off,
                    sb,
                    v0,
                    v1,
                    layer,
                    h0,
                    h1,
                ) in tables:
                    rows = self._cpu_hash_rows(
                        chunk,
                        cpos,
                        hist_newest_first,
                        depth,
                        tm,
                        pad,
                        layer,
                        h0,
                        h1,
                        v0,
                        v1,
                    )
                    e_disk = self.engrams[
                        self._pf_layers.index(layer)
                    ].embed_tokens.disk
                    e_disk._pf_expected = rows
                    import os as _os2

                    if _os2.environ.get("DSV41_ENGRAM_PF_DUMP", "0") == "1":
                        e_disk._pf_dump = (
                            list(chunk),
                            list(cpos),
                            list(hist_newest_first[:depth]),
                            sorted(rows)[:12],
                        )
                        e_disk._pf_dump_armed = True
                    for row in rows:
                        _os.posix_fadvise(
                            w_fd, w_off + row * dim, 8192, _os.POSIX_FADV_WILLNEED
                        )
                        _os.posix_fadvise(
                            s_fd, s_off + row * sb, 4096, _os.POSIX_FADV_WILLNEED
                        )
        except Exception as exc:  # noqa: BLE001
            import traceback as _tb

            if getattr(self, "_pf_err_count", 0) < 3:
                self._pf_err_count = getattr(self, "_pf_err_count", 0) + 1
                print(
                    "dsv41: engram prefetch worker error "
                    + str(self._pf_err_count)
                    + chr(10)
                    + _tb.format_exc(),
                    flush=True,
                )
            if getattr(self, "_pf_err_count", 0) >= 3:
                self.prefetch_on = False
                print(
                    f"dsv41: engram prefetch disabled after "
                    f"{self._pf_err_count} errors: {exc!r}",
                    flush=True,
                )

    def _cpu_hash_rows(
        self,
        chunk,
        cpos,
        hist_newest_first,
        depth,
        tm,
        pad,
        layer,
        head_start,
        head_end,
        vocab_start,
        vocab_end,
    ):
        """Rows this TP shard owns for `chunk` under one engram layer.

        Mirrors _hash_ids_kernel: rolling int64 xor of compressed ids times
        per-shift multipliers; one hash per (ngram size, head) column,
        `rolling % prime + offset`. Only columns in [head_start, head_end)
        belong to this rank; only hashes inside [vocab_start, vocab_end)
        live in this shard's file rows.
        """
        import torch as _torch

        ngram = self._pf_ngram
        heads = self._pf_heads
        cols = (ngram - 1) * heads
        mult = self._pf_mult[layer].tolist()
        primes = self._pf_primes
        offsets = self._pf_offsets
        # Column span this rank owns within the layer's hash columns; the
        # flat param index is layer*cols + col (primes is [L, ngram-1, H]).
        col_start, col_end = head_start, head_end
        chunk_start = cpos[0]
        rows = set()
        for t in range(len(chunk)):
            position = cpos[t]
            rolling = 0
            blocked = False
            for shift in range(ngram):
                lookback = position - shift
                if lookback >= chunk_start:
                    idx = lookback - chunk_start
                    src = chunk[idx]
                    source = tm[src] if 0 <= src < len(tm) else pad
                else:
                    col = chunk_start - 1 - lookback
                    if 0 <= col < depth:
                        tok = hist_newest_first[col]
                        source = (
                            tm[tok] if 0 <= tok < len(tm) else pad
                        )
                    else:
                        source = pad
                blocked = blocked or lookback < 0 or source == -1
                value = pad if blocked else source
                rolling ^= value * mult[shift]
                if shift > 0:
                    col_base = (shift - 1) * heads
                    for col in range(col_base, col_base + heads):
                        if not (col_start <= col < col_end):
                            continue
                        idx = layer * cols + col
                        hashed = rolling % primes[idx] + offsets[idx]
                        if vocab_start <= hashed < vocab_end:
                            # File rows are GLOBAL ids: the hash value itself
                            # (HF shards are unsharded tables).
                            rows.add(hashed)
        return rows
'''

STAGE_DEF_ANCHOR = "    @torch.inference_mode()\n    def stage("

RUNNER_ANCHOR = """        # NOTE: This is intentionally done after creating the AsyncOutput,
        # ensuring that `copy_event` is recorded before calling postprocess.
        # This sequencing may slightly reduce latency as async D2H copy does not
        # need to wait for the postprocess to finish.
        self.postprocess_sampled(
            input_batch.idx_mapping,
            sampler_output.sampled_token_ids,
            num_sampled,
            num_rejected,
            input_batch.query_start_loc,
        )
"""

RUNNER_HOOK = RUNNER_ANCHOR + MARKER + """
        if self.speculator is not None:
            _stager = getattr(self.model_state, "engram_stager", None)
            if _stager is not None:
                _stager.enqueue_prefetch(
                    input_batch.input_ids,
                    input_batch.positions,
                    self.model_state.lookback_token_ids,
                    sampler_output.sampled_token_ids,
                    num_sampled,
                    input_batch.num_reqs,
                    input_batch.num_tokens,
                )
"""


def apply(model_root: Path, runner: Path) -> None:
    engram = model_root / "common" / "engram.py"
    text = engram.read_text()
    if MARKER not in text:
        if STAGER_INIT_TAIL not in text:
            raise SystemExit("engram_prefetch: stager init anchor missing")
        text = text.replace(STAGER_INIT_TAIL, STAGER_INIT_NEW, 1)
        if STAGE_DEF_ANCHOR not in text:
            raise SystemExit("engram_prefetch: stage def anchor missing")
        text = text.replace(
            STAGE_DEF_ANCHOR, STAGER_METHODS + "\n\n" + STAGE_DEF_ANCHOR, 1
        )
        engram.write_text(text)
        print("dsv41: engram prefetch stager installed")

    rtext = runner.read_text()
    if MARKER in rtext:
        return
    if RUNNER_ANCHOR not in rtext:
        raise SystemExit("engram_prefetch: runner anchor missing")
    rtext = rtext.replace(RUNNER_ANCHOR, RUNNER_HOOK, 1)
    runner.write_text(rtext)
    print("dsv41: engram prefetch runner hook installed")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model_root", type=Path, help=".../models/deepseek_v4_1")
    p.add_argument("runner", type=Path, help=".../v1/worker/gpu/model_runner.py")
    args = p.parse_args()
    apply(args.model_root, args.runner)


if __name__ == "__main__":
    main()
