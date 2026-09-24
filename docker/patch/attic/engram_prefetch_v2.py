#!/usr/bin/env python3
"""Prefetch Engram disk rows for the next decode step — v2 (pf_hit pairing fix).

v1 (docker/patch/engram_prefetch.py) verified the CPU hash port offline
(predicted rows intersect actual staged rows) but measured pf_hit ~0% and no
read-time drop. Root causes found by line-audit (see
experiments/dsv41-opt/patches/ENGRAAM-PREFETCH-V2.md for the full ranking):

1. PUBLISH-TOO-LATE (dominant). v1's worker set `e_disk._pf_expected = rows`
   (v1 :251) only AFTER hashing ALL tables for ALL requests — a pure-Python
   big-int loop over ngram x heads x tokens x tables that takes longer than
   the ~15-17 ms (draft 3-4 ms + gap tail) before the next step's
   `EngramDiskStager.stage` gathers. The gather found `_pf_expected = None`
   (never set yet), the census block then counted nothing and cleared
   nothing, and by the time the worker published, the consuming gather had
   already run -> pf_hit ~0% with a correct prediction. v2 fadvises AND
   publishes each table's row set as soon as that table is predicted, so a
   late worker still warms pages for a *subsequent* step and the accounting
   can pair.

2. SUPERSEDED-GEN SILENT DROP (v1 :166-167). `if gen != self._pf_gen:
   return` skipped BOTH fadvise and publish whenever a newer enqueue landed
   first — under back-to-back decode steps that dropped whole predictions.
   v2 processes every generation and publishes gen-stamped (newest wins).

3. FADVISE WINDOW OVERSHOOT (v1 :263-268). Real table geometry is
   [384006168, 256] F8_E4M3 (dim=256 B/row) and scale [., 8]: v1 advised
   8192 B (32 rows) on the weight fd and 4096 B on the scale fd per row.
   Pages do land (reads should have warmed) — this cannot explain pf_hit ~0%
   (the accounting bug does) but wastes readahead across unrelated rows.
   v2 fadviseS the exact page-aligned row span.

4. ACCOUNTING DILUTION (engram_stage_census.py :88-90 applied block). The
   denominator `_pf[1] += len(_got)` grew on EVERY gather while the
   numerator only grew when `_pf_expected` was non-None, and the pf_hit
   print divided hits by that inflated denominator. Combined with (1) the
   measured number was ~0% even when a (late) prediction existed. v2's
   apply() rewrites the census block in engram_disk.py: hits and predicted
   rows are only counted when a prediction is actually paired to the
   gather; pf_hit% = covered predictions.

Also adds: one-shot first-consumption self-check log (predicted-set id,
consumed-set id, intersection size), DSV41_ENGRAM_PREFETCH_DEBUG=1 for
verbose per-pair logs bounded to the first 20 pairs, and a _pf_sync() hook
so CPU-only dry harnesses can drive the real worker.

Keeps v1's contract: DSV41_ENGRAM_PREFETCH=1 enables (default off),
advisory only, self-disables after 3 worker errors, idempotent, requires
engram_stage_fast. Apply AFTER engram_stage_census + engram_stage_fast.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "# --- engram-prefetch-v2 ---"

STAGER_INIT_TAIL = """        self.hashes_ready = torch.cuda.Event()
        self.num_staged = 0
"""

STAGER_INIT_NEW = """        self.hashes_ready = torch.cuda.Event()
        self.num_staged = 0
""" + MARKER + """
        self._prefetch_setup()
"""

# Marker v1 used; v2 replaces v1's method block if it finds one.
V1_MARKER = "# --- engram-prefetch ---"

STAGER_METHODS = MARKER + '''
    def _prefetch_setup(self) -> None:
        # One-time CPU copies of the hash constants + pinned mirrors.
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
        self._pf_pub_gen = 0
        self._pf_debug = (
            _os.environ.get("DSV41_ENGRAM_PREFETCH_DEBUG", "0") == "1"
        )
        self._pf_log_pairs = 20  # bounded verbose pairing logs
        print(
            f"dsv41: engram prefetch v2 armed (ngram={self._pf_ngram} "
            f"heads={self._pf_heads} depth={self._pf_depth} "
            f"layers={self._pf_layers} "
            f"mult={tuple(hs.multipliers.shape)} "
            f"primes={tuple(hs.primes.shape)} "
            f"offsets={tuple(hs.offsets.shape)})",
            flush=True,
        )

    def _pf_sync(self) -> None:
        # Indirection so CPU-only dry harnesses can drive the real worker.
        self._pf_event.synchronize()

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
        # Side-stream snapshot of next-chunk tokens; never blocks the step.
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

    def _fadvise_row(self, fd: int, base: int, row: int, row_bytes: int) -> None:
        # Exact page-aligned span covering just this row (real geometry:
        # weight rows are dim bytes, scale rows sb bytes — NOT 8192/4096).
        import os as _os

        start = base + row * row_bytes
        start_a = start & -4096
        end_a = (start + row_bytes + 4095) & -4096
        _os.posix_fadvise(fd, start_a, end_a - start_a, _os.POSIX_FADV_WILLNEED)

    def _pf_set_id(self, rows) -> int:
        import zlib as _zlib

        if not rows:
            return 0
        return _zlib.crc32(
            b",".join(str(r).encode() for r in sorted(rows)[:4096])
        )

    def _prefetch_worker(self, gen: int, num_reqs: int, num_tokens: int) -> None:
        # Hash the predicted chunks on CPU; fadvise AND publish each table
        # as soon as it is predicted so the next gather can pair with it.
        import os as _os

        try:
            self._pf_sync()
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
                    e.embed_tokens.disk,
                )
                for e, layer, (h0, h1) in zip(
                    self.engrams, self._pf_layers, self._pf_heads_span
                )
            ]
            last_pos = pos[-1]
            # New generation -> reset the per-table row accumulators so a
            # stale worker can never pollute a newer prediction (pool is a
            # single thread, but stay safe). The publish is a UNION of all
            # requests' rows for this generation — v1 overwrote per request
            # (last request won), which starved the hit numerator whenever
            # more than one request was in flight.
            if gen != getattr(self, "_pf_acc_gen", 0):
                self._pf_acc_gen = gen
                for _t in tables:
                    _t[11]._pf_expected = set()
                    _t[11]._pf_acc_gen = gen
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
                    e_disk,
                ) in tables:
                    rows = self._cpu_hash_rows(
                        chunk,
                        cpos,
                        hist_newest_first,
                        depth,
                        tm,
                        self._pf_pad,
                        layer,
                        h0,
                        h1,
                        v0,
                        v1,
                    )
                    # fadvise FIRST (pages start landing), then publish.
                    for row in rows:
                        self._fadvise_row(w_fd, w_off, row, dim)
                        self._fadvise_row(s_fd, s_off, row, sb)
                    # Publish gen-stamped; only the newest generation wins
                    # (single worker thread -> monotonic gens). Union with
                    # earlier requests of the same generation.
                    if gen >= self._pf_pub_gen:
                        self._pf_pub_gen = gen
                        e_disk._pf_expected = (
                            e_disk._pf_expected or set()
                        ) | rows
                        e_disk._pf_exp_gen = gen
                        e_disk._pf_exp_id = self._pf_set_id(e_disk._pf_expected)
                        if _os.environ.get("DSV41_ENGRAM_PF_DUMP", "0") == "1":
                            e_disk._pf_dump = (
                                list(chunk),
                                list(cpos),
                                list(hist_newest_first[:depth]),
                                sorted(rows)[:12],
                            )
                            e_disk._pf_dump_armed = True
                        if self._pf_debug and gen <= self._pf_log_pairs:
                            print(
                                "[pf-pub] gen=%d layer=%d |pred|=%d "
                                "pred_id=%08x"
                                % (gen, layer, len(rows), e_disk._pf_exp_id),
                                flush=True,
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
        # Rows this TP shard owns for `chunk` under one engram layer.
        #
        # Mirrors _hash_ids_kernel: rolling int64 xor of compressed ids times
        # per-shift multipliers; one hash per (ngram size, head) column,
        # `rolling % prime + offset`. Only columns in [head_start, head_end)
        # belong to this rank; only hashes inside [vocab_start, vocab_end)
        # live in this shard's file rows. (Verified vs live stage dumps,
        # round 11; unchanged from v1.)
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

# --- census accounting fix (applied to engram_disk.py) ----------------------
#
# engram_stage_census.py installs this block inside gather_dequant. Its
# accounting dilutes pf_hit: the denominator grows on every gather while the
# numerator only grows when a prediction happens to be paired. v1's late
# publish made the pairing almost never coincide -> pf_hit ~0% even with
# correct predictions. Replacement counts BOTH sides only when paired, adds
# the one-shot first-consumption self-check and bounded debug pairing logs.

CENSUS_OLD = """            _exp = getattr(self, "_pf_expected", None)
            if _exp is not None:
                _got = set(rel_l)
                _pf = _ENG_PF_HITS
                _pf[0] += len(_got & _exp)
                _pf[1] += len(_got)
                _pf[2] += 1
                self._pf_expected = None
"""

CENSUS_NEW = """            _exp = getattr(self, "_pf_expected", None)
            if _exp is not None:
                _got = set(rel_l)
                _pf = _ENG_PF_HITS
                _pf[0] += len(_got & _exp)
                _pf[1] += len(_exp)
                _pf[2] += 1
                self._pf_expected = None
""" + MARKER + """
                # Pairing self-check: hits and predicted rows only counted
                # when paired, so pf_hit% = share of predictions consumed.
                import os as _pfv2_os
                import zlib as _pfv2_zlib

                _pfv2_pairs = getattr(self, "_pfv2_pairs", 0)
                _pfv2_log = (
                    _pfv2_pairs == 0
                    or _pfv2_os.environ.get(
                        "DSV41_ENGRAM_PREFETCH_DEBUG", "0"
                    ) == "1"
                )
                if _pfv2_log and _pfv2_pairs < 20:
                    _pfv2_exp_id = int(
                        getattr(self, "_pf_exp_id", 0)
                    ) or _pfv2_zlib.crc32(
                        b",".join(str(r).encode() for r in sorted(_exp)[:4096])
                    )
                    _pfv2_got_id = _pfv2_zlib.crc32(
                        b",".join(str(r).encode() for r in sorted(_got)[:4096])
                    )
                    print(
                        "[pf-pair] gen=%s table_pred_id=%08x "
                        "consumed_id=%08x |pred|=%d |consumed|=%d "
                        "intersect=%d pair=%d"
                        % (
                            getattr(self, "_pf_exp_gen", "?"),
                            _pfv2_exp_id,
                            _pfv2_got_id,
                            len(_exp),
                            len(_got),
                            len(_got & _exp),
                            _pfv2_pairs,
                        ),
                        flush=True,
                    )
                self._pfv2_pairs = _pfv2_pairs + 1
"""


def _patch_census(disk_text: str) -> str:
    if MARKER in disk_text:
        return disk_text  # already fixed
    if CENSUS_OLD in disk_text:
        return disk_text.replace(CENSUS_OLD, CENSUS_NEW, 1)
    return disk_text  # census not installed; accounting fix is a no-op then


def apply(model_root: Path, runner: Path, vllm_root: Path | None = None) -> None:
    engram = model_root / "common" / "engram.py"
    text = engram.read_text()
    if MARKER not in text:
        if V1_MARKER in text:
            # Swap v1's method block (marker .. stage def) for v2's.
            start = text.index(V1_MARKER)
            end = text.index(STAGE_DEF_ANCHOR, start)
            text = text[:start] + STAGER_METHODS + "\n\n" + text[end:]
            print("dsv41: engram prefetch v1 methods replaced by v2")
        else:
            if STAGER_INIT_TAIL not in text:
                raise SystemExit("engram_prefetch_v2: stager init anchor missing")
            text = text.replace(STAGER_INIT_TAIL, STAGER_INIT_NEW, 1)
            if STAGE_DEF_ANCHOR not in text:
                raise SystemExit("engram_prefetch_v2: stage def anchor missing")
            text = text.replace(
                STAGE_DEF_ANCHOR, STAGER_METHODS + "\n\n" + STAGE_DEF_ANCHOR, 1
            )
        engram.write_text(text)
        print("dsv41: engram prefetch v2 stager installed")

    rtext = runner.read_text()
    if MARKER not in rtext and V1_MARKER not in rtext:
        if RUNNER_ANCHOR not in rtext:
            raise SystemExit("engram_prefetch_v2: runner anchor missing")
        rtext = rtext.replace(RUNNER_ANCHOR, RUNNER_HOOK, 1)
        runner.write_text(rtext)
        print("dsv41: engram prefetch v2 runner hook installed")

    if vllm_root is not None:
        disk = vllm_root / "models" / "deepseek_v4_1" / "common" / "engram_disk.py"
        dtext = disk.read_text()
        fixed = _patch_census(dtext)
        if fixed != dtext:
            disk.write_text(fixed)
            print("dsv41: engram census pf_hit accounting fixed (v2)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model_root", type=Path, help=".../models/deepseek_v4_1")
    p.add_argument("runner", type=Path, help=".../v1/worker/gpu/model_runner.py")
    p.add_argument(
        "vllm_root",
        type=Path,
        nargs="?",
        default=None,
        help=".../vllm (site-packages root) for the census accounting fix",
    )
    args = p.parse_args()
    apply(args.model_root, args.runner, args.vllm_root)


if __name__ == "__main__":
    main()
