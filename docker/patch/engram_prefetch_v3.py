#!/usr/bin/env python3
"""Prefetch Engram disk rows for the next decode step — v3 (prediction fix).

v2 fixed DELIVERY (per-table immediate publish; [pf-pub] now precedes
[pf-pair] every generation on both ranks) but the measured pf_hit stayed
~0% with table_pred_id != consumed_id on ~88% of pairs: the PREDICTED
token stream is not the stream the gather hashes.

Root cause (see ENGRAAM-PREFETCH-V3.md, cited lines): under dspark spec
decode the step's verify chunk is [bonus, d_1..d_k] at S..S+k, the
sampler emits A=num_sampled samples s_0..s_{A-1} (s_j commits at S+1+j),
and the NEXT step's chunk is [s_{A-1}, d'_1..d'_k] at S+A..S+A+k — the
bonus is RE-FED as row 0 and 5 of 6 tokens are NEW drafts produced by
speculator.propose() AFTER v2's postprocess hook
(v1/worker/gpu/model_runner.py:1941-1975). v2 hashed [s_0..s_{A-1}] at
pos[0]+1.. (engram_prefetch_v2.py:265-282) — a stream no gather ever
hashes; only the last element s_{A-1} lands on the true next-chunk start
(Q+A), giving the recorded boundary-gen partials of exactly 12 rows (one
token x 12 owned hash columns) and 0 on lagged mid-run pairs.

v3 moves the enqueue to AFTER propose (model_runner.py:1981-1987 anchor)
and predicts the real next chunk:
  chunk = [s_{A-1}] + draft_tokens[r]        (bonus = last_sampled)
  cpos  = S_r + A ..                        (per-request chunk start S_r
                                             = positions[query_start_loc[r]],
                                             NOT request 0's pos[0]+1)
  hist  = [s_{A-2}, s_{A-3}, ..., bonus_old, win[j-A]...]  (newest-first;
          the lookback window the next gather reads = committed seq
          [S+A-1-j] per _gather_lookback_kernel,
          models/deepseek_v4_1/nvidia/model_state.py:17-41)

Keeps v2's delivery contract: DSV41_ENGRAM_PREFETCH=1 enables (default
off), per-table fadvise+publish the moment a table is predicted,
gen-stamped union publish across requests, exact page-aligned fadvise
spans, census accounting fix, one-shot [pf-pair] self-check + bounded
[pf-pub] debug, self-disable after 3 worker errors, idempotent, replaces
v1 or v2 if installed. Apply AFTER engram_stage_census + engram_stage_fast.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "# --- engram-prefetch-v3 ---"

STAGER_INIT_TAIL = """        self.hashes_ready = torch.cuda.Event()
        self.num_staged = 0
"""

STAGER_INIT_NEW = """        self.hashes_ready = torch.cuda.Event()
        self.num_staged = 0
""" + MARKER + """
        self._prefetch_setup()
"""

# Markers v1/v2 used; v3 replaces their method block if present.
V1_MARKER = "# --- engram-prefetch ---"
V2_MARKER = "# --- engram-prefetch-v2 ---"

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
        cap = 64
        kmax = 8
        self._pf_cap = cap
        self._pf_kmax = kmax
        self._pf_rmax = 8
        self._pf_pin_ids = _torch.zeros(cap, dtype=_torch.int64, pin_memory=True)
        self._pf_pin_pos = _torch.zeros(cap, dtype=_torch.int64, pin_memory=True)
        self._pf_pin_qsl = _torch.zeros(
            self._pf_rmax + 1, dtype=_torch.int64, pin_memory=True
        )
        self._pf_pin_win = _torch.zeros(
            8 * self._pf_depth, dtype=_torch.int32, pin_memory=True
        )
        # sampled_token_ids: [max_num_reqs-ish, cap] flattened per request
        self._pf_pin_out = _torch.zeros(
            8 * (cap + 1), dtype=_torch.int64, pin_memory=True
        )
        # draft_tokens: [num_reqs, k] from speculator.propose()
        self._pf_pin_draft = _torch.zeros(
            8 * kmax, dtype=_torch.int64, pin_memory=True
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
            f"dsv41: engram prefetch v3 armed (ngram={self._pf_ngram} "
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
        query_start_loc: torch.Tensor,
        window: torch.Tensor,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        draft_tokens: torch.Tensor,
        num_reqs: int,
        num_tokens: int,
    ) -> None:
        # Side-stream snapshot for the post-propose prediction; the copies
        # order after everything already submitted to the main stream.
        if not getattr(self, "prefetch_on", False):
            return
        try:
            import torch as _torch

            if (
                num_tokens > self._pf_cap
                or num_reqs > self._pf_rmax
                or num_tokens <= 0
                or draft_tokens is None
                or draft_tokens.numel() == 0
            ):
                return
            k = min(int(draft_tokens.shape[1]), self._pf_kmax)
            # Order the side stream behind main-stream work queued so far.
            # MUST be issued while the MAIN stream is current — inside the
            # with-block below current_stream() IS the side stream and
            # wait_stream(self) is a no-op (stale-read bug, fixed round 20).
            self._pf_stream.wait_stream(_torch.cuda.current_stream())
            with _torch.cuda.stream(self._pf_stream):
                self._pf_pin_ids[:num_tokens].copy_(
                    input_ids[:num_tokens].to(_torch.int64), non_blocking=True
                )
                self._pf_pin_pos[:num_tokens].copy_(
                    positions[:num_tokens].to(_torch.int64), non_blocking=True
                )
                self._pf_pin_qsl[: num_reqs + 1].copy_(
                    query_start_loc[: num_reqs + 1].to(_torch.int64),
                    non_blocking=True,
                )
                if window.numel():
                    win = window[:num_reqs, :].reshape(-1).to(_torch.int32)
                    self._pf_pin_win[: win.numel()].copy_(win, non_blocking=True)
                out = sampled_token_ids[:num_reqs, : self._pf_cap].reshape(-1).to(
                    _torch.int64
                )
                self._pf_pin_out[: out.numel()].copy_(out, non_blocking=True)
                dr = draft_tokens[:num_reqs, :k].reshape(-1).to(_torch.int64)
                self._pf_pin_dr_n = dr.numel()
                self._pf_pin_draft[: dr.numel()].copy_(dr, non_blocking=True)
                self._pf_pin_ns[:num_reqs].copy_(
                    num_sampled[:num_reqs].to(_torch.int64), non_blocking=True
                )
                self._pf_event.record()
            self._pf_gen += 1
            self._pf_pool.submit(
                self._prefetch_worker, self._pf_gen, num_reqs, num_tokens, k
            )
        except Exception as exc:  # noqa: BLE001
            self.prefetch_on = False
            print(f"dsv41: engram prefetch disabled: {exc!r}", flush=True)

    def _fadvise_row(self, fd: int, base: int, row: int, row_bytes: int) -> None:
        # Exact page-aligned span covering just this row (real geometry:
        # weight rows are dim bytes, scale rows sb bytes).
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

    def _pf_next_chunk(self, r, A, ids, q0, q1, outs_r, drafts_r, win_r, S):
        # The chunk the NEXT gather hashes (committed-prefix advance rule,
        # derived from _prepare_dflash_inputs_kernel + post_update):
        #   this step's rows = [bonus_old, d_1..] at S..; s_j commits at
        #   S+1+j; next chunk = [s_{A-1}] + NEW drafts at S+A..; its
        #   lookback window (newest first) = seq[S+A-1-j] =
        #   [s_{A-2}, ..., s_0, bonus_old, win[0], ...].
        chunk = [int(outs_r[A - 1])] + [int(d) for d in drafts_r]
        cpos = list(range(S + A, S + A + len(chunk)))
        bonus_old = int(ids[q0])
        hist = []
        for j in range(self._pf_depth):
            if j < A - 1:
                hist.append(int(outs_r[A - 2 - j]))
            elif j == A - 1:
                hist.append(bonus_old)
            else:
                wj = j - A
                hist.append(int(win_r[wj]) if 0 <= wj < len(win_r) else -1)
        return chunk, cpos, hist

    def _prefetch_worker(self, gen: int, num_reqs: int, num_tokens: int,
                         k: int) -> None:
        # Hash the predicted NEXT chunks on CPU; fadvise AND publish each
        # table as soon as it is predicted so the next gather can pair.
        import os as _os

        try:
            self._pf_sync()
            ids = self._pf_pin_ids[:num_tokens].tolist()
            pos = self._pf_pin_pos[:num_tokens].tolist()
            qsl = self._pf_pin_qsl[: num_reqs + 1].tolist()
            win2 = self._pf_pin_win[
                : num_reqs * self._pf_depth
            ].reshape(num_reqs, self._pf_depth)
            outs = self._pf_pin_out[: num_reqs * self._pf_cap].reshape(
                num_reqs, self._pf_cap
            )
            drs = self._pf_pin_draft[: num_reqs * k].reshape(num_reqs, k)
            ns = self._pf_pin_ns[:num_reqs].tolist()
            tm = self._pf_tm.tolist()
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
            # New generation -> reset the per-table row accumulators so a
            # stale worker can never pollute a newer prediction. The
            # publish is a UNION of all requests' rows for this generation.
            if gen != getattr(self, "_pf_acc_gen", 0):
                self._pf_acc_gen = gen
                for _t in tables:
                    _t[11]._pf_expected = set()
                    _t[11]._pf_acc_gen = gen
            for r in range(num_reqs):
                A = max(int(ns[r]), 0)
                if A <= 0:
                    continue
                q0 = int(qsl[r])
                q0 = min(max(q0, 0), num_tokens - 1)
                S = int(pos[q0])
                chunk, cpos, hist = self._pf_next_chunk(
                    r, A, ids, q0, None,
                    outs[r].tolist(), drs[r].tolist(),
                    win2[r].tolist(), S,
                )
                if not chunk:
                    continue
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
                        hist,
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
                    # Publish gen-stamped; only the newest generation wins.
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
                                list(hist[:depth]),
                                sorted(rows)[:12],
                            )
                            e_disk._pf_dump_armed = True
                        if self._pf_debug and gen <= self._pf_log_pairs:
                            print(
                                "[pf-pub] gen=%d layer=%d |pred|=%d "
                                "pred_id=%08x"
                                % (gen, layer, len(rows),
                                   e_disk._pf_exp_id),
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
        # round 11; unchanged from v1/v2.)
        ngram = self._pf_ngram
        heads = self._pf_heads
        cols = (ngram - 1) * heads
        mult = self._pf_mult[layer].tolist()
        primes = self._pf_primes
        offsets = self._pf_offsets
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

# v2's runner hook (postprocess_sampled anchor + v2 marker) — removed and
# relocated post-propose when upgrading a v2-patched tree. v2's marker line
# is `MARKER + "\\n"` inserted right after the anchor text.
V2_RUNNER_BLOCK = """# --- engram-prefetch-v2 ---
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

# Fresh-install anchor: AFTER propose + set_draft_tokens (the drafts and
# last_sampled this prediction needs exist only there).
RUNNER_ANCHOR = """        if self.num_speculative_steps > 0:
            # Spec-decode and diffusion LLMs both use draft tokens but the latter does
            # not have a speculator (i.e. self.speculator is None)
            self.draft_tokens_handler.set_draft_tokens(
                input_batch,
                self.req_states.draft_tokens[input_batch.idx_mapping],
            )
"""

RUNNER_HOOK = RUNNER_ANCHOR + MARKER + """
        if self.speculator is not None:
            _stager = getattr(self.model_state, "engram_stager", None)
            if _stager is not None:
                _stager.enqueue_prefetch(
                    input_batch.input_ids,
                    input_batch.positions,
                    input_batch.query_start_loc,
                    self.model_state.lookback_token_ids,
                    sampler_output.sampled_token_ids,
                    num_sampled,
                    draft_tokens,
                    input_batch.num_reqs,
                    input_batch.num_tokens,
                )
"""

# --- census accounting fix (applied to engram_disk.py) ----------------------
#
# Same replacement as v2 (counted only when paired), with the [pf-pair]
# self-check. Unchanged from v2 apart from the marker.

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
                import os as _pfv3_os
                import zlib as _pfv3_zlib

                _pfv3_pairs = getattr(self, "_pfv3_pairs", 0)
                _pfv3_log = (
                    _pfv3_pairs == 0
                    or _pfv3_os.environ.get(
                        "DSV41_ENGRAM_PREFETCH_DEBUG", "0"
                    ) == "1"
                )
                if _pfv3_log and _pfv3_pairs < 20:
                    _pfv3_exp_id = int(
                        getattr(self, "_pf_exp_id", 0)
                    ) or _pfv3_zlib.crc32(
                        b",".join(str(r).encode() for r in sorted(_exp)[:4096])
                    )
                    _pfv3_got_id = _pfv3_zlib.crc32(
                        b",".join(str(r).encode() for r in sorted(_got)[:4096])
                    )
                    print(
                        "[pf-pair] gen=%s table_pred_id=%08x "
                        "consumed_id=%08x |pred|=%d |consumed|=%d "
                        "intersect=%d pair=%d"
                        % (
                            getattr(self, "_pf_exp_gen", "?"),
                            _pfv3_exp_id,
                            _pfv3_got_id,
                            len(_exp),
                            len(_got),
                            len(_got & _exp),
                            _pfv3_pairs,
                        ),
                        flush=True,
                    )
                self._pfv3_pairs = _pfv3_pairs + 1
"""


def _patch_census(disk_text: str) -> str:
    for marker in (MARKER, V2_MARKER):
        if marker in disk_text:
            return disk_text  # already fixed (v3 marker wins)
    if CENSUS_OLD in disk_text:
        return disk_text.replace(CENSUS_OLD, CENSUS_NEW, 1)
    return disk_text  # census not installed; accounting fix is a no-op then


def _swap_stager_methods(text: str, old_marker: str) -> str | None:
    # The METHODS block occurrence of the old marker is immediately
    # followed by a 4-space `def` (the init occurrence precedes an
    # 8-space `self._prefetch_setup()`); only swap that one.
    needle = old_marker + "\n    def "
    i = text.find(needle)
    if i < 0:
        return None
    end = text.index(STAGE_DEF_ANCHOR, i)
    text = text[:i] + STAGER_METHODS + "\n\n" + text[end:]
    # Re-stamp any remaining (init) occurrence of the old marker.
    return text.replace(old_marker, MARKER)


def apply(model_root: Path, runner: Path, vllm_root: Path | None = None) -> None:
    engram = model_root / "common" / "engram.py"
    text = engram.read_text()
    if MARKER not in text:
        swapped = None
        for marker, name in ((V2_MARKER, "v2"), (V1_MARKER, "v1")):
            new_text = _swap_stager_methods(text, marker)
            if new_text is not None:
                text = new_text
                swapped = name
                break
        if swapped is None:
            if STAGER_INIT_TAIL not in text:
                raise SystemExit("engram_prefetch_v3: stager init anchor missing")
            text = text.replace(STAGER_INIT_TAIL, STAGER_INIT_NEW, 1)
            if STAGE_DEF_ANCHOR not in text:
                raise SystemExit("engram_prefetch_v3: stage def anchor missing")
            text = text.replace(
                STAGE_DEF_ANCHOR, STAGER_METHODS + "\n\n" + STAGE_DEF_ANCHOR, 1
            )
        else:
            # swap carried the _prefetch_setup() call; ensure it survived
            if "self._prefetch_setup()" not in text:
                if STAGER_INIT_TAIL not in text:
                    raise SystemExit(
                        "engram_prefetch_v3: stager init anchor missing on swap"
                    )
                text = text.replace(STAGER_INIT_TAIL, STAGER_INIT_NEW, 1)
        engram.write_text(text)
        print(
            "dsv41: engram prefetch v3 stager installed"
            + (f" (replaced {swapped} methods)" if swapped else "")
        )

    rtext = runner.read_text()
    if MARKER not in rtext:
        if V2_RUNNER_BLOCK in rtext:
            # upgrade path: remove v2's postprocess-site hook entirely (the
            # v3 prediction needs `draft_tokens`, which only exists AFTER
            # propose), then install the v3 hook at the post-propose anchor.
            rtext = rtext.replace(V2_RUNNER_BLOCK, "", 1)
        if MARKER not in rtext:
            if V1_MARKER in rtext:
                raise SystemExit(
                    "engram_prefetch_v3: v1 runner hook present — apply v2 "
                    "first or clean the tree"
                )
            if RUNNER_ANCHOR not in rtext:
                raise SystemExit("engram_prefetch_v3: runner anchor missing")
            rtext = rtext.replace(RUNNER_ANCHOR, RUNNER_HOOK, 1)
        runner.write_text(rtext)
        print("dsv41: engram prefetch v3 runner hook installed")

    if vllm_root is not None:
        disk = vllm_root / "models" / "deepseek_v4_1" / "common" / "engram_disk.py"
        dtext = disk.read_text()
        fixed = _patch_census(dtext)
        if fixed != dtext:
            disk.write_text(fixed)
            print("dsv41: engram census pf_hit accounting fixed (v3)")


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
