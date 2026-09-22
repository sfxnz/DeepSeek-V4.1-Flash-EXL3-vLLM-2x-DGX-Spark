#!/usr/bin/env python3
"""CPU-side Engram hashes: delete the prepare_inputs D2H + event sync.

Round-18 trace attribution (results/2026-09-21-trace-attribution/
ATTRIBUTION.md): 99.9% of the ~14.2 ms/step GPU-idle pool is ONE stack —
EngramDiskStager.stage -> hashes_ready.synchronize(), a hard
cudaEventSynchronize waiting on a tiny D2H hash copy queued BEHIND the
previous step's graph. The hash data ultimately depends on the sampler
output, so any D2H of it queues behind the graph; the only way out is to
have the NEXT step's hashes already on host before prepare_inputs.

Premise correction (verified against the canonical-e12 image): the next
step's input_ids are NOT host-resident — input_batch.input_ids is a GPU
buffer filled by combine_sampled_and_draft_tokens from GPU-only
last_sampled_tokens / draft_tokens. But at post-propose time the next
chunk is fully determined: chunk_r = [s_{A-1}] + draft_r at S_r+A_r..,
window_r[j] = seq[S_r+A_r-1-j] (the prefetch-v3 reconstruction, verified
against live stage dumps in round 16 and re-verified by the bit-exact
warmup below). So:

  * post-propose hook: snapshot ids/pos/qsl/window/sampled/drafts to
    pinned mirrors on a side stream (the wait happens on a WORKER
    thread, never the main thread), then a worker reconstructs the next
    chunk, mirrors _hash_ids_kernel bit-exactly with numpy int64, writes
    hash_host, and preads+dequants all tables into rows_host (reusing
    the fast-stage thread pool) — all while the main thread runs engine
    python.
  * commit hook (host-only, no GPU): record this step's batch signature
    (num_tokens, req_ids, query_start_loc_np, has_prefill).
  * stage(): if the prediction matches the batch signature bit-for-bit
    and the worker finished, just H2D the staged rows. The GPU hash
    launch, the D2H, and hashes_ready.synchronize() are SKIPPED on the
    main thread entirely.

Safety: the first 4 fast steps ALSO compute the real GPU hash and
compare the full owned head-slice bit-exactly against the CPU mirror
(any mismatch disarms permanently and that step uses the stock path);
afterwards an async canary (side-stream GPU hash, compared off-thread)
keeps guarding. Any exception, worker error, signature mismatch,
prefill batch, or oversized batch falls back to the stock path for that
step. DSV41_ENGRAM_CPU_HASH=1 enables (default off). Idempotent.
Requires the engram_stage_fast chain (uses its thread pool); composes
with engram_prefetch_v3 (independent markers).
"""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "# --- engram-cpu-hash ---"
COMMIT_MARKER = "# --- engram-cpu-hash-commit ---"

# ---------------------------------------------------------------------------
# stage() replacement.  OLD_STAGE must be byte-identical to the stage() that
# engram_stage_fast.py installs (validated by validate_cpu_hash_chain.py
# against engram_stage_fast.STAGE_FAST, so drift is caught offline).
# ---------------------------------------------------------------------------

OLD_STAGE = '''    @torch.inference_mode()
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
        import os as _pf_os

        if _pf_os.environ.get("DSV41_ENGRAM_PF_DUMP", "0") == "1":
            _win = (
                lookback_token_ids[0].tolist()
                if lookback_token_ids is not None
                and lookback_token_ids.numel()
                else []
            )
            print(
                "[pf-dump-stage] n=%d ids=%s pos=%s win=%s"
                % (n, ids.tolist()[:8], positions[:n].tolist()[:8], _win),
                flush=True,
            )
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

        def _fast_stage_one(engram, buf):
            # CPU-side gather + dequant only; the H2D copy_ stays on the
            # calling thread so it keeps the graph replay's stream order.
            local = host[:, engram.layer_hash_index, :].to(torch.int64)
            file_rows, owned = engram.embed_tokens.disk_file_rows_owned(local)
            rows = engram.embed_tokens.disk.gather_dequant(file_rows, owned)
            staged = buf[:n]
            staged.copy_(rows.view(n, self.local_heads, self.dim))

        pairs = list(zip(self.engrams, self.rows_host))
        if (
            not _ENG_FAST_STAGE_OK[0]
            or len(pairs) == 1
        ):
            for engram, buf in pairs:
                _fast_stage_one(engram, buf)
        else:
            futs = [
                _ENG_STAGE_POOL.submit(_fast_stage_one, engram, buf)
                for engram, buf in pairs
            ]
            first_exc = None
            for fut in futs:
                exc = fut.exception()
                if exc is not None and first_exc is None:
                    first_exc = exc
            if first_exc is not None:
                _ENG_FAST_STAGE_OK[0] = False
                print(
                    "dsv41: engram fast stage disabled after worker error: "
                    f"{first_exc!r}",
                    flush=True,
                )
                raise first_exc

        if _ENG_FAST_STAGE_SELFCHECK[0]:
            _ENG_FAST_STAGE_SELFCHECK[0] = False
            try:
                ref = [
                    (
                        engram._staged_rows_for_ubatch()[:n].clone(),
                        buf[:n].clone(),
                    )
                    for engram, buf in pairs
                ]
                for (engram, buf), (dest_ref, staged_ref) in zip(pairs, ref):
                    _fast_stage_one(engram, buf)
                    dest = engram._staged_rows_for_ubatch()[:n]
                    if not torch.equal(dest, dest_ref) or not torch.equal(
                        buf[:n], staged_ref
                    ):
                        raise RuntimeError(
                            "engram fast stage self-check mismatch"
                        )
                print(
                    "dsv41: engram fast stage self-check bit-exact "
                    f"({len(pairs)} tables, n={n})",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                _ENG_FAST_STAGE_OK[0] = False
                print(
                    "dsv41: engram fast stage REVERTED to serial: "
                    f"{exc!r}",
                    flush=True,
                )

        for engram, buf in pairs:
            staged = buf[:n]
            dest = engram._staged_rows_for_ubatch()[:n]
            dest.copy_(staged, non_blocking=True)
        self.num_staged = n
        return n
'''

NEW_STAGE = '''    @torch.inference_mode()
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
        import os as _pf_os

        if _pf_os.environ.get("DSV41_ENGRAM_PF_DUMP", "0") == "1":
            _win = (
                lookback_token_ids[0].tolist()
                if lookback_token_ids is not None
                and lookback_token_ids.numel()
                else []
            )
            print(
                "[pf-dump-stage] n=%d ids=%s pos=%s win=%s"
                % (n, ids.tolist()[:8], positions[:n].tolist()[:8], _win),
                flush=True,
            )
        _ch_fast = self._ch_try_stage(
            n, ids, positions[:n], query_start_loc, lookback_token_ids
        )
        if not _ch_fast:
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
            host.copy_(
                hashes[:, :, self.head_start : self.head_end], non_blocking=True
            )
            self.hashes_ready.record()
            self.hashes_ready.synchronize()

            def _fast_stage_one(engram, buf):
                # CPU-side gather + dequant only; the H2D copy_ stays on the
                # calling thread so it keeps the graph replay's stream order.
                local = host[:, engram.layer_hash_index, :].to(torch.int64)
                file_rows, owned = engram.embed_tokens.disk_file_rows_owned(local)
                rows = engram.embed_tokens.disk.gather_dequant(file_rows, owned)
                staged = buf[:n]
                staged.copy_(rows.view(n, self.local_heads, self.dim))

            pairs = list(zip(self.engrams, self.rows_host))
            if (
                not _ENG_FAST_STAGE_OK[0]
                or len(pairs) == 1
            ):
                for engram, buf in pairs:
                    _fast_stage_one(engram, buf)
            else:
                futs = [
                    _ENG_STAGE_POOL.submit(_fast_stage_one, engram, buf)
                    for engram, buf in pairs
                ]
                first_exc = None
                for fut in futs:
                    exc = fut.exception()
                    if exc is not None and first_exc is None:
                        first_exc = exc
                if first_exc is not None:
                    _ENG_FAST_STAGE_OK[0] = False
                    print(
                        "dsv41: engram fast stage disabled after worker error: "
                        f"{first_exc!r}",
                        flush=True,
                    )
                    raise first_exc

            if _ENG_FAST_STAGE_SELFCHECK[0]:
                _ENG_FAST_STAGE_SELFCHECK[0] = False
                try:
                    ref = [
                        (
                            engram._staged_rows_for_ubatch()[:n].clone(),
                            buf[:n].clone(),
                        )
                        for engram, buf in pairs
                    ]
                    for (engram, buf), (dest_ref, staged_ref) in zip(pairs, ref):
                        _fast_stage_one(engram, buf)
                        dest = engram._staged_rows_for_ubatch()[:n]
                        if not torch.equal(dest, dest_ref) or not torch.equal(
                            buf[:n], staged_ref
                        ):
                            raise RuntimeError(
                                "engram fast stage self-check mismatch"
                            )
                    print(
                        "dsv41: engram fast stage self-check bit-exact "
                        f"({len(pairs)} tables, n={n})",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    _ENG_FAST_STAGE_OK[0] = False
                    print(
                        "dsv41: engram fast stage REVERTED to serial: "
                        f"{exc!r}",
                        flush=True,
                    )

        for engram, buf in zip(self.engrams, self.rows_host):
            staged = buf[:n]
            dest = engram._staged_rows_for_ubatch()[:n]
            dest.copy_(staged, non_blocking=True)
        if _ch_fast:
            self._ch_note_h2d()
        self.num_staged = n
        return n
'''

# ---------------------------------------------------------------------------
# Stager methods block (inserted before stage()).
# ---------------------------------------------------------------------------

STAGER_METHODS = MARKER + '''\


    # ---- CPU-side Engram hash (Round 18 attribution fix) ------------------
    # See engram_cpu_hash.py. _ch_try_stage returns True -> hash_host /
    # rows_host already hold this step's values (computed off-thread from
    # the post-propose prediction) and the stock hash+D2H+sync+gather is
    # skipped. Everything self-disarms to the stock path.

    def _ch_armed(self) -> bool:
        import os as _os

        st = getattr(self, "_ch_state", None)
        if st is not None:
            return st["on"] and st["ok"]
        import numpy as _np
        import torch as _torch

        hs = self.hash_state
        st = {
            "on": _os.environ.get("DSV41_ENGRAM_CPU_HASH", "0") == "1",
            "ok": True,
            "warm": 0,
            "warm_n": 4,
            "fast": 0,
            "gen": 0,
            "pending": None,
            "pred": None,
            "batch_sig": None,
            "pool": None,
            "stream": None,
            "snap_ev": None,
            "canary_ev": None,
            "h2d_ev": None,
            "pin": {},
        }
        self._ch_state = st
        if not st["on"]:
            return False
        try:
            if getattr(hs, "use_slot_cache", False):
                raise RuntimeError("slot-cache hash state not mirrorable")
            self._ch_tm = _np.asarray(
                hs.token_map.detach().to("cpu").to(_torch.int64).tolist(),
                dtype=_np.int64,
            )
            self._ch_pad = int(hs.pad_id)
            self._ch_ngram = int(hs.multipliers.shape[1])
            self._ch_heads = int(hs.primes.shape[-1])
            self._ch_nlayers = int(hs.multipliers.shape[0])
            self._ch_depth = int(hs.lookback_depth)
            self._ch_mult = _np.asarray(
                hs.multipliers.detach().to("cpu").to(_torch.int64).tolist(),
                dtype=_np.int64,
            )
            self._ch_primes = _np.asarray(
                hs.primes.detach().to("cpu").to(_torch.int64).reshape(-1).tolist(),
                dtype=_np.int64,
            ).reshape(self._ch_nlayers, (self._ch_ngram - 1) * self._ch_heads)
            self._ch_offs = _np.asarray(
                hs.offsets.detach().to("cpu").to(_torch.int64).reshape(-1).tolist(),
                dtype=_np.int64,
            ).reshape(self._ch_nlayers, (self._ch_ngram - 1) * self._ch_heads)
            from .mm_preprocess import (
                IMAGE_PAD_ID,
                IMAGE_SENTINEL_BASE_ID,
            )

            self._ch_dead_ids = _np.asarray(
                [IMAGE_SENTINEL_BASE_ID, IMAGE_PAD_ID], dtype=_np.int64
            )
            # Snapshot capacities (decode-shaped; larger batches fall back).
            self._ch_rmax = 8
            self._ch_cap = max(128, int(self.max_tokens))
            self._ch_omax = 64
            self._ch_kmax = 8
            span = self.head_end - self.head_start
            pin = st["pin"]
            pin["ids"] = _torch.zeros(self._ch_cap, dtype=_torch.int64, pin_memory=True)
            pin["pos"] = _torch.zeros(self._ch_cap, dtype=_torch.int64, pin_memory=True)
            pin["aid"] = _torch.zeros(self._ch_cap, dtype=_torch.int64, pin_memory=True)
            pin["apos"] = _torch.zeros(self._ch_cap, dtype=_torch.int64, pin_memory=True)
            pin["awin"] = _torch.zeros(
                self._ch_rmax * self._ch_depth, dtype=_torch.int64, pin_memory=True
            )
            pin["qsl"] = _torch.zeros(
                self._ch_rmax + 1, dtype=_torch.int64, pin_memory=True
            )
            pin["win"] = _torch.zeros(
                self._ch_rmax * self._ch_depth, dtype=_torch.int64, pin_memory=True
            )
            pin["outs"] = _torch.zeros(
                self._ch_rmax * self._ch_omax, dtype=_torch.int64, pin_memory=True
            )
            pin["draft"] = _torch.zeros(
                self._ch_kmax * self._ch_rmax, dtype=_torch.int64, pin_memory=True
            )
            pin["ns"] = _torch.zeros(self._ch_rmax, dtype=_torch.int64, pin_memory=True)
            pin["ref"] = _torch.zeros(
                (int(self.max_tokens), self._ch_nlayers, span),
                dtype=_torch.int32,
                pin_memory=True,
            )
            pin["gpu"] = _torch.zeros(
                (int(self.max_tokens), self._ch_nlayers, span),
                dtype=_torch.int32,
                pin_memory=True,
            )
            from concurrent.futures import ThreadPoolExecutor as _TPE

            st["pool"] = _TPE(max_workers=1)
            st["stream"] = _torch.cuda.Stream()
            st["snap_ev"] = _torch.cuda.Event()
            st["canary_ev"] = _torch.cuda.Event()
            st["h2d_ev"] = _torch.cuda.Event()
            st["h2d_ev"].record()
            print(
                "dsv41: engram cpu-hash armed (layers=%d ngram=%d heads=%d "
                "span=%d depth=%d)"
                % (
                    self._ch_nlayers,
                    self._ch_ngram,
                    self._ch_heads,
                    span,
                    self._ch_depth,
                ),
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            st["on"] = False
            print(
                f"dsv41: engram cpu-hash disabled at setup: {exc!r}", flush=True
            )
        return st["on"] and st["ok"]

    def _ch_drain(self) -> None:
        # Join any unconsumed prediction worker (rare: fallback step or a
        # superseded generation) so it never writes pinned buffers while
        # the stock path (or a newer worker) uses them.
        st = getattr(self, "_ch_state", None)
        if st is None:
            return
        pend = st["pending"]
        if pend is not None:
            st["pending"] = None
            try:
                pend["fut"].result()
            except Exception:  # noqa: BLE001
                pass
        st["pred"] = None

    def cpu_hash_commit_batch(self, input_batch) -> None:
        # Host-only signature of THIS step's batch (called from the runner
        # just before prepare_inputs); stage()'s fast path verifies the
        # prediction against it. Never touches the GPU.
        if not self._ch_armed():
            return
        try:
            st = self._ch_state
            nq = int(input_batch.num_reqs)
            st["batch_sig"] = (
                int(input_batch.num_tokens),
                list(input_batch.req_ids[:nq]),
                input_batch.query_start_loc_np[: nq + 1].tolist(),
                bool(input_batch.has_prefill),
            )
        except Exception:  # noqa: BLE001
            self._ch_state["batch_sig"] = None

    def enqueue_cpu_hash(
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
        req_ids=None,
        has_prefill: bool = True,
    ) -> None:
        # Post-propose: snapshot this step's outputs to pinned mirrors on a
        # side stream and predict/hash/gather the NEXT step off-thread.
        if not self._ch_armed():
            return
        st = self._ch_state
        try:
            if (
                has_prefill
                or draft_tokens is None
                or draft_tokens.numel() == 0
                or num_reqs <= 0
                or num_reqs > self._ch_rmax
                or num_tokens <= 0
                or num_tokens > self._ch_cap
                or req_ids is None
            ):
                self._ch_drain()
                return
            k = min(int(draft_tokens.shape[1]), self._ch_kmax)
            om = min(
                int(sampled_token_ids.shape[1])
                if sampled_token_ids is not None and sampled_token_ids.dim() == 2
                else self._ch_omax,
                self._ch_omax,
            )
            self._ch_drain()
            import torch as _torch

            # Order the side stream behind ALL main-stream work queued so far
            # (incl. the dspark draft-graph replay that writes draft_tokens).
            # MUST be issued while the MAIN stream is current: inside the
            # with-block below, current_stream() IS the side stream and
            # wait_stream(self) is a no-op (the Round-19 stale-draft bug).
            st["stream"].wait_stream(_torch.cuda.current_stream())
            with _torch.cuda.stream(st["stream"]):
                st["pin"]["ids"][:num_tokens].copy_(
                    input_ids[:num_tokens].to(_torch.int64), non_blocking=True
                )
                st["pin"]["pos"][:num_tokens].copy_(
                    positions[:num_tokens].to(_torch.int64), non_blocking=True
                )
                st["pin"]["qsl"][: num_reqs + 1].copy_(
                    query_start_loc[: num_reqs + 1].to(_torch.int64),
                    non_blocking=True,
                )
                if window is not None and window.numel():
                    win = window[:num_reqs, :].reshape(-1).to(_torch.int64)
                    st["pin"]["win"][: min(win.numel(), st["pin"]["win"].numel())].copy_(
                        win[: st["pin"]["win"].numel()], non_blocking=True
                    )
                outs = (
                    sampled_token_ids[:num_reqs, :om].reshape(-1).to(_torch.int64)
                )
                st["pin"]["outs"][: outs.numel()].copy_(outs, non_blocking=True)
                st["ch_om"] = om
                dr = draft_tokens[:num_reqs, :k].reshape(-1).to(_torch.int64)
                st["pin"]["draft"][: dr.numel()].copy_(dr, non_blocking=True)
                st["pin"]["ns"][:num_reqs].copy_(
                    num_sampled[:num_reqs].to(_torch.int64), non_blocking=True
                )
                st["snap_ev"].record()
            st["req_ids_snap"] = [str(r) for r in req_ids[:num_reqs]]
            st["gen"] = st.get("gen", 0) + 1
            gen = st["gen"]
            st["pending"] = {
                "gen": gen,
                "fut": st["pool"].submit(
                    self._ch_worker, gen, num_reqs, num_tokens, k
                ),
            }
        except Exception as exc:  # noqa: BLE001
            # Anchor-level surprise (e.g. a run path without a sampler):
            # fall back to stock for this step; do NOT disarm — the warmup
            # bit-exact check and the canary own that decision.
            self._ch_drain()
            st["err_n"] = st.get("err_n", 0) + 1
            if st["err_n"] <= 3:
                print(
                    f"dsv41: engram cpu-hash enqueue skipped: {exc!r}", flush=True
                )

    def _ch_predict(self, snap):
        # Reconstruct the next step's gather inputs from the snapshot
        # (prefetch-v3 rule, round-16-verified): chunk_r = [s_{A-1}] +
        # draft_r at S_r+A_r.., window_r[j] = seq[S_r+A_r-1-j].
        depth = self._ch_depth
        ids, pos, qsl, wins, reqs = [], [], [0], [], []
        n = 0
        for r in range(len(snap["ns"])):
            A = int(snap["ns"][r])
            if A <= 0:
                return None
            outs = snap["outs"][r]
            if A > len(outs):
                return None
            drafts = snap["drafts"][r]
            q0 = int(snap["qsl"][r])
            if not (0 <= q0 < len(snap["ids"])):
                return None
            S = int(snap["pos"][q0])
            chunk = [int(outs[A - 1])] + [int(d) for d in drafts]
            if S + A < 0:
                return None
            cpos = list(range(S + A, S + A + len(chunk)))
            bonus_old = int(snap["ids"][q0])
            win = []
            for j in range(depth):
                if j < A - 1:
                    win.append(int(outs[A - 2 - j]))
                elif j == A - 1:
                    win.append(bonus_old)
                else:
                    wj = j - A
                    wr = snap["win"][r]
                    win.append(int(wr[wj]) if 0 <= wj < len(wr) else -1)
            ids.extend(chunk)
            pos.extend(cpos)
            n += len(chunk)
            qsl.append(n)
            wins.append(win)
            reqs.append(snap["req_ids"][r])
        if n <= 0 or n > self._ch_cap:
            return None
        return {"n": n, "ids": ids, "pos": pos, "qsl": qsl,
                "wins": wins, "req_ids": reqs}

    def _ch_hash_layers(self, ids, pos, qsl, wins):
        # Vectorized mirror of _hash_ids_kernel (bit-exact; verified
        # against the GPU kernel at warmup and by the async canary).
        # Column/row semantics: see the Triton kernel in this file.
        import numpy as _np

        tm = self._ch_tm
        pad = self._ch_pad
        ngram = self._ch_ngram
        heads = self._ch_heads
        L = self._ch_nlayers
        h0 = self.head_start
        ncol = self.head_end - self.head_start
        ids = _np.asarray(ids, dtype=_np.int64)
        pos = _np.asarray(pos, dtype=_np.int64)
        qsl = _np.asarray(qsl, dtype=_np.int64)
        win = _np.asarray(wins, dtype=_np.int64)
        n = len(ids)
        toks = _np.arange(n, dtype=_np.int64)
        # request of each token (kernel binary search on query_start_loc)
        req = _np.minimum(
            _np.searchsorted(qsl[1:], toks, side="right"),
            len(qsl) - 2,
        )
        chunk_start = pos[_np.clip(qsl[_np.clip(req, 0, len(qsl) - 1)], 0, n - 1)]
        tm_lo = tm[_np.clip(ids, 0, len(tm) - 1)]
        out = _np.zeros((n, L, (ngram - 1) * heads), dtype=_np.int64)
        rolling = _np.zeros((n, L), dtype=_np.int64)
        blocked = _np.zeros(n, dtype=bool)
        nv = len(tm)
        for shift in range(ngram):
            lookback = pos - shift
            in_batch = lookback >= chunk_start
            # batch branch: source row is ids[token - shift] (kernel
            # indexes by ABSOLUTE token row, masked by in_batch)
            bi = _np.clip(toks - shift, 0, n - 1)
            b_tok = ids[bi]
            b_src = _np.where(
                (b_tok >= 0) & (b_tok < nv),
                tm[_np.clip(b_tok, 0, nv - 1)],
                pad,
            )
            b_src = _np.where(
                _np.isin(b_tok, self._ch_dead_ids), _np.int64(-1), b_src
            )
            # window branch: col = chunk_start - 1 - lookback (newest first)
            col = chunk_start - 1 - lookback
            in_win = (~in_batch) & (col >= 0) & (col < self._ch_depth)
            colc = _np.clip(col, 0, self._ch_depth - 1)
            w_tok = win[_np.clip(req, 0, win.shape[0] - 1), colc]
            known = in_win & (w_tok >= 0)
            w_src = _np.where(
                (w_tok >= 0) & (w_tok < nv),
                tm[_np.clip(w_tok, 0, nv - 1)],
                pad,
            )
            w_src = _np.where(
                _np.isin(w_tok, self._ch_dead_ids), _np.int64(-1), w_src
            )
            source = _np.where(in_batch, b_src, _np.where(known, w_src, pad))
            blocked = blocked | (lookback < 0) | (source == -1)
            value = _np.where(blocked, pad, source)
            rolling = rolling ^ (value[:, None] * self._ch_mult[:, shift][None, :])
            if shift > 0:
                base = (shift - 1) * heads
                p = self._ch_primes[:, base : base + heads]
                o = self._ch_offs[:, base : base + heads]
                out[:, :, base : base + heads] = (
                    rolling[:, :, None] % p[None, :, :] + o[None, :, :]
                )
        return out[:, :, h0 : h0 + ncol]

    def _ch_worker(self, gen, num_reqs, num_tokens, k):
        # Off-thread: wait the pinned snapshots, reconstruct + CPU-hash the
        # next chunk into hash_host, pread/dequant all tables into
        # rows_host. Waits (side events) happen HERE, never on the main
        # thread.
        st = self._ch_state
        try:
            st["h2d_ev"].synchronize()
            st["snap_ev"].synchronize()
            if gen != st["gen"]:
                return
            depth = self._ch_depth
            om = st.get("ch_om", self._ch_omax)
            snap = {
                "ids": st["pin"]["ids"][:num_tokens].tolist(),
                "pos": st["pin"]["pos"][:num_tokens].tolist(),
                "qsl": st["pin"]["qsl"][: num_reqs + 1].tolist(),
                "win": st["pin"]["win"][: num_reqs * depth]
                .reshape(num_reqs, depth)
                .tolist(),
                "outs": st["pin"]["outs"][: num_reqs * om]
                .reshape(num_reqs, om)
                .tolist(),
                "drafts": st["pin"]["draft"][: num_reqs * k]
                .reshape(num_reqs, k)
                .tolist(),
                "ns": st["pin"]["ns"][:num_reqs].tolist(),
                "req_ids": list(st["req_ids_snap"]),
            }
            pred = self._ch_predict(snap)
            if pred is None:
                return
            n = pred["n"]
            hashed = self._ch_hash_layers(
                pred["ids"], pred["pos"], pred["qsl"], pred["wins"]
            )
            import numpy as _np
            import torch as _torch

            self.hash_host[:n].copy_(
                _torch.from_numpy(hashed.astype(_np.int32))
            )
            hh = self.hash_host

            def _gather_one(engram, buf):
                local = hh[:n, engram.layer_hash_index, :].to(_torch.int64)
                file_rows, owned = engram.embed_tokens.disk_file_rows_owned(local)
                rows = engram.embed_tokens.disk.gather_dequant(file_rows, owned)
                buf[:n].copy_(rows.view(n, self.local_heads, self.dim))

            pool = globals().get("_ENG_STAGE_POOL")
            if pool is not None and len(self.engrams) > 1:
                futs = [
                    pool.submit(_gather_one, e, b)
                    for e, b in zip(self.engrams, self.rows_host)
                ]
                for f in futs:
                    exc = f.exception()
                    if exc is not None:
                        raise exc
            else:
                for e, b in zip(self.engrams, self.rows_host):
                    _gather_one(e, b)
            st["pred"] = {
                "n": n,
                "req_ids": list(pred["req_ids"]),
                "qsl": [int(x) for x in pred["qsl"]],
                "hash": self.hash_host[:n].clone(),
            }
            st["pred_ids_dbg"] = [int(x) for x in pred["ids"]]
            st["pred_pos_dbg"] = [int(x) for x in pred["pos"]]
        except Exception as exc:  # noqa: BLE001
            st["ok"] = False
            import traceback as _tb

            print(
                "dsv41: engram cpu-hash DISABLED (worker): "
                + repr(exc)
                + chr(10)
                + _tb.format_exc(),
                flush=True,
            )

    def _ch_canary(self, ref, n):
        # Off-thread guard: GPU hash of the ACTUAL ids vs the CPU mirror
        # that fed this step. Any mismatch disarms permanently.
        try:
            st = self._ch_state
            st["canary_ev"].synchronize()
            import torch as _torch

            if not _torch.equal(st["pin"]["gpu"][:n], ref[:n]):
                st["ok"] = False
                print(
                    "dsv41: engram cpu-hash DISABLED after canary mismatch",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            self._ch_state["ok"] = False
            print(
                f"dsv41: engram cpu-hash disabled (canary): {exc!r}", flush=True
            )

    def _ch_try_stage(self, n, ids, positions, query_start_loc, lookback_token_ids):
        # Main thread. True -> hash_host/rows_host are ready for this step
        # and the caller must skip the stock hash+D2H+sync+gather.
        if not self._ch_armed():
            return False
        st = self._ch_state
        pend = st["pending"]
        st["pending"] = None
        if pend is None:
            return False
        try:
            pend["fut"].result()
        except Exception as exc:  # noqa: BLE001
            st["ok"] = False
            print(
                f"dsv41: engram cpu-hash disabled (join): {exc!r}", flush=True
            )
            return False
        pred = st["pred"]
        st["pred"] = None
        sig = st.get("batch_sig")
        if (
            pred is None
            or sig is None
            or sig[3]
            or sig[0] != pred["n"]
            or list(sig[1]) != list(pred["req_ids"])
            or [int(x) for x in sig[2]] != [int(x) for x in pred["qsl"]]
        ):
            # Scheduler surprise or no usable prediction: stock path
            # (worker already drained; buffers will be rewritten).
            return False
        import torch as _torch

        if st["warm"] < st["warm_n"]:
            # Two-stage warmup validation on the REAL batch:
            #   A) mirror-vs-GPU on the ACTUAL ids (isolates the hash
            #      mirror from the prediction);
            #   B) predicted ids/pos/qsl/window vs actual (isolates the
            #      reconstruction rule).
            # Bounded dumps on failure name the diverging stream.
            from .mm_preprocess import image_sentinel_mask

            hashes = self.hash_state(
                ids,
                positions,
                query_start_loc,
                image_sentinel_mask(ids),
                lookback_token_ids,
                image_sentinel_mask(lookback_token_ids),
                None,
                None,
            )
            st["pin"]["ref"][:n].copy_(
                hashes[:, :, self.head_start : self.head_end], non_blocking=True
            )
            self.hashes_ready.record()
            self.hashes_ready.synchronize()

            def _dump(tag):
                m = ~_torch.eq(
                    st["pin"]["gpu"][:n], self.hash_host[:n]
                )
                bad = m.any(dim=(1, 2))
                idx = _torch.nonzero(bad).flatten()[:6].tolist()
                lines = [
                    "dsv41: engram cpu-hash warmup %s mismatch (n=%d, "
                    "pred_n=%d, mism rows=%d/%d)"
                    % (tag, n, pred["n"], int(bad.sum()), n)
                ]
                for t in idx:
                    lines.append(
                        "  row %d: aid=%s apos=%s | pid=%s ppos=%s | "
                        "gpu=%s cpu=%s"
                        % (
                            t,
                            a_ids[t],
                            a_pos[t],
                            (pred_ids[t] if t < len(pred_ids) else None),
                            (pred_pos[t] if t < len(pred_pos) else None),
                            st["pin"]["gpu"][t, 0, :4].tolist(),
                            self.hash_host[t, 0, :4].tolist(),
                        )
                    )
                print(chr(10).join(lines), flush=True)

            # stage A: snapshot the ACTUAL ids/pos/window to pinned mirrors
            # (side stream; sync via the event recorded after the copies)
            import numpy as _np

            # Order after main-stream writes of ids/pos/window (see enqueue:
            # wait_stream must be issued while the MAIN stream is current).
            st["stream"].wait_stream(_torch.cuda.current_stream())
            with _torch.cuda.stream(st["stream"]):
                st["pin"]["aid"][:n].copy_(ids.to(_torch.int64), non_blocking=True)
                st["pin"]["apos"][:n].copy_(
                    positions.to(_torch.int64), non_blocking=True
                )
                if lookback_token_ids is not None and lookback_token_ids.numel():
                    aw = lookback_token_ids.reshape(-1).to(_torch.int64)
                    st["pin"]["awin"][
                        : min(aw.numel(), st["pin"]["awin"].numel())
                    ].copy_(aw[: st["pin"]["awin"].numel()], non_blocking=True)
                st["snap_ev"].record()
            st["snap_ev"].synchronize()  # aid/apos/awin landed
            nr = int(query_start_loc.numel()) - 1
            a_ids = st["pin"]["aid"][:n].tolist()
            a_pos = st["pin"]["apos"][:n].tolist()
            a_qsl = [int(x) for x in query_start_loc.tolist()[: nr + 1]]
            depth = self._ch_depth
            awin = (
                st["pin"]["awin"][: nr * depth].reshape(nr, depth).tolist()
                if nr > 0
                else []
            )
            try:
                a_hash = self._ch_hash_layers(a_ids, a_pos, a_qsl, awin)
                st["pin"]["gpu"][:n].copy_(
                    _torch.from_numpy(a_hash.astype(_np.int32))
                )
            except Exception as exc:  # noqa: BLE001
                st["ok"] = False
                print(
                    f"dsv41: engram cpu-hash DISABLED (mirror threw): {exc!r}",
                    flush=True,
                )
                return False
            if not _torch.equal(st["pin"]["gpu"][:n], st["pin"]["ref"][:n]):
                _dump("MIRROR")
                st["ok"] = False
                return False
            # stage B: predicted ids/pos must equal actual
            pred_ids = list(st.get("pred_ids_dbg") or [])
            pred_pos = list(st.get("pred_pos_dbg") or [])
            if (
                len(pred_ids) != n
                or len(pred_pos) != n
                or any(
                    int(pred_ids[t]) != int(a_ids[t]) for t in range(n)
                )
                or any(
                    int(pred_pos[t]) != int(a_pos[t]) for t in range(n)
                )
            ):
                _dump("PREDICT")
                st["ok"] = False
                print(
                    "dsv41: engram cpu-hash DISABLED: prediction != actual "
                    "ids/pos (see dump above)",
                    flush=True,
                )
                return False
            st["warm"] += 1
            if st["warm"] == st["warm_n"]:
                print(
                    "dsv41: engram cpu-hash ACTIVE (mirror+predict bit-exact "
                    "x%d; prepare_inputs D2H+event sync removed)" % st["warm_n"],
                    flush=True,
                )
            return True
        # Steady state: async canary on a side stream; the main thread
        # never waits for it.
        from .mm_preprocess import image_sentinel_mask

        # Order the canary hash after this step's main-stream work (see
        # enqueue: wait_stream must be issued while the MAIN stream is
        # current — inside the with-block it would self-wait, a no-op).
        st["stream"].wait_stream(_torch.cuda.current_stream())
        with _torch.cuda.stream(st["stream"]):
            hashes = self.hash_state(
                ids,
                positions,
                query_start_loc,
                image_sentinel_mask(ids),
                lookback_token_ids,
                image_sentinel_mask(lookback_token_ids),
                None,
                None,
            )
            st["pin"]["gpu"][:n].copy_(
                hashes[:, :, self.head_start : self.head_end], non_blocking=True
            )
            st["canary_ev"].record()
        st["pool"].submit(self._ch_canary, pred["hash"], n)
        st["fast"] = st.get("fast", 0) + 1
        if st["fast"] % 500 == 0:
            print(
                "dsv41: engram cpu-hash fast-path steps=%d" % st["fast"],
                flush=True,
            )
        return True

    def _ch_note_h2d(self) -> None:
        # Recorded after the staged-rows H2D so the next worker knows the
        # pinned buffers are free.
        st = getattr(self, "_ch_state", None)
        if st is not None and st["on"]:
            st["h2d_ev"].record()
'''

# ---------------------------------------------------------------------------
# Runner hooks.
# ---------------------------------------------------------------------------

# Post-propose hook: appended after the prefetch-v3 hook when present,
# else installed at v3's fresh anchor.
V3_HOOK_BODY = '''        if self.speculator is not None:
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
'''

RUNNER_ANCHOR = """        if self.num_speculative_steps > 0:
            # Spec-decode and diffusion LLMs both use draft tokens but the latter does
            # not have a speculator (i.e. self.speculator is None)
            self.draft_tokens_handler.set_draft_tokens(
                input_batch,
                self.req_states.draft_tokens[input_batch.idx_mapping],
            )
"""

ENQUEUE_HOOK = MARKER + '''
        if self.speculator is not None:
            _stager = getattr(self.model_state, "engram_stager", None)
            if _stager is not None:
                _stager.enqueue_cpu_hash(
                    input_batch.input_ids,
                    input_batch.positions,
                    input_batch.query_start_loc,
                    self.model_state.lookback_token_ids,
                    sampler_output.sampled_token_ids,
                    num_sampled,
                    draft_tokens,
                    input_batch.num_reqs,
                    input_batch.num_tokens,
                    input_batch.req_ids,
                    input_batch.has_prefill,
                )
'''

COMMIT_ANCHOR = """        input_ids = input_batch.input_ids
        inputs_embeds = None
        ec_connector_output = None
"""

COMMIT_HOOK = COMMIT_MARKER + '''
        _stager = getattr(self.model_state, "engram_stager", None)
        if _stager is not None and input_batch.input_ids is not None:
            _stager.cpu_hash_commit_batch(input_batch)
'''

STAGE_DEF_ANCHOR = "    @torch.inference_mode()\n    def stage("


def apply(model_root: Path, runner: Path) -> None:
    engram = model_root / "common" / "engram.py"
    text = engram.read_text()
    if MARKER not in text:
        if OLD_STAGE not in text:
            raise SystemExit(
                "engram_cpu_hash: fast-stage stage() anchor missing "
                "(apply engram_stage_fast first)"
            )
        text = text.replace(OLD_STAGE, NEW_STAGE, 1)
        if STAGE_DEF_ANCHOR not in text:
            raise SystemExit("engram_cpu_hash: stage def anchor missing")
        text = text.replace(
            STAGE_DEF_ANCHOR, STAGER_METHODS + "\n\n" + STAGE_DEF_ANCHOR, 1
        )
        engram.write_text(text)
        print("dsv41: engram cpu-hash stager installed")

    rtext = runner.read_text()
    if MARKER not in rtext:
        v3_block = "# --- engram-prefetch-v3 ---\n" + V3_HOOK_BODY
        if v3_block in rtext:
            rtext = rtext.replace(
                v3_block, v3_block + "\n" + ENQUEUE_HOOK, 1
            )
        elif RUNNER_ANCHOR in rtext:
            rtext = rtext.replace(
                RUNNER_ANCHOR, RUNNER_ANCHOR + "\n" + ENQUEUE_HOOK, 1
            )
        else:
            raise SystemExit("engram_cpu_hash: runner enqueue anchor missing")
        if COMMIT_MARKER not in rtext:
            if COMMIT_ANCHOR not in rtext:
                raise SystemExit("engram_cpu_hash: commit anchor missing")
            rtext = rtext.replace(
                COMMIT_ANCHOR, COMMIT_HOOK + "\n" + COMMIT_ANCHOR, 1
            )
        runner.write_text(rtext)
        print("dsv41: engram cpu-hash runner hooks installed")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model_root", type=Path, help=".../models/deepseek_v4_1")
    p.add_argument("runner", type=Path, help=".../v1/worker/gpu/model_runner.py")
    args = p.parse_args()
    apply(args.model_root, args.runner)


if __name__ == "__main__":
    main()
