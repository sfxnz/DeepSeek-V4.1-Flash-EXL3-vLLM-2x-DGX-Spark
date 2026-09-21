#!/usr/bin/env python3
"""Engram stage defer — off-thread next-step gather, event-wait at replay.

Round-22/23 attribution: after gather v2 the residual GPU idle is still
anchored at `engram.stage` on the prepare_inputs critical path (post-v2
smaller — v2 recovered ~1.3 ms/step of host execution — but stage
remains the single blocking owner, plus ~2.4 ms sub-1-ms fragments).
Round 18 named the fix (option 2): move the ENTIRE next-step gather off
the critical path. It became safe when the side-stream ordering bugs
were fixed (35e05fd / Round-19/20 lesson: wait_stream(main) must be
issued while the MAIN stream is current; record an event; main waits it
at replay time).

DSV41_ENGRAM_DEFER=1 (default off). Chain: prestage -> census ->
fast-stage -> prefetch v3 (fadvise warm) -> cpu-hash text -> gather v2
(fast host gather) -> THIS (overlap). The chain composes: the defer
worker uses the v2 batched preadv gather for its table reads and the
cpu-hash numpy mirror for hashing.

Design (ONE persistent worker thread per rank):

  * The post-propose enqueue (same site as the cpu-hash hook) snapshots
    this step's sampler/draft outputs to pinned mirrors on a side stream
    ordered behind ALL main-stream work queued so far (wait issued while
    the MAIN stream is current), records `snap_ev`, submits the worker,
    and returns — zero blocking on the main thread.
  * The worker (off-thread; preadv/memcpy/numpy release the GIL) waits
    snap_ev, reconstructs the next step's chunk (the prefetch-v3 rule —
    bit-exact, proven Rounds 20/21 on live data, pf_hit 100%), CPU-hashes
    it with the cpu-hash mirror, gathers+dequants every table via the v2
    path into a DOUBLE-BUFFERED per-table pinned staging area (its own
    buffers — the critical-path hash_host/rows_host are never touched),
    waits `h2d_ev` (bounds pinned-slot reuse), issues the H2D on the
    side stream (ordered behind main via wait_stream BEFORE entering the
    with-block — the 35e05fd lesson), records `h2d_ev`, publishes a
    ready record {gen, slot, n, req/qsl signature}.
  * stage(): after the stock GPU hash+sync, if the ready record's
    generation+batch signature matches this step's batch, the main
    thread joins the (already-finished) future with a 2 s bound, waits
    `h2d_ev` (signaled in the common case), and issues ONE DtoD per
    table (defer_dev -> staged_rows, non_blocking, MAIN stream). Zero
    gather work on the critical path. Any mismatch -> the exact current
    sync v2 path runs for this step (identical behavior, never slower)
    and the worker re-syncs at the next enqueue.

Warmup: the first warm_n predicted steps ALSO run the sync gather and
bit-compare the predicted pinned slot against it (int16 bitview — fp8
NaN patterns make float equality lie). Any mismatch self-disarms.

Fallback contract: ANY anomaly (worker error, signature mismatch, late
join, warm-verify mismatch) self-disarms to the synchronous v2 path
with ONE warning line; NEVER crashes; NEVER blocks longer than sync
(the only join is bounded at 2 s and is reached only when the worker
already published ready; a wedged worker at worst makes every later
step a silent miss = today's behavior).

Apply AFTER engram_gather_v2 (requires the cpu-hash stage text, which
the live sitecustomize chain always installs). Idempotent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "# --- engram-defer ---"

# ---------------------------------------------------------------------------
# engram.py: stager init — arm defer right after the stock init tail.
# ---------------------------------------------------------------------------

STAGER_INIT_TAIL = (
    "        self.hashes_ready = torch.cuda.Event()\n"
    "        self.num_staged = 0\n"
)

STAGER_INIT_NEW = (
    "        self.hashes_ready = torch.cuda.Event()\n"
    "        self.num_staged = 0\n"
    + MARKER
    + "\n        self._defer_setup()\n"
)

# ---------------------------------------------------------------------------
# engram.py: stager methods (inserted before the stage def anchor).
# ---------------------------------------------------------------------------

STAGE_DEF_ANCHOR = "    @torch.inference_mode()\n    def stage("

STAGER_METHODS = MARKER + '''
    def _defer_setup(self) -> None:
        # ONE persistent worker + double-buffered pinned staging. The
        # critical-path buffers (hash_host / rows_host) are untouched.
        import os as _os

        self.defer_on = _os.environ.get("DSV41_ENGRAM_DEFER", "0") == "1"
        if not self.defer_on:
            return
        try:
            import numpy as _np
            import torch as _torch
            from concurrent.futures import ThreadPoolExecutor as _TPE

            from .mm_preprocess import IMAGE_PAD_ID, IMAGE_SENTINEL_BASE_ID

            hs = self.hash_state
            if getattr(hs, "use_slot_cache", False):
                raise RuntimeError("slot-cache hash state not mirrorable")
            self._df_tm = _np.asarray(
                hs.token_map.detach().to("cpu").to(_torch.int64).tolist(),
                dtype=_np.int64,
            )
            self._df_pad = int(hs.pad_id)
            self._df_dead = _np.asarray(
                [IMAGE_SENTINEL_BASE_ID, IMAGE_PAD_ID], dtype=_np.int64
            )
            self._df_ngram = int(hs.multipliers.shape[1])
            self._df_heads = int(hs.primes.shape[-1])
            self._df_nlayers = int(hs.multipliers.shape[0])
            self._df_depth = int(hs.lookback_depth)
            self._df_mult = _np.asarray(
                hs.multipliers.detach().to("cpu").to(_torch.int64).tolist(),
                dtype=_np.int64,
            )
            self._df_primes = _np.asarray(
                hs.primes.detach().to("cpu").to(_torch.int64).reshape(-1).tolist(),
                dtype=_np.int64,
            ).reshape(self._df_nlayers, (self._df_ngram - 1) * self._df_heads)
            self._df_offs = _np.asarray(
                hs.offsets.detach().to("cpu").to(_torch.int64).reshape(-1).tolist(),
                dtype=_np.int64,
            ).reshape(self._df_nlayers, (self._df_ngram - 1) * self._df_heads)

            self._df_rmax = 8
            self._df_kmax = 8
            self._df_omax = 64
            mt = int(self.max_tokens)
            self._df_cap = max(128, mt)
            r, c = int(self.local_heads), int(self.dim)
            # Per-table pinned rows per slot (the v2 gather returns one
            # host tensor per table; never share one buffer across tables).
            self._df_pin = [
                [
                    _torch.zeros((mt, r, c), dtype=_torch.bfloat16,
                                 pin_memory=True)
                    for _ in self.engrams
                ]
                for _ in range(2)
            ]
            self._df_dev = [
                _torch.zeros_like(e._staged_rows_for_ubatch())
                for e in self.engrams
            ]
            self._df_pin_ids = _torch.zeros(
                self._df_cap, dtype=_torch.int64, pin_memory=True
            )
            self._df_pin_pos = _torch.zeros(
                self._df_cap, dtype=_torch.int64, pin_memory=True
            )
            self._df_pin_qsl = _torch.zeros(
                self._df_rmax + 1, dtype=_torch.int64, pin_memory=True
            )
            self._df_pin_win = _torch.zeros(
                self._df_rmax * self._df_depth,
                dtype=_torch.int64,
                pin_memory=True,
            )
            self._df_pin_out = _torch.zeros(
                self._df_rmax * self._df_omax,
                dtype=_torch.int64,
                pin_memory=True,
            )
            self._df_pin_draft = _torch.zeros(
                self._df_kmax * self._df_rmax,
                dtype=_torch.int64,
                pin_memory=True,
            )
            self._df_pin_ns = _torch.zeros(
                self._df_rmax, dtype=_torch.int64, pin_memory=True
            )
            self._df_snap_ev = _torch.cuda.Event()
            self._df_snap_ev.record()
            self._df_h2d_ev = _torch.cuda.Event()
            self._df_h2d_ev.record()
            self._df_stream = _torch.cuda.Stream()
            self._df_pool = _TPE(max_workers=1)
            self._df_gen = 0
            self._df_slot = 0  # slot the NEXT worker writes
            self._df_ready = None  # newest completed ready record
            self._df_pending = None  # {"gen", "slot", "fut"}
            self._df_warm = 0
            self._df_warm_n = 4
            self._df_v_slot = None
            # engagement census: [hits, misses, late, pairs, worker_ms]
            self._df_stats = [0, 0, 0, 0, 0.0]
            self._df_att = 0
            self._df_miss_logs = 0
            self._df_seen_gen = 0
            self._df_census = _os.environ.get("DSV41_ENGRAM_CENSUS", "0") == "1"
            self._df_every = int(
                _os.environ.get("DSV41_ENGRAM_CENSUS_EVERY", "32")
            )
            print(
                "dsv41: engram defer armed (tables=%d rmax=%d cap=%d "
                "pin=%.1f MiB/rank)"
                % (
                    len(self.engrams),
                    self._df_rmax,
                    self._df_cap,
                    2.0 * len(self.engrams) * mt * r * c * 2 / 1048576,
                ),
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            self.defer_on = False
            print(f"dsv41: engram defer disabled at setup: {exc!r}", flush=True)

    def _defer_disarm(self, why: str) -> None:
        # ONE warning line, permanent self-disarm, never a crash. The
        # pending worker (if any) only touches its own slot, so letting
        # it finish into the void is safe without joining.
        if self.defer_on:
            print(
                f"dsv41: engram defer DISABLED -> sync v2 path: {why}",
                flush=True,
            )
        self.defer_on = False

    def enqueue_defer(
        self,
        input_ids,
        positions,
        query_start_loc,
        window,
        sampled_token_ids,
        num_sampled,
        draft_tokens,
        num_reqs: int,
        num_tokens: int,
        req_ids=None,
        has_prefill: bool = True,
    ) -> None:
        # Post-propose (same site as the cpu-hash hook). Snapshot on a
        # side stream ordered behind ALL main-stream work queued so far;
        # the wait_stream is issued while the MAIN stream is current
        # (the Round-19/20 lesson). Record snap_ev; the WORKER waits it.
        if not getattr(self, "defer_on", False):
            return
        try:
            import torch as _torch

            if (
                has_prefill
                or draft_tokens is None
                or draft_tokens.numel() == 0
                or num_reqs <= 0
                or num_reqs > self._df_rmax
                or num_tokens <= 0
                or num_tokens > self._df_cap
                or req_ids is None
            ):
                return
            k = min(int(draft_tokens.shape[1]), self._df_kmax)
            om = min(
                int(sampled_token_ids.shape[1])
                if sampled_token_ids is not None and sampled_token_ids.dim() == 2
                else self._df_omax,
                self._df_omax,
            )
            # Boot-2 evidence: an adaptive-verification micro-step calls
            # this hook with a malformed snapshot (sum(qsl[1:]) !=
            # num_tokens). Enqueueing it would churn _df_gen, make the
            # GOOD worker abandon (gen check) and make stage() reject the
            # good ready as superseded -> 0% hits. Skip malformed shapes.
            qsl_l = [
                int(x)
                for x in query_start_loc[: num_reqs + 1].tolist()
            ]
            if (
                len(qsl_l) != num_reqs + 1
                or qsl_l[0] != 0
                or sum(qsl_l[1:]) != num_tokens
                or any(
                    qsl_l[i + 1] <= qsl_l[i] for i in range(num_reqs)
                )
            ):
                return
            self._df_stream.wait_stream(_torch.cuda.current_stream())
            with _torch.cuda.stream(self._df_stream):
                self._df_pin_ids[:num_tokens].copy_(
                    input_ids[:num_tokens].to(_torch.int64), non_blocking=True
                )
                self._df_pin_pos[:num_tokens].copy_(
                    positions[:num_tokens].to(_torch.int64), non_blocking=True
                )
                self._df_pin_qsl[: num_reqs + 1].copy_(
                    _torch.tensor(
                        qsl_l, dtype=_torch.int64, pin_memory=True
                    ),
                    non_blocking=True,
                )
                if window is not None and window.numel():
                    win = window[:num_reqs, :].reshape(-1).to(_torch.int64)
                    m = min(win.numel(), self._df_pin_win.numel())
                    self._df_pin_win[:m].copy_(win[:m], non_blocking=True)
                outs = sampled_token_ids[:num_reqs, :om].reshape(-1).to(
                    _torch.int64
                )
                self._df_pin_out[: outs.numel()].copy_(outs, non_blocking=True)
                dr = draft_tokens[:num_reqs, :k].reshape(-1).to(_torch.int64)
                self._df_pin_draft[: dr.numel()].copy_(dr, non_blocking=True)
                self._df_pin_ns[:num_reqs].copy_(
                    num_sampled[:num_reqs].to(_torch.int64), non_blocking=True
                )
                self._df_snap_ev.record()
            self._df_req_snap = [str(x) for x in req_ids[:num_reqs]]
            self._df_om_snap = om
            self._df_gen += 1
            gen = self._df_gen
            slot = self._df_slot
            self._df_slot ^= 1
            self._df_pending = {
                "gen": gen,
                "slot": slot,
                "fut": self._df_pool.submit(
                    self._defer_worker, gen, num_reqs, num_tokens, k, slot
                ),
            }
        except Exception as exc:  # noqa: BLE001
            # Enqueue-side surprise: no prediction for the next step; do
            # NOT disarm (stage falls back); bounded log.
            self._df_pending = None
            n = getattr(self, "_df_enq_errs", 0)
            self._df_enq_errs = n + 1
            if n < 3:
                print(f"dsv41: engram defer enqueue skipped: {exc!r}", flush=True)

    def _defer_predict(self, snap):
        # Prefetch-v3 rule (Round 20/21 proven bit-exact): next chunk_r =
        # [s_{A-1}] + draft_r at S_r+A_r..; window_r[j] = seq[S+A-1-j].
        depth = self._df_depth
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
        if n <= 0 or n > self._df_cap or n > int(self.max_tokens):
            return None
        return {
            "n": n,
            "ids": ids,
            "pos": pos,
            "qsl": qsl,
            "wins": wins,
            "req_ids": reqs,
        }

    def _defer_hash_layers(self, ids, pos, qsl, wins):
        # engram_cpu_hash's vectorized numpy mirror of
        # _hash_ids_kernel (bit-exact; proven on live data Round 20).
        import numpy as _np

        tm = self._df_tm
        pad = self._df_pad
        ngram = self._df_ngram
        heads = self._df_heads
        L = self._df_nlayers
        h0 = self.head_start
        ncol = self.head_end - self.head_start
        ids = _np.asarray(ids, dtype=_np.int64)
        pos = _np.asarray(pos, dtype=_np.int64)
        qsl = _np.asarray(qsl, dtype=_np.int64)
        win = _np.asarray(wins, dtype=_np.int64)
        n = len(ids)
        toks = _np.arange(n, dtype=_np.int64)
        req = _np.minimum(
            _np.searchsorted(qsl[1:], toks, side="right"),
            len(qsl) - 2,
        )
        chunk_start = pos[
            _np.clip(qsl[_np.clip(req, 0, len(qsl) - 1)], 0, n - 1)
        ]
        out = _np.zeros((n, L, (ngram - 1) * heads), dtype=_np.int64)
        rolling = _np.zeros((n, L), dtype=_np.int64)
        blocked = _np.zeros(n, dtype=bool)
        nv = len(tm)
        for shift in range(ngram):
            lookback = pos - shift
            in_batch = lookback >= chunk_start
            bi = _np.clip(toks - shift, 0, n - 1)
            b_tok = ids[bi]
            b_src = _np.where(
                (b_tok >= 0) & (b_tok < nv),
                tm[_np.clip(b_tok, 0, nv - 1)],
                pad,
            )
            b_src = _np.where(
                _np.isin(b_tok, self._df_dead), _np.int64(-1), b_src
            )
            col = chunk_start - 1 - lookback
            in_win = (~in_batch) & (col >= 0) & (col < self._df_depth)
            colc = _np.clip(col, 0, self._df_depth - 1)
            w_tok = win[_np.clip(req, 0, win.shape[0] - 1), colc]
            known = in_win & (w_tok >= 0)
            w_src = _np.where(
                (w_tok >= 0) & (w_tok < nv),
                tm[_np.clip(w_tok, 0, nv - 1)],
                pad,
            )
            w_src = _np.where(
                _np.isin(w_tok, self._df_dead), _np.int64(-1), w_src
            )
            source = _np.where(in_batch, b_src, _np.where(known, w_src, pad))
            blocked = blocked | (lookback < 0) | (source == -1)
            value = _np.where(blocked, pad, source)
            rolling = rolling ^ (
                value[:, None] * self._df_mult[:, shift][None, :]
            )
            if shift > 0:
                base = (shift - 1) * heads
                p = self._df_primes[:, base : base + heads]
                o = self._df_offs[:, base : base + heads]
                out[:, :, base : base + heads] = (
                    rolling[:, :, None] % p[None, :, :] + o[None, :, :]
                )
        return out[:, :, h0 : h0 + ncol]

    def _defer_worker(self, gen, num_reqs, num_tokens, k, slot):
        # OFF the main thread. All waits live here — the main thread only
        # ever reads published state or joins a done future. preadv /
        # numpy release the GIL; the only torch ops are cheap pinned
        # copy_ calls at the end.
        import time as _time

        t0 = _time.perf_counter()
        try:
            import torch as _torch

            self._df_snap_ev.synchronize()
            if gen != self._df_gen:
                return  # superseded before the snapshot landed
            om = self._df_om_snap
            snap = {
                "ids": self._df_pin_ids[:num_tokens].tolist(),
                "pos": self._df_pin_pos[:num_tokens].tolist(),
                "qsl": self._df_pin_qsl[: num_reqs + 1].tolist(),
                "win": self._df_pin_win[: num_reqs * self._df_depth]
                .reshape(num_reqs, self._df_depth)
                .tolist(),
                "outs": self._df_pin_out[: num_reqs * om]
                .reshape(num_reqs, om)
                .tolist(),
                "drafts": self._df_pin_draft[: num_reqs * k]
                .reshape(num_reqs, k)
                .tolist(),
                "ns": self._df_pin_ns[:num_reqs].tolist(),
                "req_ids": list(self._df_req_snap),
            }
            pred = self._defer_predict(snap)
            if pred is None:
                self._df_none_n = getattr(self, "_df_none_n", 0) + 1
                if self._df_none_n <= 6:
                    print(
                        "[defer-miss] worker predict None (ns=%s qsl=%s "
                        "num_reqs=%d num_tokens=%d)"
                        % (snap["ns"], snap["qsl"], num_reqs, num_tokens),
                        flush=True,
                    )
                return
            n = pred["n"]
            hashed = self._defer_hash_layers(
                pred["ids"], pred["pos"], pred["qsl"], pred["wins"]
            )
            # Gather + dequant per table with the v2 path into this
            # slot's per-table pinned buffers. Layer index =
            # engram.layer_hash_index (engrams sorted by it; be exact).
            pins = self._df_pin[slot]
            for engram, pin in zip(self.engrams, pins):
                disk = engram.embed_tokens.disk
                local = _torch.from_numpy(
                    hashed[:, engram.layer_hash_index, :]
                )
                file_rows, owned = engram.embed_tokens.disk_file_rows_owned(
                    local
                )
                if hasattr(disk, "_gather_dequant_v2"):
                    rows = disk._gather_dequant_v2(file_rows, owned)
                else:
                    rows = disk.gather_dequant(file_rows, owned)
                pin[:n].copy_(rows.view(n, self.local_heads, self.dim))
            # Bound pinned-slot reuse: the PREVIOUS worker's H2D (same
            # slot two generations back) must have completed GPU-side.
            self._df_h2d_ev.synchronize()
            # H2D on the side stream, ordered behind ALL main-stream work
            # queued so far — this also orders the _df_dev overwrite
            # after any main-stream DtoD that reads it. wait BEFORE
            # entering the with-block (35e05fd lesson).
            self._df_stream.wait_stream(_torch.cuda.current_stream())
            with _torch.cuda.stream(self._df_stream):
                for dbuf, pin in zip(self._df_dev, pins):
                    dbuf[:n].copy_(pin[:n], non_blocking=True)
                self._df_h2d_ev.record()
            self._df_ready = {
                "gen": gen,
                "slot": slot,
                "n": n,
                "req_ids": [str(x) for x in pred["req_ids"]],
                "qsl": [int(x) for x in pred["qsl"]],
                "ms": 1000.0 * (_time.perf_counter() - t0),
            }
        except Exception as exc:  # noqa: BLE001
            self._defer_disarm(f"worker error: {exc!r}")

    def _defer_sig(self, input_batch):
        # Host-only signature of the batch stage() is about to serve
        # (same fields the cpu-hash commit signature uses).
        try:
            nq = int(input_batch.num_reqs)
            return (
                int(input_batch.num_tokens),
                list(input_batch.req_ids[:nq]),
                input_batch.query_start_loc_np[: nq + 1].tolist(),
            )
        except Exception:  # noqa: BLE001
            return None

    def _defer_miss(self, why, detail=""):
        # Bounded miss logging: first 6 reasons verbatim, then counted.
        st = self._df_stats
        st[1] += 1
        n = getattr(self, "_df_miss_logs", 0)
        self._df_miss_logs = n + 1
        if n < 6:
            print(
                "[defer-miss] %s%s" % (why, (" " + detail) if detail else ""),
                flush=True,
            )

    def _defer_try_stage(self, n, input_batch):
        # Main thread, called from stage() after the stock hash+sync.
        # True -> consume the predicted rows (DtoD only); "warm" -> also
        # run the sync gather and bit-verify; False -> sync v2 path.
        st = self._df_stats
        try:
            ready = self._df_ready
            if ready is None:
                self._defer_miss("no-ready (gen=%d armed=%s)"
                                 % (self._df_gen, self.defer_on))
                self._defer_census_tick()
                return False
            if ready["gen"] <= getattr(self, "_df_seen_gen", 0):
                self._defer_miss("already-consumed",
                                 "ready_gen=%d seen=%d"
                                 % (ready["gen"],
                                    getattr(self, "_df_seen_gen", 0)))
                self._defer_census_tick()
                return False
            if ready["gen"] != self._df_gen:
                self._defer_miss("superseded", "ready_gen=%d cur=%d"
                                 % (ready["gen"], self._df_gen))
                self._defer_census_tick()
                return False
            pend = self._df_pending
            if pend is None or pend["gen"] != ready["gen"]:
                self._defer_miss("pend-mismatch")
                self._defer_census_tick()
                return False
            sig = self._defer_sig(input_batch)
            if sig is None:
                self._defer_miss("sig-none (input_batch=%r)"
                                 % (type(input_batch).__name__,))
                self._defer_census_tick()
                return False
            if (
                sig[0] != ready["n"]
                or [str(x) for x in sig[1]] != ready["req_ids"]
                or [int(x) for x in sig[2]] != ready["qsl"]
            ):
                self._defer_miss(
                    "sig-mismatch",
                    "n=%d/%d reqs=%s/%s qsl=%s/%s"
                    % (sig[0], ready["n"],
                       sig[1], ready["req_ids"],
                       [int(x) for x in sig[2]], ready["qsl"]),
                )
                self._defer_census_tick()
                return False
            # Bound the join: `ready` is published as the worker's LAST
            # statement, so the future is done in the common case; the
            # bound only guards a worker wedged between the H2D enqueue
            # and the publish.
            try:
                pend["fut"].result(timeout=2.0)
            except TimeoutError:
                st[2] += 1  # late -> sync path, never block longer
                return False
            except Exception as exc:  # noqa: BLE001
                self._defer_disarm(f"join error: {exc!r}")
                return False
            self._df_seen_gen = ready["gen"]
            if self._df_warm < self._df_warm_n:
                self._df_warm += 1
                st[0] += 1
                self._defer_census_tick()
                return "warm"
            st[0] += 1
            st[4] += ready.get("ms", 0.0)
            self._defer_census_tick()
            return True
        except Exception as exc:  # noqa: BLE001
            self._defer_disarm(f"try_stage error: {exc!r}")
            return False

    def _defer_census_tick(self):
        # Print on a fixed ATTEMPT cadence (not only on hits) so an
        # all-miss serve is observable.
        st = self._df_stats
        self._df_att = getattr(self, "_df_att", 0) + 1
        if self._df_census and self._df_att % self._df_every == 0:
            tot = st[0] + st[1]
            print(
                "[defer-census] hits=%d misses=%d late=%d pairs=%d "
                "hit%%=%.1f worker_ms/pred=%.2f"
                % (
                    st[0], st[1], st[2], st[3],
                    100.0 * st[0] / tot if tot else 0.0,
                    st[4] / max(st[3], 1),
                ),
                flush=True,
            )
            st[0] = st[1] = st[2] = st[3] = 0
            st[4] = 0.0

    def _defer_check_warm(self, n):
        # Called by stage() AFTER the sync gather filled rows_host on a
        # warm step: the predicted slot must be bit-identical to the
        # sync rows (int16 bitview — fp8-dequanted bf16 can hold NaN
        # patterns where float equality lies).
        try:
            import torch as _torch

            slot = self._df_v_slot
            self._df_v_slot = None
            if slot is None:
                return
            for li, buf in enumerate(self.rows_host):
                a = self._df_pin[slot][li][:n]
                if not _torch.equal(
                    a.view(_torch.int16), buf[:n].view(_torch.int16)
                ):
                    self._defer_disarm(
                        "warm verify mismatch (table %d slot != sync rows)"
                        % li
                    )
                    return
            if self._df_warm == self._df_warm_n:
                print(
                    "dsv41: engram defer ACTIVE (warm-verified bit-exact "
                    "x%d; next-step gather off critical path)"
                    % self._df_warm,
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            self._defer_disarm(f"verify error: {exc!r}")

    def _defer_apply(self, n):
        # Main thread, MAIN stream: wait the worker's H2D event (already
        # signaled in the common case — the join guaranteed the H2D was
        # enqueued; the wait is bounded because the GPU is idle here and
        # the copies are KiB-scale) then ONE DtoD per table into the
        # persistent staged_rows the graph reads.
        self._df_h2d_ev.synchronize()
        for engram, dbuf in zip(self.engrams, self._df_dev):
            dest = engram._staged_rows_for_ubatch()[:n]
            dest.copy_(dbuf[:n], non_blocking=True)
        st = self._df_stats
        st[3] += 1
'''

# ---------------------------------------------------------------------------
# engram.py: stage() rewire (cpu-hash chain — the live sitecustomize text).
# ---------------------------------------------------------------------------

CH_STAGE_OLD = """        _ch_fast = self._ch_try_stage(
            n, ids, positions[:n], query_start_loc, lookback_token_ids
        )
        if not _ch_fast:
"""

CH_STAGE_NEW = """        _ch_fast = self._ch_try_stage(
            n, ids, positions[:n], query_start_loc, lookback_token_ids
        )
        # --- engram-defer ---
        _defer_hit = False
        if (
            not _ch_fast
            and getattr(self, "defer_on", False)
            and input_batch is not None
        ):
            _defer_hit = self._defer_try_stage(n, input_batch)
            if _defer_hit == "warm":
                # verification step: run the sync gather too, compare in
                # the epilogue, serve the SYNC rows this step.
                self._df_v_slot = self._df_ready["slot"]
                _defer_hit = False
            elif _defer_hit is True:
                _ch_fast = True  # skip the sync gather entirely
        if not _ch_fast:
"""

EPI_OLD = """        for engram, buf in zip(self.engrams, self.rows_host):
            staged = buf[:n]
            dest = engram._staged_rows_for_ubatch()[:n]
            dest.copy_(staged, non_blocking=True)
        if _ch_fast:
            self._ch_note_h2d()
        self.num_staged = n
        return n
"""

EPI_NEW = """        # --- engram-defer ---
        if _defer_hit is True:
            self._defer_apply(n)
            self.num_staged = n
            return n
        self._defer_check_warm(n)  # no-op unless a warm verify is open
        for engram, buf in zip(self.engrams, self.rows_host):
            staged = buf[:n]
            dest = engram._staged_rows_for_ubatch()[:n]
            dest.copy_(staged, non_blocking=True)
        if _ch_fast:
            self._ch_note_h2d()
        self.num_staged = n
        return n
"""

# stage() signature: add the input_batch parameter.
SIG_OLD = """        lookback_token_ids: torch.Tensor,
        num_tokens: int,
    ) -> int:
"""

SIG_NEW = """        lookback_token_ids: torch.Tensor,
        num_tokens: int,
        input_batch=None,
    ) -> int:
"""

# ---------------------------------------------------------------------------
# model_state.py: pass input_batch into the stage() call.
# ---------------------------------------------------------------------------

MS_CALL = "self.engram_stager.stage("


def _wire_model_state(mtext: str) -> str:
    i = mtext.index(MS_CALL)
    j = mtext.index(")", i)
    k = j - 1
    while mtext[k] in " \n\t":
        k -= 1
    tail = "" if mtext[k] == "," else ","
    return (
        mtext[: k + 1]
        + tail
        + "\n            input_batch=input_batch,\n        "
        + mtext[j:]
    )


# ---------------------------------------------------------------------------
# Runner hook (model_runner.py): enqueue_defer after the cpu-hash hook
# (live chain), else after the v3 hook tail, else at the fresh anchor.
# ---------------------------------------------------------------------------

RUNNER_CH_HOOK_END = """                    input_batch.req_ids,
                    input_batch.has_prefill,
                )
"""

RUNNER_V3_HOOK_TAIL = """                    input_batch.num_reqs,
                    input_batch.num_tokens,
                )
"""

RUNNER_DEFER_HOOK = (
    MARKER
    + """
        if self.speculator is not None:
            _stager = getattr(self.model_state, "engram_stager", None)
            if _stager is not None:
                _stager.enqueue_defer(
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
"""
)

RUNNER_ANCHOR = """        if self.num_speculative_steps > 0:
            # Spec-decode and diffusion LLMs both use draft tokens but the latter does
            # not have a speculator (i.e. self.speculator is None)
            self.draft_tokens_handler.set_draft_tokens(
                input_batch,
                self.req_states.draft_tokens[input_batch.idx_mapping],
            )
"""


def apply(model_root: Path, runner: Path, model_state: Path) -> None:
    engram = model_root / "common" / "engram.py"
    text = engram.read_text()
    if MARKER not in text:
        if STAGER_INIT_TAIL not in text:
            raise SystemExit("engram_defer: stager init anchor missing")
        text = text.replace(STAGER_INIT_TAIL, STAGER_INIT_NEW, 1)
        if STAGE_DEF_ANCHOR not in text:
            raise SystemExit("engram_defer: stage def anchor missing")
        text = text.replace(
            STAGE_DEF_ANCHOR, STAGER_METHODS + "\n\n" + STAGE_DEF_ANCHOR, 1
        )
        if "_ch_fast = self._ch_try_stage(" not in text:
            raise SystemExit(
                "engram_defer: cpu-hash stage text missing (apply the "
                "sitecustomize chain through engram_cpu_hash first)"
            )
        if SIG_OLD not in text:
            raise SystemExit("engram_defer: stage signature anchor missing")
        text = text.replace(SIG_OLD, SIG_NEW, 1)
        if CH_STAGE_OLD not in text:
            raise SystemExit(
                "engram_defer: cpu_hash stage prologue anchor missing"
            )
        text = text.replace(CH_STAGE_OLD, CH_STAGE_NEW, 1)
        if EPI_OLD not in text:
            raise SystemExit(
                "engram_defer: cpu_hash stage epilogue anchor missing"
            )
        text = text.replace(EPI_OLD, EPI_NEW, 1)
        engram.write_text(text)
        print("dsv41: engram defer stager installed")

    rtext = runner.read_text()
    if MARKER not in rtext:
        if RUNNER_CH_HOOK_END in rtext:
            rtext = rtext.replace(
                RUNNER_CH_HOOK_END, RUNNER_CH_HOOK_END + RUNNER_DEFER_HOOK, 1
            )
        elif RUNNER_V3_HOOK_TAIL in rtext:
            rtext = rtext.replace(
                RUNNER_V3_HOOK_TAIL, RUNNER_V3_HOOK_TAIL + RUNNER_DEFER_HOOK, 1
            )
        elif RUNNER_ANCHOR in rtext:
            rtext = rtext.replace(
                RUNNER_ANCHOR, RUNNER_ANCHOR + RUNNER_DEFER_HOOK, 1
            )
        else:
            raise SystemExit("engram_defer: runner anchor missing")
        runner.write_text(rtext)
        print("dsv41: engram defer runner hook installed")

    mtext = model_state.read_text()
    if MARKER not in mtext and MS_CALL in mtext:
        model_state.write_text(MARKER + "\n" + _wire_model_state(mtext))
        print("dsv41: engram defer model_state stage(input_batch=...) wired")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model_root", type=Path, help=".../models/deepseek_v4_1")
    p.add_argument("runner", type=Path, help=".../v1/worker/gpu/model_runner.py")
    p.add_argument("model_state", type=Path, help=".../nvidia/model_state.py")
    args = p.parse_args()
    apply(args.model_root, args.runner, args.model_state)


if __name__ == "__main__":
    main()
