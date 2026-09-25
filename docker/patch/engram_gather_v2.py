#!/usr/bin/env python3
"""Engram disk gather v2 — preadv run batching (Round 22 attribution fix).

Round-22 trace on the engaged 30.10 serve: residual GPU idle ~9.9 ms/step,
of which ~7.5 ms/step is HOST EXECUTION of the stage gather loop —
51 rows/call x (per-row preadv syscall + chunk-pool submit/result +
per-call alloc/dequant chain), called from prepare_inputs
(engram.stage <- model_state.py:85). Data is page-cached (census read_w
0.09-0.12 ms, pf_hit 100%); the cost is loop/syscall/pool-dispatch count.

Probe on spark1 (hot cache, real geometry w=256B s=8B rows, R=51):
  stock _read_rows (pool, 8 futures)      0.636 ms/call
  pool submit+result overhead alone       0.105 ms/call
  inline per-row pread, NO pool (102 sys) 0.053 ms/call
The chunk-pool dispatch dominates, not the syscalls.

This patch adds an env-gated path inside DiskEngramTable.gather_dequant
(DSV41_ENGRAM_GATHER_V2=1, default off):
  (a) ONE os.preadv per contiguous run of file-offset-adjacent rows in the
      same table file (scatter iovecs land each row directly in its output
      slot; duplicate row ids are preadv'd once then copied in memory;
      partial preadv resumes row-aligned). Hash rows are ~uniform over a
      384M-row space so runs are typically length 1 — preadv(1 iov) is then
      exactly one pread syscall, same count, minus the pool hop.
  (b) the stock dequant chain VERBATIM (torch fp8_e4m3->f32 *
      ue8m0-as-f32 exponent, bf16 round at the end) — bit-exact by
      construction; validated offline tensor-for-tensor anyway.
  (c) H2D stays ONE pinned-buffer copy per table per step (the existing
      stage() contract — dest.copy_(staged) on the calling thread; the
      round-11 parallel per-table workers are untouched and v2 runs inside
      them).

Safety: first v2 call ALSO runs the stock path and torch.equal's the two
outputs (self-check, one line); ANY error disarms v2 permanently with one
warning line and falls through to the stock body — the serve never crashes.
Adds a [gv2-census] line (when DSV41_ENGRAM_CENSUS=1) reporting
runs/call + preads/call so engagement (pread count drop) is observable,
plus fadv/call and pf_hit (prefetch-v3 predicted rows consumed).

Prefill (2026-09-24): the serial preadv loop is queue depth 1, so a cold
prefill call (5k-98k rows) waits ~350 us per NVMe row. With
DSV41_ENGRAM_WILLNEED=1, calls with at least DSV41_ENGRAM_WILLNEED_MIN_ROWS
rows (default 512; decode is 48-96) first issue one posix_fadvise(WILLNEED)
per merged 4 KiB page span in both files, so the reads are in flight
together. Advisory only: the bytes read do not change. Default off: only a
CPU read bench backs it; it flips on after the E0/E1 serve ABAB passes.
DSV41_ENGRAM_GATHER_V2_MAX_ROWS=N (unset/0 = off) is the fallback arm:
calls with more than N rows take the stock 32-thread pool instead.

Apply AFTER engram_stage_census (reads the chained text). Idempotent. A
file that carries the pre-v2.1 text (baked into the canonical-g8 image) has
those three blocks stripped first, then gets this version.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# v2.1 = WILLNEED pre-pass, MAX_ROWS arm, pf_hit and the census lock.
MARKER = "# --- engram-gather-v2.1 ---"
OLD_MARKER = "# --- engram-gather-v2 ---"

# Boot-log markers for tools/engagement_audit.py (run.sh post-ready audit).
LOG_ENGAGED = "dsv41: engram gather v2 self-check bit-exact"
LOG_DISARMED = "dsv41: engram gather v2 DISABLED ->"

GD_DEF = "    def gather_dequant(self, rel, owned):"

DISPATCH_OLD = (
    '        """rel: [R] int64 CPU local row ids; owned: [R] bool. '
    'Returns [R, dim] bf16 CPU."""\n'
    "        torch = _torch()"
)

DISPATCH_NEW = (
    '        """rel: [R] int64 CPU local row ids; owned: [R] bool. '
    'Returns [R, dim] bf16 CPU."""\n'
    + MARKER
    + """
        if _ENG_GATHER_V2[0] and int(rel.numel()) <= _ENG_GV2_MAX_ROWS:
            # Round-22 fix: run-batched preadv + stock dequant math. Any
            # error disarms to the stock path below with ONE warning line.
            try:
                return self._gv2_checked(rel, owned)
            except Exception as _gv2_exc:  # noqa: BLE001
                _ENG_GATHER_V2[0] = False
                print(
                    "dsv41: engram gather v2 DISABLED -> stock path: %r"
                    % (_gv2_exc,),
                    flush=True,
                )
        torch = _torch()"""
)

GV2_MODULE = (
    MARKER
    + """
import os as _gv2_os
import threading as _gv2_threading
import time as _gv2_time

_ENG_GATHER_V2 = [_gv2_os.environ.get("DSV41_ENGRAM_GATHER_V2", "0") == "1"]
_ENG_GV2_SELFCHECK = [True]
# Above this many rows the stock 32-thread pool reads instead. Unset/0 = off.
_ENG_GV2_MAX_ROWS = int(
    _gv2_os.environ.get("DSV41_ENGRAM_GATHER_V2_MAX_ROWS", "0") or 0
) or (1 << 62)
_ENG_GV2_WILLNEED = _gv2_os.environ.get("DSV41_ENGRAM_WILLNEED", "0") == "1"
_ENG_GV2_WILLNEED_MIN = int(
    _gv2_os.environ.get("DSV41_ENGRAM_WILLNEED_MIN_ROWS", "512") or 512
)
_ENG_GV2_CENSUS = _gv2_os.environ.get("DSV41_ENGRAM_CENSUS", "0") == "1"
_ENG_GV2_EVERY = int(_gv2_os.environ.get("DSV41_ENGRAM_CENSUS_EVERY", "32"))
# window counters: calls, rows, runs, preads, read seconds, fadvise calls,
# pf hits, pf predicted rows, pf paired calls. The stage threads (one per
# table) share them: update, test and reset only under _ENG_GV2_LOCK.
_ENG_GV2_SEEN = [0, 0, 0, 0, 0.0, 0, 0, 0, 0]
_ENG_GV2_LOCK = _gv2_threading.Lock()


"""
)

GV2_METHODS = (
    MARKER
    + '''
    def _gv2_read_runs(self, fd: int, base: int, rel: list, row_bytes: int, buf):
        # ONE preadv per contiguous run of file-adjacent rows (scatter
        # iovecs -> each unique row's first output slot). Duplicate row ids:
        # first occurrence preadv'd, then in-memory copies. Partial reads
        # resume row-aligned (every iov is exactly row_bytes). Returns
        # (preads, runs).
        import os as _os

        pos_of = {}
        for i, r in enumerate(rel):
            pl = pos_of.get(r)
            if pl is None:
                pos_of[r] = [i]
            else:
                pl.append(i)
        keys = sorted(pos_of)
        preads = 0
        runs = 0
        i = 0
        nk = len(keys)
        while i < nk:
            j = i
            while j + 1 < nk and keys[j + 1] == keys[j] + 1:
                j += 1
            runs += 1
            iovs = [
                buf[pos_of[k][0] * row_bytes:(pos_of[k][0] + 1) * row_bytes]
                for k in keys[i:j + 1]
            ]
            off = base + keys[i] * row_bytes
            total = (j - i + 1) * row_bytes
            got = 0
            cur = list(iovs)
            while got < total:
                n = _os.preadv(fd, cur, off + got)
                if n <= 0:
                    raise OSError("engram gather v2: short preadv")
                preads += 1
                got += n
                if got < total:
                    # consume n bytes off the front of cur; iovs stop being
                    # row-aligned after the first partial, so track actual
                    # iov lengths, not row_bytes.
                    rem = n
                    while rem > 0 and cur:
                        ln = len(cur[0])
                        if rem >= ln:
                            cur = cur[1:]
                            rem -= ln
                        else:
                            cur = [cur[0][rem:]] + cur[1:]
                            rem = 0
            for k in keys[i:j + 1]:
                ps = pos_of[k]
                if len(ps) > 1:
                    src = bytes(
                        buf[ps[0] * row_bytes:(ps[0] + 1) * row_bytes]
                    )
                    for p in ps[1:]:
                        buf[p * row_bytes:(p + 1) * row_bytes] = src
            i = j + 1
        return preads, runs

    def _gv2_willneed(self, rel: list) -> int:
        # Prefill pre-pass: one WILLNEED per merged 4 KiB page span of the
        # unique rows, w file then s file, before any serial preadv. Small
        # (decode) calls skip it. Returns the number of fadvise calls.
        import os as _os

        if not _ENG_GV2_WILLNEED or len(rel) < _ENG_GV2_WILLNEED_MIN:
            return 0
        keys = sorted(set(rel))
        calls = 0
        for fd, base, row_bytes in (
            (self.w_fd, self.w_off, self.dim),
            (self.s_fd, self.s_off, self.sb),
        ):
            lo = hi = -1
            for k in keys:
                start = (base + k * row_bytes) & -4096
                end = (base + (k + 1) * row_bytes + 4095) & -4096
                if start <= hi:
                    hi = max(hi, end)
                    continue
                if hi > lo:
                    _os.posix_fadvise(fd, lo, hi - lo, _os.POSIX_FADV_WILLNEED)
                    calls += 1
                lo, hi = start, end
            if hi > lo:
                _os.posix_fadvise(fd, lo, hi - lo, _os.POSIX_FADV_WILLNEED)
                calls += 1
        return calls

    def _gather_dequant_v2(self, rel, owned):
        # Same contract as gather_dequant; only the read strategy changes.
        # Dequant chain copied VERBATIM from the stock body.
        torch = _torch()
        r = int(rel.numel())
        w = torch.empty((r, self.dim), dtype=torch.uint8)
        s = torch.empty((r, self.sb), dtype=torch.uint8)
        rel_l = rel.tolist()
        t0 = _gv2_time.perf_counter()
        fadv = self._gv2_willneed(rel_l)
        pw, rw = self._gv2_read_runs(
            self.w_fd, self.w_off, rel_l, self.dim,
            memoryview(w.numpy()).cast("B"),
        )
        ps_, rs_ = self._gv2_read_runs(
            self.s_fd, self.s_off, rel_l, self.sb,
            memoryview(s.numpy()).cast("B"),
        )
        t1 = _gv2_time.perf_counter()
        vals = w.view(torch.float8_e4m3fn).to(torch.float32).view(r, self.sb, -1)
        scale = (s.to(torch.int32) << 23).view(torch.float32)
        out = (vals * scale[:, :, None]).reshape(r, self.dim)
        out[~owned] = 0
        out = out.to(torch.bfloat16)
        self._gv2_census(r, rw + rs_, pw + ps_, t1 - t0, fadv, rel_l, owned)
        return out

    def _gv2_census(self, r, runs, preads, secs, fadv, rel_l, owned):
        # Up to 13 stage threads call this at once. Unlocked, a thread could
        # test st[0] right after another thread reset it and divide by zero,
        # which disarmed gv2 for the process (ZeroDivisionError, 2/12157
        # steps in a CPU stress copy of the old block).
        hit = None
        if _ENG_GV2_CENSUS:
            # pf_hit = share of prefetch-v3 predicted rows this call consumed
            # (owned rows only; unowned slots read file row 0).
            exp = getattr(self, "_pf_expected", None)
            if exp is not None:
                got = {x for x, o in zip(rel_l, owned.tolist()) if o}
                hit = (len(got & exp), len(exp))
                self._pf_expected = None
        line = None
        with _ENG_GV2_LOCK:
            st = _ENG_GV2_SEEN
            st[0] += 1
            st[1] += r
            st[2] += runs
            st[3] += preads
            st[4] += secs
            st[5] += fadv
            if hit is not None:
                st[6] += hit[0]
                st[7] += hit[1]
                st[8] += 1
            if _ENG_GV2_CENSUS and st[0] % _ENG_GV2_EVERY == 0:
                line = (
                    "[gv2-census] calls=%d rows/call=%d runs/call=%.1f "
                    "preads/call=%.1f read=%.3fms fadv/call=%.1f%s"
                    % (
                        st[0],
                        st[1] // st[0],
                        st[2] / st[0],
                        st[3] / st[0],
                        1000.0 * st[4] / st[0],
                        st[5] / st[0],
                        " pf_hit=%.0f%%(%d)" % (100.0 * st[6] / st[7], st[8])
                        if st[7] > 0
                        else "",
                    )
                )
                st[0] = st[1] = st[2] = st[3] = st[5] = st[6] = st[7] = st[8] = 0
                st[4] = 0.0
        if line is not None:
            print(line, flush=True)

    def _gv2_checked(self, rel, owned):
        # First v2 call verifies bit-exactness against the stock path
        # (stock runs with the flag temporarily off, so no recursion).
        torch = _torch()
        out = self._gather_dequant_v2(rel, owned)
        if _ENG_GV2_SELFCHECK[0]:
            _ENG_GV2_SELFCHECK[0] = False
            _ENG_GATHER_V2[0] = False
            ref = self.gather_dequant(rel, owned)
            # bitwise compare: fp8 data can hold NaN patterns (0x7F/0xFF
            # bytes) and torch.equal(NaN, NaN) is False even for identical
            # bits — bit-exactness must be judged on the raw bf16 bits.
            if not torch.equal(
                out.view(torch.int16), ref.view(torch.int16)
            ):
                raise RuntimeError(
                    "engram gather v2 self-check mismatch (stock != v2)"
                )
            _ENG_GATHER_V2[0] = True
            print(
                "dsv41: engram gather v2 self-check bit-exact (r=%d)"
                % int(rel.numel()),
                flush=True,
            )
        return out
'''
)


def _strip_old(text: str) -> str:
    """Remove the module, methods and dispatch blocks an OLD_MARKER apply added."""
    for start, end in (
        (OLD_MARKER + "\nimport os as _gv2_os", "class DiskEngramTable:"),
        (OLD_MARKER + "\n    def _gv2_read_runs", GD_DEF),
        (OLD_MARKER + "\n        if _ENG_GATHER_V2[0]", "        torch = _torch()"),
    ):
        i = text.find(start)
        if i < 0:
            raise SystemExit("engram_gather_v2: baked pre-v2.1 block not recognised")
        text = text[:i] + text[text.index(end, i):]
    if OLD_MARKER in text:
        raise SystemExit("engram_gather_v2: stray pre-v2.1 marker left after strip")
    return text


def apply(vllm_root: Path) -> None:
    disk = vllm_root / "models" / "deepseek_v4_1" / "common" / "engram_disk.py"
    text = disk.read_text()
    if MARKER in text:
        return
    upgraded = OLD_MARKER in text
    if upgraded:
        text = _strip_old(text)
    if DISPATCH_OLD not in text:
        raise SystemExit(
            "engram_gather_v2: gather_dequant docstring anchor missing"
        )
    if GD_DEF not in text:
        raise SystemExit("engram_gather_v2: gather_dequant def anchor missing")
    cls = "class DiskEngramTable:"
    if cls not in text:
        raise SystemExit("engram_gather_v2: class anchor missing")

    text = text.replace(cls, GV2_MODULE + cls, 1)
    text = text.replace(GD_DEF, GV2_METHODS + "\n" + GD_DEF, 1)
    text = text.replace(DISPATCH_OLD, DISPATCH_NEW, 1)
    disk.write_text(text)
    print(
        "dsv41: engram gather v2 installed (DSV41_ENGRAM_GATHER_V2)"
        + (" (replaced baked pre-v2.1 text)" if upgraded else "")
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("vllm_root", type=Path, help=".../vllm site-packages root")
    args = p.parse_args()
    apply(args.vllm_root)


if __name__ == "__main__":
    main()
