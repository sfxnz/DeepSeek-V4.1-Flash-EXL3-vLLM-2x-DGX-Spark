"""Release cached allocator blocks after long prefill chunks (GB10 UMA hygiene).

Every prefill chunk of a DeepSeek-V4.1 sequence scores its queries against
the whole prefix, so the per-chunk indexer/attention transients grow with
the prefix and PyTorch's caching allocator cannot reuse the previous chunk's
slightly smaller blocks: reserved memory grows with the square of the
prompt. On a Spark that is host memory (her 100k prefill took the head from
4.9 GiB free to the 1.5 GiB guard, 2026-09-12). After any prefill step whose
longest sequence is >= DSV41_PREFILL_EMPTY_CACHE_TOKENS, release the unused
cached blocks — but only while the node's MemAvailable after the step is
below DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB (releasing every chunk cost
~20% prefill throughput at 50k tokens on her box, so the release is
adaptive: keep the blocks while there is headroom).

Source (READ ONLY reference): mia-exl3-ref overlay/patch_memory_log.py:153-266
(_install_prefill_empty_cache wrapping Worker.execute_model); her
.env.example:247-254 documents the two knobs.

Env guard: DSV41_PREFILL_EMPTY_CACHE_TOKENS (default 0 = no-op, our opt-in
convention; her default was 8192). Set both:
  DSV41_PREFILL_EMPTY_CACHE_TOKENS=8192
  DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB=2.5
(DSV41_PREFILL_END_EMPTY_CACHE — one extra release at prefill->decode handover
— measured no decode gain by mia and is intentionally NOT ported.)
Never breaks a step: all hook errors are swallowed after first report.
"""

from __future__ import annotations

MARK = "dsv41-prefill-empty-cache"


def install() -> bool:
    import functools
    import os

    threshold = int(os.environ.get("DSV41_PREFILL_EMPTY_CACHE_TOKENS", "0") or 0)
    if threshold <= 0:
        return False
    memavail_gib = float(os.environ.get("DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB", "2.5") or 0)

    from vllm.v1.worker.gpu_worker import Worker

    orig = getattr(Worker, "execute_model", None)
    if orig is None or getattr(orig, "_dsv41_empty_cache", False):
        return True  # idempotent: another copy is already armed

    state = {"calls": 0, "skipped": 0}

    def _say(msg: str) -> None:
        print(f"[{MARK}] {msg}", flush=True)

    def _memavail_gib() -> float:
        try:
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / (1024 * 1024)
        except OSError:
            pass
        return 0.0

    @functools.wraps(orig)
    def execute_model(self, scheduler_output, *args, **kwargs):
        out = orig(self, scheduler_output, *args, **kwargs)
        try:
            nst = getattr(scheduler_output, "num_scheduled_tokens", None)
            if nst and max(nst.values()) > 64:
                longest = 0
                for req in getattr(scheduler_output, "scheduled_new_reqs", None) or ():
                    longest = max(
                        longest,
                        int(getattr(req, "num_computed_tokens", 0) or 0)
                        + int(nst.get(getattr(req, "req_id", ""), 0)),
                    )
                cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
                if cached is not None:
                    for rid, nc in zip(cached.req_ids, cached.num_computed_tokens):
                        longest = max(longest, int(nc) + int(nst.get(rid, 0)))
                if longest >= threshold:
                    import torch

                    avail = _memavail_gib() if memavail_gib > 0 else 0.0
                    if memavail_gib > 0 and avail >= memavail_gib:
                        state["skipped"] += 1
                        if state["skipped"] in (1, 100, 1000):
                            _say(
                                f"skipped #{state['skipped']} (longest seq {longest}, "
                                f"MemAvailable {avail:.2f} GiB >= {memavail_gib} GiB)"
                            )
                    else:
                        torch.cuda.empty_cache()
                        state["calls"] += 1
                        if state["calls"] in (1, 10, 100):
                            _say(
                                f"empty_cache #{state['calls']} (longest seq {longest}, "
                                f"MemAvailable {avail:.2f} GiB)"
                            )
        except Exception as exc:  # never break a step
            if state["calls"] == 0:
                _say(f"hook error: {exc!r}")
                state["calls"] = -1
        return out

    execute_model._dsv41_empty_cache = True
    Worker.execute_model = execute_model
    _say(
        f"armed (threshold {threshold} tokens, release when MemAvailable < "
        f"{memavail_gib} GiB{' [always]' if memavail_gib <= 0 else ''})"
    )
    return True
