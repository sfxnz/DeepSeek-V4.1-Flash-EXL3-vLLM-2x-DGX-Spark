"""Drop EXL3 shard page cache after weight load (GB10 UMA memory hygiene).

The mmap weight load leaves the shard pages in the host page cache. On GB10
the GPU driver allocates from MemFree and does not reclaim clean page cache,
so a later CUDA allocation can fail with NVRM out-of-memory while MemFree
~1 GiB but MemAvailable ~9 GiB (mia-exl3-ref boot 11, 2026-09-12; her
.env.example:225-229). posix_fadvise(POSIX_FADV_DONTNEED) on every shard
right after load_model returns those clean pages to the OS.

Source (READ ONLY reference): mia-exl3-ref overlay/patch_memory_log.py:73-100
(_drop_page_cache), hooked before compile_or_warm_up_model (lines 134-138)
and after load_model (lines 300-303).

Env guard: DSV41_DROP_PAGE_CACHE=1 enables (default off = no-op, our chain's
opt-in convention; mia's default was on with "0" to skip). Never raises.
"""

from __future__ import annotations

MARK = "dsv41-drop-page-cache"

# Boot-log markers for tools/engagement_audit.py (run.sh post-ready audit).
LOG_ENGAGED = "dropped page cache of"
LOG_DISARMED = "drop failed: "


def _mem_free_gib() -> float:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemFree:"):
                    return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return -1.0


def _drop_page_cache(worker) -> None:
    import glob
    import os

    paths: list[str] = []
    try:
        paths.append(str(worker.vllm_config.model_config.model))
    except Exception:
        pass
    before = _mem_free_gib()
    n = 0
    for d in paths:
        for f in glob.glob(os.path.join(d, "*.safetensors")):
            try:
                fd = os.open(f, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
                n += 1
            except OSError as exc:
                print(f"[{MARK}] fadvise {f}: {exc!r}", flush=True)
    print(
        f"[{MARK}] dropped page cache of {n} shard files: "
        f"MemFree {before:.2f}GiB -> {_mem_free_gib():.2f}GiB",
        flush=True,
    )


def install() -> bool:
    """Wrap Worker.load_model + compile_or_warm_up_model. Idempotent."""
    import functools
    import os

    if os.environ.get("DSV41_DROP_PAGE_CACHE", "0") != "1":
        return False

    from vllm.v1.worker.gpu_worker import Worker

    if getattr(Worker, "_dsv41_drop_page_cache", False):
        return True

    def _wrap(name: str, drop_before: bool) -> None:
        orig = getattr(Worker, name, None)
        if orig is None or getattr(orig, "_dsv41_dpc_wrapped", False):
            return

        @functools.wraps(orig)
        def wrapped(self, *args, **kwargs):
            if drop_before:
                try:
                    _drop_page_cache(self)
                except Exception as exc:  # never break the boot
                    print(f"[{MARK}] drop failed: {exc!r}", flush=True)
            try:
                return orig(self, *args, **kwargs)
            finally:
                if not drop_before:
                    try:
                        _drop_page_cache(self)
                    except Exception as exc:  # never break the load
                        print(f"[{MARK}] drop failed: {exc!r}", flush=True)

        wrapped._dsv41_dpc_wrapped = True
        setattr(Worker, name, wrapped)

    # Before graph capture / warm-up is where the big KV + NCCL allocs land,
    # and after load_model for the eager path. Idempotent by marker flags.
    _wrap("compile_or_warm_up_model", drop_before=True)
    _wrap("load_model", drop_before=False)
    Worker._dsv41_drop_page_cache = True
    print(f"[{MARK}] armed (fadvise DONTNEED on shards after weight load)", flush=True)
    return True
