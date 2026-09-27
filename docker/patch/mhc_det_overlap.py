"""DSV41_MHC_DET_OVERLAP=1: run the det mHC coefficient half under the next TP all-reduce.

With DSV41_MHC_DET_SPLITS=16 each decode pre is det prenorm GEMM -> det fused norm (layer_input
plus the coefficients post_mix, comb_mix and the next pre-mix). The coefficients feed only the
NEXT post and the next pre's layer_input, and in the model every post follows a TP all-reduce
(attention wo_b, MoE final reduce: NCCL RING_LL over RoCE, ~22.8 us, DRAM nearly idle while it
runs). With DSV41_MHC_DET_OVERLAP=1 (needs DSV41_MHC_DET_SPLITS=16) the pre launches only
layer_input (mhc_det_norm_li) and defers the prenorm GEMM + mhc_det_norm_coef (defer). The next
GroupCoordinator.all_reduce call on the same stream forks a side stream and launches them there
(fork_at_all_reduce); the next post, or any next pre, first joins the side stream (settle). With
no all-reduce in between (TP=1, an all-reduce on another stream) settle launches them in place.
Same kernels and arithmetic as the fused path, so the same bits
(kernel_study/mhc_det/overlap_probe.py, race_check.py, lever_smoke.py --overlap).

Every tensor the deferred kernels touch was allocated on the main stream before the fork and is
held by the pending record until settle has joined the side stream, so the caching allocator
cannot hand that memory to anything else while the side stream uses it (no record_stream).
A pending record never crosses a CUDA-graph capture boundary in this model (every forward ends
with an mHC post: tests/test_mhc_det.py pins it); if one did, settle raises instead of joining
or launching work that belongs to another graph.

Top-level imports are stdlib only.
"""

from __future__ import annotations

import inspect
import os

ENV = "DSV41_MHC_DET_OVERLAP"
LOG_ENGAGED = "dsv41: mhc det overlap engaged"
LOG_DISARMED = "dsv41: mhc det overlap is OFF"
GC_MOD = "vllm.distributed.parallel_state"
CO_MOD = "vllm.distributed.communication_op"
LINEAR_MOD = "vllm.model_executor.layers.linear"
RUNNER_MOD = "vllm.model_executor.layers.fused_moe.runner.moe_runner"
# The two all-reduces that precede the mHC posts, and the path both take to GroupCoordinator.
ANCHORS = (
    (CO_MOD, None, "tensor_model_parallel_all_reduce", "return get_tp_group().all_reduce(input_)"),
    (LINEAR_MOD, "RowParallelLinear", "forward", "output = tensor_model_parallel_all_reduce(output_parallel)"),
    (RUNNER_MOD, "MoERunner", "_maybe_reduce_final_output", "states = tensor_model_parallel_all_reduce(states)"),
)


def on_from_env(env) -> bool:
    raw = (env.get(ENV, "0") or "0").strip()
    if raw not in ("0", "1"):
        raise ValueError(f"{ENV}={raw}: only 0 or 1")
    return raw == "1"


class Pending:
    """Deferred coefficient half of one pre: launch() enqueues it on the current stream."""

    __slots__ = ("launch", "keep", "stream", "capturing", "forked")

    def __init__(self, launch, keep, stream, capturing: bool) -> None:
        self.launch = launch
        self.keep = keep  # every tensor the deferred kernels read or write
        self.stream = stream
        self.capturing = capturing
        self.forked = False


class _State:
    def __init__(self) -> None:
        self.torch = None
        self.armed = False  # all-reduce hook installed
        self.on = False  # engaged: the split path passed its self-test
        self.failed = False
        self.side = None
        self.pending: Pending | None = None
        self.forks = 0
        self.in_place = 0


_S = _State()


def _disarm(msg: str) -> None:
    _S.on = False
    _S.failed = True
    print(f"dsv41: mhc det overlap is OFF: {msg}", flush=True)


def active() -> bool:
    return _S.on


def defer(launch, keep) -> None:
    """Called by the det pre after it launched layer_input: hold its coefficient half."""
    settle()
    torch = _S.torch
    _S.pending = Pending(launch, keep, torch.cuda.current_stream(), torch.cuda.is_current_stream_capturing())


def _fork(p: Pending) -> None:
    torch = _S.torch
    cur = torch.cuda.current_stream()
    if _S.side is None:
        _S.side = torch.cuda.Stream(device=cur.device)
    _S.side.wait_stream(cur)
    with torch.cuda.stream(_S.side):
        p.launch()
    p.forked = True
    _S.forks += 1


def fork_at_all_reduce(input_) -> None:
    """GroupCoordinator.all_reduce hook: launch the pending coefficient half on the side stream
    right before the all-reduce, if it was deferred on this stream in this capture state."""
    p = _S.pending
    if p is None or p.forked or not _S.on or not getattr(input_, "is_cuda", False):
        return
    torch = _S.torch
    if torch.cuda.current_stream() != p.stream or torch.cuda.is_current_stream_capturing() != p.capturing:
        return  # settle launches it in place
    _fork(p)


def settle() -> None:
    """Before anything reads a pending pre's coefficients: join the side stream, or launch the
    deferred kernels in place if no all-reduce forked them."""
    p = _S.pending
    if p is None:
        return
    _S.pending = None
    torch = _S.torch
    if torch.cuda.is_current_stream_capturing() != p.capturing:
        _disarm("a deferred mHC coefficient half crossed a CUDA-graph capture boundary; the fused det pre stays")
        raise RuntimeError("mhc det overlap: deferred work crossed a CUDA-graph capture boundary")
    if p.forked:
        torch.cuda.current_stream().wait_stream(_S.side)
    else:
        p.launch()
        _S.in_place += 1


def check_anchors(mods: dict) -> None:
    for mod, cls, fn, needle in ANCHORS:
        obj = mods[mod]
        if cls is not None:
            obj = getattr(obj, cls)
        src = inspect.getsource(getattr(obj, fn))
        if needle not in src:
            raise RuntimeError(f"{mod}:{cls or ''}.{fn} anchor missing: {needle!r}")
    gc = mods[GC_MOD].GroupCoordinator
    params = list(inspect.signature(gc.all_reduce).parameters)
    if params[:2] != ["self", "input_"]:
        raise RuntimeError(f"GroupCoordinator.all_reduce signature {params}")


def wrap_all_reduce(gc_cls) -> None:
    orig = gc_cls.all_reduce

    def all_reduce(self, input_, *args, **kwargs):
        if _S.pending is not None:
            fork_at_all_reduce(input_)
        return orig(self, input_, *args, **kwargs)

    all_reduce.__wrapped__ = orig
    all_reduce._dsv41_mhc_ovl = True
    gc_cls.all_reduce = all_reduce


def install(env=None) -> str:
    """'off' | 'armed' | 'disarmed'. Engages later, in mhc_det.prepare(), after its self-test."""
    env = os.environ if env is None else env
    try:
        if not on_from_env(env):
            return "off"
    except ValueError as exc:
        _disarm(str(exc))
        return "disarmed"
    if _S.failed:
        return "disarmed"
    try:
        import importlib

        import torch

        mods = {m: importlib.import_module(m) for m in (GC_MOD, CO_MOD, LINEAR_MOD, RUNNER_MOD)}
        gc = mods[GC_MOD].GroupCoordinator
        if not getattr(gc.all_reduce, "_dsv41_mhc_ovl", False):
            check_anchors(mods)
            wrap_all_reduce(gc)
        _S.torch = torch
        _S.armed = True
    except Exception as exc:  # noqa: BLE001 - the lever never breaks the load
        _disarm(f"install failed: {exc!r}; the fused det pre stays")
        return "disarmed"
    print(f"dsv41: mhc det overlap armed ({ENV}=1): the next all-reduce runs the det prenorm GEMM + "
          "sinkhorn of each decode pre on a side stream", flush=True)
    return "armed"


def engage(label: str, detail: str) -> None:
    if _S.armed and not _S.failed:
        _S.on = True
        print(f"dsv41: mhc det overlap engaged: {label}: {detail}", flush=True)
