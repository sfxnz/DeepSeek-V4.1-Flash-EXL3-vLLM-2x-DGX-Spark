"""NCCL graph mixing off, made safe by an eager-only twin communicator.

DSV41_NCCL_EAGER_TWIN=1 (default 0, run.sh FORWARD_ENVS). sitecustomize calls install().

Why. With NCCL's default NCCL_GRAPH_MIXING_SUPPORT=1 every collective captured into a
CUDA graph carries external serialEvent wait/record nodes on NCCL's device and host
strong streams (nccl 2.30.7 src/misc/strongstream.cc ncclStrongStreamAcquire/Release,
mixing == graphUsageMode 2), and an eager launch on a communicator that owns graph
plans can be queued as a host task behind the graph's host nodes (src/enqueue.cc
ncclLaunchPrepare, persistentRefs). The k3 profile measured, two nodes, exact decode
sizes, 2x-L2 flush before each call, profiler off: critical-path cost of an in-graph
all-reduce at m=4 57.6 us with mixing on vs 22.5 us off (the bandwidth-bound neighbour
runs 13.5% slower with mixing on), modeled -3.08 ms/step at c=1
(results/2026-09-25-kernels/profile/nccl-sweep.json, nccl-graph-diag.txt).

Why not the env var alone. With mixing off NCCL does not support a non-captured
collective launched during an outstanding graph launch that uses the same
communicator, regardless of stream ordering (NCCL env docs). The serve launches the
eager logits all-gather ~45 ms and the eager embed all-reduce ~32 ms before the
target graph ends, in every step.

What this does.
- NCCL_GRAPH_MIXING_SUPPORT=0 is written into this process's environment before any
  NCCL communicator exists. NCCL maps it to graphUsageMode 0 in ncclCommInitRank
  (src/init.cc); the only behaviour keyed on the mode is mixing == (mode == 2).
- CudaCommunicator.__init__ is wrapped. For the TP group ('tp:*') a second
  PyNcclCommunicator, the eager twin, is built on the same CPU group right after the
  stock one, and self.pynccl_comm becomes a GraphEagerRouter: a call whose stream is
  capturing goes to the stock (graph) communicator, every other call goes to the twin.
  So the graph communicator only ever runs captured collectives, and the twin never
  runs captured ones. Both ranks build and route identically (same program order).
- Any other group with a PyNccl communicator (here 'ep:0', unused by the TP MoE path)
  gets a router without a twin: eager calls pass through, a captured call raises, so
  an unexpected captured use fails the boot at capture instead of mixing.
NCCL's other unsupported case, parallel graph launches on different streams without
dependencies, does not arise: vLLM replays the target and draft graphs on one stream.

Numerics: unchanged. The same NCCL algorithm, protocol and channel choice run on the
same buffers; only which communicator object issues each call changes.

Not applied (self-disarm, mixing stays on): the lever is off, the vLLM classes are
missing, or their shape differs from what the router covers (see check_anchors).
NCCL_GRAPH_MIXING_SUPPORT=0 set without this lever is reset to 1.
Top-level imports are stdlib only.
"""

from __future__ import annotations

import inspect
import os

LOG_ENGAGED = "dsv41: nccl eager twin engaged"
LOG_DISARMED = ("dsv41: nccl eager twin DISARMED", "dsv41: nccl eager twin REFUSED")

ENV = "DSV41_NCCL_EAGER_TWIN"
MIXING_ENV = "NCCL_GRAPH_MIXING_SUPPORT"

# PyNcclCommunicator methods that enqueue NCCL work on a stream: routed per call.
ROUTED = (
    "all_reduce",
    "all_gather",
    "all_gatherv",
    "reduce_scatter",
    "reduce_scatterv",
    "reduce",
    "scatter",
    "send",
    "recv",
    "broadcast",
    "batch_isend_irecv",
)
# Lifecycle calls that must reach both communicators.
FAN_OUT = ("destroy", "suspend", "resume")
# ncclGroupStart/End take no communicator: forwarding them to either comm is the same.
PASS = ("group_start", "group_end")
# Symmetric-window registration binds a buffer to one communicator; vLLM only uses it
# with NCCL symmetric memory, which is off on this platform. Refused rather than guessed.
REFUSED = ("register_comm_window", "register_comm_window_raw", "deregister_comm_window")
# Public PyNcclCommunicator names that need no routing (classmethod constructor).
IGNORED = ("from_unique_id_bytes",)
# Source anchors in CudaCommunicator.__init__ (canonical-e13).
INIT_ANCHORS = (
    "self.pynccl_comm = PyNcclCommunicator(",
    "group=self.cpu_group if tcp_store_group is None else tcp_store_group",
)
INIT_PARAMS = ("cpu_group", "device", "unique_name", "tcp_store_group")


def lever_on(env) -> bool:
    return (env.get(ENV, "0") or "0") == "1"


def group_kind(unique_name: str) -> str:
    """'tp' for vLLM's TP group ('tp:0'), else the name's prefix."""
    return (unique_name or "").split(":", 1)[0]


def stream_positions(pynccl_cls) -> dict[str, int]:
    """Positional index (self excluded) of each routed method's 'stream' parameter."""
    out = {}
    for name in ROUTED:
        params = list(inspect.signature(getattr(pynccl_cls, name)).parameters)
        if "stream" not in params:
            raise TypeError(f"PyNcclCommunicator.{name} has no stream parameter")
        out[name] = params.index("stream") - 1
    return out


def check_anchors(comm_cls, pynccl_cls) -> None:
    """Raise if CudaCommunicator/PyNcclCommunicator differ from what the router covers."""
    init = comm_cls.__init__
    src = inspect.getsource(init)
    for anchor in INIT_ANCHORS:
        if anchor not in src:
            raise RuntimeError(f"CudaCommunicator.__init__ anchor missing: {anchor!r}")
    params = inspect.signature(init).parameters
    for name in INIT_PARAMS:
        if name not in params:
            raise RuntimeError(f"CudaCommunicator.__init__ has no {name!r} parameter")
    public = {
        n
        for klass in pynccl_cls.__mro__[:-1]
        for n, v in vars(klass).items()
        if not n.startswith("_") and (callable(v) or isinstance(v, (classmethod, staticmethod)))
    }
    known = set(ROUTED) | set(FAN_OUT) | set(PASS) | set(REFUSED) | set(IGNORED)
    missing = sorted((set(ROUTED) | set(FAN_OUT) | set(PASS)) - public)
    unknown = sorted(public - known)
    if missing or unknown:
        raise RuntimeError(f"PyNcclCommunicator API changed: missing {missing}, unrouted {unknown}")
    stream_positions(pynccl_cls)


class GraphEagerRouter:
    """Stands in for CudaCommunicator.pynccl_comm.

    graph: the stock communicator, used only for calls whose stream is capturing.
    eager: the twin, used for every other call; None means this group has no twin and
    a captured call raises.
    capturing(stream) -> bool decides per call, on the stream PyNccl will launch on.
    Attribute reads (disabled, world_size, rank, device, nccl, ...) come from graph.
    """

    def __init__(self, graph, eager, capturing, name: str, positions: dict[str, int], log=print):
        d = self.__dict__
        d["_graph"], d["_eager"], d["_capturing"] = graph, eager, capturing
        d["_name"], d["_positions"], d["_log"] = name, positions, log
        d["_n_graph"], d["_n_eager"], d["_reported"] = 0, 0, False

    def _pick(self, method: str, args, kwargs):
        stream = kwargs.get("stream")
        pos = self._positions[method]
        if stream is None and len(args) > pos:
            stream = args[pos]
        d = self.__dict__
        if self._capturing(stream):
            if self._eager is None:
                raise RuntimeError(
                    f"dsv41: nccl eager twin REFUSED: {method} on group {self._name} was captured into a "
                    f"CUDA graph, but {MIXING_ENV}=0 is only safe for communicators with an "
                    f"eager twin (tp). Set {ENV}=0 or add a twin for this group."
                )
            d["_n_graph"] += 1
            return self._graph
        d["_n_eager"] += 1
        if self._eager is None:
            return self._graph
        if not self._reported and self._n_graph:
            d["_reported"] = True
            self._log(
                f"dsv41: nccl eager twin routing on {self._name}: {self._n_graph} captured "
                f"calls on the graph comm, eager calls on the twin ({self._n_eager} so far)",
            )
        return self._eager

    def __getattr__(self, name):
        if name in REFUSED:
            raise RuntimeError(f"dsv41: nccl eager twin REFUSED: {name} is not routed by {ENV}")
        return getattr(self._graph, name)

    def __setattr__(self, name, value):
        raise AttributeError(f"GraphEagerRouter is read-only ({name})")


def _routed(method: str):
    def call(self, *args, **kwargs):
        return getattr(self._pick(method, args, kwargs), method)(*args, **kwargs)

    call.__name__ = method
    return call


def _fan_out(method: str):
    def call(self, *args, **kwargs):
        out = getattr(self._graph, method)(*args, **kwargs)
        if self._eager is not None:
            getattr(self._eager, method)(*args, **kwargs)
        return out

    call.__name__ = method
    return call


def _passed(method: str):
    def call(self, *args, **kwargs):
        return getattr(self._graph, method)(*args, **kwargs)

    call.__name__ = method
    return call


for _m in ROUTED:
    setattr(GraphEagerRouter, _m, _routed(_m))
for _m in FAN_OUT:
    setattr(GraphEagerRouter, _m, _fan_out(_m))
for _m in PASS:
    setattr(GraphEagerRouter, _m, _passed(_m))
del _m


def stream_is_capturing(stream=None) -> bool:
    """Whether the stream PyNccl will launch on (explicit, else vLLM's current) is capturing."""
    import torch
    from vllm.utils.torch_utils import current_stream

    s = current_stream() if stream is None else stream
    with torch.cuda.stream(s):
        return bool(torch.cuda.is_current_stream_capturing())


def wrap_init(orig_init, pynccl_cls, capturing, log=print):
    """CudaCommunicator.__init__ that installs the router after the stock init."""
    sig = inspect.signature(orig_init)
    positions = stream_positions(pynccl_cls)

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        comm = getattr(self, "pynccl_comm", None)
        if comm is None or getattr(comm, "disabled", True) or isinstance(comm, GraphEagerRouter):
            return
        name = getattr(self, "unique_name", "") or "?"
        if group_kind(name) != "tp":
            self.pynccl_comm = GraphEagerRouter(comm, None, capturing, name, positions, log)
            log(f"dsv41: nccl eager twin guard on {name}: eager-only, a captured call raises")
            return
        tcp = sig.bind(self, *args, **kwargs).arguments.get("tcp_store_group")
        twin = pynccl_cls(group=self.cpu_group if tcp is None else tcp, device=self.device)
        if getattr(twin, "disabled", True):
            raise RuntimeError(f"dsv41: nccl eager twin DISARMED: eager twin for {name} came up disabled")
        self.pynccl_comm = GraphEagerRouter(comm, twin, capturing, name, positions, log)
        log(
            f"dsv41: nccl eager twin engaged on {name}: captured collectives -> stock comm, eager -> twin; "
            f"{MIXING_ENV}={os.environ.get(MIXING_ENV)}"
        )

    __init__._dsv41_eager_twin = True
    __init__.__wrapped__ = orig_init
    return __init__


def install(env=None, log=print) -> str:
    """'off' | 'armed' | 'disarmed'. Sets NCCL_GRAPH_MIXING_SUPPORT=0 only when armed."""
    env = os.environ if env is None else env

    def say(msg):
        log(msg, flush=True) if log is print else log(msg)

    if not lever_on(env):
        if env.get(MIXING_ENV) == "0":
            env[MIXING_ENV] = "1"
            say(f"dsv41: nccl eager twin REFUSED: {MIXING_ENV}=0 without {ENV}=1 is unsafe here; reset to 1")
        return "off"
    try:
        from vllm.distributed.device_communicators import cuda_communicator as cc
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        init = cc.CudaCommunicator.__init__
        if not getattr(init, "_dsv41_eager_twin", False):
            check_anchors(cc.CudaCommunicator, PyNcclCommunicator)
            cc.CudaCommunicator.__init__ = wrap_init(init, PyNcclCommunicator, stream_is_capturing, say)
    except Exception as exc:  # noqa: BLE001 - disarm, never half-apply
        say(f"dsv41: nccl eager twin DISARMED: {exc!r}; NCCL graph mixing stays on")
        return "disarmed"
    env[MIXING_ENV] = "0"
    return "armed"
