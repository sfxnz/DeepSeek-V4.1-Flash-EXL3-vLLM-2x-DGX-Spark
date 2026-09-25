"""Warm L2 with the next layer's qkv_a weights while the MoE all-reduce runs.

DSV41_AR_L2_PREFETCH=1 (default 0, run.sh FORWARD_ENVS); DSV41_AR_L2_PREFETCH_MIB
(default 10, which covers all of the next layer's fused_wqa_wkv: 9.02 MiB at TP=2) is
the byte budget per window. sitecustomize calls install().

Why (k3 comm, overlap the AR with independent work). After each MoE all-reduce the
target layer runs AR (~22.7 us, 5 NCCL CTAs) -> mhc_post -> mhc_pre -> act quant ->
qkv_a (fused_wqa_wkv, 9.46 MB MXFP8 at TP=2), and qkv_a streams its weights cold (47.6
us in the r3 serve profile). A side stream forked right before the AR issues a TMA L2
prefetch (cp.async.bulk.prefetch.L2, one CTA; l2pf_kernel.py) of the first budget bytes
of the next layer's fused_wqa_wkv weight and scale.

What it buys is a range (k3 comm fix pass). The AR does not leave DRAM idle: NCCL's LL
all-reduce over the net transport with GDR off polls host-memory buffers and waits on NIC
DMA in the same LPDDR5x, so the prefetch burst delays the late rank's AR, which is the
critical path.
- Optimistic end: the AR replaced by a 5-CTA clock spin that touches no memory
  (l2_prefetch_window.py, ar_l2_prefetch_gpu.py). 5.5 MiB gives -20.2 us per prefetched
  window (qkv_a 47.1 -> 23.1 us), about -0.75 ms/step over 37 windows.
- Pessimistic end: a real NCCL AR between two ranks on one GB10 (MPS, a different
  NCCL_HOSTID per rank -> NET/IB loopback, serve NCCL env), a p2b stand-in before the AR,
  and the lever's own join after the AR (kernel_study/comm/ar_window_nccl.py). The late
  rank's AR slows by +16..+18 us at 5.5 MiB and +23..+27 us with the whole qkv_a, while
  qkv_a saves 20..23 and 30..33 us. Net per window at m=4: -1.9..-4.5 (5.5 MiB) and
  -7.2..-8.7 us (whole qkv_a); at m=8: -0.2..-4.3 and -1.0..-2.0 us. With the whole
  qkv_a that is -0.27..-0.32 ms/step at c=1 and -0.04..-0.07 at c=2. The emulated
  ranks share one GPU, DRAM and NIC (the emulated m=4 AR is 72-83 us against 22.7 in
  the serve).
Placement, issue and budget come from that real-AR data at m=4 and m=8 (two arm orders,
1000 replays each):
- Fork at the AR start. A fork after the AR moves the slowdown onto mHC (+20 us at
  5.5 MiB, +34 us for the whole qkv_a).
- Keep the 1-CTA burst. A paced issue, a closed-loop TMA ring or multi-CTA issue loses
  more on the AR or warms less.
- Prefetch the whole qkv_a: best at m=4 in both orders, a tie at m=8.
Moving the prefetch before p2b does not win. As a separate kernel its run time is
serialized, because p2b fills every SM's register file. Issued by p2b itself it is
zero-sum, because p2b is DRAM-bound where it streams (p2b_prefetch_probe.py).
The serve mechanism boot checks the rest (results/2026-09-25-kernels/comm/
serve_arm_plan.txt). E fails if the late rank's MoE AR slows by at least what qkv_a saves.

Hooks (all in the target model; the draft model's layers never get a plan):
- DeepseekV4Model.forward builds, once per model, a plan for layer i: the next
  layer's fused_wqa_wkv weight and weight_scale, the same fraction of each, total
  <= budget. No plan when the next layer has an Engram (its 162 MB wkv GEMM owns that
  window) or for the last layer. After the forward, a pending prefetch is joined.
- DeepseekV4DecoderLayer.forward joins any pending prefetch on entry (keeps
  breakable-graph segments free of forked streams), arms its plan, runs, disarms. The
  join is right after the AR: the 1-CTA kernel stays resident until the TMA has taken
  its requests (22 us at 3 MiB, 31 at 5.5, 44 for the whole qkv_a), so a prefetch that
  outlasts the AR stalls mhc_post (+1.4..+2.2 us measured with the real AR).
- MoERunner._maybe_reduce_final_output (the call that all-reduces routed + shared)
  forks the side stream and launches the prefetch before running the AR, when a plan
  is armed and the batch has <= 64 tokens (decode).
Numerics: unchanged (cache warm-up only). Self-disarm on anchor drift or when the
Triton prefetch kernel fails its eager trial launch.
Top-level imports are stdlib only.
"""

from __future__ import annotations

import inspect
import os

LOG_ENGAGED = "dsv41: ar l2 prefetch engaged"
LOG_DISARMED = "dsv41: ar l2 prefetch DISARMED"

ENV = "DSV41_AR_L2_PREFETCH"
MIB_ENV = "DSV41_AR_L2_PREFETCH_MIB"
DEFAULT_MIB = 10.0  # >= fused_wqa_wkv's 9.02 MiB at TP=2: all of the next qkv_a
MAX_TOKENS = 64
CHUNK = 16384
LAYER_ANCHORS = ("x = self.attn(positions, x, None)", "x = self.ffn(x, input_ids)")
MODEL_ANCHORS = ("islice(self.layers, self.start_layer, self.end_layer)",)
RUNNER_ANCHORS = ("states = tensor_model_parallel_all_reduce(states)",)


def lever_on(env) -> bool:
    return (env.get(ENV, "0") or "0") == "1"


def budget_bytes(env) -> int:
    mib = float(env.get(MIB_ENV, "") or DEFAULT_MIB)
    if not 0 < mib <= 20:
        raise ValueError(f"{MIB_ENV}={mib} outside (0, 20]")
    return int(mib * 2**20)


def split_budget(sizes: list[int], budget: int) -> list[int]:
    """Bytes to prefetch from each tensor: the same fraction of each, 16 B aligned."""
    total = sum(sizes)
    if total <= 0:
        return [0 for _ in sizes]
    frac = min(1.0, budget / total)
    return [int(n * frac) & ~15 for n in sizes]


def layer_plans(layers, budget: int) -> list:
    """Per layer: [(tensor, nbytes), ...] for the next layer's qkv_a, or None."""
    out = []
    for i, layer in enumerate(layers):
        nxt = layers[i + 1] if i + 1 < len(layers) else None
        attn = getattr(nxt, "attn", None)
        proj = getattr(attn, "fused_wqa_wkv", None)
        if proj is None or getattr(nxt, "engram", None) is not None:
            out.append(None)
            continue
        tensors = [t for t in (getattr(proj, "weight", None), getattr(proj, "weight_scale", None)) if t is not None]
        sizes = [t.numel() * t.element_size() for t in tensors]
        out.append([(t, n) for t, n in zip(tensors, split_budget(sizes, budget)) if n > 0] or None)
    return out


class Prefetcher:
    """Fork/launch/join state for one process (the model runs on one thread)."""

    def __init__(self, launch, torch_mod):
        self.launch = launch  # launch(tensor, nbytes) on the current stream
        self.torch = torch_mod
        self.side = None
        self.pending = False
        self.armed = None
        self.forks = 0

    def join(self):
        if self.pending:
            self.torch.cuda.current_stream().wait_stream(self.side)
            self.pending = False

    def fork(self, plan):
        cur = self.torch.cuda.current_stream()
        if self.side is None:
            self.side = self.torch.cuda.Stream()
        self.side.wait_stream(cur)
        with self.torch.cuda.stream(self.side):
            for t, n in plan:
                self.launch(t, n)
        self.pending = True
        self.forks += 1


def check_anchors(layer_cls, model_cls, runner_cls) -> None:
    for cls, meth, anchors in ((layer_cls, "forward", LAYER_ANCHORS), (model_cls, "forward", MODEL_ANCHORS),
                               (runner_cls, "_maybe_reduce_final_output", RUNNER_ANCHORS)):
        src = inspect.getsource(getattr(cls, meth))
        for a in anchors:
            if a not in src:
                raise RuntimeError(f"{cls.__name__}.{meth} anchor missing: {a!r}")


def wrap(layer_cls, model_cls, runner_cls, pf: Prefetcher, budget: int, log=print) -> None:
    layer_fwd, model_fwd = layer_cls.forward, model_cls.forward
    reduce_final = runner_cls._maybe_reduce_final_output

    def build_plans(model) -> None:
        """First eager forward: plans plus one trial launch (compiles the kernel)."""
        layers = list(model.layers)[model.start_layer:model.end_layer]
        plans = layer_plans(layers, budget)
        first = next((p for p in plans if p), None)
        try:
            if first:
                pf.fork(first[:1])
                pf.join()
                pf.torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001 - the lever turns itself off
            log(f"dsv41: ar l2 prefetch DISARMED: trial launch failed: {exc!r}")
            plans = [None] * len(layers)
        for layer, plan in zip(layers, plans):
            layer._dsv41_l2pf_plan = plan
        model._dsv41_l2pf_plans = plans
        if any(plans):
            mb = max(sum(b for _, b in p) for p in plans if p) / 2**20
            log(f"dsv41: ar l2 prefetch engaged: {sum(p is not None for p in plans)}/{len(plans)} layers, "
                f"{mb:.2f} MiB per MoE all-reduce window (next layer fused_wqa_wkv)")

    def model_forward(self, *args, **kwargs):
        if not hasattr(self, "_dsv41_l2pf_plans") and not pf.torch.cuda.is_current_stream_capturing():
            build_plans(self)
        try:
            return model_fwd(self, *args, **kwargs)
        finally:
            pf.armed = None
            pf.join()

    def layer_forward(self, *args, **kwargs):
        pf.join()
        pf.armed = getattr(self, "_dsv41_l2pf_plan", None)
        try:
            return layer_fwd(self, *args, **kwargs)
        finally:
            pf.armed = None

    def maybe_reduce_final_output(self, states, *args, **kwargs):
        plan, pf.armed = pf.armed, None
        if plan and states.is_cuda and states.dim() >= 1 and states.shape[0] <= MAX_TOKENS:
            pf.fork(plan)
        return reduce_final(self, states, *args, **kwargs)

    for fn, orig in ((model_forward, model_fwd), (layer_forward, layer_fwd),
                     (maybe_reduce_final_output, reduce_final)):
        fn.__wrapped__ = orig
        fn._dsv41_l2pf = True
    model_cls.forward = model_forward
    layer_cls.forward = layer_forward
    runner_cls._maybe_reduce_final_output = maybe_reduce_final_output


def install(env=None, log=print) -> str:
    """'off' | 'armed' | 'disarmed'."""
    env = os.environ if env is None else env

    def say(msg):
        log(msg, flush=True) if log is print else log(msg)

    if not lever_on(env):
        return "off"
    try:
        budget = budget_bytes(env)
        import torch
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
        from vllm.models.deepseek_v4_1.nvidia.model import DeepseekV4DecoderLayer, DeepseekV4Model

        if getattr(DeepseekV4Model.forward, "_dsv41_l2pf", False):
            return "armed"
        check_anchors(DeepseekV4DecoderLayer, DeepseekV4Model, MoERunner)
        import l2pf_kernel

        pf = Prefetcher(l2pf_kernel.launcher(torch), torch)
        wrap(DeepseekV4DecoderLayer, DeepseekV4Model, MoERunner, pf, budget, say)
    except Exception as exc:  # noqa: BLE001 - never half-apply
        say(f"dsv41: ar l2 prefetch DISARMED: {exc!r}")
        return "disarmed"
    return "armed"
