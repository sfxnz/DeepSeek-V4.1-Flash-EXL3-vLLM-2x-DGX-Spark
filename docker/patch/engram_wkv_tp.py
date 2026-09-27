"""Engram wkv column-parallel across the TP ranks (DSV41_ENGRAM_WKV_TP=1, default off).

Each Engram layer (layers 1 and 14) projects its gathered n-gram rows [T, 6144]
with wkv, a ReplicatedLinear [25600, 6144] MXFP8: every TP rank reads the whole
157 MB weight per layer (r3 profile: 758 us per call, 2 calls a step, 1.57 ms at
214 GB/s). With the lever the Engram module gets a ColumnParallelLinear
(gather_output=True) instead: each rank loads (the stock column-parallel loader
narrows the checkpoint tensors) and reads its 12800 output rows, and one
all-gather puts the [T, 25600] kv on every rank. The production MXFP8 GEMM
(mm_mxfp8, backend auto -> b12x on sm_121) computes each output column the same
way whatever N is, so the gathered kv is bit-identical to the replicated one
(kernel_study/fusion_host/engram_wkv_shard_check.py: layers 1 and 14, M 1..2048,
128 cases, 0 bits differ). The GEMM goes 802 -> 443 us cold at M=4; the gather
of [4, 12800] bf16 per rank is ~20-30 us (the Engram embed gather of 24 KiB takes
18.4 us in the r3 trace). Each rank also holds half the weight (-79 MB a layer).

It applies only with TP > 1, no sequence parallelism (every rank must hold every
token), no Engram DP and an output width that splits into whole 128-row tiles
per rank; otherwise the stock ReplicatedLinear stays and one line says why.

Safety: right after the sharded layer's weights are processed (vLLM's
process_weights_after_loading, before the profile run; every rank processes the
modules in the same order) the layer checks itself. Each rank all-gathers its
weight and swizzled-scale shards, which is the stock replicated weight byte for
byte (the F8_128x4 scale swizzle is 128-row-tile major and a shard is a whole
number of tiles), runs probe rows at M 1, 4, 8 and 64 through the sharded GEMM
plus the all-gather and through the stock kernel on the gathered full weight,
and compares raw bf16 bits. The collectives run in the same order on every rank
whatever happens locally (a local failure contributes zeros and a failed
verdict), and the verdict is all-reduced, so the ranks cannot diverge. Pass ->
LOG_ENGAGED and the full copy is freed. Fail -> LOG_DISARMED and the layer keeps
the gathered full weight and runs the stock replicated GEMM (no gather) from
then on.

Top-level imports are stdlib only. decode_levers.install() calls install() when
the env is on.
"""

from __future__ import annotations

import os

# Boot-log markers for tools/engagement_audit.py.
LOG_ENGAGED = "dsv41: engram wkv column-parallel self-check bit-exact"
LOG_DISARMED = "dsv41: engram wkv column-parallel DISABLED ->"

TILE_ROWS = 128  # F8_128x4 scale swizzle tile (mxfp8_utils.swizzle_mxfp8_scale)
PROBE_M = (1, 4, 8, 64)
PROBE_SEED = 20260925
_STATE = {"sharded": 0, "engaged": 0, "disarmed": 0}


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return (env.get("DSV41_ENGRAM_WKV_TP", "0") or "0") == "1"


def refuse_reason(tp_size: int, sequence_parallel: bool, dp_size: int, out_features: int) -> str | None:
    """Why wkv must stay replicated (None: it can be column-parallel)."""
    if tp_size < 2:
        return f"TP={tp_size}"
    if sequence_parallel:
        return "sequence parallelism (a rank holds only its own tokens)"
    if dp_size != 1:
        return f"Engram DP={dp_size}"
    if out_features % (tp_size * TILE_ROWS):
        return f"{out_features} rows do not split into {tp_size} shards of whole {TILE_ROWS}-row tiles"
    return None


def probe_inputs(torch, k: int, device):
    """The same bf16 rows on every rank (fixed-seed fp32 CPU draws, whatever the default dtype)."""
    g = torch.Generator(device="cpu").manual_seed(PROBE_SEED)
    return [
        (torch.randn(m, k, generator=g, dtype=torch.float32) * 2.0).to(torch.bfloat16).to(device)
        for m in PROBE_M
    ]


def contributions(torch, layer, xs):
    """This rank's side of every all-gather, in order: weight and scale bytes, then
    the sharded GEMM output per probe. A local failure gives zeros of the right
    shape (the gathers must still run) and an error string."""
    err = None
    out = [layer.weight.data.contiguous().view(torch.uint8), layer.weight_scale.data.contiguous().view(torch.uint8).view(-1)]
    for x in xs:
        try:
            if err is not None:
                raise RuntimeError("skipped after an earlier failure")
            y = layer.quant_method.apply(layer, x)
        except Exception as exc:  # noqa: BLE001 - reported through the verdict
            err = err or repr(exc)
            y = torch.zeros((x.shape[0], layer.output_size_per_partition), dtype=torch.bfloat16, device=x.device)
        out.append(y.contiguous())
    return out, err


def verdict(torch, full_layer, xs, gathered_outputs, err):
    """(ok, why): the gathered sharded outputs vs the stock GEMM on the full weight."""
    if err is not None:
        return False, err
    try:
        for x, got in zip(xs, gathered_outputs):
            ref = full_layer.forward(x)
            if got.shape != ref.shape or not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
                return False, f"sharded + all-gather != replicated GEMM (M={x.shape[0]})"
    except Exception as exc:  # noqa: BLE001
        return False, repr(exc)
    return True, ""


def make_full_layer_class(torch):
    class ReplicatedWkv(torch.nn.Module):
        """The stock replicated GEMM on a full weight: the sharded layer's own
        quant method (the stock kernel's apply_weights) on the gathered weight and
        swizzled scale. No quant_method attribute on purpose: vLLM's post-load pass
        must not process these tensors a second time."""

        def __init__(self, sharded, weight, weight_scale):
            super().__init__()
            self._method = sharded.quant_method
            self.input_size = sharded.input_size
            self.output_size = sharded.output_size
            self.prefix = sharded.prefix
            self.weight = torch.nn.Parameter(weight, requires_grad=False)
            self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)

        def forward(self, x):
            return self._method.apply(self, x)

    return ReplicatedWkv


def self_check(torch, engram, layer, all_gather, all_reduce_sum, full_layer_cls) -> bool:
    """Rank-symmetric check of one sharded wkv (see the module docstring). True: engaged."""
    xs = probe_inputs(torch, layer.input_size, layer.weight.device)
    local, err = contributions(torch, layer, xs)
    w_full = all_gather(local[0], 0).view(layer.weight.dtype)
    s_full = all_gather(local[1], 0).view(layer.weight_scale.dtype)
    gathered = [all_gather(y, -1) for y in local[2:]]
    full = full_layer_cls(layer, w_full, s_full)
    ok, why = verdict(torch, full, xs, gathered, err)
    bad = torch.tensor([0 if ok else 1], dtype=torch.int32, device=layer.weight.device)
    ranks_bad = int(all_reduce_sum(bad).item())
    if ranks_bad == 0:
        _STATE["engaged"] += 1
        print(
            "dsv41: engram wkv column-parallel self-check bit-exact (%s: %d of %d rows per rank, M %s)"
            % (layer.prefix, layer.output_size_per_partition, layer.output_size, "/".join(map(str, PROBE_M))),
            flush=True,
        )
        return True
    engram.wkv = full
    _STATE["disarmed"] += 1
    print(
        "dsv41: engram wkv column-parallel DISABLED -> stock replicated GEMM on the gathered weight "
        "(%s; %d rank(s) failed; here: %s)" % (layer.prefix, ranks_bad, why or "ok"),
        flush=True,
    )
    return False


def install() -> str:
    """Wrap Engram.__init__: a column-parallel wkv that checks itself after loading."""
    import torch
    from vllm.distributed import (
        get_tensor_model_parallel_world_size,
        tensor_model_parallel_all_gather,
        tensor_model_parallel_all_reduce,
    )
    from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear
    from vllm.models.deepseek_v4_1.common import engram as eng

    cls = eng.Engram
    if getattr(cls.__init__, "_dsv41_wkv_tp", False):
        return "already installed"
    orig_init = cls.__init__
    full_layer_cls = make_full_layer_class(torch)

    def arm_check(engram, layer):
        method = layer.quant_method
        orig_process = method.process_weights_after_loading

        def process_weights_after_loading(lyr):
            orig_process(lyr)
            if lyr is layer and not getattr(layer, "_dsv41_wkv_checked", False):
                layer._dsv41_wkv_checked = True
                self_check(
                    torch, engram, layer, tensor_model_parallel_all_gather,
                    tensor_model_parallel_all_reduce, full_layer_cls,
                )

        method.process_weights_after_loading = process_weights_after_loading

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        old = self.wkv
        why = None
        if type(old) is not ReplicatedLinear or old.bias is not None or old.return_bias:
            why = f"wkv is {type(old).__name__} (bias {old.bias is not None}, return_bias {old.return_bias})"
        why = why or refuse_reason(
            get_tensor_model_parallel_world_size(),
            bool(self.use_sequence_parallel),
            int(self.embed_tokens.dp_size),
            int(old.output_size),
        )
        if why is not None:
            print(f"dsv41: engram wkv stays replicated ({old.prefix}): {why}", flush=True)
            return
        new = ColumnParallelLinear(
            old.input_size,
            old.output_size,
            bias=False,
            gather_output=True,
            quant_config=old.quant_config,
            prefix=old.prefix,
            return_bias=False,
        )
        self.wkv = new  # the replicated parameters are freed here
        del old
        arm_check(self, new)
        _STATE["sharded"] += 1
        print(
            "dsv41: engram wkv column-parallel armed (%s: %d of %d rows per rank, gathered)"
            % (new.prefix, new.output_size_per_partition, new.output_size),
            flush=True,
        )

    __init__._dsv41_wkv_tp = True
    cls.__init__ = __init__
    return "Engram.__init__ wrapped (wkv column-parallel, checked after loading)"
