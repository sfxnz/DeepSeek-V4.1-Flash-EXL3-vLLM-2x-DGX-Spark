#!/usr/bin/env python3
"""End-to-end check of the serve wiring (run in the image with the repo's patch dir
mounted like run.sh does and DSV41_DENSE_GEMV=1):

- sitecustomize rewrote vllm's mxfp8/flashinfer.py (hooks present);
- FlashInferCutlassMxfp8LinearKernel.process_weights_after_loading on a layer with
  real rank-0 weights arms the GEMV (load-time self-test passes);
- apply_weights at M = 1..8 (bf16 and QuantizedActivation inputs) goes through the
  GEMV (call counted) and is bit-identical to the stock b12x path (disarmed layer);
- apply_weights captured in a CUDA graph replays bit-identically;
- M > 8 falls through to b12x;
- the lm_head path (prepare_lmhead) with the stock lm_head apply as reference.
"""
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weights  # noqa: E402

assert os.environ.get("DSV41_DENSE_GEMV") == "1", "run with DSV41_DENSE_GEMV=1"
import dense_gemv  # noqa: E402  (/opt/dsv41-patch)
from torch.nn import Parameter  # noqa: E402
from vllm.model_executor.kernels.linear.mxfp8 import flashinfer as fi_mod  # noqa: E402
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation  # noqa: E402
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize  # noqa: E402
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic  # noqa: E402

res = {"hooks_in_vllm_file": "_dsv41_gemv_apply" in open(fi_mod.__file__).read(), "shapes": {}}
assert res["hooks_in_vllm_file"], "sitecustomize did not patch flashinfer.py"
calls = {"n": 0}
_orig_run = dense_gemv._run


def _counting_run(*a, **k):
    out = _orig_run(*a, **k)
    calls["n"] += out is not None  # None = no bucket for this M: maybe_apply falls back to b12x
    return out


dense_gemv._run = _counting_run
Kern = fi_mod.FlashInferCutlassMxfp8LinearKernel
gen = torch.Generator(device="cuda").manual_seed(11)
ok_all = True
for name in ("qkv_a", "wq_b", "wo_b", "shared_gate_up", "shared_down", "engram_wkv", "main_proj"):
    _, N, K, layers = weights.SHAPES[name]
    w, s2d = weights.load(name, layers[-1])
    layer = torch.nn.Module()
    layer.weight = Parameter(w, requires_grad=False)
    layer.weight_scale = Parameter(s2d, requires_grad=False)
    kern = object.__new__(Kern)
    kern.process_weights_after_loading(layer)
    armed = getattr(layer, "_dsv41_gemv", None)
    row = {"armed": armed is not None, "cases": 0, "bitwise": 0, "gemv_calls": 0}
    if armed is None:
        ok_all = False
        res["shapes"][name] = row
        print(f"{name}: NOT ARMED", flush=True)
        continue
    for M in range(1, 10):
        x = torch.randn(M, K, generator=gen, device="cuda").to(torch.bfloat16)
        q, s = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
        qa = QuantizedActivation(data=q, scale=s, orig_dtype=torch.bfloat16, orig_shape=x.shape,
                                 quant_key=kMxfp8Dynamic)
        for inp in (x, qa):
            before = calls["n"]
            y = kern.apply_weights(layer, inp)
            used = calls["n"] > before
            layer._dsv41_gemv = None
            yref = kern.apply_weights(layer, inp)
            layer._dsv41_gemv = armed
            torch.cuda.synchronize()
            eq = torch.equal(y.view(torch.int16), yref.view(torch.int16))
            expect_used = M <= 8 and armed.plan(M) is not None
            row["cases"] += 1
            row["bitwise"] += eq
            row["gemv_calls"] += used
            if not eq or used != expect_used:
                ok_all = False
                print(f"{name} M={M} {type(inp).__name__}: equal={eq} used={used} expected_used={expect_used}")
    # CUDA graph capture of the hooked apply_weights (M = 4)
    x = torch.randn(4, K, generator=gen, device="cuda").to(torch.bfloat16)
    eager = kern.apply_weights(layer, x)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        yg = kern.apply_weights(layer, x)
    x.copy_(torch.randn(4, K, generator=gen, device="cuda").to(torch.bfloat16))
    g.replay()
    ye = kern.apply_weights(layer, x)
    torch.cuda.synchronize()
    row["graph_replay_eq_eager"] = bool(torch.equal(yg.view(torch.int16), ye.view(torch.int16)))
    ok_all &= row["graph_replay_eq_eager"]
    res["shapes"][name] = row
    print(f"{name}: {row}", flush=True)
    del g, layer, w, s2d
    torch.cuda.empty_cache()

# lm_head: stock lm_head apply (quant + flashinfer.mm_mxfp8 auto) as the reference
import flashinfer  # noqa: E402
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import swizzle_mxfp8_scale  # noqa: E402

w, s2d = weights.load("lm_head", 0)
N, K = w.shape
lm = torch.nn.Module()
lm.weight = Parameter(w, requires_grad=False)
lm.weight_scale = Parameter(swizzle_mxfp8_scale(s2d, M=N, K=K).contiguous(), requires_grad=False)


class _StockHead:
    def apply(self, layer, x, bias=None):
        out = dense_gemv.maybe_apply(layer, x, bias)
        if out is not None:
            return out
        q, s = mxfp8_e4m3_quantize(x.reshape(-1, K), is_sf_swizzled_layout=True)
        return flashinfer.mm_mxfp8(q, layer.weight.t(), s, layer.weight_scale, out_dtype=x.dtype,
                                   backend="auto").view(*x.shape[:-1], N)


head = _StockHead()
dense_gemv.prepare_lmhead(head, lm, s2d)
armed = getattr(lm, "_dsv41_gemv", None)
row = {"armed": armed is not None, "cases": 0, "bitwise": 0}
if armed is not None:
    for M in (1, 3, 4, 6, 8):
        x = torch.randn(M, K, generator=gen, device="cuda").to(torch.bfloat16)
        y = head.apply(lm, x)
        lm._dsv41_gemv = None
        yref = head.apply(lm, x)
        lm._dsv41_gemv = armed
        torch.cuda.synchronize()
        row["cases"] += 1
        row["bitwise"] += bool(torch.equal(y.view(torch.int16), yref.view(torch.int16)))
ok_all &= row["armed"] and row["bitwise"] == row["cases"]
res["shapes"]["lm_head"] = row
print(f"lm_head: {row}", flush=True)
res["all_ok"] = bool(ok_all)
json.dump(res, open("/repo/results/2026-09-25-kernels/dense-gemv/serve-hook-test.json", "w"), indent=1)
print("ALL_OK" if ok_all else "FAILURES", flush=True)
sys.exit(0 if ok_all else 1)
