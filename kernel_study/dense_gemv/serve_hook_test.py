#!/usr/bin/env python3
"""End-to-end check of the serve wiring (run in the image with the repo's patch dir
mounted like run.sh does and DSV41_DENSE_GEMV=1):

- sitecustomize rewrote vllm's mxfp8/flashinfer.py (hooks present);
- serve load order FIRST (fresh process, nothing armed yet): every dense layer arms in
  the image's named_modules order before any call, the first eager forward arms wo_a
  and checks wq_b's pre-quantized input, and only then is every armed layer called at
  M = 8..1 (bf16 and QuantizedActivation) bitwise vs stock, plus a captured forward
  replayed vs eager (k3 review blocker: arming a smaller-K shape used to lower a shared
  instantiation's dynamic-smem attribute and break the larger K's launches);
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
import flashinfer  # noqa: E402
from torch.nn import Parameter  # noqa: E402
from vllm.model_executor.kernels.linear.mxfp8 import flashinfer as fi_mod  # noqa: E402
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation  # noqa: E402
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (  # noqa: E402
    mxfp8_e4m3_quantize, swizzle_mxfp8_scale)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic  # noqa: E402
from vllm.models.deepseek_v4.nvidia.ops import o_proj as oproj_mod  # noqa: E402

res = {"hooks_in_vllm_file": "_dsv41_gemv_apply" in open(fi_mod.__file__).read(), "shapes": {}}
assert res["hooks_in_vllm_file"], "sitecustomize did not patch flashinfer.py"
calls = {"n": 0}
_orig_run = dense_gemv._run


def _counting_run(*a, **k):
    out = _orig_run(*a, **k)
    calls["n"] += out is not None  # None = no bucket for this M: maybe_apply falls back to b12x
    return out


dense_gemv._run = _counting_run
woa_calls = {"n": 0}
_orig_woa_run = dense_gemv._woa_run


def _counting_woa_run(*a, **k):
    out = _orig_woa_run(*a, **k)
    woa_calls["n"] += bool(out)
    return out


dense_gemv._woa_run = _counting_woa_run
Kern = fi_mod.FlashInferCutlassMxfp8LinearKernel
gen = torch.Generator(device="cuda").manual_seed(11)
ok_all = True


class _StockHead:
    """lm_head apply as lmhead_mxfp8 does it: GEMV hook, else quant + flashinfer.mm_mxfp8 auto."""

    def apply(self, layer, x, bias=None):
        out = dense_gemv.maybe_apply(layer, x, bias)
        if out is not None:
            return out
        n, k = layer.weight.shape
        q, s = mxfp8_e4m3_quantize(x.reshape(-1, k), is_sf_swizzled_layout=True)
        return flashinfer.mm_mxfp8(q, layer.weight.t(), s, layer.weight_scale, out_dtype=x.dtype,
                                   backend="auto").view(*x.shape[:-1], n)


class _Ident(torch.nn.Module):
    def forward(self, t):
        return t


# o_proj wo_a through the real (patched) call site: deep_gemm_fp8_o_proj with the serve's
# producer, the prepacked weight scale (DSV41_WOA_PREPACK=1) and an identity wo_b.
G, D, WK = dense_gemv.WOA_GROUPS, dense_gemv.WOA_D, dense_gemv.WOA_K
res["hooks_in_o_proj"] = "_dsv41_gemv_woa(" in open(oproj_mod.__file__).read()
ok_all &= res["hooks_in_o_proj"]
wo_b = _Ident()
wo_b.weight = torch.empty(0, dtype=torch.bfloat16)  # o_proj's wo_a probe prints wo_b.weight.dtype
cos_sin = torch.randn(8192, 64, device="cuda", dtype=torch.float32)
kw = dict(n_groups=G, heads_per_group=8, nope_dim=448, rope_dim=64, o_lora_rank=D, einsum_recipe=(1, 1, 32),
          tma_aligned_scales=True)


def make_dense(name, li):
    w, s2d = weights.load(name, li)
    layer = torch.nn.Module()
    layer.weight = Parameter(w, requires_grad=False)
    layer.weight_scale = Parameter(s2d, requires_grad=False)
    kern = object.__new__(Kern)
    kern.process_weights_after_loading(layer)  # -> dense_gemv.prepare(): plan_grid, self-test, arm
    return kern, layer


def make_lmhead():
    w, s2d = weights.load("lm_head", 0)
    n, k = w.shape
    lm = torch.nn.Module()
    lm.weight = Parameter(w, requires_grad=False)
    lm.weight_scale = Parameter(swizzle_mxfp8_scale(s2d, M=n, K=k).contiguous(), requires_grad=False)
    head = _StockHead()
    dense_gemv.prepare_lmhead(head, lm, s2d)
    return head, lm


def make_woa(li):
    w, s2d = weights.load("wo_a", li)
    wo_a = torch.nn.Module()
    wo_a.weight = Parameter(w.view(G, D, WK).contiguous(), requires_grad=False)
    wo_a.weight_scale = Parameter(torch.exp2(s2d.view(G, D, WK // 32).float() - 127.0), requires_grad=False)
    return wo_a


# ---------------------------------------------------------------- serve load order
# The image's registration order (vLLM process_weights_after_loading walks
# model.named_modules()): DeepseekV4DecoderLayer = [engram (layers 1, 14)], attn
# (fused_wqa_wkv, wq_b, wo_a, wo_b), ffn (shared gate_up, down); the target lm_head
# after the layers; then the DSpark draft (main_proj, then its decoder layers; it aliases
# the target lm_head). wo_a is not armed at load: its o_proj hook arms at the first eager
# decode-size call. Layer 38 stands in for a draft decoder layer (same shapes).
SERVE_ORDER = [("engram_wkv", 1), ("qkv_a", 1), ("wq_b", 1), ("wo_a", 1), ("wo_b", 1), ("shared_gate_up", 1),
               ("shared_down", 1),
               ("qkv_a", 39), ("wq_b", 39), ("wo_a", 39), ("wo_b", 39), ("shared_gate_up", 39), ("shared_down", 39),
               ("lm_head", 0), ("main_proj", 0),
               ("qkv_a", 38), ("wq_b", 38), ("wo_a", 38), ("wo_b", 38), ("shared_gate_up", 38), ("shared_down", 38)]


def _clear_cuda_error() -> int:
    """cudaGetLastError() from the process's libcudart: a failed launch leaves its error pending,
    and torch's next launch check would raise it instead of this test recording it."""
    import ctypes

    for line in open("/proc/self/maps"):
        if "libcudart.so" in line:
            return int(ctypes.CDLL(line.split()[-1]).cudaGetLastError())
    return -1


def _inputs(e, M):
    """Fresh random inputs for entry e at M rows."""
    if e["name"] == "wo_a":
        return (torch.randn(M, 32, 512, generator=gen, device="cuda").to(torch.bfloat16),
                torch.randint(0, 8000, (M,), generator=gen, device="cuda"))
    return (torch.randn(M, e["K"], generator=gen, device="cuda").to(torch.bfloat16),)


def _call(e, inp, quant=False):
    if e["name"] == "wo_a":
        return oproj_mod.deep_gemm_fp8_o_proj(inp[0], inp[1], cos_sin, e["mod"], wo_b, **kw)
    x = inp[0]
    if quant:
        q, s = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
        x = QuantizedActivation(data=q, scale=s, orig_dtype=torch.bfloat16, orig_shape=x.shape,
                                quant_key=kMxfp8Dynamic)
    if e["name"] == "lm_head":
        return e["kern"].apply(e["mod"], x)
    return e["kern"].apply_weights(e["mod"], x)


def _check(e, M, quant, row):
    """One call through the hook vs the stock path on the same input (layer disarmed)."""
    inp = _inputs(e, M)
    counter = woa_calls if e["name"] == "wo_a" else calls
    before = counter["n"]
    try:
        y = _call(e, inp, quant)
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - a launch failure is a test failure; keep going
        err = _clear_cuda_error()
        row["failures"].append(f"{e['tag']} M={M} quant={quant}: {str(exc).splitlines()[0][:160]} "
                               f"(cudaGetLastError={err})")
        return
    used = counter["n"] > before
    armed = e["mod"]._dsv41_gemv  # read after the call: wo_a arms at its first eager call
    e["mod"]._dsv41_gemv = False if e["name"] == "wo_a" else None
    yref = _call(e, inp, quant)
    e["mod"]._dsv41_gemv = armed
    torch.cuda.synchronize()
    eq = torch.equal(y.view(torch.int16), yref.view(torch.int16))
    if e["name"] == "wo_a":
        expect = M <= dense_gemv.MAX_M
    else:
        expect = armed is not None and armed.plan(M) is not None
    row["cases"] += 1
    row["bitwise"] += eq
    row["used_as_expected"] += used == expect
    if not eq or used != expect:
        row["failures"].append(f"{e['tag']} M={M} quant={quant}: equal={eq} used={used} expected_used={expect}")


def serve_order_phase() -> dict:
    row = {"order": [f"{n}@L{li}" for n, li in SERVE_ORDER], "not_armed": [], "cases": 0, "bitwise": 0,
           "used_as_expected": 0, "failures": []}
    ents = []
    for name, li in SERVE_ORDER:  # load: every dense layer arms before any call
        e = {"name": name, "tag": f"{name}@L{li}"}
        if name == "wo_a":
            e["mod"] = make_woa(li)
        elif name == "lm_head":
            e["kern"], e["mod"] = make_lmhead()
            e["K"] = e["mod"].weight.shape[1]
        else:
            e["kern"], e["mod"] = make_dense(name, li)
            e["K"] = e["mod"].weight.shape[1]
        if name != "wo_a" and getattr(e["mod"], "_dsv41_gemv", None) is None:
            row["not_armed"].append(e["tag"])
        ents.append(e)
    print(f"serve order: {len(ents)} layers created in load order, not armed: {row['not_armed']}", flush=True)
    # first eager forward at the largest capture size (vLLM warms each size up eagerly before
    # capturing it, largest first): wo_a arms here, wq_b's QuantizedActivation is checked here
    for e in ents:
        _check(e, 8, e["name"] == "wq_b", row)
    # then every layer at every decode M, largest first, both input kinds where the hook takes both
    for M in range(dense_gemv.MAX_M, 0, -1):
        for e in ents:
            _check(e, M, False, row)
            if e["name"] not in ("wo_a", "lm_head"):
                _check(e, M, True, row)
    # one captured forward-order pass at M = 4, replayed on new inputs vs eager
    try:
        static = [_inputs(e, 4) for e in ents]
        for e, inp in zip(ents, static):
            _call(e, inp, e["name"] == "wq_b")
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outs = [_call(e, inp, e["name"] == "wq_b") for e, inp in zip(ents, static)]
        for inp in static:
            inp[0].copy_(torch.randn(inp[0].shape, generator=gen, device="cuda").to(inp[0].dtype))
        g.replay()
        eager = [_call(e, inp, e["name"] == "wq_b") for e, inp in zip(ents, static)]
        torch.cuda.synchronize()
        row["graph_replay_eq_eager"] = all(torch.equal(a.view(torch.int16), b.view(torch.int16))
                                           for a, b in zip(outs, eager))
    except Exception as exc:  # noqa: BLE001 - recorded as a failure
        err = _clear_cuda_error()
        row["graph_replay_eq_eager"] = False
        row["failures"].append(f"graph capture/replay: {str(exc).splitlines()[0][:160]} (cudaGetLastError={err})")
    row["ok"] = (not row["not_armed"] and not row["failures"] and row["bitwise"] == row["cases"]
                 and row["used_as_expected"] == row["cases"] and row["graph_replay_eq_eager"])
    for f in row["failures"]:
        print(f"serve order FAIL {f}", flush=True)
    print(f"serve order: {row['bitwise']}/{row['cases']} bitwise, {row['used_as_expected']}/{row['cases']} "
          f"GEMV use as expected, {len(row['failures'])} failures, graph replay == eager "
          f"{row['graph_replay_eq_eager']}", flush=True)
    return row


res["serve_order"] = serve_order_phase()  # FIRST: nothing may be armed before it
ok_all &= res["serve_order"]["ok"]
torch.cuda.empty_cache()

# ---------------------------------------------------------------- one shape at a time
for name in ("qkv_a", "wq_b", "wo_b", "shared_gate_up", "shared_down", "engram_wkv", "main_proj"):
    _, N, K, layers = weights.SHAPES[name]
    kern, layer = make_dense(name, layers[-1])
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
    del g, layer, kern
    torch.cuda.empty_cache()

# lm_head: stock lm_head apply (quant + flashinfer.mm_mxfp8 auto) as the reference
head, lm = make_lmhead()
armed = getattr(lm, "_dsv41_gemv", None)
row = {"armed": armed is not None, "cases": 0, "bitwise": 0}
if armed is not None:
    for M in (1, 3, 4, 6, 8):
        x = torch.randn(M, lm.weight.shape[1], generator=gen, device="cuda").to(torch.bfloat16)
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
del head, lm

row = {"cases": 0, "bitwise": 0, "armed": False, "graph_replay_eq_eager": None}
wo_a = make_woa(5)
for M in (4, 1, 3, 6, 8, 9):
    for rep in range(3):
        o_att = torch.randn(M, 32, 512, generator=gen, device="cuda").to(torch.bfloat16)
        pos = torch.randint(0, 8000, (M,), generator=gen, device="cuda")
        before = woa_calls["n"]
        out = oproj_mod.deep_gemm_fp8_o_proj(o_att, pos, cos_sin, wo_a, wo_b, **kw)
        used = woa_calls["n"] > before
        armed = getattr(wo_a, "_dsv41_gemv", None)
        wo_a._dsv41_gemv = False  # stock einsum for the reference
        ref = oproj_mod.deep_gemm_fp8_o_proj(o_att, pos, cos_sin, wo_a, wo_b, **kw)
        wo_a._dsv41_gemv = armed
        torch.cuda.synchronize()
        eq = torch.equal(out.view(torch.int16), ref.view(torch.int16))
        row["cases"] += 1
        row["bitwise"] += eq
        row["armed"] = armed not in (None, False)
        if not eq or used != (M <= 8):
            ok_all = False
            print(f"wo_a M={M}: equal={eq} used={used}", flush=True)
# graph capture of the hooked call site at M = 4
o_att = torch.randn(4, 32, 512, generator=gen, device="cuda").to(torch.bfloat16)
pos = torch.arange(4, device="cuda") + 50
eager = oproj_mod.deep_gemm_fp8_o_proj(o_att, pos, cos_sin, wo_a, wo_b, **kw)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    yg = oproj_mod.deep_gemm_fp8_o_proj(o_att, pos, cos_sin, wo_a, wo_b, **kw)
o_att.copy_(torch.randn(4, 32, 512, generator=gen, device="cuda").to(torch.bfloat16))
g.replay()
ye = oproj_mod.deep_gemm_fp8_o_proj(o_att, pos, cos_sin, wo_a, wo_b, **kw)
torch.cuda.synchronize()
row["graph_replay_eq_eager"] = bool(torch.equal(yg.view(torch.int16), ye.view(torch.int16)))
ok_all &= row["armed"] and row["bitwise"] == row["cases"] and row["graph_replay_eq_eager"]
res["shapes"]["wo_a"] = row
print(f"wo_a: {row}", flush=True)
res["all_ok"] = bool(ok_all)
json.dump(res, open("/repo/results/2026-09-25-kernels/dense-gemv/serve-hook-test.json", "w"), indent=1)
print("ALL_OK" if ok_all else "FAILURES", flush=True)
sys.exit(0 if ok_all else 1)
