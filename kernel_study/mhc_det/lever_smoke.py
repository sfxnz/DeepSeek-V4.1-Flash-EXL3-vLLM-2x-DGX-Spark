#!/usr/bin/env python3
"""GPU smoke of the DSV41_MHC_DET_SPLITS lever as the serve wires it (serve image, spark2).

install() against the real vLLM modules, prepare() on a module tree holding real mHC weights,
then the patched model-module names (what DeepseekV4DecoderLayer.forward calls) vs the stock
functions, bitwise, eager and inside a CUDA graph; T > 16 and an unpacked fn during capture
must take the stock path.
"""
from __future__ import annotations

import importlib
import json
import sys

import torch

sys.path.insert(0, "/repo/kernel_study/mhc_det")
import common as C  # noqa: E402
import mhc_det  # noqa: E402


class DeepseekV4DecoderLayer(torch.nn.Module):  # name matched by mhc_det._fn_attrs
    def __init__(self, attn, ffn, bc=None):
        super().__init__()
        self.hc_attn_fn = torch.nn.Parameter(attn, requires_grad=False)
        self.hc_ffn_fn = torch.nn.Parameter(ffn, requires_grad=False)
        self.hc_attn_fn_broadcast = bc


def ints(t):
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


def same(a, b) -> bool:
    if isinstance(a, (tuple, list)):
        return all(same(x, y) for x, y in zip(a, b))
    return bool((ints(a) == ints(b)).all())


def main() -> int:
    res = {}
    mhc_det.install({"DSV41_MHC_DET_SPLITS": "16"})
    m = importlib.import_module(mhc_det.MODEL_MOD)
    d = importlib.import_module(mhc_det.DSPARK_MOD)
    res["patched"] = [m.mhc_post_tilelang is mhc_det._post, m.mhc_pre_delayed_tilelang is mhc_det._pre,
                      d.mhc_post_tilelang is mhc_det._post]
    names = ["layers.0.hc_attn_fn", "layers.0.hc_ffn_fn", "layers.7.hc_attn_fn", "layers.7.hc_ffn_fn",
             "layers.7.hc_attn_scale", "layers.7.hc_attn_base", "layers.7.attn_norm.weight"]
    w = C.load(names)
    bc = w["layers.0.hc_attn_fn"].view(-1, 4, 5120).sum(dim=1).contiguous()
    tree = torch.nn.Sequential(DeepseekV4DecoderLayer(w["layers.0.hc_attn_fn"], w["layers.0.hc_ffn_fn"], bc),
                               DeepseekV4DecoderLayer(w["layers.7.hc_attn_fn"], w["layers.7.hc_ffn_fn"]))
    mhc_det.prepare(tree, "smoke")
    res["on"] = mhc_det._S.on
    calls = {"gemm": 0, "post": 0}
    gemm0, post0 = mhc_det._S.dk.gemm_pk, mhc_det._S.dk.post

    def gemm_count(*a, **k):
        calls["gemm"] += 1
        return gemm0(*a, **k)

    def post_count(*a, **k):
        calls["post"] += 1
        return post0(*a, **k)

    mhc_det._S.dk.gemm_pk, mhc_det._S.dk.post = gemm_count, post_count
    fn = tree[1].hc_attn_fn
    scale, base, nw = w["layers.7.hc_attn_scale"].float(), w["layers.7.hc_attn_base"].float(), w["layers.7.attn_norm.weight"].bfloat16()
    emb = C.embeddings(32)
    rows = []
    for t in (1, 3, 4, 6, 8, 16, 17, 64):
        g = torch.Generator(device="cuda").manual_seed(t)
        residual = (torch.randn(t, 4, 5120, device="cuda", generator=g) * 3).bfloat16()
        x = (torch.randn(t, 5120, device="cuda", generator=g)).bfloat16()
        pmix = torch.softmax(torch.randn(t, 4, device="cuda", generator=g), -1).contiguous()
        post_mix = (2 * torch.sigmoid(torch.randn(t, 4, 1, device="cuda", generator=g))).contiguous()
        comb = torch.softmax(torch.randn(t, 4, 4, device="cuda", generator=g), -1).contiguous()
        before = dict(calls)
        r_d = m.mhc_post_tilelang(x, residual, post_mix, comb)
        r_s = mhc_det._S.stock_post(x, residual, post_mix, comb)
        args = (scale, base, 1e-20, 1e-6, 1e-6, 2.0, 20)
        o_d = m.mhc_pre_delayed_tilelang(r_d, fn, *args, pre_mix=pmix, norm_weight=nw, norm_eps=1e-20)
        o_s = mhc_det._S.stock_pre(r_s, fn, *args, pre_mix=pmix, norm_weight=nw, norm_eps=1e-20)
        e = emb[: min(t, 32)] if t <= 32 else emb[:32].repeat(2, 1)[:t]
        rb = e.unsqueeze(1).expand(-1, 4, -1).contiguous()
        b_d = m.mhc_pre_delayed_tilelang(rb, bc, *args, x=e, norm_weight=nw, norm_eps=1e-20)
        b_s = mhc_det._S.stock_pre(rb, bc, *args, x=e, norm_weight=nw, norm_eps=1e-20)
        rows.append({"T": t, "post_bitwise": same(r_d, r_s), "pre_bitwise": same(o_d, o_s),
                     "broadcast_bitwise": same(b_d, b_s), "det_gemm_calls": calls["gemm"] - before["gemm"],
                     "det_post_calls": calls["post"] - before["post"]})
    res["dispatch"] = rows
    # capture through the patched names; an unpacked fn in capture must fall back to stock
    t = 4
    g = torch.Generator(device="cuda").manual_seed(99)
    residual = (torch.randn(t, 4, 5120, device="cuda", generator=g) * 3).bfloat16()
    x = torch.randn(t, 5120, device="cuda", generator=g).bfloat16()
    pmix = torch.softmax(torch.randn(t, 4, device="cuda", generator=g), -1).contiguous()
    post_mix = (2 * torch.sigmoid(torch.randn(t, 4, 1, device="cuda", generator=g))).contiguous()
    comb = torch.softmax(torch.randn(t, 4, 4, device="cuda", generator=g), -1).contiguous()
    fresh = (fn.clone() * 1.0001).contiguous()  # never packed
    args = (scale, base, 1e-20, 1e-6, 1e-6, 2.0, 20)

    def body():
        r = m.mhc_post_tilelang(x, residual, post_mix, comb)
        o = m.mhc_pre_delayed_tilelang(r, fn, *args, pre_mix=pmix, norm_weight=nw, norm_eps=1e-20)
        o2 = m.mhc_pre_delayed_tilelang(r, fresh, *args, pre_mix=pmix, norm_weight=nw, norm_eps=1e-20)
        return (r,) + tuple(o) + tuple(o2)

    ref = [o.clone() for o in body()]
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    mhc_det._S.packed.pop(fresh.data_ptr(), None)  # the eager call above packed it; unpack again
    graph = torch.cuda.CUDAGraph()
    before = dict(calls)
    misses0 = mhc_det._S.misses
    with torch.cuda.graph(graph):
        out = body()
    res["capture"] = {"det_gemm_calls_captured": calls["gemm"] - before["gemm"],
                      "misses_logged": mhc_det._S.misses - misses0}
    bad = 0
    for _ in range(50):
        graph.replay()
        torch.cuda.synchronize()
        bad += int(not same(out, ref))
    res["capture"]["mismatching_replays_of_50"] = bad
    print(json.dumps(res, indent=1), flush=True)
    ok = (all(res["patched"]) and res["on"] and bad == 0
          and all(r["post_bitwise"] and r["pre_bitwise"] and r["broadcast_bitwise"] for r in rows)
          and all((r["det_gemm_calls"] == 2) == (r["T"] <= 16) for r in rows)
          and res["capture"]["det_gemm_calls_captured"] == 1 and res["capture"]["misses_logged"] == 1)
    res["pass"] = ok
    with open("/repo/results/2026-09-25-kernels/mhc-det/lever_smoke.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
