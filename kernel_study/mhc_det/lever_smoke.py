#!/usr/bin/env python3
"""GPU smoke of the DSV41_MHC_DET_SPLITS lever as the serve wires it (serve image, spark2).

install() against the real vLLM modules, prepare() on a module tree holding real mHC weights,
then the patched model-module names (what DeepseekV4DecoderLayer.forward calls) vs the stock
functions, bitwise, eager and inside a CUDA graph; T > 16 and an unpacked fn during capture
must take the stock path.

--overlap: the same with DSV41_MHC_DET_OVERLAP=1. A layer-shaped chain through the patched names
(pre -> all-reduce -> post -> pre ... -> final post, 2 layers) where the all-reduce is the real,
wrapped GroupCoordinator.all_reduce on a world-size-1 group (it returns its input after the
hook forked the deferred coefficient half): bitwise vs the stock chain at T 1/3/4/6/8/16, fork
and join counts, the no-all-reduce variant (in-place launch), T > 16 stock, and the chain
captured in one CUDA graph (50 replays vs eager). Writes lever_smoke_overlap.json.
"""
from __future__ import annotations

import importlib
import json
import sys

import torch

sys.path.insert(0, "/repo/kernel_study/mhc_det")
import common as C  # noqa: E402
import mhc_det  # noqa: E402


class _Norm(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.weight = torch.nn.Parameter(w, requires_grad=False)
        self.variance_epsilon = 1e-20


class DeepseekV4DecoderLayer(torch.nn.Module):  # name matched by mhc_det._layers
    def __init__(self, w, prefix, bc=None):
        super().__init__()
        for sub in ("attn", "ffn"):
            setattr(self, f"hc_{sub}_fn", torch.nn.Parameter(w[f"{prefix}.hc_{sub}_fn"].float(), requires_grad=False))
            setattr(self, f"hc_{sub}_scale", torch.nn.Parameter(w[f"{prefix}.hc_{sub}_scale"].float(), requires_grad=False))
            setattr(self, f"hc_{sub}_base", torch.nn.Parameter(w[f"{prefix}.hc_{sub}_base"].float(), requires_grad=False))
            setattr(self, f"{sub}_norm", _Norm(w[f"{prefix}.{sub}_norm.weight"].bfloat16()))
        self.hc_attn_fn_broadcast = bc
        self.rms_norm_eps, self.hc_eps, self.hc_post_alpha, self.hc_sinkhorn_iters = 1e-20, 1e-6, 2.0, 20


def ints(t):
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


def same(a, b) -> bool:
    if isinstance(a, (tuple, list)):
        return all(same(x, y) for x, y in zip(a, b))
    return bool((ints(a) == ints(b)).all())


def overlap_main() -> int:
    import mhc_det_overlap as ovl
    from vllm.distributed import parallel_state as ps

    res = {}
    mhc_det.install({"DSV41_MHC_DET_SPLITS": "16", "DSV41_MHC_DET_OVERLAP": "1"})
    m = importlib.import_module(mhc_det.MODEL_MOD)
    res["patched"] = [m.mhc_post_tilelang is mhc_det._post, m.mhc_pre_delayed_tilelang is mhc_det._pre,
                      getattr(ps.GroupCoordinator.all_reduce, "_dsv41_mhc_ovl", False), mhc_det._S.ovl is ovl]
    names = []
    for p in ("layers.0", "layers.7"):
        for sub in ("attn", "ffn"):
            names += [f"{p}.hc_{sub}_fn", f"{p}.hc_{sub}_scale", f"{p}.hc_{sub}_base", f"{p}.{sub}_norm.weight"]
    w = C.load(names)
    bc = w["layers.0.hc_attn_fn"].float().view(-1, 4, 5120).sum(dim=1).contiguous()
    tree = torch.nn.Sequential(DeepseekV4DecoderLayer(w, "layers.0", bc), DeepseekV4DecoderLayer(w, "layers.7"))
    mhc_det.prepare(tree, "smoke")
    res["det_on"], res["overlap_on"] = mhc_det._S.on, ovl.active()
    emb = C.embeddings(64)

    class _G:  # GroupCoordinator.all_reduce returns its input at world size 1 (after the hook)
        world_size = 1

    def ar(x):
        return ps.GroupCoordinator.all_reduce(_G(), x)

    def chain(post_f, pre_f, ar_f, t, xs):
        l0, l1 = tree[0], tree[1]
        args = lambda layer, sub: (getattr(layer, f"hc_{sub}_scale"), getattr(layer, f"hc_{sub}_base"),  # noqa: E731
                                   1e-20, 1e-6, 1e-6, 2.0, 20)
        e = emb[:t] if t <= 64 else emb.repeat(2, 1)[:t]
        rb = e.unsqueeze(1).expand(-1, 4, -1).contiguous()
        pm, cm, li, pr = pre_f(rb, bc, *args(l0, "attn"), x=e, norm_weight=l0.attn_norm.weight, norm_eps=1e-20)
        outs = [rb, pm, cm, li, pr]
        residual = rb
        for k, (layer, sub) in enumerate(((l0, "ffn"), (l1, "attn"), (l1, "ffn"))):
            residual = post_f(ar_f(xs[k]), residual, pm, cm)
            pm, cm, li, pr = pre_f(residual, getattr(layer, f"hc_{sub}_fn"), *args(layer, sub), pre_mix=pr,
                                   norm_weight=getattr(layer, f"{sub}_norm").weight, norm_eps=1e-20)
            outs += [residual, pm, cm, li, pr]
        outs.append(post_f(ar_f(xs[3]), residual, pm, cm))  # the model's final post
        return outs

    rows = []
    for t in (1, 3, 4, 6, 8, 16, 17, 64):
        g = torch.Generator(device="cuda").manual_seed(500 + t)
        xs = [(torch.randn(t, 5120, device="cuda", generator=g) * 2).bfloat16() for _ in range(4)]
        ref = chain(mhc_det._S.stock_post, mhc_det._S.stock_pre, lambda x: x, t, xs)
        f0, i0 = ovl._S.forks, ovl._S.in_place
        got = chain(m.mhc_post_tilelang, m.mhc_pre_delayed_tilelang, ar, t, xs)
        f1, i1 = ovl._S.forks, ovl._S.in_place
        got2 = chain(m.mhc_post_tilelang, m.mhc_pre_delayed_tilelang, lambda x: x, t, xs)  # no all-reduce
        torch.cuda.synchronize()
        rows.append({"T": t, "bitwise_with_all_reduce": same(ref, got), "bitwise_without": same(ref, got2),
                     "forks": f1 - f0, "in_place": ovl._S.in_place - i1, "pending_after": ovl._S.pending is not None})
    res["chain"] = rows
    t = 4
    g = torch.Generator(device="cuda").manual_seed(99)
    xs = [(torch.randn(t, 5120, device="cuda", generator=g) * 2).bfloat16() for _ in range(4)]
    eager = [o.clone() for o in chain(m.mhc_post_tilelang, m.mhc_pre_delayed_tilelang, ar, t, xs)]
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        chain(m.mhc_post_tilelang, m.mhc_pre_delayed_tilelang, ar, t, xs)
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    f0 = ovl._S.forks
    with torch.cuda.graph(graph):
        out = chain(m.mhc_post_tilelang, m.mhc_pre_delayed_tilelang, ar, t, xs)
    res["capture"] = {"forks_captured": ovl._S.forks - f0, "pending_after": ovl._S.pending is not None}
    bad = 0
    for _ in range(50):
        for o in out:
            if o.dtype == torch.bfloat16 and o.dim() == 2:
                o.zero_()  # layer_input
        graph.replay()
        torch.cuda.synchronize()
        bad += int(not same(out, eager))
    res["capture"]["mismatching_replays_of_50"] = bad
    print(json.dumps(res, indent=1), flush=True)
    ok = (all(res["patched"]) and res["det_on"] and res["overlap_on"] and bad == 0
          and res["capture"]["forks_captured"] == 4 and not res["capture"]["pending_after"]
          and all(r["bitwise_with_all_reduce"] and r["bitwise_without"] and not r["pending_after"] for r in rows)
          and all((r["forks"], r["in_place"]) == ((4, 4) if r["T"] <= 16 else (0, 0)) for r in rows))
    res["pass"] = ok
    with open("/repo/results/2026-09-25-kernels/mhc-det/lever_smoke_overlap.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


def main() -> int:
    if "--overlap" in sys.argv:
        return overlap_main()
    res = {}
    mhc_det.install({"DSV41_MHC_DET_SPLITS": "16"})
    m = importlib.import_module(mhc_det.MODEL_MOD)
    d = importlib.import_module(mhc_det.DSPARK_MOD)
    res["patched"] = [m.mhc_post_tilelang is mhc_det._post, m.mhc_pre_delayed_tilelang is mhc_det._pre,
                      d.mhc_post_tilelang is mhc_det._post]
    names = []
    for p in ("layers.0", "layers.7"):
        for sub in ("attn", "ffn"):
            names += [f"{p}.hc_{sub}_fn", f"{p}.hc_{sub}_scale", f"{p}.hc_{sub}_base", f"{p}.{sub}_norm.weight"]
    w = C.load(names)
    bc = w["layers.0.hc_attn_fn"].float().view(-1, 4, 5120).sum(dim=1).contiguous()
    tree = torch.nn.Sequential(DeepseekV4DecoderLayer(w, "layers.0", bc), DeepseekV4DecoderLayer(w, "layers.7"))
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
    scale, base, nw = tree[1].hc_attn_scale, tree[1].hc_attn_base, tree[1].attn_norm.weight
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
