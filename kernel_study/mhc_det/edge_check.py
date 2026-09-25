#!/usr/bin/env python3
"""Adversarial bitwise checks of the det kernels vs stock (serve image, spark2; real weights).

From the review of 4f661a5 (reviewer's rv_edge.py), plus the DSV41_MHC_DET_OVERLAP split path.
GEMM (det vs DeepGEMM 16-split): x families the bench_gemm families do not reach: denormal-range
squares, signed zeros, all-zero rows, huge (square overflow), inf/nan rows.
Whole pre at EVERY T in 1..16 on several real sublayers, carried pre-mix families (softmax,
extreme, signed zeros) and residual families: the fused det pre (det GEMM + det norm) AND the
split pre (mhc_det_norm_li + det GEMM + mhc_det_norm_coef, forked to a side stream and joined by
mhc_det_overlap, and launched in place), each vs the stock pre.
Post at every T in 1..16 with signed zeros / tiny / huge values.
Gate, fixed before any run: equal bits, or both NaN at the same positions (NaN payload
differences are counted separately and reported). Writes results/.../edge_check.json.
"""
from __future__ import annotations

import json
import sys

import torch

sys.path.insert(0, "/repo/kernel_study/mhc_det")
sys.path.insert(0, "/repo/docker/patch")
import bench_path as BP  # noqa: E402
import common as C  # noqa: E402
import mhc_det  # noqa: E402
import mhc_det_overlap as O  # noqa: E402

RMS_EPS, HC_EPS, POST_ALPHA, SINKHORN = 1e-20, 1e-6, 2.0, 20


def ints(t):
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


def cmp(a, b) -> dict:
    ia, ib = ints(a), ints(b)
    eq = ia == ib
    na, nb = torch.isnan(a.float()), torch.isnan(b.float())
    nan_both = na & nb
    bad = ~eq & ~nan_both
    return {"bit_diff": int((~eq).sum()), "non_nan_diff": int(bad.sum()), "nan_pos_mismatch": int((na ^ nb).sum()),
            "nan_both": int(nan_both.sum())}


def xfam(kind, t, k, g):
    z = torch.randn(t, k, device="cuda", generator=g)
    if kind == "denorm_sq":  # squares land in the fp32 denormal range
        return (torch.sign(z) * torch.exp(torch.empty(t, k, device="cuda").uniform_(-50.0, -43.5, generator=g))).bfloat16()
    if kind == "negzero":
        x = (z * 3).bfloat16()
        m = torch.rand(t, k, device="cuda", generator=g) < 0.3
        x[m] = -0.0
        return x
    if kind == "zero_rows":
        x = (z * 2).bfloat16()
        x[::2] = 0
        return x
    if kind == "negzero_rows":
        x = (z * 2).bfloat16()
        x[1::2] = -0.0
        return x
    if kind == "huge":
        return (z * 3e19).bfloat16()  # squares overflow to inf
    if kind == "infnan":
        x = (z * 2).bfloat16()
        if t > 1:
            x[t - 1, 5] = float("inf")
            x[0, 7] = float("nan")
        return x
    if kind == "mixed_scale":
        s = torch.exp(torch.randn(t, 1, device="cuda", generator=g) * 6)
        return (z * s).bfloat16()
    raise ValueError(kind)


def main() -> int:
    from vllm.model_executor.kernels.mhc import tilelang as tl
    from vllm.utils import deep_gemm as vdg

    vdg._lazy_init()
    res: dict = {"device": torch.cuda.get_device_name(), "l2_bytes": torch.cuda.get_device_properties(0).L2_cache_size}
    w = BP.load_weights()
    dk = mhc_det.DetKernels()
    g = torch.Generator(device="cuda").manual_seed(4242)

    # ---------------- GEMM ----------------
    fns = ["layers.0.hc_attn_fn_broadcast", "layers.3.hc_attn_fn", "layers.21.hc_ffn_fn", "mtp.2.hc_ffn_fn"]
    kinds = ["denorm_sq", "negzero", "zero_rows", "negzero_rows", "huge", "infnan", "mixed_scale"]
    gemm = {"cases": 0, "bitwise_fail": 0, "nan_only_diff": 0, "first": []}
    for name in fns:
        fn = w[name]
        pk = mhc_det.pack_fn(fn)
        k = fn.shape[1]
        for t in range(1, 17):
            for kind in kinds:
                x = xfam(kind, t, k, g)
                rm, rs = torch.empty(16, t, 24, device="cuda"), torch.empty(16, t, device="cuda")
                vdg.tf32_hc_prenorm_gemm(x, fn, rm, rs, 16)
                dm, ds = torch.full_like(rm, 7.0), torch.full_like(rs, 7.0)
                dk.gemm_pk(x, pk, dm, ds)
                c1, c2 = cmp(rm, dm), cmp(rs, ds)
                gemm["cases"] += 1
                if c1["bit_diff"] or c2["bit_diff"]:
                    if c1["non_nan_diff"] or c2["non_nan_diff"] or c1["nan_pos_mismatch"] or c2["nan_pos_mismatch"]:
                        gemm["bitwise_fail"] += 1
                        if len(gemm["first"]) < 8:
                            gemm["first"].append({"fn": name, "T": t, "kind": kind, "mixes": c1, "sqr": c2})
                    else:
                        gemm["nan_only_diff"] += 1
    res["gemm"] = gemm
    print("gemm", json.dumps(gemm), flush=True)

    # ---------------- whole pre, every T ----------------
    O._S.torch = torch
    O._S.armed = O._S.on = True
    pre = {"cases": 0, "fail": 0, "nan_only_diff": 0, "first": []}
    split = {"cases": 0, "fail": 0, "nan_only_diff": 0, "first": [], "forks": 0, "in_place": 0}
    subl = [("layers.0", "attn", True), ("layers.1", "attn", False), ("layers.9", "ffn", False),
            ("layers.39", "ffn", False), ("mtp.0", "attn", False), ("mtp.1", "ffn", False)]
    rfams = ["randn4", "negzero", "tiny", "zero_rows", "emb_like"]
    pfams = ["softmax", "extreme", "negzero", None]
    for prefix, sub, bc in subl:
        fname = "layers.0.hc_attn_fn_broadcast" if bc else f"{prefix}.hc_{sub}_fn"
        fn = w[fname]
        pk = mhc_det.pack_fn(fn)
        args = (w[f"{prefix}.hc_{sub}_scale"], w[f"{prefix}.hc_{sub}_base"], RMS_EPS, HC_EPS, HC_EPS, POST_ALPHA,
                SINKHORN)
        nw = w[f"{prefix}.{sub}_norm.weight"]
        for t in range(1, 17):
            for rf in rfams:
                for pf in (pfams if not bc else [None]):
                    xx = None
                    if bc:
                        xx = xfam("negzero" if rf == "negzero" else "mixed_scale", t, C.HIDDEN, g)
                        residual = xx.unsqueeze(1).expand(-1, C.HC, -1).contiguous()
                    else:
                        z = torch.randn(t, C.HC, C.HIDDEN, device="cuda", generator=g)
                        if rf == "randn4":
                            residual = (z * 4).bfloat16()
                        elif rf == "negzero":
                            residual = (z * 4).bfloat16()
                            residual[torch.rand(residual.shape, device="cuda", generator=g) < 0.3] = -0.0
                        elif rf == "tiny":
                            residual = (z * 1e-19).bfloat16()
                        elif rf == "zero_rows":
                            residual = (z * 4).bfloat16()
                            residual[::2] = 0
                        else:
                            residual = (z * torch.exp(torch.randn(t, C.HC, 1, device="cuda", generator=g) * 3)).bfloat16()
                    pm = None
                    if pf == "softmax":
                        pm = torch.softmax(torch.randn(t, 4, device="cuda", generator=g) * 2, -1).contiguous()
                    elif pf == "extreme":
                        pm = torch.tensor([[1e-30, 1e30, 0.5, -2.0]], device="cuda").repeat(t, 1).contiguous()
                    elif pf == "negzero":
                        pm = torch.tensor([[-0.0, 1.0, -0.0, 0.25]], device="cuda").repeat(t, 1).contiguous()
                    kw = dict(pre_mix=pm, x=xx, norm_weight=nw, norm_eps=RMS_EPS)
                    ref = tl.mhc_pre_delayed_tilelang(residual, fn, *args, **kw)
                    for how in ("fork", "in_place"):
                        sgot = mhc_det.det_pre_delayed(dk, pk, residual, fn, *args, **kw, defer=O.defer)
                        if how == "fork":
                            O._fork(O._S.pending)
                        O.settle()
                        split["cases"] += 1
                        sworst, snan = None, False
                        for nm, a, b in zip(("post_mix", "comb_mix", "layer_input", "pre_mix"), ref, sgot):
                            c = cmp(a, b)
                            if c["bit_diff"]:
                                if c["non_nan_diff"] or c["nan_pos_mismatch"]:
                                    sworst = sworst or {"out": nm, **c}
                                else:
                                    snan = True
                        if sworst:
                            split["fail"] += 1
                            if len(split["first"]) < 10:
                                split["first"].append({"sub": f"{prefix}.{sub}", "T": t, "rfam": rf, "pfam": pf,
                                                       "how": how, **sworst})
                        elif snan:
                            split["nan_only_diff"] += 1
                    got = mhc_det.det_pre_delayed(dk, pk, residual, fn, *args, **kw)
                    pre["cases"] += 1
                    worst = None
                    nan_only = False
                    for nm, a, b in zip(("post_mix", "comb_mix", "layer_input", "pre_mix"), ref, got):
                        c = cmp(a, b)
                        if c["bit_diff"]:
                            if c["non_nan_diff"] or c["nan_pos_mismatch"]:
                                worst = worst or {"out": nm, **c}
                            else:
                                nan_only = True
                    if worst:
                        pre["fail"] += 1
                        if len(pre["first"]) < 10:
                            pre["first"].append({"sub": f"{prefix}.{sub}", "T": t, "rfam": rf, "pfam": pf, **worst})
                    elif nan_only:
                        pre["nan_only_diff"] += 1
    res["pre"] = pre
    print("pre", json.dumps(pre), flush=True)
    split["forks"], split["in_place"] = O._S.forks, O._S.in_place
    res["pre_split_overlap"] = split
    print("pre_split_overlap", json.dumps(split), flush=True)

    # ---------------- post, every T ----------------
    post = {"cases": 0, "fail": 0, "first": []}
    for t in range(1, 17):
        for fam in ("negzero", "tiny", "huge", "randn"):
            z = torch.randn(t, 4, 5120, device="cuda", generator=g)
            zx = torch.randn(t, 5120, device="cuda", generator=g)
            if fam == "negzero":
                residual, x = (z * 3).bfloat16(), (zx * 2).bfloat16()
                residual[torch.rand(residual.shape, device="cuda", generator=g) < 0.4] = -0.0
                x[torch.rand(x.shape, device="cuda", generator=g) < 0.4] = -0.0
            elif fam == "tiny":
                residual, x = (z * 1e-20).bfloat16(), (zx * 1e-20).bfloat16()
            elif fam == "huge":
                residual, x = (z * 1e36).bfloat16(), (zx * 1e36).bfloat16()
            else:
                residual, x = (z * 3).bfloat16(), (zx * 2).bfloat16()
            post_mix = (2 * torch.sigmoid(torch.randn(t, 4, 1, device="cuda", generator=g))).contiguous()
            comb = torch.softmax(torch.randn(t, 4, 4, device="cuda", generator=g) * 3, -1).contiguous()
            if fam == "negzero":
                comb[:, 0, :] = -0.0
            a = tl.mhc_post_tilelang(x, residual, post_mix, comb)
            b = mhc_det.det_post(dk, x, residual, post_mix, comb)
            c = cmp(a, b)
            post["cases"] += 1
            if c["non_nan_diff"] or c["nan_pos_mismatch"]:
                post["fail"] += 1
                if len(post["first"]) < 8:
                    post["first"].append({"T": t, "fam": fam, **c})
    res["post"] = post
    print("post", json.dumps(post), flush=True)
    res["gate"] = "equal bits, or both NaN at the same positions"
    with open(f"{BP.OUT}/edge_check.json", "w") as fh:
        json.dump(res, fh, indent=1)
    ok = gemm["bitwise_fail"] == 0 and pre["fail"] == 0 and split["fail"] == 0 and post["fail"] == 0
    print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
