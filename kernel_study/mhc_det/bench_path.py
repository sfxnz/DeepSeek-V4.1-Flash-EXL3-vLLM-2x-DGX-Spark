#!/usr/bin/env python3
"""Whole decode mHC path: det (post + det GEMM + stock fused norm) vs stock (serve image, spark2).

--mode check : (1) det post vs TileLang mhc_post, bitwise, T = 1..16;
               (2) a 43-layer mHC recurrence on real weights (86 fn/scale/base, attn/ffn norm
                   weights, real embeddings; synthetic sublayer outputs), target layer 0 via
                   the K=5120 broadcast path, draft layer 0 without a carried pre-mix; every
                   output of every sublayer compared bitwise, T in {1, 3, 4, 6, 8, 16};
               (3) 1000 runs of the whole det recurrence, bitwise identical;
               (4) the whole recurrence captured in one CUDA graph, 100 replays vs eager.
--mode time  : the recurrence in a CUDA graph (86 sublayers, each preceded by an AR-proxy copy
               of the sublayer output), stock vs det, per-sublayer us = (t[graph] -
               t[AR-proxy-only graph]) / 86, alternating replays, 200 reps.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch

sys.path.insert(0, "/repo/kernel_study/mhc_det")
import common as C  # noqa: E402
import mhc_det  # noqa: E402

OUT = "/repo/results/2026-09-25-kernels/mhc-det"
RMS_EPS, HC_EPS, POST_ALPHA, SINKHORN = 1e-20, 1e-6, 2.0, 20


def load_weights():
    names = C.mhc_names()
    norms = []
    for prefix in [f"layers.{i}" for i in range(40)] + [f"mtp.{i}" for i in range(3)]:
        norms += [f"{prefix}.attn_norm.weight", f"{prefix}.ffn_norm.weight"]
    w = C.load(names + norms)
    for k in list(w):
        if k.endswith("_fn") or k.endswith("_scale") or k.endswith("_base"):
            w[k] = w[k].float().contiguous()
        else:
            w[k] = w[k].bfloat16().contiguous()
    l0 = w["layers.0.hc_attn_fn"]
    w["layers.0.hc_attn_fn_broadcast"] = l0.view(-1, C.HC, C.HIDDEN).sum(dim=1).contiguous()
    return w


def sublayers():
    """(prefix, sub, first) in execution order: 40 target layers, then the 3 draft layers."""
    out = []
    for grp in ([f"layers.{i}" for i in range(40)], [f"mtp.{i}" for i in range(3)]):
        for li, p in enumerate(grp):
            out.append((p, "attn", li == 0))
            out.append((p, "ffn", False))
    return out


class Path:
    def __init__(self, w, dk, packed, det: bool, det_norm: bool = True):
        self.w, self.dk, self.packed, self.det, self.det_norm = w, dk, packed, det, det_norm
        from vllm.model_executor.kernels.mhc import tilelang as tl

        self.tl = tl

    def post(self, x, residual, post_mix, comb):
        if self.det:
            return mhc_det.det_post(self.dk, x, residual, post_mix, comb)
        return self.tl.mhc_post_tilelang(x, residual, post_mix, comb)

    def pre(self, residual, fn_name, prefix, sub, pre_mix, x=None):
        w = self.w
        args = (w[f"{prefix}.hc_{sub}_scale"], w[f"{prefix}.hc_{sub}_base"], RMS_EPS, HC_EPS, HC_EPS,
                POST_ALPHA, SINKHORN)
        kw = dict(pre_mix=pre_mix, x=x, norm_weight=w[f"{prefix}.{sub}_norm.weight"], norm_eps=RMS_EPS)
        if self.det:
            return mhc_det.det_pre_delayed(self.dk, self.packed[fn_name], residual, w[fn_name], *args, **kw,
                                           det_norm=self.det_norm)
        return self.tl.mhc_pre_delayed_tilelang(residual, w[fn_name], *args, **kw)


def run_chain(path: Path, t: int, emb, xouts, record=None):
    """43-layer mHC recurrence. xouts[i]: synthetic sublayer output [t, H] for sublayer i."""
    outs = []
    state = None
    for i, (prefix, sub, first) in enumerate(sublayers()):
        if first and prefix == "layers.0":
            e = emb[:t]
            residual = e.unsqueeze(1).expand(-1, C.HC, -1).contiguous()
            post_mix, comb, li, pre = path.pre(residual, "layers.0.hc_attn_fn_broadcast", prefix, sub, None, x=e)
        elif first:  # draft layer 0: residual = embeddings repeated, no carried pre-mix
            residual = emb[:t].unsqueeze(-2).repeat(1, C.HC, 1)
            post_mix, comb, li, pre = path.pre(residual, f"{prefix}.hc_{sub}_fn", prefix, sub, None)
        else:
            residual_prev, post_prev, comb_prev, pre_prev = state
            residual = path.post(xouts[i], residual_prev, post_prev, comb_prev)
            post_mix, comb, li, pre = path.pre(residual, f"{prefix}.hc_{sub}_fn", prefix, sub, pre_prev)
        state = (residual, post_mix, comb, pre)
        if record is not None:
            record.append((residual, post_mix, comb, li, pre))
    return state


def ints(tsr):
    return tsr.contiguous().view(torch.int16 if tsr.element_size() == 2 else torch.int32)


def check(args) -> dict:
    res: dict = {}
    w = load_weights()
    emb = C.embeddings(64)
    dk = mhc_det.DetKernels()
    fn_names = [k for k in w if k.endswith("_fn") or k.endswith("_broadcast")]
    packed = {k: mhc_det.pack_fn(w[k]) for k in fn_names}
    stock, det = Path(w, dk, packed, False), Path(w, dk, packed, True)
    det_tl = Path(w, dk, packed, True, det_norm=False)

    # (1) post alone
    fails, n = [], 0
    for t in range(1, 17):
        for seed in range(6):
            g = torch.Generator(device="cuda").manual_seed(seed * 100 + t)
            residual = (torch.randn(t, C.HC, C.HIDDEN, device="cuda", generator=g) * (1 + 10 * seed)).bfloat16()
            x = (torch.randn(t, C.HIDDEN, device="cuda", generator=g) * 3).bfloat16()
            post_mix = (2 * torch.sigmoid(torch.randn(t, C.HC, 1, device="cuda", generator=g))).contiguous()
            comb = torch.softmax(torch.randn(t, C.HC, C.HC, device="cuda", generator=g) * 3, -1).contiguous()
            a = stock.post(x, residual, post_mix, comb)
            b = det.post(x, residual, post_mix, comb)
            n += 1
            if not bool((ints(a) == ints(b)).all()):
                fails.append({"T": t, "seed": seed, "n_diff": int((ints(a) != ints(b)).sum())})
    res["post_bitwise"] = {"cases": n, "failures": len(fails), "first": fails[:5]}
    print(json.dumps(res["post_bitwise"]), flush=True)

    # (2) whole recurrence, every output of every sublayer
    names = ("residual", "post_mix", "comb_mix", "layer_input", "pre_mix")
    res["chain_bitwise"] = {}
    for t in (1, 3, 4, 6, 8, 16):
        g = torch.Generator(device="cuda").manual_seed(1234 + t)
        xouts = [(torch.randn(t, C.HIDDEN, device="cuda", generator=g) * (0.5 + (i % 7))).bfloat16()
                 for i in range(len(sublayers()))]
        rec_s, rec_d, rec_t = [], [], []
        run_chain(stock, t, emb, xouts, rec_s)
        run_chain(det, t, emb, xouts, rec_d)
        run_chain(det_tl, t, emb, xouts, rec_t)
        out = {}
        for arm, rec in (("det", rec_d), ("det_stock_norm", rec_t)):
            bad = []
            for i, (os_, od) in enumerate(zip(rec_s, rec)):
                for nm, a, b in zip(names, os_, od):
                    if not bool((ints(a) == ints(b)).all()):
                        bad.append({"sublayer": i, "out": nm, "n_diff": int((ints(a) != ints(b)).sum())})
            out[arm] = {"outputs_compared": 5 * len(rec_s), "mismatches": len(bad), "first": bad[:5]}
        res["chain_bitwise"][f"T{t}"] = {"sublayers": len(rec_s), **out}
        print(t, json.dumps(res["chain_bitwise"][f"T{t}"]), flush=True)

    # (3) determinism of the whole det recurrence
    res["determinism"] = {}
    for t, runs in ((4, args.det_runs), (8, max(1, args.det_runs // 5))):
        g = torch.Generator(device="cuda").manual_seed(77 + t)
        xouts = [(torch.randn(t, C.HIDDEN, device="cuda", generator=g) * 2).bfloat16() for _ in sublayers()]
        ref = []
        run_chain(det, t, emb, xouts, ref)
        bad = torch.zeros((), dtype=torch.int64, device="cuda")
        for _ in range(runs):
            rec = []
            run_chain(det, t, emb, xouts, rec)
            for os_, od in zip(ref, rec):
                for a, b in zip(os_, od):
                    bad += (ints(a) != ints(b)).sum()
        res["determinism"][f"T{t}"] = {"runs": runs, "sublayers": len(ref), "differing_elements": int(bad)}
        print(json.dumps(res["determinism"][f"T{t}"]), flush=True)

    # (4) the whole det recurrence in one CUDA graph vs eager
    res["graph"] = {}
    for t in (1, 3, 4, 6, 8):
        g = torch.Generator(device="cuda").manual_seed(555 + t)
        xouts = [(torch.randn(t, C.HIDDEN, device="cuda", generator=g) * 2).bfloat16() for _ in sublayers()]
        eager = []
        run_chain(det, t, emb, xouts, eager)
        eager = [[o.clone() for o in r] for r in eager]
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            run_chain(det, t, emb, xouts)
        torch.cuda.current_stream().wait_stream(st)
        graph = torch.cuda.CUDAGraph()
        cap = []
        with torch.cuda.graph(graph):
            run_chain(det, t, emb, xouts, cap)
        bad_replays = 0
        for _ in range(100):
            for r in cap:
                r[3].fill_(0)  # layer_input
            graph.replay()
            torch.cuda.synchronize()
            ok = all(bool((ints(a) == ints(b)).all()) for re, rc in zip(eager, cap) for a, b in zip(re, rc))
            bad_replays += int(not ok)
        res["graph"][f"T{t}"] = {"replays": 100, "mismatching_replays": bad_replays}
    print(json.dumps(res["graph"]), flush=True)
    return res


def _graph(body):
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        body()
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()
    return g


def timing(args) -> dict:
    w = load_weights()
    emb = C.embeddings(64)
    dk = mhc_det.DetKernels()
    fn_names = [k for k in w if k.endswith("_fn") or k.endswith("_broadcast")]
    packed = {k: mhc_det.pack_fn(w[k]) for k in fn_names}
    res: dict = {"chain": []}
    subs = sublayers()
    for t in (1, 3, 4, 6, 8):
        g = torch.Generator(device="cuda").manual_seed(9 + t)
        src = [(torch.randn(t, C.HIDDEN, device="cuda", generator=g) * 2).bfloat16() for _ in subs]
        xouts = [s.clone() for s in src]

        def copies_only():
            for i in range(len(subs)):
                xouts[i].copy_(src[i])

        graphs = {"copies": _graph(copies_only)}
        for arm in ("stock", "det_stock_norm", "det"):
            path = Path(w, dk, packed, arm != "stock", det_norm=(arm == "det"))

            def with_copies(path=path):
                state = None
                for i, (prefix, sub, first) in enumerate(subs):
                    xouts[i].copy_(src[i])  # AR-proxy: the kernel that writes the sublayer output
                    if first and prefix == "layers.0":
                        e = emb[:t]
                        residual = e.unsqueeze(1).expand(-1, C.HC, -1).contiguous()
                        pm, cm, li, pr = path.pre(residual, "layers.0.hc_attn_fn_broadcast", prefix, sub, None, x=e)
                    elif first:
                        residual = emb[:t].unsqueeze(-2).repeat(1, C.HC, 1)
                        pm, cm, li, pr = path.pre(residual, f"{prefix}.hc_{sub}_fn", prefix, sub, None)
                    else:
                        rp, pp, cp, prp = state
                        residual = path.post(xouts[i], rp, pp, cp)
                        pm, cm, li, pr = path.pre(residual, f"{prefix}.hc_{sub}_fn", prefix, sub, prp)
                    state = (residual, pm, cm, pr)

            graphs[arm] = _graph(with_copies)
        for _ in range(3):
            for gr in graphs.values():
                gr.replay()
        torch.cuda.synchronize()
        samples = {k: [] for k in graphs}
        for _ in range(args.iters):
            for k, gr in graphs.items():  # alternate stock / det / copies every rep
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
                gr.replay()
                b.record()
                b.synchronize()
                samples[k].append(a.elapsed_time(b) * 1000.0)
        base = C.summarize(samples["copies"])["median_us"]
        for arm in ("stock", "det_stock_norm", "det"):
            per = [x - y for x, y in zip(samples[arm], samples["copies"])]
            st = C.summarize(per)
            row = {"T": t, "arm": arm, "sublayers": len(subs), "graph_us_median": C.summarize(samples[arm])["median_us"],
                   "copies_only_us": base, "mhc_us_per_pass_median": st["median_us"], "p10": st["p10_us"],
                   "p90": st["p90_us"], "us_per_sublayer": round(st["median_us"] / len(subs), 3)}
            res["chain"].append(row)
            print(json.dumps(row), flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("check", "time"), required=True)
    ap.add_argument("--det-runs", type=int, default=1000)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    res = check(args) if args.mode == "check" else timing(args)
    with open(f"{OUT}/path_{args.mode}{args.tag}.json", "w") as fh:
        json.dump(res, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
