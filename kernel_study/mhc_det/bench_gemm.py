#!/usr/bin/env python3
"""mHC prenorm GEMM: det kernels vs the stock DeepGEMM 16-split kernel (serve image, spark2).

--mode check : bitwise equality vs stock on all 86 real fn matrices + the layer-0 broadcast fn,
               T = 1..16, several input families; 1000-run determinism of det and of stock
               (16 and 40 splits, GEMM and the full stock pre); CUDA-graph replay equality.
--mode time  : CUDA-event timing, cold L2 (64 MB flush, activation re-written so it is
               L2-hot as in the serve) and warm, 200 iterations after 20 warmup, arms
               alternating; plus an 86-call graph sweeping the real fn matrices.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import torch

sys.path.insert(0, "/repo/kernel_study/mhc_det")
import common as C  # noqa: E402
import mhc_det  # noqa: E402

OUT = "/repo/results/2026-09-25-kernels/mhc-det"


def stock_gemm(vdg, x, fn, mixes, sqr, splits):
    vdg.tf32_hc_prenorm_gemm(x, fn, mixes, sqr, splits)


def check(args) -> dict:
    from vllm.utils import deep_gemm as vdg

    vdg._lazy_init()
    res: dict = {"bitwise": {}, "determinism": {}, "graph": {}}
    fns = C.real_fns()
    emb = C.embeddings(512)
    dk = mhc_det.DetKernels()
    t0 = time.time()
    packed = {n: mhc_det.pack_fn(f) for n, f in fns}
    res["pack_s"] = round(time.time() - t0, 2)
    kinds_all = ["emb", "randn", "heavy"]
    kinds_edge = ["big", "tiny", "sparse"]
    fails = []
    n_cmp = 0
    for li, (name, fn) in enumerate(fns):
        k = fn.shape[1]
        kinds = kinds_all + (kinds_edge if li % 10 == 0 or k != C.K_FULL else [])
        for t in range(1, 17):
            for kind in kinds:
                x = C.make_x(kind, t, k, seed=1000 * li + 17 * t + len(kind), emb=emb)
                ref_m = torch.empty(16, t, 24, device="cuda")
                ref_s = torch.empty(16, t, device="cuda")
                stock_gemm(vdg, x, fn, ref_m, ref_s, 16)
                for arm in ("pk",):
                    m = torch.full((16, t, 24), float("nan"), device="cuda")
                    s = torch.full((16, t), float("nan"), device="cuda")
                    dk.gemm_pk(x, packed[name], m, s)
                    ok = C.bitwise_equal(m, ref_m) and C.bitwise_equal(s, ref_s)
                    n_cmp += 1
                    if not ok:
                        fails.append({"fn": name, "T": t, "kind": kind, "arm": arm,
                                      "mixes": C.maxdiff(m, ref_m), "sqr": C.maxdiff(s, ref_s)})
    res["bitwise"] = {"comparisons": n_cmp, "failures": len(fails), "first_failures": fails[:10],
                      "fns": len(fns), "T": "1..16", "kinds": kinds_all + kinds_edge}
    print(json.dumps(res["bitwise"])[:2000], flush=True)

    # determinism: 1000 runs, every output compared bitwise with run 0 (on device, one sync)
    name, fn = next((n, f) for n, f in fns if n == "layers.20.hc_ffn_fn")
    from vllm.model_executor.kernels.mhc import tilelang as mhc_tl
    from vllm.model_executor.kernels.mhc import warmup as mhc_wu

    scale_base = C.load(["layers.20.hc_ffn_scale", "layers.20.hc_ffn_base"])
    norm_w = (torch.rand(C.HIDDEN, device="cuda") + 0.5).bfloat16()
    for t in (4, 8):
        x = C.make_x("emb", t, C.K_FULL, seed=7 + t, emb=emb)
        arms = {
            "det_pk": lambda m, s: dk.gemm_pk(x, packed[name], m, s),
            "stock_dg16": lambda m, s: stock_gemm(vdg, x, fn, m, s, 16),
        }
        for arm, call in arms.items():
            m0 = torch.empty(16, t, 24, device="cuda")
            s0 = torch.empty(16, t, device="cuda")
            call(m0, s0)
            bad = torch.zeros((), dtype=torch.int64, device="cuda")
            m = torch.empty_like(m0)
            s = torch.empty_like(s0)
            for _ in range(args.det_runs):
                m.fill_(float("nan"))
                call(m, s)
                bad += (m.view(torch.int32) != m0.view(torch.int32)).sum() + (s.view(torch.int32) != s0.view(torch.int32)).sum()
            res["determinism"][f"{arm}_T{t}"] = {"runs": args.det_runs, "differing_elements": int(bad)}
        m40 = torch.empty(40, t, 24, device="cuda")
        s40 = torch.empty(40, t, device="cuda")
        stock_gemm(vdg, x, fn, m40, s40, 40)
        bad = torch.zeros((), dtype=torch.int64, device="cuda")
        m = torch.empty_like(m40)
        s = torch.empty_like(s40)
        for _ in range(args.det_runs):
            stock_gemm(vdg, x, fn, m, s, 40)
            bad += (m.view(torch.int32) != m40.view(torch.int32)).sum() + (s.view(torch.int32) != s40.view(torch.int32)).sum()
        res["determinism"][f"stock_dg40_T{t}"] = {"runs": args.det_runs, "differing_elements": int(bad)}
        # full stock pre (GEMM + TileLang fused norm), 16 and 40 splits
        residual = x.view(t, C.HC, C.HIDDEN)
        pre_mix = torch.softmax(torch.randn(t, C.HC, device="cuda"), -1).float().contiguous()
        stock_splits = mhc_wu.compute_mhc_pre_num_splits
        try:
            for sp in (16, 40):
                mhc_wu.compute_mhc_pre_num_splits = lambda _k, _t, sp=sp: sp

                def pre():
                    return mhc_tl.mhc_pre_delayed_tilelang(
                        residual, fn, scale_base["layers.20.hc_ffn_scale"], scale_base["layers.20.hc_ffn_base"],
                        1e-20, 1e-6, 1e-6, 2.0, 20, pre_mix=pre_mix, norm_weight=norm_w, norm_eps=1e-20)

                ref = [o.clone() for o in pre()]
                bad = torch.zeros((), dtype=torch.int64, device="cuda")
                for _ in range(args.det_runs):
                    out = pre()
                    for o, r in zip(out, ref):
                        bad += (o.contiguous().view(torch.int16 if o.dtype == torch.bfloat16 else torch.int32)
                                != r.contiguous().view(torch.int16 if r.dtype == torch.bfloat16 else torch.int32)).sum()
                res["determinism"][f"stock_pre_s{sp}_T{t}"] = {"runs": args.det_runs, "differing_elements": int(bad)}
        finally:
            mhc_wu.compute_mhc_pre_num_splits = stock_splits
    print(json.dumps(res["determinism"]), flush=True)

    # CUDA graph capture + replay equality (static buffers, capture sizes)
    for t in (1, 3, 4, 6, 8):
        x = C.make_x("emb", t, C.K_FULL, seed=99 + t, emb=emb)
        for arm in ("pk",):
            m_e = torch.empty(16, t, 24, device="cuda")
            s_e = torch.empty(16, t, device="cuda")
            m_g = torch.empty_like(m_e)
            s_g = torch.empty_like(s_e)
            call = lambda m, s: dk.gemm_pk(x, packed[name], m, s)  # noqa: E731
            call(m_e, s_e)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                call(m_g, s_g)
            torch.cuda.current_stream().wait_stream(st)
            with torch.cuda.graph(g):
                call(m_g, s_g)
            bad = 0
            for _ in range(100):
                m_g.fill_(float("nan"))
                g.replay()
                torch.cuda.synchronize()
                bad += int(not (C.bitwise_equal(m_g, m_e) and C.bitwise_equal(s_g, s_e)))
            res["graph"][f"{arm}_T{t}"] = {"replays": 100, "mismatching_replays": bad}
    print(json.dumps(res["graph"]), flush=True)
    return res


def timing(args) -> dict:
    from vllm.utils import deep_gemm as vdg

    vdg._lazy_init()
    fns = C.real_fns()
    emb = C.embeddings(512)
    dk = mhc_det.DetKernels()
    packed = {n: mhc_det.pack_fn(f) for n, f in fns}
    timer = C.Timer()
    res: dict = {"single": [], "sweep86": []}
    name, fn = next((n, f) for n, f in fns if n == "layers.20.hc_ffn_fn")
    bname, bfn = fns[-1]
    shapes = [(t, C.K_FULL, name, fn) for t in (1, 3, 4, 6, 8, 16)] + [(4, 5120, bname, bfn)]
    for t, k, fname, f in shapes:
        x_src = C.make_x("emb", t, k, seed=5 + t, emb=emb)
        x = x_src.clone()
        outs = {sp: (torch.empty(sp, t, 24, device="cuda"), torch.empty(sp, t, device="cuda")) for sp in (16, 40)}
        arms = {
            "stock_dg16": lambda: stock_gemm(vdg, x, f, *outs[16], 16),
            "det_pk": lambda: dk.gemm_pk(x, packed[fname], *outs[16]),
        }
        if k == C.K_FULL:
            arms["stock_dg40"] = lambda: stock_gemm(vdg, x, f, *outs[40], 40)
        fn_bytes = {"stock_dg16": 24 * k * 4, "stock_dg40": 24 * k * 4, "det_pk": 24 * k * 2.5}
        for cold in (True, False):
            r = timer.run(arms, iters=args.iters, warmup=20, cold=cold, pre=(lambda: x.copy_(x_src)) if cold else None)
            for arm, st in r.items():
                nbytes = fn_bytes[arm] + t * k * 2 + (40 if arm == "stock_dg40" else 16) * t * 25 * 4
                row = {"T": t, "K": k, "arm": arm, "l2": "cold" if cold else "warm", **st,
                       "bytes": int(nbytes), "GBps_median": round(C.gbps(nbytes, st["median_us"]), 1),
                       "pct_of_250": round(100 * C.gbps(nbytes, st["median_us"]) / C.PEAK_GBPS, 1)}
                res["single"].append(row)
                print(json.dumps(row), flush=True)
    # Serve-like: 86 real decode fn matrices (169 MB > L2, so every call streams cold), each call
    # preceded by the kernel that writes its activation (a copy, like mhc_post writing the new
    # residual). Per-call = (t[copy; gemm] - t[copy]) / 86 from alternating graph replays.
    dec = [(n, f) for n, f in fns if f.shape[1] == C.K_FULL]
    for t in (1, 3, 4, 6, 8, 16):
        xsrc = [C.make_x("emb", t, C.K_FULL, seed=300 + i, emb=emb) for i in range(len(dec))]
        xs = [x.clone() for x in xsrc]
        m = torch.empty(16, t, 24, device="cuda")
        s = torch.empty(16, t, device="cuda")
        base = lambda c: xs[c].copy_(xsrc[c])  # noqa: E731
        arms = {
            "stock_dg16": lambda c: stock_gemm(vdg, xs[c], dec[c][1], m, s, 16),
            "det_pk": lambda c: dk.gemm_pk(xs[c], packed[dec[c][0]], m, s),
        }
        for arm, call in arms.items():
            st = C.graph_delta_time(call, base, reps=args.iters, calls=len(dec))
            nbytes = (24 * C.K_FULL * (2.5 if arm == "det_pk" else 4)) + t * C.K_FULL * 2 + 16 * t * 25 * 4
            row = {"T": t, "arm": arm, "calls": len(dec), **st,
                   "GBps_median": round(C.gbps(nbytes, st["median_us"]), 1),
                   "pct_of_250": round(100 * C.gbps(nbytes, st["median_us"]) / C.PEAK_GBPS, 1),
                   "ms_per_step_86": round(st["median_us"] * 86 / 1000, 3)}
            res["sweep86"].append(row)
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
    res["device"] = torch.cuda.get_device_name()
    with open(f"{OUT}/gemm_{args.mode}{args.tag}.json", "w") as fh:
        json.dump(res, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
