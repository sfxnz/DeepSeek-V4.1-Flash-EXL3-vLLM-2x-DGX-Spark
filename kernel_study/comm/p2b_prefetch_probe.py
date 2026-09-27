#!/usr/bin/env python3
"""p2b window: warm the next layer's qkv_a (and part of wq_b) with a prefetch issued right
before the routed MoE (p2b), instead of inside the latency-bound MoE-AR window. Single GPU.

Why (k3 comm fix pass). With a real NCCL LL all-reduce the AR-window prefetch slows the AR
(or, forked after it, mHC) by about what it saves on qkv_a (review; ar_window_nccl.py). p2b
streams ~106 MB per layer at m=4 with ld.global.cs (evict-first) at ~78% of DRAM peak, so it
has slack, and lines prefetched with normal priority may survive its stream. Two facts
constrain the design, both measured here on the served p2b kernel (SORT=0,
kernel_study/p2b_coop bench build = the canonical-e13 chain, set_coop(0)):
  1. p2b is a cooperative launch that fills every SM's register file (4 x 256 threads x 64
     registers): nothing runs beside it, and it cannot start while any other CTA is
     resident. A prefetch kernel therefore runs before p2b, and its own run time is on the
     critical path. A 1-CTA TMA prefetch stays resident until its requests are taken
     (~transfer time), so the engines here also spread the issue over many CTAs.
  2. p2b's fp32 slot sum uses atomicAdd: its output is not bitwise reproducible with
     normalized routing weights. With one-hot routing weights every slot but one adds an
     exact zero, so arms are compared bit for bit in a one-hot pass.

Per replay, one CUDA graph per arm (arms alternate replay by replay; each replay index uses
the same census routing for every arm, copied into the static ids first):
  2x-L2 streaming read (the rest of the model; normal loads) ; [warm: budget read in] ;
  e0 ; [pre: prefetch of the budget] ; e_pf ; p2b ; e_p2b ; AR stand-in (5-CTA clock spin,
  no memory) ; mhc_post ; mhc_pre ; act quant ; e1 ; qkv_a b12x ; e2 ; act quant ; wq_b b12x ;
  e3 ; [demote: applypriority evict_normal over the budget, +last engines only]
Spans: pf e0-e_pf, p2b e_pf-e_p2b, window e_p2b-e1, qkv_a e1-e2, wq_b e2-e3, total e0-e3.

Arms (--arms, comma list): none | warm/<budget> | pre/<engine>/<budget> | in0 | in/<budget>
  pre:    a separate prefetch kernel launched right before p2b (engine: see
          l2pf_variants.parse_engine; tri = the lever's Triton kernel, burst:c<N> TMA bulk
          prefetch from N CTAs, pf:c<N> per-thread prefetch.global.L2, +last = evict_last)
  in:     p2b itself issues the prefetch in its prologue (p2b_pf_bench.py: the same kernel +
          a TMA prefetch prologue, bench build); in0 = that kernel with no prefetch ranges
  budget: full (all of qkv_a: weight + scale) | full+wqb:<MiB> (plus that much of wq_b,
          the same fraction of its weight and scale) | <MiB> (of qkv_a, as the lever splits it)
Weights: --pack DIR --layer L loads the real rank-0 qkv_a (wq_a | wkv) and wq_b rows 0..16383
(e4m3 + 32x32-block e8m0 scales, expanded per row as vLLM's loader does); else random MXFP8.
Trellis is random (same bytes and layout as the pack; p2b's time does not depend on the
values), routing is the s10 census (m > 4: two independent 4-token windows, as c=2 decode).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "docker" / "patch"))
sys.path.insert(0, str(ROOT / "kernel_study" / "p2b_coop"))
os.environ.pop("DSV41_P2B_SRC_SORT", None)
os.environ.pop("DSV41_P2B_COOP", None)

HIDDEN, HC = 5120, 4
QKV_A = (5120, 1792)  # K, N (fused wq_a + wkv, replicated at TP=2)
WQ_B = (1280, 16384)  # K, N per rank
EXPERT_BYTES = 3 * (HIDDEN // 16) * (1152 // 16) * 32 * 2  # gate + up + down trellis, one expert
DEFAULT_ARMS = ("none,warm/full,in0,in/full,in/full+wqb:4,in/full+wqb:8,in/full+wqb:12,pre/tri/full,"
                "pre/burst:c48/full")
EVENTS = ("e0", "e_pf", "e_p2b", "e1", "e2", "e3")
SPANS = {"pf": ("e0", "e_pf"), "p2b": ("e_pf", "e_p2b"), "window": ("e_p2b", "e1"), "qkv_a": ("e1", "e2"),
         "wq_b": ("e2", "e3"), "total": ("e0", "e3")}


def stats(xs):
    s = sorted(xs)
    q = lambda f: s[min(len(s) - 1, int(round(f * (len(s) - 1))))]  # noqa: E731
    return {"median": round(statistics.median(s), 2), "p10": round(q(0.1), 2), "p90": round(q(0.9), 2),
            "mean": round(statistics.fmean(s), 2), "n": len(s)}


def parse_budget(spec: str) -> dict:
    """'full' | 'full+wqb:<MiB>' | '<MiB>' -> {'qkv': 'full' | MiB, 'wqb_mib': float}."""
    base, _, extra = spec.partition("+")
    out = {"qkv": "full" if base == "full" else float(base), "wqb_mib": 0.0}
    if extra:
        kind, _, mib = extra.partition(":")
        if kind != "wqb" or out["qkv"] != "full":
            raise ValueError(spec)
        out["wqb_mib"] = float(mib)
    return out


def parse_arm(name: str) -> dict:
    import l2pf_variants

    if name == "none":
        return {"name": name, "place": None, "warm": None}
    parts = name.split("/")
    if parts[0] == "warm" and len(parts) == 2:
        return {"name": name, "place": None, "warm": parse_budget(parts[1])}
    if parts[0] == "pre" and len(parts) == 3:
        return {"name": name, "place": "pre", "warm": None, "engine": l2pf_variants.parse_engine(parts[1]),
                "budget": parse_budget(parts[2])}
    if name == "in0":
        return {"name": name, "place": "in", "warm": None, "budget": None}
    if parts[0] == "in" and len(parts) == 2:
        return {"name": name, "place": "in", "warm": None, "budget": parse_budget(parts[1])}
    raise ValueError(name)


def build_p2b(build_dir: str):
    import make_bench
    from torch.utils.cpp_extension import load

    make_bench.main()
    exl = "/usr/local/lib/python3.12/dist-packages/exllamav3/exllamav3_ext"
    os.makedirs(build_dir, exist_ok=True)
    return load(name="p2b_coop_bench", sources=[str(ROOT / "kernel_study/p2b_coop/build/bench_coop.cu")],
                extra_include_paths=[exl, os.path.join(exl, "quant")], extra_cuda_cflags=["-O3", "-std=c++17"],
                build_directory=build_dir, verbose=False)


def build_p2b_pf(build_dir: str):
    import p2b_pf_bench
    from torch.utils.cpp_extension import load

    p2b_pf_bench.main()
    exl = "/usr/local/lib/python3.12/dist-packages/exllamav3/exllamav3_ext"
    os.makedirs(build_dir, exist_ok=True)
    return load(name="p2b_pf_bench", sources=[str(p2b_pf_bench.OUT)],
                extra_include_paths=[exl, os.path.join(exl, "quant")], extra_cuda_cflags=["-O3", "-std=c++17"],
                build_directory=build_dir, verbose=False)


def pack_weight(pack: str, names: list[str], rows: tuple[int, int] | None, dev):
    """Rank-0 e4m3 weight [N, K] and per-row uint8 scale [N, K/32] (32x32 checkpoint blocks)."""
    import torch
    from safetensors import safe_open

    index = json.load(open(os.path.join(pack, "model.safetensors.index.json")))["weight_map"]
    ws, ss = [], []
    for name in names:
        with safe_open(os.path.join(pack, index[name + ".weight"]), framework="pt") as fh:
            sl = fh.get_slice(name + ".weight")
            ws.append(sl[slice(*rows) if rows else slice(None), :])
        with safe_open(os.path.join(pack, index[name + ".scale"]), framework="pt") as fh:
            sl = fh.get_slice(name + ".scale")
            srows = slice(rows[0] // 32, -(-rows[1] // 32)) if rows else slice(None)
            ss.append(sl[srows, :].view(torch.uint8).repeat_interleave(32, dim=0))
    return torch.cat(ws).contiguous().to(dev), torch.cat(ss).contiguous().to(dev)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default=DEFAULT_ARMS)
    ap.add_argument("--m", type=int, nargs="+", default=[1, 3, 4, 6, 8])
    ap.add_argument("--replays", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--ar-us", type=float, default=22.7)
    ap.add_argument("--sm-mhz", type=float, default=2190.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pack", default="", help="pack snapshot dir: real rank-0 qkv_a / wq_b weights")
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--var-build-dir", default="/repo/kernel_study/comm/.l2pf_var_build")
    ap.add_argument("--p2b-build-dir", default="/repo/kernel_study/comm/.p2b_build")
    ap.add_argument("--p2b-pf-build-dir", default="/repo/kernel_study/comm/.p2b_pf_build")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--json")
    args = ap.parse_args()

    import l2pf_variants

    arms = [parse_arm(a) for a in args.arms.split(",")]
    if arms[0]["name"] != "none":
        raise SystemExit("the first arm must be 'none' (the reference)")
    if args.compile_only:
        l2pf_variants.build(args.var_build_dir)
        build_p2b(args.p2b_build_dir)
        build_p2b_pf(args.p2b_pf_build_dir)
        print("compiled")
        return 0

    import torch
    from driver import CENSUS, INTER, SWIGLU_LIMIT, TOPK, Routings, make_weights
    from vllm.model_executor.kernels.mhc import tilelang as mhc_tl
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
        swizzle_mxfp8_scale,
    )
    from vllm.utils import deep_gemm as vdg
    from vllm.utils import flashinfer as vfi

    import ar_l2_prefetch as alp
    import l2pf_kernel

    ext = l2pf_variants.build(args.var_build_dir)
    launch = l2pf_variants.launcher(ext, l2pf_kernel.launcher(torch), args.sm_mhz)
    p2b = build_p2b(args.p2b_build_dir)
    p2b.set_coop(0)
    p2b_pf = build_p2b_pf(args.p2b_pf_build_dir) if any(a["place"] == "in" for a in arms) else None
    if p2b_pf is not None:
        p2b_pf.set_coop(0)
        p2b_pf.set_prefetch([], [])
    vdg._lazy_init()
    dev = "cuda"
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    props = torch.cuda.get_device_properties(0)
    l2 = int(props.L2_cache_size)
    trellis, tables = make_weights(dev)  # keep the tensors: the tables only hold their pointers

    def mx_weight(k, n):
        w = torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02
        return mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=False)

    if args.pack:
        L = f"layers.{args.layer}.attn"
        wa, sca = pack_weight(args.pack, [f"{L}.wq_a", f"{L}.wkv"], None, dev)
        wb, scb = pack_weight(args.pack, [f"{L}.wq_b"], (0, WQ_B[1]), dev)
        weights_src = f"pack {args.pack} layer {args.layer} rank 0"
    else:
        wa, sca = mx_weight(*QKV_A)
        wb, scb = mx_weight(*WQ_B)
        weights_src = "random bf16 -> mxfp8"
    assert tuple(wa.shape) == (QKV_A[1], QKV_A[0]) and tuple(wb.shape) == (WQ_B[1], WQ_B[0]), (wa.shape, wb.shape)
    sa = swizzle_mxfp8_scale(sca, M=QKV_A[1], K=QKV_A[0]).contiguous()
    sb = swizzle_mxfp8_scale(scb, M=WQ_B[1], K=WQ_B[0]).contiguous()
    nbytes = lambda t: t.numel() * t.element_size()  # noqa: E731
    qkv_bytes, wqb_bytes = nbytes(wa) + nbytes(sa), nbytes(wb) + nbytes(sb)

    def plan_for(b):
        if b["qkv"] == "full":
            plan = [(wa, nbytes(wa) & ~127), (sa, nbytes(sa) & ~127)]
        else:
            plan = [(t, n & ~127) for t, n in zip((wa, sa), alp.split_budget([nbytes(wa), nbytes(sa)],
                                                                          int(b["qkv"] * 2**20))) if n > 0]
        if b["wqb_mib"] > 0:
            plan += [(t, n & ~127) for t, n in zip((wb, sb), alp.split_budget([nbytes(wb), nbytes(sb)],
                                                                          int(b["wqb_mib"] * 2**20))) if n > 0]
        return plan

    k_hc, mix = HC * HIDDEN, HC * (HC + 2)
    fn = torch.randn(mix, k_hc, device=dev, dtype=torch.float32) * 0.02
    hc_scale = torch.rand(3, device=dev, dtype=torch.float32) + 0.5
    hc_base = torch.randn(mix, device=dev, dtype=torch.float32) * 0.1
    norm_w = (torch.rand(HIDDEN, device=dev) + 0.5).to(torch.bfloat16)
    flush = torch.empty(2 * l2, dtype=torch.uint8, device=dev)
    sink = torch.zeros(1, dtype=torch.int32, device=dev)
    results = {}
    all_ok = True
    for m in args.m:
        routes = Routings(min(m, 4), CENSUS, rng)
        x = torch.randn(m, HIDDEN, dtype=torch.half, device=dev)
        rw_full = torch.rand(m, TOPK, device=dev)
        rw_full = (rw_full / rw_full.sum(dim=1, keepdim=True)).half()
        rw_onehot = torch.zeros(m, TOPK, device=dev, dtype=torch.half)
        rw_onehot[:, 0] = 1.0
        rw = rw_full.clone()
        ids = torch.zeros(m, TOPK, dtype=torch.int32, device=dev)

        def draw():
            rows = routes.draw("census")
            while len(rows) < m:  # m > 4: two independent 4-token windows (two sequences)
                rows = rows + routes.draw("census")
            return rows[:m]

        ids.copy_(torch.tensor(draw(), dtype=torch.int32, device=dev))
        out = torch.empty_like(x)
        residual = torch.randn(m, HC, HIDDEN, device=dev, dtype=torch.bfloat16)
        pre_mix = torch.softmax(torch.randn(m, HC, device=dev), -1).float().contiguous()
        post_mix, res_mix, _, _ = mhc_tl.mhc_pre_delayed_tilelang(
            residual, fn, hc_scale, hc_base, 1e-6, 1e-6, 1e-6, 2.0, 20, pre_mix=pre_mix,
            norm_weight=norm_w, norm_eps=1e-6)
        x_ar = torch.randn(m, HIDDEN, device=dev, dtype=torch.bfloat16)
        xb_in = torch.randn(m, WQ_B[0], device=dev, dtype=torch.bfloat16)
        ev = {a["name"]: {e: torch.cuda.Event(enable_timing=True, external=True) for e in EVENTS} for a in arms}
        outs = {}

        def body(a, timed=True):
            plan = plan_for(a["budget"]) if a["place"] and a["budget"] else None
            warm = plan_for(a["warm"]) if a["warm"] else None

            def b():
                e = ev[a["name"]]
                rec = (lambda k: e[k].record()) if timed else (lambda k: None)  # noqa: E731
                ext.k_read(flush, flush.numel(), 48, 0, sink)
                for t, n in warm or ():
                    ext.k_read(t, n, 48, 0, sink)
                rec("e0")
                if a["place"] == "pre":
                    for t, n in plan:
                        launch(a["engine"], t, n)
                rec("e_pf")
                if a["place"] == "in":  # the ranges are kernel arguments, read at this launch
                    p2b_pf.set_prefetch([t.data_ptr() for t, _ in plan or ()], [n for _, n in plan or ()])
                    p2b_pf.p2b_fused_moe(x, out, *tables, ids, rw, 2, 2, 2, True, INTER, SWIGLU_LIMIT)
                    p2b_pf.set_prefetch([], [])
                else:
                    p2b.p2b_fused_moe(x, out, *tables, ids, rw, 2, 2, 2, True, INTER, SWIGLU_LIMIT)
                rec("e_p2b")
                ext.k_spin(args.ar_us * args.sm_mhz, 5)
                res2 = mhc_tl.mhc_post_tilelang(x_ar, residual, post_mix, res_mix)
                _, _, h, _ = mhc_tl.mhc_pre_delayed_tilelang(
                    res2, fn, hc_scale, hc_base, 1e-6, 1e-6, 1e-6, 2.0, 20, pre_mix=pre_mix,
                    norm_weight=norm_w, norm_eps=1e-6)
                q, s = mxfp8_e4m3_quantize(h, is_sf_swizzled_layout=True)
                rec("e1")
                oa = vfi.mm_mxfp8(q, wa.t(), s, sa, out_dtype=torch.bfloat16, backend="auto")
                rec("e2")
                qb, sbq = mxfp8_e4m3_quantize(xb_in, is_sf_swizzled_layout=True)
                ob = vfi.mm_mxfp8(qb, wb.t(), sbq, sb, out_dtype=torch.bfloat16, backend="auto")
                rec("e3")
                if a["place"] == "pre" and a["engine"]["last"]:
                    for t, n in plan:
                        ext.k_demote(t, n, 48)
                return out.clone(), oa, ob

            return b

        graphs = {}
        for a in arms:
            b = body(a)
            b()
            b()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                outs[a["name"]] = b()
            torch.cuda.synchronize()
            graphs[a["name"]] = g
        res = {a["name"]: {k: [] for k in SPANS} for a in arms}
        for i in range(args.warmup + args.replays):
            ids.copy_(torch.tensor(draw(), dtype=torch.int32, device=dev))
            for a in arms:
                graphs[a["name"]].replay()
                torch.cuda.synchronize()
                if i < args.warmup:
                    continue
                e = ev[a["name"]]
                for k, (s0, s1) in SPANS.items():
                    res[a["name"]][k].append(e[s0].elapsed_time(e[s1]) * 1e3)

        # Correctness (tolerance fixed up front): qkv_a / wq_b outputs bitwise equal to 'none' in
        # every arm (deterministic GEMMs on identical inputs); p2b bitwise equal under one-hot
        # routing weights; the graph replay of every arm bitwise equal to an eager run of 'none'.
        # Normalized weights: p2b's atomic slot-sum order varies, so max |d| vs 'none' is reported
        # next to 'none' replayed twice (the kernel's own run-to-run spread), not gated.
        check = {}
        fixed = torch.tensor(draw(), dtype=torch.int32, device=dev)
        for mode, w in (("onehot", rw_onehot), ("full", rw_full)):
            rw.copy_(w)
            ids.copy_(fixed)
            got = {}
            for a in arms:
                graphs[a["name"]].replay()
                torch.cuda.synchronize()
                got[a["name"]] = tuple(t.clone() for t in outs[a["name"]])
            graphs["none"].replay()
            torch.cuda.synchronize()
            again = tuple(t.clone() for t in outs["none"])
            eager = tuple(t.clone() for t in body(arms[0], timed=False)())
            torch.cuda.synchronize()
            ref = got["none"]
            row = {"none_vs_none_p2b_max_abs": float((again[0].float() - ref[0].float()).abs().max()),
                   "graph_vs_eager_bitwise": all(torch.equal(x_, y_) for x_, y_ in zip(eager[1:], ref[1:]))
                   and (mode == "full" or torch.equal(eager[0], ref[0]))}
            for a in arms[1:]:
                o = got[a["name"]]
                row[a["name"]] = {"gemm_bitwise": bool(torch.equal(o[1], ref[1]) and torch.equal(o[2], ref[2])),
                                  "p2b_bitwise": bool(torch.equal(o[0], ref[0])),
                                  "p2b_max_abs": float((o[0].float() - ref[0].float()).abs().max())}
            check[mode] = row
        rw.copy_(rw_full)
        ok = check["onehot"]["graph_vs_eager_bitwise"] and check["full"]["graph_vs_eager_bitwise"] and all(
            v["gemm_bitwise"] and (mode == "full" or v["p2b_bitwise"])
            for mode, row in check.items() for k, v in row.items() if isinstance(v, dict))
        all_ok &= ok

        summary = {}
        for a in arms:
            s = {k: stats(v) for k, v in res[a["name"]].items()}
            s["qkv_a_gbps"] = round(qkv_bytes / (s["qkv_a"]["median"] * 1e3), 1)
            s["wq_b_gbps"] = round(wqb_bytes / (s["wq_b"]["median"] * 1e3), 1)
            s["p2b_gbps_nodedup"] = round(m * TOPK * EXPERT_BYTES / (s["p2b"]["median"] * 1e3), 1)
            if a["place"] and a["budget"]:
                s["prefetch_bytes"] = sum(n for _, n in plan_for(a["budget"]))
            summary[a["name"]] = s
        base = summary["none"]
        for a in arms:
            for k in ("pf", "p2b", "window", "qkv_a", "wq_b", "total"):
                summary[a["name"]][f"d_{k}"] = round(summary[a["name"]][k]["median"] - base[k]["median"], 2)
        results[str(m)] = {"arms": summary, "check": check, "check_ok": ok}
        for a in arms:
            s = summary[a["name"]]
            print(f"m={m} {a['name']:28s} pf {s['pf']['median']:6.1f} p2b {s['p2b']['median']:7.1f} "
                  f"[{s['p2b']['p10']:.1f}, {s['p2b']['p90']:.1f}] qkv_a {s['qkv_a']['median']:6.1f} "
                  f"wq_b {s['wq_b']['median']:6.1f} total {s['total']['median']:7.1f} "
                  f"[{s['total']['p10']:.1f}, {s['total']['p90']:.1f}] d_total {s['d_total']:+6.1f} "
                  f"(pf {s['d_pf']:+5.1f} p2b {s['d_p2b']:+5.1f} win {s['d_window']:+5.1f} "
                  f"qkv_a {s['d_qkv_a']:+5.1f} wq_b {s['d_wq_b']:+5.1f})", flush=True)
        print(f"m={m} check ok={ok} {json.dumps(check)}", flush=True)
        del graphs
    out_doc = {"what": "next-layer qkv_a (+ wq_b) L2 prefetch around the served p2b (SORT=0): a separate "
                       "kernel right before it ('pre', its time inside the span) or issued by p2b itself "
                       "('in', bench build p2b_pf_bench.py), single GPU",
               "device": props.name, "l2_bytes": l2, "weights": weights_src, "qkv_a_bytes": qkv_bytes,
               "wq_b_bytes": wqb_bytes, "expert_bytes": EXPERT_BYTES, "replays": args.replays,
               "warmup": args.warmup, "arms": [a["name"] for a in arms], "results": results,
               "p2b_occupancy_blocks_per_sm": p2b.occupancy(0), "all_checks_ok": all_ok,
               "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if args.json:
        Path(args.json).write_text(json.dumps(out_doc, indent=1) + "\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
