#!/usr/bin/env python3
"""The MoE-AR window with a REAL NCCL all-reduce: does an L2 prefetch still pay?

k3 comm fix pass. l2_prefetch_window.py stood the AR in with a 5-CTA clock spin that
touches no memory. The serve's AR is NCCL LL over the net transport with GDR off: LL
lines, flags and the NIC's DMA all live in host memory (the same LPDDR5x on GB10), and
the proxy threads poll it. The review measured the shipped burst prefetch slowing that
AR by ~19 us (rev_nccl_pf.py). This harness measures prefetch engines and
placements against the real AR, on one GB10:

  two_rank_mps.sh runs two ranks on this GPU under MPS; NCCL_HOSTID differs per rank,
  so NCCL connects them through NET/IB (RoCE loopback on the serve's HCA), with the
  serve's NCCL env. Rank 1 replays a graph holding only the matching AR per arm (it
  arrives first and waits, like the serve's early rank: rank 0 is late on 72% of the
  serve's all-reduces); rank 0 replays, per arm:

    2x-L2 streaming read (the rest of the model: normal loads, L2 cold) ;
    [warm: budget read into L2] ; e0 ; [pre: prefetch on the stream] ; e_pf ;
    p2b stand-in (m x 6 x 4.42 MB read with ld.global.cs = evict-first, like p2b's trellis
    loads; 48 CTAs, not cooperative: the real p2b cannot share the GPU with rank 1's
    resident AR kernel, see p2b_prefetch_probe.py for the real kernel) ; e_p2b ;
    [start: fork side stream: s0 ; prefetch ; s1] ; AR ; e_ar ; [after: fork here instead] ;
    mhc_post ; e_post ; mhc_pre ; e_pre ; act quant ; e1 ; qkv_a b12x ; e2 ;
    act quant ; wq_b b12x ; e3 ; join

Arms alternate replay by replay in one process; every arm's qkv_a / wq_b outputs and
AR output are checked bitwise against arm 'none'. Arm grammar (--arms, comma list):
  none | warm/<budget> (budget read into L2 before e0: the in-situ ceiling)
  <pre|start|startj|startq|after>/<engine>/<budget>   engine: l2pf_variants.parse_engine;
                                          start forks at the AR start and joins at the end;
                                          startj joins right after the AR, before mhc_post
                                          (the lever's first join point, the next layer's
                                          entry); startq joins right after qkv_a (the lever's
                                          join point now: DeepseekV4Attention._split_qkv_and_norm)
  in/<budget> | in0                       the p2b stand-in's own CTAs issue the TMA L2 prefetch
                                          in their prologue (k_read_pf = p2b_pf_bench.py's
                                          prologue); in0 = that kernel with no ranges
  spin:<arm>                              the same with the 5-CTA spin instead of the AR
Budget: <MiB> of qkv_a (the lever's split_budget: the same fraction of weight and scale) |
full (all of qkv_a) | full+wqb:<MiB> (plus that much of wq_b). Reported per arm: spans
(median, p10, p90, mean, n), prefetch kernel time and GB/s, qkv_a / wq_b GB/s, rank 1's AR.
The review's E mechanism spans are here as 'ar' (rank 0 = the late rank's AR kernel window)
and 'ar_to_qkv_a_end' (AR start to qkv_a end).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "docker" / "patch"))

HIDDEN, HC = 5120, 4
QKV_A = (5120, 1792)  # K, N per rank (fused wq_a + wkv)
WQ_B = (1280, 16384)
EXPERT_BYTES = 3 * (HIDDEN // 16) * (1152 // 16) * 32 * 2  # one routed expert's trellis (4.42 MB)
DEFAULT_ARMS = ("none,warm/full,start/tri/5.5,start/tri/full,after/tri/5.5,start/paced:100/5.5,"
                "start/ring:6x16/5.5,start/burst:c48/5.5,pre/tri/full,pre/burst:c48/full,pre/pf:c192/full,"
                "pre/burst:c48/full+wqb:8,spin:none,spin:start/tri/5.5")
EVENTS = ("e0", "e_pf", "e_p2b", "ar", "post", "pre", "e1", "e2", "e3", "s0", "s1")
SPANS = {
    "pf": ("e0", "e_pf"), "p2b": ("e_pf", "e_p2b"), "ar": ("e_p2b", "ar"), "mhc_post": ("ar", "post"),
    "mhc_pre": ("post", "pre"), "quant": ("pre", "e1"), "window": ("e_p2b", "e1"), "qkv_a": ("e1", "e2"),
    "wq_b": ("e2", "e3"), "ar_to_qkv_a_end": ("e_p2b", "e2"), "total": ("e0", "e3"), "prefetch": ("s0", "s1"),
    "post_p2b": ("e_p2b", "e3"),
}


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
    spin = name.startswith("spin:")
    body = name[5:] if spin else name
    arm = {"name": name, "ar": "spin" if spin else "nccl", "warm": None, "place": None}
    if body == "none":
        return arm
    if body == "in0":
        arm.update(place="in", budget=None)
        return arm
    parts = body.split("/")
    if parts[0] == "in" and len(parts) == 2:
        arm.update(place="in", budget=parse_budget(parts[1]))
        return arm
    if parts[0] == "warm" and len(parts) == 2:
        arm["warm"] = parse_budget(parts[1])
        return arm
    if len(parts) != 3 or parts[0] not in ("pre", "start", "startj", "startq", "after"):
        raise ValueError(name)
    import l2pf_variants

    arm.update(place=parts[0], engine=l2pf_variants.parse_engine(parts[1]), budget=parse_budget(parts[2]))
    return arm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--master", default="127.0.0.1:29711")
    ap.add_argument("--arms", default=DEFAULT_ARMS)
    ap.add_argument("--m", type=int, nargs="+", default=[4])
    ap.add_argument("--replays", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--ar-us", type=float, default=22.7, help="spin stand-in length (spin: arms)")
    ap.add_argument("--sm-mhz", type=float, default=2190.0)
    ap.add_argument("--build-dir", default="/repo/kernel_study/comm/.l2pf_var_build")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--no-p2b", action="store_true", help="drop the p2b stand-in (the review's harness shape)")
    ap.add_argument("--hang-s", type=float, default=150.0,
                    help="watchdog: exit(3) when no progress for this long (a dead peer leaves the other "
                         "rank spinning inside an NCCL kernel forever)")
    ap.add_argument("--json")
    args = ap.parse_args()

    import l2pf_variants

    if args.compile_only:
        l2pf_variants.build(args.build_dir)
        print("compiled", args.build_dir)
        return 0

    import torch
    import torch.distributed as dist
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    arms = [parse_arm(a) for a in args.arms.split(",")]
    n_nccl = sum(a["ar"] == "nccl" for a in arms)
    torch.cuda.set_device(0)
    dev = torch.device("cuda:0")
    progress = {"t": time.monotonic(), "what": "start"}

    def tick(what):
        progress.update(t=time.monotonic(), what=what)

    def watchdog():
        while True:
            time.sleep(1.0)
            if time.monotonic() - progress["t"] > args.hang_s:
                print(f"HANG: rank {args.rank} no progress for {args.hang_s}s after {progress['what']}", flush=True)
                os._exit(3)

    threading.Thread(target=watchdog, daemon=True).start()
    if args.rank == 0:  # every engine must launch (eager and captured) before any collective exists
        import l2pf_kernel

        ext0 = l2pf_variants.build(args.build_dir)
        launch0 = l2pf_variants.launcher(ext0, l2pf_kernel.launcher(torch), args.sm_mhz)
        scratch = torch.zeros(1 << 20, dtype=torch.uint8, device=dev)
        for a in arms:
            if a["place"] in ("pre", "start", "startj", "startq", "after"):
                launch0(a["engine"], scratch, scratch.numel())
        torch.cuda.synchronize()
        tick("engine pre-validation")
    dist.init_process_group("gloo", init_method=f"tcp://{args.master}", rank=args.rank, world_size=2,
                            timeout=timedelta(seconds=args.hang_s))
    comm = PyNcclCommunicator(group=dist.group.WORLD, device=dev)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    results = {}

    if args.rank == 1:
        for m in args.m:
            x_in = torch.full((m, HIDDEN), 1.0, device=dev, dtype=torch.bfloat16)
            x_out = torch.empty_like(x_in)
            ea, eb = (torch.cuda.Event(enable_timing=True, external=True) for _ in range(2))
            for _ in range(2 * n_nccl):  # rank 0's eager warm-up calls
                comm.all_reduce(x_in, x_out)
            torch.cuda.synchronize()
            tick(f"rank 1 eager warm-up m={m}")
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream):
                ea.record()
                comm.all_reduce(x_in, x_out)
                eb.record()
            torch.cuda.synchronize()
            dist.barrier()
            per_arm = {a["name"]: [] for a in arms if a["ar"] == "nccl"}
            for i in range(args.warmup + args.replays):
                for a in arms:
                    if a["ar"] != "nccl":
                        continue
                    g.replay()
                    torch.cuda.synchronize()
                    tick(f"rank 1 replay {i} m={m}")
                    if i >= args.warmup:
                        per_arm[a["name"]].append(ea.elapsed_time(eb) * 1e3)
            dist.barrier()
            obj = [{k: stats(v) for k, v in per_arm.items()}]
            dist.broadcast_object_list(obj, src=1)
            del g
        return 0

    from vllm.model_executor.kernels.mhc import tilelang as mhc_tl
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
        swizzle_mxfp8_scale,
    )
    from vllm.utils import deep_gemm as vdg
    from vllm.utils import flashinfer as vfi

    import ar_l2_prefetch as alp
    import l2pf_kernel

    ext = l2pf_variants.build(args.build_dir)
    launch = l2pf_variants.launcher(ext, l2pf_kernel.launcher(torch), args.sm_mhz)
    vdg._lazy_init()
    props = torch.cuda.get_device_properties(0)
    l2 = int(props.L2_cache_size)
    torch.manual_seed(0)

    def mx_weight(k, n):
        w = torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02
        w8, sc = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=False)
        return w8, swizzle_mxfp8_scale(sc, M=n, K=k).contiguous()

    wa, sa = mx_weight(*QKV_A)
    wb, sb = mx_weight(*WQ_B)
    nb = lambda t: t.numel() * t.element_size()  # noqa: E731
    qkv_bytes, wqb_bytes = nb(wa) + nb(sa), nb(wb) + nb(sb)
    standin = torch.empty(0 if args.no_p2b else max(args.m) * 6 * EXPERT_BYTES, dtype=torch.uint8, device=dev)
    k_hc, mix = HC * HIDDEN, HC * (HC + 2)
    fn = torch.randn(mix, k_hc, device=dev, dtype=torch.float32) * 0.02
    hc_scale = torch.rand(3, device=dev, dtype=torch.float32) + 0.5
    hc_base = torch.randn(mix, device=dev, dtype=torch.float32) * 0.1
    norm_w = (torch.rand(HIDDEN, device=dev) + 0.5).to(torch.bfloat16)
    flush = torch.empty(2 * l2, dtype=torch.uint8, device=dev)
    sink = torch.zeros(1, dtype=torch.int32, device=dev)
    side = torch.cuda.Stream()

    def plan_for(b):
        if b["qkv"] == "full":
            plan = [(wa, nb(wa) & ~127), (sa, nb(sa) & ~127)]
        else:
            plan = [(t, n & ~127) for t, n in zip((wa, sa), alp.split_budget([nb(wa), nb(sa)], int(b["qkv"] * 2**20)))
                    if n > 0]
        if b["wqb_mib"] > 0:
            plan += [(t, n & ~127) for t, n in zip((wb, sb), alp.split_budget([nb(wb), nb(sb)],
                                                                          int(b["wqb_mib"] * 2**20))) if n > 0]
        return plan

    for m in args.m:
        x_in = torch.full((m, HIDDEN), 1.0, device=dev, dtype=torch.bfloat16)
        x_out = torch.empty_like(x_in)
        residual = torch.randn(m, HC, HIDDEN, device=dev, dtype=torch.bfloat16)
        pre_mix = torch.softmax(torch.randn(m, HC, device=dev), -1).float().contiguous()
        post_mix, res_mix, _, _ = mhc_tl.mhc_pre_delayed_tilelang(
            residual, fn, hc_scale, hc_base, 1e-6, 1e-6, 1e-6, 2.0, 20, pre_mix=pre_mix,
            norm_weight=norm_w, norm_eps=1e-6)
        xb_in = torch.randn(m, WQ_B[0], device=dev, dtype=torch.bfloat16)
        ev = {a["name"]: {e: torch.cuda.Event(enable_timing=True, external=True) for e in EVENTS} for a in arms}
        outs = {}

        def body(a):
            plan = plan_for(a["budget"]) if a["place"] and a.get("budget") else None

            def fork(e):
                cur = torch.cuda.current_stream()
                side.wait_stream(cur)
                with torch.cuda.stream(side):
                    e["s0"].record()
                    for t, n in plan:
                        launch(a["engine"], t, n)
                    e["s1"].record()

            def b():
                e = ev[a["name"]]
                cur = torch.cuda.current_stream()
                ext.k_read(flush, flush.numel(), 48, 0, sink)
                for t, n in plan_for(a["warm"]) if a["warm"] else ():
                    ext.k_read(t, n, 48, 0, sink)
                e["e0"].record()
                if a["place"] == "pre":
                    for t, n in plan:
                        launch(a["engine"], t, n)
                e["e_pf"].record()
                if standin.numel() and a["place"] == "in":
                    ext.k_read_pf(standin, m * 6 * EXPERT_BYTES, 48, sink, [t.data_ptr() for t, _ in plan or ()],
                                  [n for _, n in plan or ()])
                elif standin.numel():
                    ext.k_read(standin, m * 6 * EXPERT_BYTES, 48, 2, sink)
                e["e_p2b"].record()
                if a["place"] in ("start", "startj", "startq"):
                    fork(e)
                if a["ar"] == "nccl":
                    comm.all_reduce(x_in, x_out)
                else:
                    ext.k_spin(args.ar_us * args.sm_mhz, 5)
                    x_out.copy_(x_in)
                e["ar"].record()
                if a["place"] == "startj":  # the lever joins here: the next layer's entry
                    cur.wait_stream(side)
                if a["place"] == "after":
                    fork(e)
                res2 = mhc_tl.mhc_post_tilelang(x_out, residual, post_mix, res_mix)
                e["post"].record()
                _, _, x, _ = mhc_tl.mhc_pre_delayed_tilelang(
                    res2, fn, hc_scale, hc_base, 1e-6, 1e-6, 1e-6, 2.0, 20, pre_mix=pre_mix,
                    norm_weight=norm_w, norm_eps=1e-6)
                e["pre"].record()
                q, s = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
                e["e1"].record()
                oa = vfi.mm_mxfp8(q, wa.t(), s, sa, out_dtype=torch.bfloat16, backend="auto")
                e["e2"].record()
                if a["place"] == "startq":  # the lever joins here now: after the next layer's qkv_a
                    cur.wait_stream(side)
                qb, sbq = mxfp8_e4m3_quantize(xb_in, is_sf_swizzled_layout=True)
                ob = vfi.mm_mxfp8(qb, wb.t(), sbq, sb, out_dtype=torch.bfloat16, backend="auto")
                e["e3"].record()
                if a["place"] in ("start", "after"):  # join the side stream ('pre' never forks)
                    cur.wait_stream(side)
                outs[a["name"]] = (oa, ob, x_out.clone() if a["ar"] == "nccl" else None)

            return b

        graphs = {}
        for a in arms:
            b = body(a)
            b()
            b()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream):
                b()
            torch.cuda.synchronize()
            graphs[a["name"]] = g
            tick(f"captured {a['name']} m={m}")
        dist.barrier()
        res = {a["name"]: {k: [] for k in (*SPANS, "decision")} for a in arms}
        for i in range(args.warmup + args.replays):
            for a in arms:
                graphs[a["name"]].replay()
                torch.cuda.synchronize()
                tick(f"replay {i} {a['name']} m={m}")
                if i < args.warmup:
                    continue
                e = ev[a["name"]]
                for k, (x, y) in SPANS.items():
                    if k == "prefetch" and a["place"] not in ("start", "startj", "startq", "after"):
                        continue
                    res[a["name"]][k].append(e[x].elapsed_time(e[y]) * 1e3)
                # per replay: everything this arm adds except the p2b stand-in's own span
                res[a["name"]]["decision"].append(res[a["name"]]["pf"][-1] + res[a["name"]]["post_p2b"][-1])
        dist.barrier()
        obj = [None]
        dist.broadcast_object_list(obj, src=1)
        r1 = obj[0]
        def ref_of(name):  # spin arms feed mHC the un-reduced input: compare them with spin:none
            key = "spin:none" if name.startswith("spin:") and "spin:none" in outs else "none"
            return outs[key] if key in outs else outs[arms[0]["name"]]
        summary = {}
        for a in arms:
            name = a["name"]
            s = {k: stats(v) for k, v in res[name].items() if v}
            o = outs[name]
            ref = ref_of(name)
            s["gemm_bitwise_equal_to_none"] = bool(torch.equal(o[0], ref[0]) and torch.equal(o[1], ref[1]))
            if o[2] is not None and ref[2] is not None:
                s["ar_out_bitwise_equal_to_none"] = bool(torch.equal(o[2], ref[2]))
            s["qkv_a_gbps"] = round(qkv_bytes / (s["qkv_a"]["median"] * 1e3), 1)
            s["wq_b_gbps"] = round(wqb_bytes / (s["wq_b"]["median"] * 1e3), 1)
            if standin.numel():
                s["p2b_standin_gbps"] = round(m * 6 * EXPERT_BYTES / (s["p2b"]["median"] * 1e3), 1)
            if a["place"] and a.get("budget"):
                nbytes = sum(n for _, n in plan_for(a["budget"]))
                span = s["prefetch"] if a["place"] in ("start", "startj", "startq", "after") else s["pf"]
                s["prefetch_bytes"] = nbytes
                s["prefetch_kernel_us"] = span["median"]
            if name in r1:
                s["rank1_ar"] = r1[name]
            summary[name] = s
        a_by_name = {a["name"]: a for a in arms}
        for name, s in summary.items():  # deltas vs the same-AR-kind baseline (none / spin:none)
            b = summary.get("spin:none" if name.startswith("spin:") else "none")
            if b is None:
                continue
            for k in ("pf", "p2b", "ar", "mhc_post", "mhc_pre", "qkv_a", "wq_b", "ar_to_qkv_a_end", "total"):
                s[f"d_{k}"] = round(s[k]["median"] - b[k]["median"], 2)
            # The decision metric: per replay, pf + (p2b end -> wq_b end), i.e. everything but the p2b
            # stand-in's own span (its speed varies with the arm's slot in the replay cycle in this
            # one-GPU emulation, before any fork); median per arm, minus the baseline's median.
            # Not for 'in' arms: their prefetch cost is inside the stand-in span (read d_total, which
            # carries the slot noise, or the real-p2b probe p2b_prefetch_probe.py).
            s["d_net_excl_p2b"] = (None if a_by_name[name]["place"] == "in"
                                   else round(s["decision"]["median"] - b["decision"]["median"], 2))
        results[str(m)] = summary
        for name, s in summary.items():
            print(f"m={m} {name:28s} pf {s['pf']['median']:5.1f} p2b {s['p2b']['median']:6.1f} "
                  f"ar {s['ar']['median']:6.1f} [{s['ar']['p10']:.1f}, {s['ar']['p90']:.1f}] "
                  f"mhc {s['mhc_post']['median'] + s['mhc_pre']['median']:5.1f} qkv_a {s['qkv_a']['median']:5.1f} "
                  f"wq_b {s['wq_b']['median']:6.1f} total {s['total']['median']:7.1f} "
                  f"[{s['total']['p10']:.1f}, {s['total']['p90']:.1f}] | net {s.get('d_net_excl_p2b') or 0:+6.1f} = "
                  f"pf {s.get('d_pf', 0):+5.1f} p2b {s.get('d_p2b', 0):+5.1f} ar {s.get('d_ar', 0):+5.1f} "
                  f"mhc {s.get('d_mhc_post', 0) + s.get('d_mhc_pre', 0):+5.1f} qkv_a {s.get('d_qkv_a', 0):+5.1f} "
                  f"wq_b {s.get('d_wq_b', 0):+5.1f} eq {s['gemm_bitwise_equal_to_none']}", flush=True)
        del graphs

    out = {"what": "MoE-AR window with a real NCCL LL all-reduce (2 ranks on one GB10 under MPS, NET/IB RoCE "
                   "loopback, serve NCCL env) vs L2 prefetch engines and placements",
           "device": props.name, "l2_bytes": l2, "qkv_a_bytes": qkv_bytes, "wq_b_bytes": wqb_bytes,
           "p2b_standin": None if args.no_p2b else "m x 6 x %d B, ld.global.cs, 48 CTAs" % EXPERT_BYTES,
           "replays": args.replays, "warmup": args.warmup,
           "arms": [{k: v for k, v in a.items()} for a in arms],
           "nccl": comm.nccl.ncclGetVersion(),
           "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(("NCCL_", "DSV41_PM", "CUDA_MPS"))},
           "results": results, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0 if all(s["gemm_bitwise_equal_to_none"] for r in results.values() for s in r.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
