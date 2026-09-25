#!/usr/bin/env python3
"""DSV41_MOE_PREP_FUSED: bit-exactness, graph replay and timing of the p2b input prep.

Runs in the serving image on a GPU (patch dir mounted at /opt/dsv41-patch):

  docker run --rm --gpus all --network none -v <repo>:/repo \
      -v <repo>/docker/patch:/opt/dsv41-patch:ro --entrypoint python3 <image> \
      /repo/kernel_study/fusion_host/moe_prep_check.py

1. sweep: m in 1..8 and 64 rows x top-6; random routing (normalized, x1.5 like
   dsv4_topk) and adversarial values (invalid ids -1 / 384 / +-2^40, weights
   NaN / +-inf / -0 / fp16 overflow / fp16 subnormal, x the same in bf16);
   stock glue (the image's ops) vs the fused kernel, raw bits of all three
   outputs; int64 and int32 ids; fp32 and bf16 weights.
2. graph: the fused kernel captured in a CUDA graph, replayed after new inputs
   are copied into the static buffers; every replay must equal the eager stock
   ops bit for bit.
3. function: the image's _apply_native_fused_moe and apply_exl3_experts as
   rewritten (moe_prep_fused.patch_source / patch_experts_source) vs the stock
   functions, with a stand-in p2b op that records its inputs; recorded inputs
   and the bf16 routed output must be bitwise equal; plus all 65536 fp16 bit
   patterns: one cast to bf16 == through fp32.
4. timing: each arm captured in a CUDA graph at m = 1, 3, 4, 6, 8 (the decode
   capture sizes), replayed >= 300 times, CUDA events per replay, arms
   alternating; alone (graph launch inside the events), warm and cold L2
   (a ~90 us GPU sleep before the start event hides the graph launch, as
   inside the serve's target graph; cold writes 128 MiB first, GB10 L2 is
   24 MiB), and beside a 512 MiB device copy on another stream (the shared
   expert's GEMM saturating DRAM in the serve). The p2b epilogue (fp16 ->
   bf16) as 40 casts in one graph, launch hidden the same way.
5. sabotage: a fused result with one wrong id; the per-call verify must
   return the stock result and disarm (the next call is stock), and the
   first-call self-test must disarm.
Prints one JSON line; exit 1 on any mismatch.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import types

import torch

sys.path.insert(0, "/opt/dsv41-patch")
import moe_prep_fused as mpf  # noqa: E402

N_EXP, TOPK, HIDDEN = 384, 6, 5120


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def summarize(v):
    return dict(n=len(v), median=statistics.median(v), p10=pct(v, 0.10), p90=pct(v, 0.90),
                mean=statistics.fmean(v))


def same(a, b):
    """Raw bits (torch.equal is False on identical NaN floats)."""
    ok = True
    for x, y in zip(a, b):
        if x.dtype in (torch.float16, torch.bfloat16):
            x, y = x.view(torch.int16), y.view(torch.int16)
        elif x.dtype == torch.float32:
            x, y = x.view(torch.int32), y.view(torch.int32)
        ok &= x.shape == y.shape and x.dtype == y.dtype and torch.equal(x, y)
    return ok


def inputs(m, dev, g, adversarial=False, ids_dtype=torch.int64, w_dtype=torch.float32):
    ids = torch.stack([torch.randperm(N_EXP, generator=g)[:TOPK] for _ in range(m)]).to(dev)
    w = torch.rand(m, TOPK, generator=g).to(dev)
    w = w / w.sum(1, keepdim=True) * 1.5
    x = (torch.randn(m, HIDDEN, generator=g) * 3).to(dev)
    if adversarial:
        flat = ids.view(-1)
        for i, v in enumerate((-1, N_EXP, 1000, -(1 << 40), 1 << 40)):
            flat[(i * 7) % flat.numel()] = v
        wf = w.view(-1)
        for i, v in enumerate((float("nan"), float("inf"), -float("inf"), -0.0, 65520.0, 1e-8, 6e-5, -3e-7)):
            wf[(i * 5 + 1) % wf.numel()] = v
        xf = x.view(-1)
        for i, v in enumerate((float("nan"), float("inf"), -0.0, 7e4, -1e5, 1e-8, 5.9e-8, 3e-5)):
            xf[(i * 997) % xf.numel()] = v
    return ids.to(ids_dtype), w.to(w_dtype).contiguous(), x.to(torch.bfloat16).contiguous()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=30)
    args = ap.parse_args()

    import vllm_exl3.exl3 as exl3
    from vllm.triton_utils import tl, triton

    dev = torch.device("cuda")
    prep = mpf.MoePrep(torch, mpf._build_kernel(tl, triton), triton, exl3.map_topk_to_local)
    mpf._STATE.update(armed=True, verify_left=0, engaged=False)
    out = {"torch": torch.__version__, "device": torch.cuda.get_device_name()}
    bad = []

    # 1. sweep
    g = torch.Generator().manual_seed(7)
    n_cases = 0
    for m in (1, 2, 3, 4, 5, 6, 7, 8, 64):
        for adv in (False, True):
            for ids_dtype in (torch.int64, torch.int32):
                for w_dtype in (torch.float32, torch.bfloat16):
                    for _ in range(8):
                        ids, w, x = inputs(m, dev, g, adv, ids_dtype, w_dtype)
                        ref = prep.stock(ids, w, x, N_EXP, None)
                        got = prep.fused(ids, w, x, N_EXP)
                        n_cases += 1
                        if not same(got, ref):
                            bad.append(f"sweep m={m} adv={adv} {ids_dtype} {w_dtype}")
    torch.cuda.synchronize()
    out["sweep_cases"] = n_cases

    # 2. graph capture + replay
    for m in (1, 3, 4, 6, 8):
        ids, w, x = inputs(m, dev, g)
        s_ids, s_w, s_x = ids.clone(), w.clone(), x.clone()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        prep.fused(s_ids, s_w, s_x, N_EXP)  # warm the Triton cache outside capture
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            g_out = prep.fused(s_ids, s_w, s_x, N_EXP)
        for rep in range(20):
            ids, w, x = inputs(m, dev, g, adversarial=rep % 2 == 1)
            s_ids.copy_(ids)
            s_w.copy_(w)
            s_x.copy_(x)
            graph.replay()
            ref = prep.stock(ids, w, x, N_EXP, None)
            torch.cuda.synchronize()
            if not same(g_out, ref):
                bad.append(f"graph m={m} rep={rep}")
    out["graph_replays"] = 5 * 20

    # 3. the patched function vs the stock function, stand-in p2b
    import inspect

    src = inspect.getsource(exl3._apply_native_fused_moe)
    ns = dict(exl3.__dict__)
    ns["_dsv41_moe_prep"] = prep
    exec(compile(mpf.patch_source(src), "patched", "exec"), ns)
    exec(compile(mpf.patch_experts_source(inspect.getsource(exl3.apply_exl3_experts)), "patched", "exec"), ns)
    patched = ns["_apply_native_fused_moe"]
    patched_experts = ns["apply_exl3_experts"]
    rec = []

    class FakeExt:
        P2B_MOE_ABI_VERSION = 2

        @staticmethod
        def p2b_fused_moe(xh, native_out, *rest):
            ids_, rw_ = rest[9], rest[10]
            rec.append((xh.clone(), ids_.clone(), rw_.clone()))
            native_out.copy_(xh * 0.5)
            return native_out

    layer = types.SimpleNamespace(
        _exl3_intermediate_local=1152, _exl3_ptrs={k: torch.zeros(1, device=dev) for k in (
            "gate_trellis", "gate_suh", "gate_svh", "up_trellis", "up_suh", "up_svh",
            "down_trellis", "down_suh", "down_svh")},
        _exl3_k=2, _exl3_codebook_flags=(True, False, True, False, True, False),
    )
    inners = [None] * N_EXP
    saved = (exl3._load_native_exl3_ext, exl3._native_moe_dimensions_supported,
             exl3.get_moe_kernel_backend, exl3.pin_exl3_expert_map)
    for d in (exl3.__dict__, ns):
        d["_load_native_exl3_ext"] = lambda: FakeExt
        d["_native_moe_dimensions_supported"] = lambda *a: True
        d["get_moe_kernel_backend"] = lambda: "native"
        d["pin_exl3_expert_map"] = lambda layer, dev: None
    layer._exl3_inners = inners
    try:
        for m in (1, 4, 8):
            for adv in (False, True):
                ids, w, x = inputs(m, dev, g, adv)
                rec.clear()
                a = exl3._apply_native_fused_moe(x, ids, w, layer, inners, None, 10.0)
                b = patched(x, ids, w, layer, inners, None, 10.0)
                torch.cuda.synchronize()
                if len(rec) != 2 or not same(rec[0], rec[1]) or not same((a,), (b,)):
                    bad.append(f"function m={m} adv={adv}")
                # the whole routed apply: stock (fp16 -> fp32 -> bf16) vs rewritten (one cast)
                rec.clear()
                ea = exl3.apply_exl3_experts(x, ids, w, layer, limit=10.0)
                eb = patched_experts(x, ids, w, layer, limit=10.0)
                torch.cuda.synchronize()
                if ea.dtype != torch.bfloat16 or not same((ea,), (eb,)) or not same(rec[0], rec[1]):
                    bad.append(f"experts m={m} adv={adv}")
    finally:
        (exl3._load_native_exl3_ext, exl3._native_moe_dimensions_supported,
         exl3.get_moe_kernel_backend, exl3.pin_exl3_expert_map) = saved

    # fp16 -> bf16 in one cast vs through fp32, every fp16 bit pattern
    allh = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.float16).to(dev)
    one, two = allh.to(torch.bfloat16), allh.to(torch.float32).to(torch.bfloat16)
    if not torch.equal(one.view(torch.int16), two.view(torch.int16)):
        bad.append("fp16->bf16 single cast != via fp32")
    out["cast_patterns"] = int(allh.numel())

    # 4. timing in CUDA graphs
    side = torch.cuda.Stream()
    big_src = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
    big_dst = torch.empty_like(big_src)
    flush = torch.empty(128 << 20, dtype=torch.uint8, device=dev)
    timing = {}
    for m in (1, 3, 4, 6, 8):
        ids, w, x = inputs(m, dev, g)
        graphs = {}
        for name in ("stock", "fused"):
            fn = (lambda: prep.stock(ids, w, x, N_EXP, None)) if name == "stock" else (lambda: prep.fused(ids, w, x, N_EXP))
            fn()
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                fn()
            graphs[name] = gr
        for load in ("alone", "warm", "cold", "beside_copy"):
            t = {"stock": [], "fused": []}
            for it in range(args.warmup + args.iters):
                order = ("stock", "fused") if it % 2 == 0 else ("fused", "stock")
                for name in order:
                    if load == "beside_copy":
                        side.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(side):
                            big_dst.copy_(big_src)
                        torch.cuda._sleep(20000)  # let the copy saturate DRAM first
                    elif load in ("warm", "cold"):
                        if load == "cold":
                            flush.fill_(it & 0xFF)
                        torch.cuda._sleep(200_000)  # ~90 us: the graph launch lands behind it
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    graphs[name].replay()
                    e1.record()
                    torch.cuda.synchronize()
                    if it >= args.warmup:
                        t[name].append(e0.elapsed_time(e1) * 1e3)
            timing[f"m{m}_{load}"] = {k: summarize(v) for k, v in t.items()}
    # p2b epilogue: fp16 -> fp32 -> bf16 (stock) vs fp16 -> bf16. One graph
    # holds 40 of them (one per routed layer), so the graph launch cost does
    # not swamp a ~1-2 us difference; reported per step (40 layers).
    for m in (4, 8):
        hs = [torch.randn(m, HIDDEN, device=dev).half() for _ in range(40)]
        graphs = {}
        for name, conv in (("stock", lambda h: h.to(torch.float32).to(torch.bfloat16)), ("single", lambda h: h.to(torch.bfloat16))):
            fn = lambda conv=conv: [conv(h) for h in hs]
            fn()
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                fn()
            graphs[name] = gr
        t = {"stock": [], "single": []}
        for it in range(args.warmup + args.iters):
            for name in (("stock", "single") if it % 2 == 0 else ("single", "stock")):
                torch.cuda._sleep(200_000)  # the graph launch lands behind it
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                graphs[name].replay()
                e1.record()
                torch.cuda.synchronize()
                if it >= args.warmup:
                    t[name].append(e0.elapsed_time(e1) * 1e3)
        timing[f"epilogue_x40_m{m}"] = {k: summarize(v) for k, v in t.items()}
    out["timing_us"] = timing
    kernels = {}
    for name, fn in (("stock", lambda: prep.stock(ids, w, x, N_EXP, None)), ("fused", lambda: prep.fused(ids, w, x, N_EXP))):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        kernels[name] = sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)
    out["gpu_ops_per_call"] = kernels

    # 5. sabotage: a fused result with one wrong id. (a) per-call verify: the
    # stock result is returned, the lever disarms, the next call is stock;
    # (b) the first-call self-test disarms before any fused result is used.
    class Sabotaged(mpf.MoePrep):
        def fused(self, ids, weights, x2d, n_exp):
            res = super().fused(ids, weights, x2d, n_exp)
            res[0].view(-1)[:1].add_(1)
            return res

    sab_prep = Sabotaged(torch, prep.kernel, triton, exl3.map_topk_to_local)
    ids, w, x = inputs(4, dev, g)
    ref = prep.stock(ids, w, x, N_EXP, None)
    mpf._STATE.update(armed=True, verify_left=2, engaged=True)
    first = sab_prep(ids, w, x, N_EXP, None)
    disarmed_a = not mpf._STATE["armed"]
    second = sab_prep(ids, w, x, N_EXP, None)
    mpf._STATE.update(armed=True, verify_left=2, engaged=False)
    sab_prep.selftest_once(dev)
    torch.cuda.synchronize()
    out["sabotage"] = {
        "verify_disarmed": disarmed_a,
        "verify_returned_stock": same(first, ref),
        "next_call_stock": same(second, ref),
        "selftest_disarmed": not mpf._STATE["armed"] and not mpf._STATE["engaged"],
    }
    if not all(out["sabotage"].values()):
        bad.append(f"sabotage {out['sabotage']}")
    mpf._STATE.update(armed=True, verify_left=0, engaged=False)
    out["mismatches"] = bad
    print(json.dumps(out))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
