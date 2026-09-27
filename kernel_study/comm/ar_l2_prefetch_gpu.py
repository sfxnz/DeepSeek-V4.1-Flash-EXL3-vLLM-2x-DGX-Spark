#!/usr/bin/env python3
"""GPU check of docker/patch/ar_l2_prefetch.py through its own hooks (single GPU).

A stand-in decoder stack runs the lever's real wrap() / Prefetcher / l2pf_kernel code
inside a CUDA graph. Per layer, in the serve's order:
  mHC (the real mhc_post + mhc_pre TileLang kernels) -> qkv_a b12x GEMM on this layer's
  weights -> DeepseekV4Attention._split_qkv_and_norm stand-in (where the lever joins) ->
  p2b stand-in (a 2x-L2 streaming read, so every layer's qkv_a starts cold as after the
  routed MoE) -> MoE AR stand-in (MoERunner._maybe_reduce_final_output: a 22.7 us 5-CTA
  spin, where the lever forks) -> next layer.
The spin touches no memory, so this is the lever's optimistic bound (a real NCCL AR slows
under the prefetch: ar_window_nccl.py).
Two model instances, lever on and off (separate classes, only one wrapped), captured
once each and replayed alternately. Reports per-layer qkv_a time and the forward time
(median, p10/p90), checks the GEMM outputs bitwise equal, and that capture + replay
work with the side-stream fork/join inside the graph.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docker" / "patch"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

HIDDEN, HC = 5120, 4
QKV_A = (5120, 1792)


def stats(xs):
    s = sorted(xs)
    q = lambda f: s[min(len(s) - 1, int(round(f * (len(s) - 1))))]  # noqa: E731
    return {"median": round(statistics.median(s), 2), "p10": round(q(0.1), 2), "p90": round(q(0.9), 2), "n": len(s)}


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--layers", type=int, default=8)
    ap_.add_argument("--m", type=int, default=4)
    ap_.add_argument("--replays", type=int, default=200)
    ap_.add_argument("--ar-us", type=float, default=22.7)
    ap_.add_argument("--sm-mhz", type=float, default=2190.0)
    ap_.add_argument("--build-dir", default="/repo/kernel_study/comm/.l2pf_build3")
    ap_.add_argument("--mib", type=float, default=0.0, help="prefetch budget (default: the lever's default)")
    ap_.add_argument("--json")
    args = ap_.parse_args()

    import torch
    from vllm.model_executor.kernels.mhc import tilelang as mhc_tl
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize, swizzle_mxfp8_scale
    from vllm.utils import deep_gemm as vdg
    from vllm.utils import flashinfer as vfi

    import ar_l2_prefetch as ap
    import l2pf_kernel
    from l2_prefetch_window import build_ext

    ext = build_ext(args.build_dir)  # spin + streaming read kernels only
    vdg._lazy_init()
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    l2 = int(torch.cuda.get_device_properties(0).L2_cache_size)
    m, n_layers = args.m, args.layers
    flush = torch.empty(2 * l2, dtype=torch.uint8, device=dev)
    sink = torch.zeros(1, dtype=torch.int32, device=dev)
    k_hc, mix = HC * HIDDEN, HC * (HC + 2)
    fn = torch.randn(mix, k_hc, device=dev, dtype=torch.float32) * 0.02
    hc_scale = torch.rand(3, device=dev, dtype=torch.float32) + 0.5
    hc_base = torch.randn(mix, device=dev, dtype=torch.float32) * 0.1
    norm_w = (torch.rand(HIDDEN, device=dev) + 0.5).to(torch.bfloat16)
    residual = torch.randn(m, HC, HIDDEN, device=dev, dtype=torch.bfloat16)
    pre_mix = torch.softmax(torch.randn(m, HC, device=dev), -1).float().contiguous()
    post_mix, res_mix, x0, _ = mhc_tl.mhc_pre_delayed_tilelang(
        residual, fn, hc_scale, hc_base, 1e-6, 1e-6, 1e-6, 2.0, 20, pre_mix=pre_mix, norm_weight=norm_w, norm_eps=1e-6)
    x_ar = torch.randn(m, HIDDEN, device=dev, dtype=torch.bfloat16)
    weights = []
    for _ in range(n_layers):
        w = torch.randn(QKV_A[1], QKV_A[0], device=dev, dtype=torch.bfloat16) * 0.02
        w8, sc = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=False)
        weights.append((w8, swizzle_mxfp8_scale(sc, M=QKV_A[1], K=QKV_A[0]).contiguous()))

    def make_model(tag):
        events = [(torch.cuda.Event(enable_timing=True, external=True),
                   torch.cuda.Event(enable_timing=True, external=True)) for _ in range(n_layers)]
        outs = [None] * n_layers

        class Runner:
            def _maybe_reduce_final_output(self, states, *a, **k):
                ext.spin(args.ar_us * args.sm_mhz, 5)  # the MoE all-reduce stand-in
                return states

        class Attn:
            def _split_qkv_and_norm(self, qr_kv):  # the lever joins a pending prefetch here
                return qr_kv

        class Layer:
            def __init__(self, i):
                self.i, self.engram, self.runner = i, None, Runner()
                self.attn = types.SimpleNamespace(fused_wqa_wkv=types.SimpleNamespace(
                    weight=weights[i][0], weight_scale=weights[i][1]))
                self.attn_impl = Attn()

            def forward(self, x):
                res2 = mhc_tl.mhc_post_tilelang(x_ar, residual, post_mix, res_mix)
                _, _, h, _ = mhc_tl.mhc_pre_delayed_tilelang(
                    res2, fn, hc_scale, hc_base, 1e-6, 1e-6, 1e-6, 2.0, 20, pre_mix=pre_mix,
                    norm_weight=norm_w, norm_eps=1e-6)
                q, s = mxfp8_e4m3_quantize(h, is_sf_swizzled_layout=True)
                e = events[self.i]
                e[0].record()
                outs[self.i] = vfi.mm_mxfp8(q, self.attn.fused_wqa_wkv.weight.t(), s,
                                            self.attn.fused_wqa_wkv.weight_scale, out_dtype=torch.bfloat16,
                                            backend="auto")
                e[1].record()
                self.attn_impl._split_qkv_and_norm(outs[self.i])
                ext.read_l2(flush, flush.numel(), 48, False, sink)  # p2b stand-in: L2 cold again
                return self.runner._maybe_reduce_final_output(x)

        class Model:
            def __init__(self):
                self.layers = [Layer(i) for i in range(n_layers)]
                self.start_layer, self.end_layer = 0, n_layers

            def forward(self, x):
                for layer in self.layers:
                    x = layer.forward(x)
                return x

        return Layer, Model, Runner, Attn, events, outs

    budget = ap.budget_bytes({ap.MIB_ENV: str(args.mib)} if args.mib else {})
    arms = {}
    for tag in ("off", "on"):
        Layer, Model, Runner, Attn, events, outs = make_model(tag)
        pf = None
        if tag == "on":
            pf = ap.Prefetcher(l2pf_kernel.launcher(torch), torch)
            ap.wrap(Layer, Model, Runner, Attn, pf, budget, lambda msg: print(msg, flush=True))
        model = Model()
        total = (torch.cuda.Event(enable_timing=True, external=True), torch.cuda.Event(enable_timing=True, external=True))
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            model.forward(x0)  # eager: builds the plans + trial launch (on arm)
            model.forward(x0)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            ext.read_l2(flush, flush.numel(), 48, False, sink)
            total[0].record()
            model.forward(x0)
            total[1].record()
        torch.cuda.synchronize()
        arms[tag] = {"g": g, "events": events, "outs": outs, "total": total, "pf": pf}

    res = {t: {"forward": [], "qkv_a_first": [], "qkv_a_rest": []} for t in arms}
    for i in range(args.replays + 10):
        for tag, a in arms.items():
            a["g"].replay()
            torch.cuda.synchronize()
            if i < 10:
                continue
            res[tag]["forward"].append(a["total"][0].elapsed_time(a["total"][1]) * 1e3)
            ts = [e0.elapsed_time(e1) * 1e3 for e0, e1 in a["events"]]
            res[tag]["qkv_a_first"].append(ts[0])
            res[tag]["qkv_a_rest"] += ts[1:]
    bitwise = all(torch.equal(a, b) for a, b in zip(arms["off"]["outs"], arms["on"]["outs"]))
    out = {
        "what": "ar_l2_prefetch through its own hooks, single GPU, stand-in decoder stack in one CUDA graph",
        "layers": n_layers, "m": m, "replays": args.replays, "budget_bytes": budget,
        "capture_replay_ok": True, "gemm_outputs_bitwise_equal": bitwise,
        "forks_on": arms["on"]["pf"].forks,
        "summary": {t: {k: stats(v) for k, v in r.items()} for t, r in res.items()},
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    on, off = out["summary"]["on"], out["summary"]["off"]
    out["delta_forward_us"] = round(on["forward"]["median"] - off["forward"]["median"], 2)
    out["delta_per_prefetched_layer_us"] = round(out["delta_forward_us"] / (n_layers - 1), 2)
    print(json.dumps(out, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0 if bitwise else 1


if __name__ == "__main__":
    sys.exit(main())
