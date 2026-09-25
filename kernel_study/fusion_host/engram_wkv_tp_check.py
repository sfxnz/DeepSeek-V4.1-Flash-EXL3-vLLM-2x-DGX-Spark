#!/usr/bin/env python3
"""DSV41_ENGRAM_WKV_TP on one GPU: TP=2 emulated with the image's own classes.

Runs in the serving image on a GPU (patch dir at /opt/dsv41-patch, pack read-only):

1. layers: for model.layers.{1,14}.engram.wkv, a stock ReplicatedLinear [25600, 6144]
   (what the serve builds today) and the lever's ColumnParallelLinear
   (gather_output=True) for tp_rank 0 and 1 of tp_size 2, all built by the pack's
   quant config (Exl3Config -> non_routed delegate -> ModelOpt MXFP8), loaded
   through their own parameter weight loaders with the checkpoint tensors (what
   load_weights passes for .weight / .scale), processed by their own
   process_weights_after_loading. One process holds both "ranks"; the all-gather is
   emulated by concatenation in rank order (a byte copy either way).
2. bitwise: stock(x) vs cat(rank 0, rank 1) for M 1..8 and 16..2048, four activation
   distributions; and the two processed shards concatenated vs the stock layer's
   processed weight and swizzled scale, byte for byte.
3. the serve's self_check (engram_wkv_tp.self_check) on the rank-0 layer, the peer's
   gather contributions from the rank-1 layer: must engage. Sabotage (one flipped
   bit in rank 0's GEMM output): must disarm and install the replicated fallback,
   whose output must equal stock bit for bit at every M.
4. capture: the rank-0 layer call + the emulated gather captured in a CUDA graph and
   replayed with new rows, equal to stock each time.
5. timing: one rank's layer call vs the stock layer call, CUDA graphs, launch hidden
   behind a ~90 us sleep, cold (128 MiB write before each replay) and warm, >= 300
   replays alternating. The all-gather itself needs two GPUs and is modeled from
   the r3 trace (Engram embed gather, 24 KiB/rank: 18.4 us) and the comm sweep.
Prints one JSON line; exit 1 on any difference.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import types

import torch

sys.path.insert(0, "/opt/dsv41-patch")
import engram_wkv_tp as wtp  # noqa: E402

N, K, TP = 25600, 6144, 2


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def summarize(v):
    return dict(n=len(v), median=round(statistics.median(v), 2), p10=round(pct(v, 0.10), 2),
                p90=round(pct(v, 0.90), 2))


def make_x(m, dist, g):
    x = torch.randn(m, K, generator=g, device="cuda")
    if dist == "lognormal":
        x = x * torch.exp(1.5 * torch.randn(m, K, generator=g, device="cuda"))
    elif dist == "outlier":
        x[:, torch.randint(0, K, (8,), generator=g, device="cuda")] *= 300.0
    elif dist == "edge":
        pick = torch.randint(0, 6, (m, K), generator=g, device="cuda")
        e = torch.randint(-20, 20, (m, K), generator=g, device="cuda").float()
        x = torch.where(pick == 0, torch.exp2(e), x)
        x = torch.where(pick == 1, 448.0 * torch.exp2(e), x)
        x = torch.where(pick == 2, torch.zeros_like(x), x)
        x = torch.where(pick == 3, x * 1e-30, x)
        x[:, :32] = 0.0
    return x.to(torch.bfloat16)


def bits(t):
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    torch.set_default_dtype(torch.bfloat16)
    from safetensors import safe_open
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear
    from vllm_exl3.exl3 import Exl3Config

    init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                 distributed_init_method="tcp://127.0.0.1:29571", backend="nccl")
    initialize_model_parallel(tensor_model_parallel_size=1)
    cfg = json.load(open(os.path.join(args.snapshot, "config.json")))
    qc = Exl3Config.from_config(cfg.get("quantization_config") or cfg["text_config"]["quantization_config"])
    idx = json.load(open(os.path.join(args.snapshot, "model.safetensors.index.json")))["weight_map"]
    dev = torch.device("cuda")
    out = {"torch": torch.__version__, "bitwise": {}, "self_check": {}, "timing_us": {}}
    bad = []
    g = torch.Generator(device="cuda").manual_seed(3)

    def ckpt(name):
        with safe_open(os.path.join(args.snapshot, idx[name]), framework="pt") as fh:
            return fh.get_tensor(name)

    def build(layer_id, kind, rank=None):
        prefix = f"model.layers.{layer_id}.engram.wkv"
        with torch.device(dev):
            if kind == "stock":
                lyr = ReplicatedLinear(K, N, bias=False, quant_config=qc, return_bias=False, prefix=prefix)
            else:
                lyr = ColumnParallelLinear(K, N, bias=False, gather_output=True, quant_config=qc,
                                           prefix=prefix, return_bias=False, tp_rank=rank, tp_size=TP)
        w, s = ckpt(f"layers.{layer_id}.engram.wkv.weight"), ckpt(f"layers.{layer_id}.engram.wkv.scale")
        lyr.weight.weight_loader(lyr.weight, w)
        lyr.weight_scale.weight_loader(lyr.weight_scale, s)
        lyr.quant_method.process_weights_after_loading(lyr)
        return lyr

    def shard_call(lyr, x):
        return lyr.quant_method.apply(lyr, x)

    for layer_id in (1, 14):
        stock = build(layer_id, "stock")
        ranks = [build(layer_id, "tp", r) for r in range(TP)]
        out["quant_method"] = type(stock.quant_method).__name__
        out["kernel"] = type(stock.quant_method.kernel).__name__
        # processed shards vs the stock processed tensors, byte for byte
        w_cat = torch.cat([r.weight.data.view(torch.uint8) for r in ranks], 0)
        s_cat = torch.cat([r.weight_scale.data.view(torch.uint8).reshape(-1) for r in ranks], 0)
        same_w = torch.equal(w_cat, stock.weight.data.view(torch.uint8))
        same_s = torch.equal(s_cat, stock.weight_scale.data.view(torch.uint8).reshape(-1))
        if not (same_w and same_s):
            bad.append(f"layer {layer_id}: shards != stock tensors (weight {same_w}, scale {same_s})")
        n_cases = n_diff = 0
        for m in list(range(1, 9)) + [16, 32, 64, 128, 256, 512, 1024, 2048]:
            for dist in ("normal", "lognormal", "outlier", "edge"):
                x = make_x(m, dist, g)
                ref = stock(x)
                got = torch.cat([shard_call(r, x) for r in ranks], -1)
                n_cases += 1
                if not torch.equal(bits(got), bits(ref)):
                    n_diff += 1
                    bad.append(f"layer {layer_id} m {m} {dist}: sharded != stock")
        out["bitwise"][f"layer{layer_id}"] = {"cases": n_cases, "differ": n_diff,
                                             "shards_equal_stock_weight": same_w,
                                             "shards_equal_stock_scale": same_s}

        # the serve's self_check, rank 0's view; the peer contributes rank 1's tensors
        xs = wtp.probe_inputs(torch, K, dev)
        full_cls = wtp.make_full_layer_class(torch)

        def run_check(r0, r1):
            peer, _ = wtp.contributions(torch, r1, xs)
            calls = iter(peer)

            def all_gather(t, dim):
                return torch.cat([t, next(calls)], dim=dim if dim >= 0 else t.dim() + dim)

            eng = types.SimpleNamespace(wkv=r0)
            ok = wtp.self_check(torch, eng, r0, all_gather, lambda t: t * TP, full_cls)
            return ok, eng

        ok, eng = run_check(ranks[0], ranks[1])
        res = {"engaged": ok, "wkv_kept": eng.wkv is ranks[0]}
        # sabotage: one flipped bit in rank 0's GEMM output
        method = ranks[0].quant_method
        real_apply = method.apply

        def flipped(lyr, x, bias=None):
            y = real_apply(lyr, x, bias)
            if lyr is ranks[0]:
                y.view(torch.int16).view(-1)[:1].bitwise_xor_(1)
            return y

        method.apply = flipped
        ok2, eng2 = run_check(ranks[0], ranks[1])
        method.apply = real_apply
        res["sabotage_disarmed"] = not ok2
        res["fallback_installed"] = type(eng2.wkv).__name__ == "ReplicatedWkv"
        fb_diff = 0
        for m in (1, 3, 4, 6, 8, 64, 2048):
            x = make_x(m, "normal", g)
            if not torch.equal(bits(eng2.wkv(x)), bits(stock(x))):
                fb_diff += 1
        res["fallback_equals_stock_cases_differ"] = fb_diff
        out["self_check"][f"layer{layer_id}"] = res
        if not (res["engaged"] and res["wkv_kept"] and res["sabotage_disarmed"] and res["fallback_installed"]
                and fb_diff == 0):
            bad.append(f"layer {layer_id} self_check {res}")

        # capture: rank-0 call + the gather (emulated) in a graph, new rows each replay
        if layer_id == 1:
            xs_s = torch.zeros(4, K, dtype=torch.bfloat16, device=dev)
            peer_out = torch.zeros(4, N // TP, dtype=torch.bfloat16, device=dev)
            shard_call(ranks[0], xs_s)
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                g_out = torch.cat([shard_call(ranks[0], xs_s), peer_out], -1)
            n_rep = 0
            for rep in range(20):
                x = make_x(4, ("normal", "lognormal", "outlier", "edge")[rep % 4], g)
                xs_s.copy_(x)
                peer_out.copy_(shard_call(ranks[1], x))
                gr.replay()
                torch.cuda.synchronize()
                if not torch.equal(bits(g_out), bits(stock(x))):
                    bad.append(f"graph replay {rep}")
                n_rep += 1
            out["graph_replays"] = n_rep

            # timing: one rank's call vs the stock call
            flush = torch.empty(128 << 20, dtype=torch.uint8, device=dev)
            wbytes = {"stock": N * K + N * K // 32, "rank0": N // TP * K + N // TP * K // 32}
            for m in (1, 3, 4, 6, 8):
                x = make_x(m, "normal", g)
                graphs = {}
                for name, fn in (("stock", lambda: stock(x)), ("rank0", lambda: shard_call(ranks[0], x))):
                    fn()
                    torch.cuda.synchronize()
                    gg = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(gg):
                        fn()
                    graphs[name] = gg
                for l2 in ("cold", "warm"):
                    t = {"stock": [], "rank0": []}
                    for it in range(args.warmup + args.iters):
                        for name in (("stock", "rank0") if it % 2 == 0 else ("rank0", "stock")):
                            if l2 == "cold":
                                flush.fill_(it & 0xFF)
                            torch.cuda._sleep(200_000)
                            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            e0.record()
                            graphs[name].replay()
                            e1.record()
                            torch.cuda.synchronize()
                            if it >= args.warmup:
                                t[name].append(e0.elapsed_time(e1) * 1e3)
                    cell = {k: summarize(v) for k, v in t.items()}
                    for k in cell:
                        cell[k]["GBps"] = round(wbytes[k] / (cell[k]["median"] * 1e-6) / 1e9, 1)
                        cell[k]["pct_of_250"] = round(100 * cell[k]["GBps"] / 250, 1)
                    cell["delta_us"] = round(cell["stock"]["median"] - cell["rank0"]["median"], 2)
                    out["timing_us"][f"m{m}_{l2}"] = cell
        del stock, ranks
        torch.cuda.empty_cache()
    out["state"] = dict(wtp._STATE)
    out["mismatches"] = bad
    print(json.dumps(out))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
