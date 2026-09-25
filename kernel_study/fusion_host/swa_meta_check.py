#!/usr/bin/env python3
"""DSV41_SWA_META_FUSED: the rewritten SWA builder vs the image's, bit for bit; timing.

Runs in the serving image on a GPU (patch dir mounted at /opt/dsv41-patch).

1. build: DeepseekSparseSWAMetadataBuilder instances made with object.__new__
   and the attributes the build reads (window 128, block 64, 8192-token
   buffers), the image's build vs the rewritten one on the same batches (c=1
   m=1/4, c=2 m=8, a padded request with two padded tokens, a request shorter
   than the window, random block tables): every buffer the build writes (SWA
   index rows, the whole lens buffer, the validity row, the token map) and the
   returned metadata's tensors equal. The first 8 fused builds take the
   in-serve verify path, which must finish armed.
2. fallbacks: prefill, non-causal and a tensor-valued causal flag take the
   stock path (fused_ok False).
3. capture: the fused launch captured in a CUDA graph, replayed 20 times with
   new seq_lens / slot mappings / block tables, equal to the stock ops each time.
4. timing, as in the serve: 15 builds (a target decode step's causal SWA
   groups; each with its own token map, as stock) queued behind a calibrated
   torch.cuda._sleep of busy_us so the device runs them back to back (the host
   runs ~40 ms ahead in the serve); CUDA events around the 15 builds; the host
   enqueue time of every timed iteration must stay below busy_us (else the
   events would time host launch speed and the run fails); arms alternate;
   warm and cold L2 (a 128 MiB write before the sleep); after every iteration
   each fused builder's buffers must equal its stock twin's.
Prints one JSON line; exit 1 on any mismatch or host-bound sample.
"""
from __future__ import annotations

import argparse
import inspect
import json
import statistics
import sys
import textwrap
import time

import torch

sys.path.insert(0, "/opt/dsv41-patch")
import swa_meta_fused as smf  # noqa: E402

MAX_TOKENS, WINDOW, BLOCK = 8192, 128, 64


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def summarize(v):
    return dict(n=len(v), median=statistics.median(v), p10=pct(v, 0.10), p90=pct(v, 0.90), mean=statistics.fmean(v))


def make_builder(mod, dev):
    b = object.__new__(mod.DeepseekSparseSWAMetadataBuilder)
    b.decode_threshold = 4
    b.window_size = WINDOW
    b.block_size = BLOCK
    b.device = dev
    b.max_image_tokens = 0
    b.prefill_index_width = WINDOW
    b._max_tokens = MAX_TOKENS
    b._layer_types = set()
    b.is_dspark = True
    b.noncausal_index_width = 256
    b.decode_swa_indices_noncausal = None
    b.max_model_len = 1 << 20
    b.max_num_batched_tokens = MAX_TOKENS
    b.token_to_req_indices = torch.zeros(MAX_TOKENS, dtype=torch.int32, device=dev)
    b.decode_swa_indices = torch.full((MAX_TOKENS, 1, WINDOW), 99, dtype=torch.int32, device=dev)
    b.decode_swa_lens = torch.full((MAX_TOKENS,), 99, dtype=torch.int32, device=dev)
    b.prefill_swa_indices = torch.zeros((MAX_TOKENS, 1, WINDOW), dtype=torch.int32, device=dev)
    b.prefill_swa_lens = torch.zeros(MAX_TOKENS, dtype=torch.int32, device=dev)
    b.is_valid_token = torch.zeros(MAX_TOKENS, dtype=torch.bool, device=dev)
    return b


def cam(qsl, seq, num_tokens, max_q, block_table, slots, causal=True):
    from vllm.v1.attention.backend import CommonAttentionMetadata as CAM

    dev = block_table.device
    return CAM(
        query_start_loc=torch.tensor(qsl, dtype=torch.int32, device=dev),
        query_start_loc_cpu=torch.tensor(qsl, dtype=torch.int32),
        seq_lens=torch.tensor(seq, dtype=torch.int32, device=dev),
        seq_lens_cpu_upper_bound=torch.tensor(seq, dtype=torch.int32),
        max_seq_len=max(seq), num_reqs=len(seq), num_actual_tokens=num_tokens,
        max_query_len=max_q, block_table_tensor=block_table, slot_mapping=slots, causal=causal,
    )


def written(b, nd, ntok):
    return (b.decode_swa_indices[:nd], b.decode_swa_lens, b.is_valid_token[:ntok], b.token_to_req_indices[:ntok])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--groups", type=int, default=15)
    ap.add_argument("--busy-us", type=float, default=8000.0)
    args = ap.parse_args()

    from vllm.triton_utils import tl, triton
    from vllm.v1.attention.backends.mla import sparse_swa as mod

    dev = torch.device("cuda")
    stock_build = mod.DeepseekSparseSWAMetadataBuilder.build
    ns = dict(mod.__dict__)
    helper = smf.SwaFused(torch, triton, smf._build_kernel(tl, triton), mod._COMPUTE_SWA_INDICES_AND_LENS_KERNEL)
    ns["_dsv41_swa"] = helper
    src = textwrap.dedent(smf.patch_source(inspect.getsource(stock_build)))
    exec(compile(src, "patched", "exec", dont_inherit=True), ns)
    fused_build = ns["build"]
    smf._STATE.update(armed=True, verify_left=8, engaged=False)
    g = torch.Generator().manual_seed(3)
    bad = []
    out = {"torch": torch.__version__}

    # 1. build-level equality
    n_builds = 0
    for _ in range(12):
        bt = torch.randint(0, 1 << 20, (4, 4096), generator=g, dtype=torch.int32).to(dev)
        for qsl, seq, ntok, maxq in (
            ([0, 1], [int(torch.randint(1, 5000, (1,), generator=g))], 1, 1),
            ([0, 4], [int(torch.randint(4, 9000, (1,), generator=g))], 4, 4),
            ([0, 4, 8], [int(torch.randint(4, 9000, (1,), generator=g)), 50], 8, 4),
            ([0, 3, 6, 6], [700, 20, 1], 8, 3),  # padded request + 2 padded tokens
        ):
            slots = torch.randint(0, 1 << 26, (ntok,), generator=g, dtype=torch.int64)
            slots[qsl[-1]:] = -1
            slots = slots.to(dev)
            b0, b1 = make_builder(mod, dev), make_builder(mod, dev)
            m0 = stock_build(b0, 0, cam(qsl, seq, ntok, maxq, bt[: len(seq)], slots))
            m1 = fused_build(b1, 0, cam(qsl, seq, ntok, maxq, bt[: len(seq)], slots))
            torch.cuda.synchronize()
            nd = m0.num_decode_tokens
            if nd != ntok or m1.num_decode_tokens != nd:
                bad.append(f"split qsl={qsl}")
            if not all(torch.equal(a, c) for a, c in zip(written(b0, nd, ntok), written(b1, nd, ntok))):
                bad.append(f"buffers qsl={qsl} seq={seq}")
            for f in ("decode_swa_indices", "decode_swa_lens", "is_valid_token", "token_to_req_indices"):
                if not torch.equal(getattr(m0, f), getattr(m1, f)):
                    bad.append(f"metadata {f} qsl={qsl}")
            if m0.decode_swa_width != m1.decode_swa_width or m1.prefill_swa_indices is not None:
                bad.append(f"metadata scalars qsl={qsl}")
            n_builds += 1
    out["builds_compared"] = n_builds
    out["state_after_builds"] = dict(smf._STATE)
    if not smf._STATE["armed"] or smf._STATE["verify_left"] != 0 or not smf._STATE["engaged"]:
        bad.append("verify path did not finish armed")

    # 2. fallbacks
    bt = torch.randint(0, 1 << 20, (2, 64), generator=g, dtype=torch.int32).to(dev)
    b = make_builder(mod, dev)
    ar = lambda n: torch.arange(n, dtype=torch.int64, device=dev)  # noqa: E731
    out["fallback"] = {
        "prefill": not helper.fused_ok(b, cam([0, 20], [20], 20, 20, bt[:1], ar(20)), 0, 20),
        "mixed": not helper.fused_ok(b, cam([0, 1, 21], [9, 20], 21, 20, bt, ar(21)), 1, 20),
        "noncausal": not helper.fused_ok(b, cam([0, 4], [100], 4, 4, bt[:1], ar(4), causal=False), 4, 0),
        "tensor_causal": not helper.fused_ok(
            b, cam([0, 4], [100], 4, 4, bt[:1], ar(4), causal=torch.ones(4, dtype=torch.bool, device=dev)), 4, 0
        ),
        "short_slots": not helper.fused_ok(b, cam([0, 4], [100], 4, 4, bt[:1], ar(3)), 4, 0),
    }
    if not all(out["fallback"].values()):
        bad.append("fallback")

    # 3. graph capture + replay with new inputs vs the stock ops
    bg, br = make_builder(mod, dev), make_builder(mod, dev)
    qsl_d = torch.tensor([0, 4, 8], dtype=torch.int32, device=dev)
    seq_d = torch.tensor([3000, 60], dtype=torch.int32, device=dev)
    t2r_d = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32, device=dev)
    slots_d = torch.arange(8, dtype=torch.int64, device=dev)
    bt_d = torch.randint(0, 1 << 20, (2, 256), generator=g, dtype=torch.int32).to(dev)
    run = lambda bb: helper._launch(  # noqa: E731
        bb, bb.decode_swa_indices, qsl_d, seq_d, t2r_d, slots_d, bb.is_valid_token[:8], bt_d, 8
    )
    run(bg)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(bg)
    for rep in range(20):
        seq_d.copy_(torch.randint(4, 256 * BLOCK, (2,), generator=g, dtype=torch.int32))
        s = torch.randint(0, 1 << 26, (8,), generator=g, dtype=torch.int64)
        s[torch.randint(0, 8, (rep % 3,), generator=g)] = -1
        slots_d.copy_(s)
        bt_d.copy_(torch.randint(0, 1 << 20, (2, 256), generator=g, dtype=torch.int32))
        graph.replay()
        helper._stock(br, br.decode_swa_indices, qsl_d, seq_d, t2r_d, slots_d, br.is_valid_token[:8], bt_d, 8)
        torch.cuda.synchronize()
        if not all(torch.equal(a, c) for a, c in zip(written(bg, 8, 8)[:3], written(br, 8, 8)[:3])):
            bad.append(f"graph replay {rep}")
    out["graph_replays"] = 20

    # 4. eager timing behind a calibrated GPU sleep, 15 builds per step
    smf._STATE.update(verify_left=0)
    torch.cuda._sleep(1000)
    torch.cuda.synchronize()
    c0, c1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    c0.record()
    torch.cuda._sleep(20_000_000)
    c1.record()
    torch.cuda.synchronize()
    cycles_per_us = 20_000_000 / (c0.elapsed_time(c1) * 1e3)
    busy_cycles = int(args.busy_us * cycles_per_us)
    out["cycles_per_us"] = round(cycles_per_us, 1)
    flush = torch.empty(128 << 20, dtype=torch.uint8, device=dev)
    arms = {"stock": ([make_builder(mod, dev) for _ in range(args.groups)], stock_build),
            "fused": ([make_builder(mod, dev) for _ in range(args.groups)], fused_build)}
    tables = [torch.randint(0, 1 << 20, (2, 4096), generator=g, dtype=torch.int32).to(dev) for _ in range(args.groups)]
    timing = {}
    host_bound = 0
    for l2 in ("warm", "cold"):
        for label, qsl, seq, ntok in (("c1_m4", [0, 4], [3000], 4), ("c2_m8", [0, 4, 8], [3000, 5000], 8)):
            slot_rows = [torch.randint(0, 1 << 26, (ntok,), generator=g, dtype=torch.int64).to(dev) for _ in tables]
            t = {"stock": [], "fused": []}
            host = {"stock": [], "fused": []}
            ev = {k: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for k in t}
            for it in range(args.warmup + args.iters):
                for name in (("stock", "fused") if it % 2 == 0 else ("fused", "stock")):
                    builders, fn = arms[name]
                    cms = [cam(qsl, seq, ntok, 4, tb[: len(seq)], sr) for tb, sr in zip(tables, slot_rows)]
                    if l2 == "cold":
                        flush.fill_(it & 0xFF)
                    torch.cuda.synchronize()
                    torch.cuda._sleep(busy_cycles)
                    h0 = time.perf_counter()
                    ev[name][0].record()
                    for bb, cm in zip(builders, cms):
                        fn(bb, 0, cm)
                    ev[name][1].record()
                    h1 = time.perf_counter()
                    if it >= args.warmup:
                        host[name].append((h1 - h0) * 1e6)
                        host_bound += (h1 - h0) * 1e6 >= args.busy_us
                torch.cuda.synchronize()
                for gi in range(args.groups):
                    if not all(torch.equal(a, c) for a, c in zip(
                            written(arms["stock"][0][gi], ntok, ntok), written(arms["fused"][0][gi], ntok, ntok))):
                        bad.append(f"timing {label} it={it} group={gi}")
                if it >= args.warmup:
                    for name in t:
                        t[name].append(ev[name][0].elapsed_time(ev[name][1]) * 1e3)
            timing[f"{l2}_{label}"] = {k: summarize(v) for k, v in t.items()}
            timing[f"{l2}_{label}"]["host_enqueue_us"] = {k: summarize(v) for k, v in host.items()}
    out["timing_15_builds_us"] = timing
    out["host_bound_samples"] = host_bound
    if host_bound:
        bad.append(f"{host_bound} timed samples host-bound (enqueue >= busy)")

    counts = {}
    for name in ("stock", "fused"):
        builders, fn = arms[name]
        cm = cam([0, 4], [3000], 4, 4, tables[0][:1], torch.arange(4, dtype=torch.int64, device=dev))
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            fn(builders[0], 0, cm)
            torch.cuda.synchronize()
        counts[name] = sorted(e.name[:60] for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)
    out["gpu_ops_per_build"] = {k: len(v) for k, v in counts.items()}
    out["gpu_op_names"] = counts

    # 5. the verify path's safety net: a fused launch that writes one wrong
    # element must leave the stock values in place and disarm
    smf._STATE.update(armed=True, verify_left=8, engaged=True)
    real_launch = helper._launch

    def bad_launch(bb, *a):
        real_launch(bb, *a)
        bb.decode_swa_lens[1] += 1

    helper._launch = bad_launch
    bs, bf = make_builder(mod, dev), make_builder(mod, dev)
    cm_args = ([0, 4], [3000], 4, 4, tables[0][:1], torch.arange(4, dtype=torch.int64, device=dev))
    stock_build(bs, 0, cam(*cm_args))
    fused_build(bf, 0, cam(*cm_args))
    torch.cuda.synchronize()
    helper._launch = real_launch
    out["sabotage"] = {
        "disarmed": not smf._STATE["armed"],
        "stock_values_kept": all(torch.equal(a, c) for a, c in zip(written(bs, 4, 4), written(bf, 4, 4))),
        "next_build_stock": not helper.fused_ok(bf, cam(*cm_args), 4, 0),
    }
    if not all(out["sabotage"].values()):
        bad.append("sabotage")

    # 6. install() on the image class: patched build via the class, equal to stock
    smf._STATE.update(armed=True, verify_left=2, engaged=False)
    cls = mod.DeepseekSparseSWAMetadataBuilder
    msg = smf.install()
    b0, b1 = make_builder(mod, dev), make_builder(mod, dev)
    cm_args = ([0, 4, 8], [3000, 90], 8, 4, tables[1], torch.arange(8, dtype=torch.int64, device=dev))
    stock_build(b0, 0, cam(*cm_args))
    for _ in range(3):  # 2 verify builds + 1 fast build
        b1.build(0, cam(*cm_args))
    torch.cuda.synchronize()
    out["install"] = {
        "msg": msg,
        "patched": bool(getattr(cls.build, "_dsv41_swa_fused", False)),
        "qualname": cls.build.__qualname__,
        "equal": all(torch.equal(a, c) for a, c in zip(written(b0, 8, 8), written(b1, 8, 8))),
        "state": dict(smf._STATE),
        "again": smf.install(),
    }
    if not (out["install"]["patched"] and out["install"]["equal"] and smf._STATE["armed"]):
        bad.append("install")
    out["mismatches"] = bad[:20]
    out["n_mismatches"] = len(bad)
    print(json.dumps(out))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
