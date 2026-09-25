#!/usr/bin/env python3
"""Two-node NCCL latency at the exact decode collective sizes, serve down.

The serve's TP all-reduce backend is vLLM's PyNcclCommunicator (engine log:
"Using ['PYNCCL'] all-reduce backends"), so this drives the same class and the
same libnccl from the serve image. Sizes come from the model config (hidden
5120, vocab 129280, TP 2, bf16):

  all_reduce  m x hidden        m in AR_ROWS (target verify m=4/8, draft m=3/6)
  all_gather  m x vocab/TP in   m in AG_ROWS (lm_head logits, target + draft)

Modes per size:
  eager   one launch per call, CUDA events around each call; a GPU spin first so
          all calls are queued before the GPU reaches them (device time, not
          host launch rate)
  graph   G calls back to back in one CUDA graph
  gapped  G x [flush 2x L2 ; call] in one CUDA graph, against the same graph
          without the calls: each decode collective in the serve follows a
          bandwidth-bound kernel inside a CUDA graph
Graph modes time every slot of every replay with CUDA events captured into the
graph (profiler off: a torch.profiler pass changes NCCL-in-graph timing by tens
of percent here), and replay the graph with and without the calls alternately
in one process (A/B). Per slot s: cost = period_with[s] - median period_ref[s].
  steady   median slot cost at slots >= STEADY_FROM: the critical-path cost of
           one call inside a replay, a slowed neighbour included
  startup  what one replay pays once inside the graph for holding network
           collectives: excess of the first slots over steady. NCCL uploads each
           call's proxy ops from a host node; the first call waits for it.
  launch_excess  extra GPU wait from a mark before the launch to the first node
           (the GPU is idle here, so this is the extra host launch time)
  host_launch  cudaGraphLaunch wall time on the host (with vs without calls)
A decode step pays steady per collective plus startup per graph replay.

  run:        python3 -S nccl_decode_sweep.py run --rank R --master HOST:PORT --arm NAME --json OUT
  arm-env:    python3 nccl_decode_sweep.py arm-env NAME     (KEY=VAL per line, for the driver)
  summarize:  python3 nccl_decode_sweep.py summarize DIR --out nccl-sweep.json [--weights W.json]

tools/nccl_decode_sweep.sh drives both ranks per arm and rep.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys
import time
from pathlib import Path

HIDDEN = 5120
VOCAB = 129280
TP = 2
BF16 = 2
AR_ROWS = (1, 3, 4, 6, 8)
AG_ROWS = (3, 4, 6, 8)

# Serve env that is not a tuning knob (run.sh docker args), identical in every arm.
BASE_ENV = {
    "NCCL_IB_HCA": "rocep1s0f1",
    "NCCL_CROSS_NIC": "1",
    "NCCL_NET": "IB",
    "NCCL_IB_DISABLE": "0",
    "NCCL_NVLS_ENABLE": "0",
    "NCCL_CUMEM_ENABLE": "0",
}
# recipe.yaml serve.env NCCL AR-tail set (tests assert they stay equal).
KEEP_ENV = {
    "NCCL_BUFFSIZE": "1048576",
    "NCCL_LL128_BUFFSIZE": "262144",
    "NCCL_PROTO": "^LL128",
    "NCCL_MAX_NCHANNELS": "8",
}


def _keep(**over: str) -> dict:
    env = dict(KEEP_ENV)
    env.update(over)
    return env


# Phase 1: one knob at a time on top of KEEP. Phase 2 combines the winners.
ARMS: dict[str, dict] = {
    "keep": _keep(),
    "bare": {},
    "proto_ll": _keep(NCCL_PROTO="LL"),
    "proto_simple": _keep(NCCL_PROTO="Simple"),
    "proto_ll128": _keep(NCCL_PROTO="LL128"),
    "algo_ring": _keep(NCCL_ALGO="Ring"),
    "algo_tree": _keep(NCCL_ALGO="allreduce:tree"),  # AllGather has no Tree; a global Tree is invalid usage
    "ch1": _keep(NCCL_MAX_NCHANNELS="1", NCCL_MIN_NCHANNELS="1"),
    "ch2": _keep(NCCL_MAX_NCHANNELS="2", NCCL_MIN_NCHANNELS="2"),
    "ch4": _keep(NCCL_MAX_NCHANNELS="4", NCCL_MIN_NCHANNELS="4"),
    "minch8": _keep(NCCL_MIN_NCHANNELS="8"),
    "nt64": _keep(NCCL_NTHREADS="64"),
    "nt128": _keep(NCCL_NTHREADS="128"),
    "nt256": _keep(NCCL_NTHREADS="256"),
    "nt512": _keep(NCCL_NTHREADS="512"),
    "ib_inline": _keep(NCCL_IB_USE_INLINE="1"),
    "proxy_big": _keep(NCCL_PROXY_CPUSET="15-19"),
    "proxy_little": _keep(NCCL_PROXY_CPUSET="10-14"),
    "ignore_cpu_aff": _keep(NCCL_IGNORE_CPU_AFFINITY="1"),
    "graph_mixing0": _keep(NCCL_GRAPH_MIXING_SUPPORT="0"),
    "gdr_c2c": _keep(NCCL_NET_GDR_C2C="1"),
    # Phase 2 (after phase 1: only graph_mixing0 beat KEEP)
    "mix0_proxy_big": _keep(NCCL_GRAPH_MIXING_SUPPORT="0", NCCL_PROXY_CPUSET="15-19"),
    "mix0_bare": {"NCCL_GRAPH_MIXING_SUPPORT": "0"},
    # 2.30+: no capture-time serialization of comm kernels; valid only with mixing off
    "mix0_so0": _keep(NCCL_GRAPH_MIXING_SUPPORT="0", NCCL_GRAPH_STREAM_ORDERING="0"),
}

MODES = ("eager", "graph", "gapped")
GRAPH_CALLS = 30
STEADY_FROM = 5
EAGER_ITERS = 300
GRAPH_REPLAYS = int(os.environ.get("NCCL_SWEEP_REPLAYS", "200"))
WARMUP = 20
EAGER_SPIN_CYCLES = 100_000_000  # ~45 ms at 2.2 GHz, longer than queuing 300 calls


def arm_env(name: str) -> dict:
    """Full NCCL env for one arm: serve base plus the arm's knobs."""
    if name not in ARMS:
        raise KeyError(f"unknown arm {name!r}; known: {', '.join(ARMS)}")
    env = dict(BASE_ENV)
    env.update(ARMS[name])
    return env


def decode_specs(hidden: int = HIDDEN, vocab: int = VOCAB, tp: int = TP) -> list[dict]:
    """(op, rows, bytes in per rank) for every decode collective size."""
    out = [{"op": "all_reduce", "rows": m, "bytes": m * hidden * BF16} for m in AR_ROWS]
    out += [{"op": "all_gather", "rows": m, "bytes": m * (vocab // tp) * BF16} for m in AG_ROWS]
    return out


def quantile(xs: list[float], q: float) -> float:
    """Linear-interpolated quantile (numpy 'linear'), stdlib only."""
    s = sorted(xs)
    if not s:
        raise ValueError("empty")
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def stats(us: list[float]) -> dict:
    return {
        "median_us": round(statistics.median(us), 3),
        "p10_us": round(quantile(us, 0.10), 3),
        "p90_us": round(quantile(us, 0.90), 3),
        "mean_us": round(statistics.fmean(us), 3),
        "n": len(us),
    }


def graph_costs(with_slots: list[list[float]], ref_slots: list[list[float]],
                with_first: list[float], ref_first: list[float]) -> dict:
    """Per-slot costs of the calls from in-graph event timings.

    with_slots / ref_slots: per replay, the G slot periods (us) of the graph with
    and without the calls. with_first / ref_first: per replay, GPU time from a
    mark recorded just before the launch to the graph's first node.
    """
    calls = len(with_slots[0])
    if any(len(r) != calls for r in with_slots + ref_slots) or calls <= STEADY_FROM:
        raise ValueError("every replay needs the same number of slots, more than STEADY_FROM")
    ref_med = [statistics.median(r[s] for r in ref_slots) for s in range(calls)]
    cost = [[r[s] - ref_med[s] for s in range(calls)] for r in with_slots]
    steady = [c[s] for c in cost for s in range(STEADY_FROM, calls)]
    st = statistics.median(steady)
    first = [statistics.median(c[s] for c in cost) - st for s in range(STEADY_FROM)]
    launch = statistics.median(with_first) - statistics.median(ref_first)
    return {"steady": steady, "first_slots_excess_us": [round(x, 2) for x in first],
            "launch_excess_us": round(launch, 2), "startup_us": round(sum(first), 2),
            "replay_cost": [sum(c) for c in cost]}


def run(args) -> int:
    import torch
    import torch.distributed as dist
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    torch.cuda.set_device(0)
    dev = torch.device("cuda:0")
    dist.init_process_group("gloo", init_method=f"tcp://{args.master}", rank=args.rank, world_size=2)
    comm = PyNcclCommunicator(group=dist.group.WORLD, device=dev)
    l2 = int(torch.cuda.get_device_properties(0).L2_cache_size)
    flush = torch.empty(max(32 << 20, 2 * l2), dtype=torch.uint8, device=dev)
    ev = lambda: torch.cuda.Event(enable_timing=True)  # noqa: E731

    def sync():
        torch.cuda.synchronize()
        dist.barrier()

    def op_fn(spec: dict):
        n = spec["bytes"] // BF16
        x = torch.randn(n, dtype=torch.bfloat16, device=dev)
        if spec["op"] == "all_reduce":
            out = torch.empty_like(x)
            return lambda: comm.all_reduce(x, out)
        out = torch.empty(n * 2, dtype=torch.bfloat16, device=dev)
        return lambda: comm.all_gather(out, x)

    def capture(body) -> "torch.cuda.CUDAGraph":
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            body()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            body()
        torch.cuda.synchronize()
        return g

    def body(f, n: int, gap: bool, call: bool, marks: list):
        def b():
            for i in range(n):
                marks[i].record()
                if gap:
                    flush.zero_()
                if call:
                    f()
            marks[n].record()
        return b

    def replay_ab(graphs: dict, marks: dict) -> dict:
        """Alternate replays of every graph; per replay: slot periods, first-node
        wait after the launch, host launch wall time."""
        for _ in range(WARMUP // 2):
            for g in graphs.values():
                g.replay()
        sync()
        out = {k: {"slots": [], "first": [], "host": []} for k in graphs}
        for _ in range(GRAPH_REPLAYS):
            for k, g in graphs.items():
                pre = ev()
                pre.record()
                t0 = time.perf_counter()
                g.replay()
                host = time.perf_counter() - t0
                torch.cuda.synchronize()
                m = marks[k]
                out[k]["slots"].append([m[i].elapsed_time(m[i + 1]) * 1e3 for i in range(len(m) - 1)])
                out[k]["first"].append(pre.elapsed_time(m[0]) * 1e3)
                out[k]["host"].append(host * 1e6)
        return out

    rows = []
    for spec in decode_specs():
        f = op_fn(spec)
        for _ in range(WARMUP):
            f()
        sync()
        pairs = [(ev(), ev()) for _ in range(EAGER_ITERS)]
        torch.cuda._sleep(EAGER_SPIN_CYCLES)
        for a, b in pairs:
            a.record()
            f()
            b.record()
        torch.cuda.synchronize()
        rows.append({**spec, "mode": "eager", **stats([a.elapsed_time(b) * 1e3 for a, b in pairs])})
        sync()

        for mode in ("graph", "gapped"):
            gap = mode == "gapped"
            marks = {k: [torch.cuda.Event(enable_timing=True, external=True) for _ in range(GRAPH_CALLS + 1)]
                     for k in ("with", "ref")}
            gs = {k: capture(body(f, GRAPH_CALLS, gap, k == "with", marks[k])) for k in ("with", "ref")}
            t = replay_ab(gs, marks)
            c = graph_costs(t["with"]["slots"], t["ref"]["slots"], t["with"]["first"], t["ref"]["first"])
            rows.append({**spec, "mode": mode, "calls_per_graph": GRAPH_CALLS,
                         "startup_us": c["startup_us"], "first_slots_excess_us": c["first_slots_excess_us"],
                         "launch_excess_us": c["launch_excess_us"],
                         "replay_cost": stats(c["replay_cost"]),
                         "ref_slot_us": round(statistics.median(x for r in t["ref"]["slots"] for x in r), 3),
                         "host_launch_with": stats(t["with"]["host"]), "host_launch_ref": stats(t["ref"]["host"]),
                         **stats(c["steady"])})
            del gs
            sync()
        if args.rank == 0:
            for r in rows[-3:]:
                fx = (f" startup {r['startup_us']:8.1f} host launch {r['host_launch_with']['median_us']:7.0f}"
                      if "startup_us" in r else "")
                print(f"{r['op']:<10} m={r['rows']} {r['bytes']:>8d} B {r['mode']:<6} "
                      f"med {r['median_us']:8.2f} p10 {r['p10_us']:8.2f} p90 {r['p90_us']:8.2f} us{fx}", flush=True)
    res = {
        "arm": args.arm,
        "rank": args.rank,
        "host": socket.gethostname(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "nccl_version": comm.nccl.ncclGetVersion(),
        "torch": torch.__version__,
        "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("NCCL_")},
        "l2_bytes": l2,
        "flush_bytes": flush.numel(),
        "graph_replays": GRAPH_REPLAYS,
        "rows": rows,
    }
    comm.destroy()
    dist.destroy_process_group()
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1) + "\n")
    return 0


def key(r: dict) -> tuple:
    return (r["op"], r["rows"], r["mode"])


def summarize(root: Path, weights: dict | None = None) -> dict:
    """Median over reps of each (arm, op, rows, mode) cell, rank 0 JSONs.

    weights: {"c1": {(op, rows): calls_per_step, ("eager", op, rows): calls_per_step,
                     ("startup", op, rows): replays_per_step}}
    -> modeled ms/step = sum(calls x gapped steady) + sum(eager calls x eager median)
    + sum(replays x gapped startup measured at the replay's first collective size).
    """
    arms: dict[str, dict] = {}
    for f in sorted(root.glob("*/rep*.rank0.json")):
        d = json.loads(f.read_text())
        drop = ("NCCL_DEBUG", "NCCL_DEBUG_SUBSYS", "NCCL_SOCKET_IFNAME", "NCCL_DEBUG_FILE")
        a = arms.setdefault(d["arm"], {"reps": 0, "cells": {}, "nccl_version": d["nccl_version"],
                                       "env": {k: v for k, v in d["env"].items() if k not in drop}})
        a["reps"] += 1
        for r in d["rows"]:
            a["cells"].setdefault(key(r), []).append(r)
    out = {"arms": {}}
    base = arms.get("keep")

    def med(rs, *path):
        vals = []
        for r in rs:
            for p in path:
                r = r[p]
            vals.append(r)
        return round(statistics.median(vals), 2)

    for name, a in arms.items():
        cells = []
        for k, rs in sorted(a["cells"].items()):
            c = {"op": k[0], "rows": k[1], "mode": k[2], "bytes": rs[0]["bytes"], "reps": len(rs),
                 "median_us": med(rs, "median_us"), "p10_us": med(rs, "p10_us"), "p90_us": med(rs, "p90_us"),
                 "rep_medians_us": [r["median_us"] for r in rs]}
            if "startup_us" in rs[0]:
                c["startup_us"] = med(rs, "startup_us")
                c["rep_startup_us"] = [r["startup_us"] for r in rs]
                c["host_launch_us"] = med(rs, "host_launch_with", "median_us")
            if base and name != "keep" and k in base["cells"]:
                b = statistics.median(r["median_us"] for r in base["cells"][k])
                c["vs_keep_pct"] = round(100.0 * (c["median_us"] / b - 1.0), 1) if b > 0 else None
            cells.append(c)
        entry = {"reps": a["reps"], "env": a["env"], "nccl_version": a["nccl_version"], "cells": cells}
        if weights:
            gap = {(c["op"], c["rows"]): c for c in cells if c["mode"] == "gapped"}
            eag = {(c["op"], c["rows"]): c for c in cells if c["mode"] == "eager"}
            model = {}
            for w, wt in weights.items():
                us = 0.0
                for k, n in wt.items():
                    if k[0] == "startup":
                        us += n * gap[k[1:]]["startup_us"]
                    elif k[0] == "eager":
                        us += n * eag[k[1:]]["median_us"]
                    else:
                        us += n * gap[k]["median_us"]
                model[w] = round(us / 1e3, 3)
            entry["modeled_ms_per_step"] = model
        out["arms"][name] = entry
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--rank", type=int, required=True)
    r.add_argument("--master", required=True, help="host:port reachable from both ranks")
    r.add_argument("--arm", required=True)
    r.add_argument("--json")
    e = sub.add_parser("arm-env")
    e.add_argument("arm")
    sub.add_parser("arms")
    s = sub.add_parser("summarize")
    s.add_argument("dir")
    s.add_argument("--out")
    s.add_argument("--weights", help='JSON {name: [[op, rows, calls], ["eager", op, rows, calls], ["startup", op, rows, replays]]}')
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return run(args)
    if args.cmd == "arm-env":
        for k, v in arm_env(args.arm).items():
            print(f"{k}={v}")
        return 0
    if args.cmd == "arms":
        print(" ".join(ARMS))
        return 0
    weights = None
    if args.weights:
        raw = json.loads(Path(args.weights).read_text())
        weights = {}
        for w, rows in raw.items():
            weights[w] = {}
            for row in rows:
                if row[0] in ("startup", "eager"):
                    weights[w][(row[0], row[1], int(row[2]))] = float(row[3])
                else:
                    weights[w][(row[0], int(row[1]))] = float(row[2])
    res = summarize(Path(args.dir), weights)
    text = json.dumps(res, indent=1) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
