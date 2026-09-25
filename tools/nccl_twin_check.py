#!/usr/bin/env python3
"""Two-node check of DSV41_NCCL_EAGER_TWIN on the serve's decode collective pattern.

Serve DOWN, both GPUs exclusive. One process per rank per arm (NCCL reads
NCCL_GRAPH_MIXING_SUPPORT at communicator init), arms alternated by
tools/nccl_twin_check.sh:

  keep       one PyNcclCommunicator for everything, mixing on (the serve today)
  twin       stock comm + eager twin behind nccl_eager_twin.GraphEagerRouter,
             NCCL_GRAPH_MIXING_SUPPORT=0 (the lever)
  twin_so0   twin + NCCL_GRAPH_STREAM_ORDERING=0 (diagnostic)
  keep_qos / twin_qos  keep / twin with docker/patch/pm_qos.py holding a 20 us
             /dev/cpu_dma_latency request (the graph-startup wake-up, iteration #7);
             the driver runs these as root with the device
  mix0_single  one communicator, mixing off, eager calls during outstanding graphs:
             the case NCCL does not support. Diagnostic only, never in the default
             arm list; a hang here is the reason the twin exists.

Decode step (sizes from the model config, counts from the r3 c=1 profile, m=4/3 at
c=1, --m 8 --md 6 for c=2):
  target graph  40 x [work A ; AR m ; work C ; work D ; AR m], engram AG at layers 1, 14
  eager         logits AG (m x vocab/TP) right after the target launch (graph outstanding)
  draft graph   embed AR md + 3 x [work ; AR md ; work ; AR md] + logits AG (md)
  eager         embed AR m for the next step, while the draft graph is outstanding
work(us) is a memset sized from a per-rank bandwidth calibration, so the gaps between
collectives match the serve's (A 431 us, C 149 us, D 547 us per layer; --scale shrinks
them). The host stays one step ahead, like the serve.

Every collective input is a small integer derived from a per-step seed that the graph
reads at replay (exact in bf16): each checked step verifies every output bitwise.
Per-step GPU time comes from CUDA events around the step, reported as median/p10/p90
over --steps after --warmup; a watchdog aborts on a hang (the driver's timeout too).

  run:      python3 -S nccl_twin_check.py run --rank R --master HOST:PORT --arm ARM --json OUT
  arms:     python3 nccl_twin_check.py arms
  arm-env:  python3 nccl_twin_check.py arm-env ARM
  compare:  python3 nccl_twin_check.py compare DIR
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys
import threading
import time
from pathlib import Path

HIDDEN = 5120
VOCAB = 129280
TP = 2
LAYERS = 40
DRAFT_LAYERS = 3
ENGRAM_LAYERS = (1, 14)
ENGRAM_ROW = 3072  # engram rows per token per rank (12 hash columns x 256)
# Per-layer serve gaps (r3 c=1 layer budget, median us): attention block, FFN prologue, p2b.
WORK_US = {"A": 431.5, "C": 149.4, "D": 546.6, "draft": 300.0}

ARMS = {
    "keep": {},
    "twin": {"NCCL_GRAPH_MIXING_SUPPORT": "0"},
    "twin_so0": {"NCCL_GRAPH_MIXING_SUPPORT": "0", "NCCL_GRAPH_STREAM_ORDERING": "0"},
    "mix0_single": {"NCCL_GRAPH_MIXING_SUPPORT": "0"},
    "keep_qos": {"DSV41_PM_QOS_US": "20"},
    "twin_qos": {"NCCL_GRAPH_MIXING_SUPPORT": "0", "DSV41_PM_QOS_US": "20"},
}
DEFAULT_ARMS = ("keep", "twin")
BASE_ENV = {  # serve NCCL env (run.sh docker args + recipe KEEP set)
    "NCCL_IB_HCA": "rocep1s0f1",
    "NCCL_CROSS_NIC": "1",
    "NCCL_NET": "IB",
    "NCCL_IB_DISABLE": "0",
    "NCCL_NVLS_ENABLE": "0",
    "NCCL_CUMEM_ENABLE": "0",
    "NCCL_BUFFSIZE": "1048576",
    "NCCL_LL128_BUFFSIZE": "262144",
    "NCCL_PROTO": "^LL128",
    "NCCL_MAX_NCHANNELS": "8",
}


def arm_env(name: str) -> dict:
    if name not in ARMS:
        raise KeyError(f"unknown arm {name!r}; known: {', '.join(ARMS)}")
    env = dict(BASE_ENV)
    env.update(ARMS[name])
    return env


def seed_of(step: int) -> int:
    """Per-step seed 1..40: rank r sends seed*(r+1), so AR sums stay <= 120 (exact in bf16)."""
    return 1 + step % 40


def expected_ar(seed: int, tp: int = TP) -> int:
    return seed * sum(r + 1 for r in range(tp))


def expected_ag_halves(seed: int, tp: int = TP) -> list[int]:
    return [seed + r for r in range(tp)]


def is_checked(step: int, every: int, total: int) -> bool:
    """Steps whose outputs are verified (the pipeline is drained after each)."""
    return step >= 0 and (step % every == 0 or step == total - 1)


def plan(m: int, md: int, layers: int = LAYERS, draft_layers: int = DRAFT_LAYERS) -> dict:
    """Ordered collectives per step: (phase, op, rows, bytes per rank)."""
    ar = lambda rows: rows * HIDDEN * 2  # noqa: E731
    target = []
    for layer in range(layers):
        if layer in ENGRAM_LAYERS:
            target.append(("target", "all_gather", m, m * ENGRAM_ROW * 2))
        target += [("target", "all_reduce", m, ar(m)), ("target", "all_reduce", m, ar(m))]
    draft = [("draft", "all_reduce", md, ar(md))]
    for _ in range(draft_layers):
        draft += [("draft", "all_reduce", md, ar(md)), ("draft", "all_reduce", md, ar(md))]
    draft.append(("draft", "all_gather", md, md * (VOCAB // TP) * 2))
    eager = [("eager", "all_gather", m, m * (VOCAB // TP) * 2), ("eager", "all_reduce", m, ar(m))]
    return {"target": target, "draft": draft, "eager": eager}


def stats(xs: list[float]) -> dict:
    s = sorted(xs)
    q = lambda f: s[min(len(s) - 1, int(round(f * (len(s) - 1))))]  # noqa: E731
    return {"median": round(statistics.median(s), 3), "p10": round(q(0.10), 3), "p90": round(q(0.90), 3),
            "mean": round(statistics.fmean(s), 3), "n": len(s)}


def one_rank_comm(dev):
    """A real one-rank PyNcclCommunicator (the class refuses world_size 1 on its own)."""
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary

    c = PyNcclCommunicator.__new__(PyNcclCommunicator)
    c.rank, c.world_size, c.group = 0, 1, None
    c.nccl = NCCLLibrary(None)
    c.available, c.disabled, c._suspended = True, False, False
    c.nccl_version = c.nccl.ncclGetRawVersion()
    c.unique_id = c.nccl.ncclGetUniqueId()
    c._init_comm(dev)
    return c


def run(args) -> int:
    import torch
    import torch.distributed as dist

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docker" / "patch"))
    import nccl_eager_twin as nt
    import pm_qos

    qos_state = pm_qos.install()  # arms with DSV41_PM_QOS_US; must hold before the first NCCL wake-up
    if os.environ.get(pm_qos.ENV) and qos_state != "armed":
        raise RuntimeError(f"{args.arm}: PM QoS request not held ({qos_state})")
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    torch.cuda.set_device(0)
    dev = torch.device("cuda:0")
    world = 1 if args.selftest_1rank else TP
    if args.selftest_1rank:  # one GPU: real one-rank NCCL comms, no network (harness check only)
        new_comm = lambda: one_rank_comm(dev)  # noqa: E731
        barrier = torch.cuda.synchronize
    else:
        dist.init_process_group("gloo", init_method=f"tcp://{args.master}", rank=args.rank, world_size=TP)
        new_comm = lambda: PyNcclCommunicator(group=dist.group.WORLD, device=dev)  # noqa: E731
        barrier = dist.barrier
    graph_comm = new_comm()
    if args.arm.startswith("twin"):
        twin = new_comm()
        comm = nt.GraphEagerRouter(graph_comm, twin, nt.stream_is_capturing, "tp:check",
                                   nt.stream_positions(PyNcclCommunicator), lambda m: print(m, flush=True))
    else:
        comm = graph_comm

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    # --- work calibration: memset GB/s on this rank, then bytes per work slot
    big = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    for _ in range(5):
        big.zero_()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(20):
        big.zero_()
    e1.record()
    e1.synchronize()
    gbps = 20 * big.numel() / (e0.elapsed_time(e1) * 1e-3) / 1e9
    work_bytes = {k: max(1 << 20, int(v * args.scale * 1e-6 * gbps * 1e9) & ~0xFFFF) for k, v in WORK_US.items()}
    work = {k: big[:n] for k, n in work_bytes.items()}

    p = plan(args.m, args.md)
    seed_t = torch.ones(1, dtype=torch.bfloat16, device=dev)  # read by the graphs at replay
    mult = float(args.rank + 1)

    def make_calls(items):
        calls = []
        for phase, op, rows, nbytes in items:
            n = nbytes // 2
            x = torch.zeros(n, dtype=torch.bfloat16, device=dev)
            if op == "all_reduce":
                out = torch.empty_like(x)
                calls.append({"phase": phase, "op": op, "rows": rows, "x": x, "out": out})
            else:
                out = torch.empty(n * world, dtype=torch.bfloat16, device=dev)
                calls.append({"phase": phase, "op": op, "rows": rows, "x": x, "out": out})
        return calls

    def issue(c):
        # input from the device seed: captured as a kernel that reads seed_t at replay
        if c["op"] == "all_reduce":
            torch.mul(seed_t.expand_as(c["x"]), mult, out=c["x"])
            comm.all_reduce(c["x"], c["out"])
        else:
            torch.add(seed_t.expand_as(c["x"]), float(args.rank), out=c["x"])
            comm.all_gather(c["out"], c["x"])

    target_calls, draft_calls, eager_calls = (make_calls(p[k]) for k in ("target", "draft", "eager"))

    def target_body():
        it = iter(target_calls)
        for layer in range(LAYERS):
            if layer in ENGRAM_LAYERS:
                issue(next(it))
            work["A"].zero_()
            issue(next(it))
            work["C"].zero_()
            work["D"].zero_()
            issue(next(it))

    def draft_body():
        it = iter(draft_calls)
        issue(next(it))
        for _ in range(DRAFT_LAYERS):
            work["draft"].zero_()
            issue(next(it))
            work["draft"].zero_()
            issue(next(it))
        issue(next(it))

    def capture(body):
        body()  # eager warm-up on the eager path (twin arm: twin)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            body()
        torch.cuda.synchronize()
        return g

    g_target, g_draft = capture(target_body), capture(draft_body)
    barrier()

    stop = threading.Event()
    progress = {"step": -1, "t": time.monotonic()}

    def watchdog():
        while not stop.wait(5.0):
            if time.monotonic() - progress["t"] > args.hang_s:
                print(f"HANG: no step completed for {args.hang_s}s after step {progress['step']}", flush=True)
                os._exit(3)

    threading.Thread(target=watchdog, daemon=True).start()

    total = args.warmup + args.steps
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True),
           torch.cuda.Event(enable_timing=True)) for _ in range(total)]
    mismatches, checked = [], 0
    prev_end = None
    for step in range(total):
        seed_t.fill_(float(seed_of(step)))
        s0, s_mid, s1 = ev[step]
        s0.record()
        g_target.replay()
        issue(eager_calls[0])  # logits AG while the target graph is outstanding
        s_mid.record()
        g_draft.replay()
        issue(eager_calls[1])  # next-step embed AR while the draft graph is outstanding
        s1.record()
        if prev_end is not None:
            prev_end.synchronize()
        prev_end = s1
        progress.update(step=step - 1, t=time.monotonic())
        if is_checked(step, args.check_every, total):
            s1.synchronize()
            seed = seed_of(step)
            for c in target_calls + draft_calls + eager_calls:
                got = c["out"].float()
                if c["op"] == "all_reduce":
                    ok = bool((got == expected_ar(seed, world)).all())
                else:
                    halves = got.view(world, -1)
                    ok = all(bool((halves[r] == v).all()) for r, v in enumerate(expected_ag_halves(seed, world)))
                if not ok:
                    mismatches.append({"step": step, "phase": c["phase"], "op": c["op"], "rows": c["rows"]})
            checked += 1
    torch.cuda.synchronize()
    stop.set()
    # A checked step drains the pipeline; the step after it starts from an idle GPU.
    timed = [i for i in range(args.warmup, total) if not is_checked(i - 1, args.check_every, total)]
    step_us = [ev[i][0].elapsed_time(ev[i][2]) * 1e3 for i in timed]
    target_us = [ev[i][0].elapsed_time(ev[i][1]) * 1e3 for i in timed]
    draft_us = [ev[i][1].elapsed_time(ev[i][2]) * 1e3 for i in timed]
    res = {
        "arm": args.arm,
        "rank": args.rank,
        "world": world,
        "pm_qos": qos_state,
        "host": socket.gethostname(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "nccl_version": graph_comm.nccl.ncclGetVersion(),
        "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("NCCL_")},
        "m": args.m,
        "md": args.md,
        "scale": args.scale,
        "memset_gbps": round(gbps, 1),
        "work_bytes": work_bytes,
        "collectives_per_step": {k: len(v) for k, v in p.items()},
        "steps": args.steps,
        "warmup": args.warmup,
        "timed_steps": len(timed),
        "checked_steps": checked,
        "mismatches": mismatches[:20],
        "n_mismatches": len(mismatches),
        "step_us": stats(step_us),
        "target_plus_eager_ag_us": stats(target_us),
        "draft_plus_eager_ar_us": stats(draft_us),
        "router_counts": ({"graph": comm._n_graph, "eager": comm._n_eager}
                          if isinstance(comm, nt.GraphEagerRouter) else None),
    }
    print(json.dumps({k: res[k] for k in ("arm", "rank", "step_us", "n_mismatches", "router_counts")}), flush=True)
    del g_target, g_draft
    comm.destroy()
    if not args.selftest_1rank:
        dist.destroy_process_group()
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1) + "\n")
    return 0 if not mismatches else 2


def compare(root: Path) -> dict:
    """Pool rank-0 JSONs per arm; delta of per-rep medians vs keep."""
    arms: dict[str, list[dict]] = {}
    for f in sorted(root.glob("*/rep*.rank0.json")):
        d = json.loads(f.read_text())
        arms.setdefault(d["arm"], []).append(d)
    out = {}
    for arm, runs in arms.items():
        meds = [r["step_us"]["median"] for r in runs]
        out[arm] = {
            "reps": len(runs),
            "step_us_rep_medians": meds,
            "step_us_median_of_medians": round(statistics.median(meds), 2),
            "p10_p90_of_first_rep": [runs[0]["step_us"]["p10"], runs[0]["step_us"]["p90"]],
            "mismatches": sum(r["n_mismatches"] for r in runs),
        }
    if "keep" in out:
        base = out["keep"]["step_us_median_of_medians"]
        for arm, v in out.items():
            v["delta_vs_keep_ms"] = round((v["step_us_median_of_medians"] - base) / 1e3, 3)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--rank", type=int, required=True)
    r.add_argument("--master", required=True)
    r.add_argument("--arm", required=True, choices=sorted(ARMS))
    r.add_argument("--json")
    r.add_argument("--m", type=int, default=4, help="target verify rows (4 at c=1, 8 at c=2)")
    r.add_argument("--md", type=int, default=3, help="draft rows (3 at c=1, 6 at c=2)")
    r.add_argument("--steps", type=int, default=300)
    r.add_argument("--warmup", type=int, default=30)
    r.add_argument("--check-every", type=int, default=10)
    r.add_argument("--scale", type=float, default=1.0, help="scale the work gaps between collectives")
    r.add_argument("--hang-s", type=float, default=60.0)
    r.add_argument("--selftest-1rank", action="store_true", help="one GPU, one-rank comms: harness check only")
    sub.add_parser("arms")
    e = sub.add_parser("arm-env")
    e.add_argument("arm")
    c = sub.add_parser("compare")
    c.add_argument("dir")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return run(args)
    if args.cmd == "arms":
        print(" ".join(DEFAULT_ARMS))
        return 0
    if args.cmd == "arm-env":
        for k, v in arm_env(args.arm).items():
            print(f"{k}={v}")
        return 0
    print(json.dumps(compare(Path(args.dir)), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
