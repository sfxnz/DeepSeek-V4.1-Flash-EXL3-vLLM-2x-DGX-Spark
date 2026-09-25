#!/usr/bin/env python3
"""Single-GPU smoke test of docker/patch/nccl_eager_twin.py on the real stack.

What one GPU can check (the network hazard itself needs two nodes, see
tools/nccl_twin_check.py): the router's capture predicate under real
torch.cuda.graph capture (current stream, joined side stream, explicit stream),
routing of real PyNcclCommunicator objects, NCCL graph capture with
NCCL_GRAPH_MIXING_SUPPORT=0, and graph replays interleaved with eager calls on the
twin while the graph is outstanding. The communicators are real one-rank NCCL
communicators (PyNcclCommunicator refuses world_size 1, so the test builds them
through the class's own _init_comm).

  docker run --rm --gpus all --network none -v <repo>:/repo -w /repo \
    -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages --entrypoint python3 \
    dsv41-flash-exl3-sm121:canonical-e13 -S kernel_study/comm/twin_smoke_1gpu.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("NCCL_GRAPH_MIXING_SUPPORT", "0")  # before any NCCL comm exists
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docker" / "patch"))

import torch  # noqa: E402

import nccl_eager_twin as nt  # noqa: E402


def one_rank_comm(dev):
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--replays", type=int, default=500)
    args = ap.parse_args()
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    res = {"nccl_graph_mixing_support": os.environ.get("NCCL_GRAPH_MIXING_SUPPORT"), "checks": {}}
    chk = res["checks"]

    # 1. capture predicate
    s_main = torch.cuda.Stream()
    s_side = torch.cuda.Stream()
    chk["predicate_outside"] = nt.stream_is_capturing() is False
    g0 = torch.cuda.CUDAGraph()
    x0 = torch.zeros(16, device=dev)
    seen = {}
    with torch.cuda.stream(s_main):
        with torch.cuda.graph(g0, stream=s_main):
            x0.add_(1)
            seen["current"] = nt.stream_is_capturing()
            seen["explicit_capture_stream"] = nt.stream_is_capturing(s_main)
            s_side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s_side):
                x0.add_(1)
                seen["joined_side_current"] = nt.stream_is_capturing()
            torch.cuda.current_stream().wait_stream(s_side)
            seen["idle_stream_explicit"] = nt.stream_is_capturing(torch.cuda.Stream())
    chk["predicate_in_capture"] = seen == {
        "current": True,
        "explicit_capture_stream": True,
        "joined_side_current": True,
        "idle_stream_explicit": False,
    }
    res["predicate_seen"] = seen

    # 2. routing of real communicators
    a, b = one_rank_comm(dev), one_rank_comm(dev)
    logs: list[str] = []
    r = nt.GraphEagerRouter(a, b, nt.stream_is_capturing, "tp:0", nt.stream_positions(PyNcclCommunicator), logs.append)
    n = 5120 * 4
    xe = torch.randn(n, device=dev, dtype=torch.bfloat16)
    oe = r.all_reduce(xe)
    torch.cuda.synchronize()
    chk["eager_goes_to_twin"] = (r._n_eager, r._n_graph) == (1, 0) and torch.equal(oe, xe)

    xg = torch.randn(n, device=dev, dtype=torch.bfloat16)
    og = torch.empty_like(xg)
    ag_in = torch.randn(n, device=dev, dtype=torch.bfloat16)
    ag_out = torch.empty(n, device=dev, dtype=torch.bfloat16)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s_main):
        r.all_reduce(xg, og)  # warm the pattern eagerly (twin)
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s_main):
            xg.mul_(1.0)
            r.all_reduce(xg, og)
            r.all_gather(ag_out, ag_in)
    chk["captured_go_to_graph_comm"] = r._n_graph == 2

    # 3. replays interleaved with eager twin calls while graphs are outstanding
    ok = True
    ev = torch.randn(n, device=dev, dtype=torch.bfloat16)
    eo = torch.empty_like(ev)
    with torch.cuda.stream(s_main):
        for i in range(args.replays):
            xg.copy_(torch.full_like(xg, float(i % 97)))
            g.replay()
            r.all_reduce(ev, eo)  # eager, twin, graph still outstanding on the device
            if i % 50 == 49:
                torch.cuda.synchronize()
                ok &= torch.equal(og, xg) and torch.equal(eo, ev) and torch.equal(ag_out, ag_in)
    torch.cuda.synchronize()
    chk["replays_with_eager_twin_calls_exact"] = bool(ok)
    res["counts"] = {"graph": r._n_graph, "eager": r._n_eager}
    res["router_log"] = logs
    res["nccl_version"] = a.nccl.ncclGetVersion()
    res["torch"] = torch.__version__
    res["device"] = torch.cuda.get_device_name(0)
    res["ok"] = all(chk.values())
    del g, g0
    r.destroy()
    text = json.dumps(res, indent=1)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
