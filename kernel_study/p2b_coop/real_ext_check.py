#!/usr/bin/env python3
"""The real vllm_exl3_c (built from the docker/Dockerfile.e15 recipe) vs the bench module (GPU).

  python3 kernel_study/p2b_coop/real_ext_check.py --so-dir kernel_study/p2b_coop/build/e15so \
      --out results/2026-09-25-kernels/coop-moe/real-ext-check.json

The parent builds the inputs (layer 20, census routings at m = 1, 4, 8; real and one-hot routing
weights), runs the bench module (bench_r3: variants 0, 1, 2) and saves both. For DSV41_P2B_COOP
unset, "0", "1" and "2" it then starts a child with PYTHONPATH=<so-dir> that imports that
vllm_exl3_c (path checked), runs the same p2b_fused_moe calls and returns the outputs and its
stderr. Pass when:
  unset / "0": one-hot outputs == bench v0 (p2b; with one-hot weights its atomics add exact zeros)
  "1": real-weight outputs == bench v1, "2": == bench v2 and == real "1" (all bitwise)
  the engaged line is on stderr for "2" only, the disarm line never
Cold timing (flush before every call, 100 per arm) of real "2" vs bench v2 is reported as a sanity
check that the image build runs the same code at the same speed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

LOG_ENGAGED = "dsv41: p2b coop dataflow kernel engaged (DSV41_P2B_COOP=2)"
LOG_DISARMED = "dsv41: p2b coop dataflow lever is OFF for this call"


def child(inputs: Path, out: Path, so_dir: Path) -> int:
    import torch

    mode = os.environ.get("DSV41_P2B_COOP")
    import bench_r3 as br  # pops DSV41_P2B_COOP for its own module; put it back for the real .so

    if mode is not None:
        os.environ["DSV41_P2B_COOP"] = mode
    import vllm_exl3_c

    got = Path(vllm_exl3_c.__file__).resolve().parent
    if got != so_dir.resolve():
        raise SystemExit(f"imported {vllm_exl3_c.__file__}, not the build under {so_dir}")
    data = torch.load(inputs)
    layer = br.Layer(data["layer"], 0, "cuda")
    res = {}
    for name, case in data["cases"].items():
        x, ids = case["x"].cuda(), case["ids"].cuda()
        for wname in ("rw", "onehot"):
            o = torch.empty_like(x)
            vllm_exl3_c.p2b_fused_moe(x, o, *layer.tables, ids, case[wname].cuda(), 2, 2, 2, True, br.INTER, br.SWIGLU_LIMIT)
            res[f"{name}/{wname}"] = o.cpu()
    timing = {}
    if os.environ.get("DSV41_P2B_COOP") == "2":
        flush = br.Flusher("cuda")
        for name, case in data["cases"].items():
            if not name.endswith("t0"):
                continue
            x, ids, rw = case["x"].cuda(), case["ids"].cuda(), case["rw"].cuda()
            o = torch.empty_like(x)
            ts = []
            for _ in range(100):
                flush()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                vllm_exl3_c.p2b_fused_moe(x, o, *layer.tables, ids, rw, 2, 2, 2, True, br.INTER, br.SWIGLU_LIMIT)
                e1.record()
                ts.append((e0, e1))
            torch.cuda.synchronize()
            timing[name] = statistics.median(a.elapsed_time(b) * 1000.0 for a, b in ts)
    torch.save({"out": res, "timing": timing, "so": str(vllm_exl3_c.__file__)}, out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--so-dir", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--routings", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--child", nargs=2, type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        return child(args.child[0], args.child[1], args.so_dir)

    import torch

    import bench_r3 as br

    work = HERE / "build" / "real_ext_check"
    work.mkdir(parents=True, exist_ok=True)
    ext = br.build_ext(False)
    layer = br.Layer(args.layer, 0, "cuda")
    routes = br.Routings(args.layer, random.Random(5))
    b = br.Bench(ext, layer, "cuda", 5)
    cases = {}
    for m in (1, 4, 8):
        for t in range(args.routings):
            rows = routes.draw("census", m)
            rw = b.rw(m)
            onehot = torch.zeros_like(rw)
            onehot[:, t % br.TOPK] = 1.0
            cases[f"m{m}/t{t}"] = {"x": b.x(m).cpu(), "ids": b.ids(rows).cpu(), "rw": rw.cpu(), "onehot": onehot.cpu()}
    inputs = work / "inputs.pt"
    torch.save({"layer": args.layer, "cases": cases}, inputs)

    bench = {}
    for v in (0, 1, 2):
        for name, case in cases.items():
            x, ids = case["x"].cuda(), case["ids"].cuda()
            for wname in ("rw", "onehot"):
                bench[(v, f"{name}/{wname}")] = b.once(v, x, ids, case[wname].cuda()).cpu()
    flush = br.Flusher("cuda")
    bench_time = {}
    for name, case in cases.items():
        if not name.endswith("t0"):
            continue
        x, ids, rw = case["x"].cuda(), case["ids"].cuda(), case["rw"].cuda()
        o = torch.empty_like(x)
        ts = []
        for _ in range(100):
            flush()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            b.call(2, x, ids, rw, o)
            e1.record()
            ts.append((e0, e1))
        torch.cuda.synchronize()
        bench_time[name] = statistics.median(a.elapsed_time(c) * 1000.0 for a, c in ts)
    del layer, b
    torch.cuda.empty_cache()

    real = {}
    result = {"so_dir": str(args.so_dir), "cases": len(cases), "modes": {}}
    for mode in ("unset", "0", "1", "2"):
        env = {k: v for k, v in os.environ.items() if k not in ("DSV41_P2B_COOP", "DSV41_P2B_SRC_SORT")}
        if mode != "unset":
            env["DSV41_P2B_COOP"] = mode
        env["PYTHONPATH"] = str(args.so_dir.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
        out = work / f"real_{mode}.pt"
        p = subprocess.run([sys.executable, __file__, "--so-dir", str(args.so_dir), "--out", str(out), "--child", str(inputs), str(out)],
                           env=env, capture_output=True, text=True)
        if p.returncode != 0:
            print(p.stdout[-2000:], p.stderr[-4000:])
            raise SystemExit(f"child DSV41_P2B_COOP={mode} failed ({p.returncode})")
        real[mode] = torch.load(out)
        result["modes"][mode] = {"so": real[mode]["so"], "engaged_line": LOG_ENGAGED in p.stderr,
                                 "disarm_line": LOG_DISARMED in p.stderr, "timing_us": real[mode]["timing"]}

    def eq(a, c):
        return torch.equal(a.view(torch.int16), c.view(torch.int16))

    checks = {
        "unset_onehot_eq_bench_v0": all(eq(real["unset"]["out"][f"{n}/onehot"], bench[(0, f"{n}/onehot")]) for n in cases),
        "mode0_onehot_eq_bench_v0": all(eq(real["0"]["out"][f"{n}/onehot"], bench[(0, f"{n}/onehot")]) for n in cases),
        "mode1_rw_eq_bench_v1": all(eq(real["1"]["out"][f"{n}/rw"], bench[(1, f"{n}/rw")]) for n in cases),
        "mode2_rw_eq_bench_v2": all(eq(real["2"]["out"][f"{n}/rw"], bench[(2, f"{n}/rw")]) for n in cases),
        "mode2_rw_eq_real_mode1": all(eq(real["2"]["out"][f"{n}/rw"], real["1"]["out"][f"{n}/rw"]) for n in cases),
        "mode2_onehot_eq_bench_v0": all(eq(real["2"]["out"][f"{n}/onehot"], bench[(0, f"{n}/onehot")]) for n in cases),
        "engaged_only_mode2": [result["modes"][k]["engaged_line"] for k in ("unset", "0", "1", "2")] == [False, False, False, True],
        "no_disarm_line": not any(result["modes"][k]["disarm_line"] for k in result["modes"]),
    }
    result["checks"] = checks
    result["timing_cold_median_us"] = {"bench_v2": bench_time, "real_mode2": result["modes"]["2"]["timing_us"]}
    result["pass"] = all(checks.values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
