#!/usr/bin/env python3
"""p2b vs coop (DSV41_P2B_COOP) microbench: m=4 x top-6, dup {0, 0.3, 0.5} + census routings.

GPU only, serve DOWN (exclusive GPU). Run inside the serve image with the repo mounted:

  docker run --rm --gpus all --memory 16g -v $PWD:/repo -w /repo --entrypoint python3 \
    dsv41-flash-exl3-sm121:canonical-e13 kernel_study/p2b_coop/driver.py \
    --out results/2026-09-24-review/moe-dedup/coop-microbench.json

Geometry is one V4.1 TP=2 rank per routed layer: hidden 5120, local inter 1152,
384 experts, top-6, K=2 MCG, swiglu_limit 10 (every routed layer has this shape).
Trellis data is random, the same bytes for both arms. Baseline = the served
SORT=0 kernel (DSV41_P2B_SRC_SORT unset); coop = SORT=2.

Routings: synth (tools/moe_census.synth_routing: exactly round(dup*m*k) repeats of
the previous row's experts) and census (windows of m consecutive decode tokens of
one routed layer from the s10 capture, mean dup 0.2987 at m=4).

check, per routing source:
  onehot: one routing weight 1.0 per slot (TOPK passes). Both final sums then add
    exact zeros, so p2b == coop bit for bit unless the tile math differs.
  full: normalized routing weights. Fixed-order vs atomic-order fp32 slot sum, so
    only a tolerance holds: max |coop - p2b| / max |p2b| <= --rel-tol.
  determinism: coop twice on the same inputs must be bitwise equal (p2b's own
    repeat is reported: its atomics may reorder).
capture: coop captured in a CUDA graph, replayed after new routings are copied into
  the static ids; every replay must equal an eager coop call bit for bit.
time: cold = a fresh routing per timed call; warm = one routing repeated.
  Arms alternate per repeat. saving_ms_per_step = (p2b - coop) cold us x --layers.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")
os.environ.pop("DSV41_P2B_SRC_SORT", None)
os.environ.pop("DSV41_P2B_COOP", None)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402

from moe_census import routed_layers, synth_routing  # noqa: E402

EXL = "/usr/local/lib/python3.12/dist-packages/exllamav3/exllamav3_ext"
HIDDEN, INTER, EXPERTS, TOPK = 5120, 1152, 384, 6
SWIGLU_LIMIT = 10.0
CENSUS = ROOT / "results/2026-09-24-review/campaign/s10-diag-census-skew/census_npy"


def build_ext():
    sys.path.insert(0, str(HERE))
    import make_bench

    make_bench.main()
    return load(
        name="p2b_coop_bench",
        sources=[str(HERE / "build" / "bench_coop.cu")],
        extra_include_paths=[EXL, os.path.join(EXL, "quant")],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def make_weights(dev: str):
    kt, nt = HIDDEN // 16, INTER // 16

    def trellis(shape):
        return [torch.randint(-32768, 32767, shape, dtype=torch.int16, device=dev) for _ in range(EXPERTS)]

    def vec(n):
        # +-[0.9, 1.1] like upstream tests/test_moe_coop.py rand_scale: keeps outputs finite.
        def one():
            sign = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
            return ((torch.rand(n, device=dev) * 0.2 + 0.9) * sign).half()

        return [one() for _ in range(EXPERTS)]

    tensors = (
        trellis((kt, nt, 32)), vec(HIDDEN), vec(INTER),
        trellis((kt, nt, 32)), vec(HIDDEN), vec(INTER),
        trellis((nt, kt, 32)), vec(INTER), vec(HIDDEN),
    )
    tables = [torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device=dev) for ts in tensors]
    return tensors, tables


class Routings:
    """[m, TOPK] expert-id lists: synth at a dup rate, or census windows."""

    def __init__(self, m: int, census_dir: Path | None, rng: random.Random) -> None:
        self.m, self.rng = m, rng
        self.windows: list[np.ndarray] = []
        if census_dir is not None and census_dir.is_dir():
            for path in sorted(census_dir.glob("*.npy")):
                arr = np.load(path, allow_pickle=False)
                for layer in routed_layers(arr):
                    self.windows.append(arr[:, layer, :].astype(np.int64))

    def sources(self, dups: list[float]) -> list[str]:
        return [f"dup{d:g}" for d in dups] + (["census"] if self.windows else [])

    def draw(self, source: str) -> list[list[int]]:
        if source == "census":
            ids = self.windows[self.rng.randrange(len(self.windows))]
            t0 = self.rng.randrange(ids.shape[0] - self.m + 1)
            return ids[t0 : t0 + self.m].tolist()
        return synth_routing(self.m, TOPK, EXPERTS, float(source[3:]), self.rng)


def unique_ratio(rows: list[list[int]]) -> float:
    return len({x for r in rows for x in r}) / sum(len(r) for r in rows)


class Bench:
    def __init__(self, ext, tables, m: int, dev: str) -> None:
        self.ext, self.tables, self.m, self.dev = ext, tables, m, dev
        self.x = torch.randn(m, HIDDEN, dtype=torch.half, device=dev)
        rw = torch.rand(m, TOPK, device=dev)
        self.rw = (rw / rw.sum(dim=1, keepdim=True)).half()

    def ids(self, rows: list[list[int]]) -> torch.Tensor:
        return torch.tensor(rows, dtype=torch.int32, device=self.dev)

    def call(self, coop: int, ids: torch.Tensor, rw: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        self.ext.set_coop(coop)
        self.ext.p2b_fused_moe(self.x, out, *self.tables, ids, rw, 2, 2, 2, True, INTER, SWIGLU_LIMIT)
        return out

    def once(self, coop: int, ids: torch.Tensor, rw: torch.Tensor) -> torch.Tensor:
        return self.call(coop, ids, rw, torch.empty_like(self.x)).clone()


def bitwise(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def ulp_dist(a: torch.Tensor, b: torch.Tensor) -> int:
    """Max fp16 ulp distance (sign-magnitude bits mapped to a monotone integer line)."""

    def line(t: torch.Tensor) -> torch.Tensor:
        i = t.view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -(i & 0x7FFF), i)

    return int((line(a) - line(b)).abs().max().item())


def check(b: Bench, routes: Routings, sources, routings: int, rel_tol: float) -> dict:
    res = {}
    for src in sources:
        onehot_ok, onehot_ulp = True, 0
        coop_repeat_ok = p2b_repeat_ok = finite = True
        max_abs = max_rel = 0.0
        max_ulp = 0
        frac_diff = []
        uniq = []
        for _ in range(routings):
            rows = routes.draw(src)
            uniq.append(unique_ratio(rows))
            ids = b.ids(rows)
            for slot in range(TOPK):
                rw = torch.zeros_like(b.rw)
                rw[:, slot] = 1.0
                p2b, coop = b.once(0, ids, rw), b.once(1, ids, rw)
                onehot_ok &= bitwise(p2b, coop)
                onehot_ulp = max(onehot_ulp, ulp_dist(p2b, coop))
                finite &= bool(torch.isfinite(p2b).all()) and bool(torch.isfinite(coop).all())
            p2b, coop = b.once(0, ids, b.rw), b.once(1, ids, b.rw)
            coop_repeat_ok &= bitwise(coop, b.once(1, ids, b.rw))
            p2b_repeat_ok &= bitwise(p2b, b.once(0, ids, b.rw))
            d = (coop.float() - p2b.float()).abs()
            max_abs = max(max_abs, float(d.max()))
            max_rel = max(max_rel, float(d.max() / p2b.float().abs().max().clamp_min(1e-30)))
            max_ulp = max(max_ulp, ulp_dist(p2b, coop))
            frac_diff.append(float((d > 0).float().mean()))
        row = {
            "unique_ratio_mean": statistics.mean(uniq),
            "onehot_bitexact": onehot_ok,
            "onehot_max_ulp": onehot_ulp,
            "full_max_abs": max_abs,
            "full_max_rel": max_rel,
            "full_max_ulp": max_ulp,
            "full_frac_elems_differ": statistics.mean(frac_diff),
            "full_within_tol": max_rel <= rel_tol,
            "coop_repeat_bitexact": coop_repeat_ok,
            "p2b_repeat_bitexact": p2b_repeat_ok,
            "outputs_finite": finite,
        }
        row["pass"] = bool(
            (row["onehot_bitexact"] or row["onehot_max_ulp"] <= 1)
            and row["full_within_tol"]
            and row["coop_repeat_bitexact"]
            and row["outputs_finite"]
        )
        res[src] = row
        print(f"[check] {src} {json.dumps(row)}", flush=True)
    return res


def capture(b: Bench, routes: Routings, source: str, replays: int) -> dict:
    """Coop inside a CUDA graph: replays with new routings must equal eager coop."""
    ids = b.ids(routes.draw(source))
    out = torch.empty_like(b.x)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            b.call(1, ids, b.rw, out)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    b.ext.set_coop(1)
    with torch.cuda.graph(graph):
        b.ext.p2b_fused_moe(b.x, out, *b.tables, ids, b.rw, 2, 2, 2, True, INTER, SWIGLU_LIMIT)
    ok = True
    for _ in range(replays):
        ids.copy_(b.ids(routes.draw(source)))
        graph.replay()
        torch.cuda.synchronize()
        ok &= bitwise(out.clone(), b.once(1, ids, b.rw))
    res = {"source": source, "replays": replays, "replay_equals_eager": ok}
    print(f"[capture] {json.dumps(res)}", flush=True)
    return res


def time_cold(b: Bench, coop: int, pool, out) -> list[float]:
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    b.ext.set_coop(coop)
    for ids in pool:
        ev0.record()
        b.ext.p2b_fused_moe(b.x, out, *b.tables, ids, b.rw, 2, 2, 2, True, INTER, SWIGLU_LIMIT)
        ev1.record()
        torch.cuda.synchronize()
        times.append(ev0.elapsed_time(ev1) * 1000.0)
    return times


def time_warm(b: Bench, coop: int, ids, out, iters: int) -> float:
    for _ in range(5):
        b.call(coop, ids, b.rw, out)
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(iters):
        b.ext.p2b_fused_moe(b.x, out, *b.tables, ids, b.rw, 2, 2, 2, True, INTER, SWIGLU_LIMIT)
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) * 1000.0 / iters


def timing(b: Bench, routes: Routings, sources, args) -> dict:
    out = torch.empty_like(b.x)
    res = {}
    for src in sources:
        cold = {0: [], 1: []}
        warm = {0: [], 1: []}
        warm_ids = b.ids(routes.draw(src))
        uniq = []
        for rep in range(args.reps):
            rows = [routes.draw(src) for _ in range(args.cold_calls)]
            uniq.extend(unique_ratio(r) for r in rows)
            pool = [b.ids(r) for r in rows]
            time_cold(b, 0, pool[:5], out)  # settle clocks
            for coop in ((0, 1) if rep % 2 == 0 else (1, 0)):
                cold[coop].append(statistics.median(time_cold(b, coop, pool, out)))
                warm[coop].append(time_warm(b, coop, warm_ids, out, args.warm_iters))
        row = {"unique_ratio_mean": statistics.mean(uniq)}
        for kind, data in (("cold", cold), ("warm", warm)):
            p2b, coop = statistics.median(data[0]), statistics.median(data[1])
            row[kind] = {
                "p2b_us": p2b,
                "coop_us": coop,
                "reduction_pct": 100.0 * (p2b - coop) / p2b,
                "p2b_reps_us": data[0],
                "coop_reps_us": data[1],
            }
        row["saving_ms_per_step_cold"] = (row["cold"]["p2b_us"] - row["cold"]["coop_us"]) * args.layers / 1000.0
        res[src] = row
        print(
            f"[time] {src} uniq {row['unique_ratio_mean']:.3f} cold p2b {row['cold']['p2b_us']:.1f} "
            f"coop {row['cold']['coop_us']:.1f} us ({row['cold']['reduction_pct']:+.2f}%, "
            f"{row['saving_ms_per_step_cold']:+.2f} ms/step)  warm p2b {row['warm']['p2b_us']:.1f} "
            f"coop {row['warm']['coop_us']:.1f} us ({row['warm']['reduction_pct']:+.2f}%)",
            flush=True,
        )
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=4, help="verify rows (DSpark-3: 4 per sequence)")
    ap.add_argument("--dups", default="0,0.3,0.5")
    ap.add_argument("--census", default=str(CENSUS), help="s10 census_npy dir ('' skips census routings)")
    ap.add_argument("--mode", default="all", choices=["all", "check", "time"])
    ap.add_argument("--reps", type=int, default=5, help="alternating p2b/coop repeats")
    ap.add_argument("--cold-calls", type=int, default=60, help="fresh routings per cold repeat")
    ap.add_argument("--warm-iters", type=int, default=200)
    ap.add_argument("--check-routings", type=int, default=8)
    ap.add_argument("--capture-replays", type=int, default=16)
    ap.add_argument("--rel-tol", type=float, default=1e-3, help="full-weights gate: max|diff| / max|p2b|")
    ap.add_argument("--layers", type=int, default=40, help="routed MoE layers per verify step (s10 capture: 40)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    dups = [float(d) for d in args.dups.split(",") if d != ""]
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dev = "cuda"
    ext = build_ext()
    _, tables = make_weights(dev)
    b = Bench(ext, tables, args.m, dev)
    routes = Routings(args.m, Path(args.census) if args.census else None, rng)
    sources = routes.sources(dups)
    props = torch.cuda.get_device_properties(0)
    result: dict = {
        "geometry": {"m": args.m, "top_k": TOPK, "experts": EXPERTS, "hidden": HIDDEN, "inter": INTER,
                     "k_bits": 2, "mcg": True, "swiglu_limit": SWIGLU_LIMIT},
        "device": {"name": props.name, "sms": props.multi_processor_count},
        "occupancy_blocks_per_sm": {"p2b_sort0": ext.occupancy(0), "coop_sort2": ext.occupancy(2)},
        "census_windows": len(routes.windows),
    }
    print(f"[setup] {json.dumps(result)}", flush=True)

    if args.mode in ("all", "check"):
        result["check"] = check(b, routes, sources, args.check_routings, args.rel_tol)
        result["capture"] = capture(b, routes, sources[-1], args.capture_replays)
        result["check_pass"] = all(r["pass"] for r in result["check"].values()) and result["capture"]["replay_equals_eager"]

    if args.mode in ("all", "time"):
        result["time"] = timing(b, routes, sources, args)

    text = json.dumps(result, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    return 0 if result.get("check_pass", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
