#!/usr/bin/env python3
"""p2b src-sort microbench: m=4 x top-6, dup {0, 25, 50}%, cold/warm, off vs on.

GPU only, serve DOWN. Run inside the serve image with the repo mounted:

  docker run --rm --gpus all -v $PWD:/repo -w /repo --entrypoint python3 \
    dsv41-flash-exl3-sm121:canonical-e12 kernel_study/p2b_srcsort/driver.py \
    --out results/2026-09-24-review/moe-dedup/microbench.json

Geometry is the V4.1 TP=2 rank: hidden 5120, inter 1152, 384 experts,
K=2 MCG, swiglu_limit 10. Trellis data is random (same bytes both arms).

check: off vs on must be bitwise equal. One-hot routing weight per slot
  (6 passes) makes the float atomics order-free, so it is strict; the full
  weights row is also compared and reported next to an off-vs-off repeat.
cold: every timed call uses a fresh routing (no weights left in L2).
warm: one routing repeated.
Arms alternate off/on per repeat to cancel drift.
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

import torch  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402

from moe_census import synth_routing  # noqa: E402

EXL = "/usr/local/lib/python3.12/dist-packages/exllamav3/exllamav3_ext"
HIDDEN, INTER, EXPERTS, TOPK, M = 5120, 1152, 384, 6, 4
SWIGLU_LIMIT = 10.0


def build_ext():
    sys.path.insert(0, str(HERE))
    import make_bench

    make_bench.main()
    return load(
        name="p2b_srcsort_bench",
        sources=[str(HERE / "build" / "bench_srcsort.cu")],
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


class Bench:
    def __init__(self, ext, tables, dev: str) -> None:
        self.ext, self.tables, self.dev = ext, tables, dev
        self.x = torch.randn(M, HIDDEN, dtype=torch.half, device=dev)
        rw = torch.rand(M, TOPK, device=dev)
        self.rw = (rw / rw.sum(dim=1, keepdim=True)).half()

    def ids(self, dup: float, rng: random.Random) -> torch.Tensor:
        rows = synth_routing(M, TOPK, EXPERTS, dup, rng)
        return torch.tensor(rows, dtype=torch.int32, device=self.dev)

    def run(self, sort: int, ids: torch.Tensor, rw: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        self.ext.set_sort(sort)
        self.ext.p2b_fused_moe(self.x, out, *self.tables, ids, rw, 2, 2, 2, True, INTER, SWIGLU_LIMIT)
        return out

    def once(self, sort: int, ids: torch.Tensor, rw: torch.Tensor) -> torch.Tensor:
        return self.run(sort, ids, rw, torch.empty_like(self.x)).clone()


def bitwise(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def check(b: Bench, dups, rng: random.Random, routings: int) -> dict:
    res = {}
    for dup in dups:
        onehot_ok = full_ok = repeat_ok = True
        finite = True
        for _ in range(routings):
            ids = b.ids(dup, rng)
            for slot in range(TOPK):
                rw = torch.zeros_like(b.rw)
                rw[:, slot] = 1.0
                off, on = b.once(0, ids, rw), b.once(1, ids, rw)
                onehot_ok &= bitwise(off, on)
                finite &= bool(torch.isfinite(off).all())
            off, on, off2 = b.once(0, ids, b.rw), b.once(1, ids, b.rw), b.once(0, ids, b.rw)
            full_ok &= bitwise(off, on)
            repeat_ok &= bitwise(off, off2)
        res[str(dup)] = {
            "onehot_bitexact": onehot_ok,
            "full_rw_bitexact": full_ok,
            "full_rw_off_repeat_bitexact": repeat_ok,
            "outputs_finite": finite,
        }
        print(f"[check] dup={dup:.2f} {json.dumps(res[str(dup)])}", flush=True)
    return res


def time_cold(b: Bench, sort: int, pool, out) -> list[float]:
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for ids in pool:
        b.ext.set_sort(sort)
        ev0.record()
        b.ext.p2b_fused_moe(b.x, out, *b.tables, ids, b.rw, 2, 2, 2, True, INTER, SWIGLU_LIMIT)
        ev1.record()
        torch.cuda.synchronize()
        times.append(ev0.elapsed_time(ev1) * 1000.0)
    return times


def time_warm(b: Bench, sort: int, ids, out, iters: int) -> float:
    b.ext.set_sort(sort)
    for _ in range(5):
        b.run(sort, ids, b.rw, out)
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(iters):
        b.ext.p2b_fused_moe(b.x, out, *b.tables, ids, b.rw, 2, 2, 2, True, INTER, SWIGLU_LIMIT)
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) * 1000.0 / iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dups", default="0,0.25,0.5")
    ap.add_argument("--mode", default="all", choices=["all", "check", "time"])
    ap.add_argument("--reps", type=int, default=5, help="alternating off/on repeats")
    ap.add_argument("--cold-calls", type=int, default=60, help="fresh routings per cold repeat")
    ap.add_argument("--warm-iters", type=int, default=200)
    ap.add_argument("--check-routings", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    dups = [float(d) for d in args.dups.split(",")]
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dev = "cuda"
    ext = build_ext()
    _, tables = make_weights(dev)
    b = Bench(ext, tables, dev)
    result: dict = {"geometry": {"m": M, "top_k": TOPK, "experts": EXPERTS, "hidden": HIDDEN, "inter": INTER}}

    if args.mode in ("all", "check"):
        result["check"] = check(b, dups, rng, args.check_routings)

    if args.mode in ("all", "time"):
        out = torch.empty_like(b.x)
        result["time"] = {}
        for dup in dups:
            cold = {0: [], 1: []}
            warm = {0: [], 1: []}
            warm_ids = b.ids(dup, rng)
            for rep in range(args.reps):
                pool = [b.ids(dup, rng) for _ in range(args.cold_calls)]
                time_cold(b, 0, pool[:5], out)  # settle clocks
                order = (0, 1) if rep % 2 == 0 else (1, 0)
                for sort in order:
                    cold[sort].append(statistics.median(time_cold(b, sort, pool, out)))
                    warm[sort].append(time_warm(b, sort, warm_ids, out, args.warm_iters))
            row = {}
            for kind, data in (("cold", cold), ("warm", warm)):
                off, on = statistics.median(data[0]), statistics.median(data[1])
                row[kind] = {
                    "off_us": off,
                    "on_us": on,
                    "reduction_pct": 100.0 * (off - on) / off,
                    "off_reps_us": data[0],
                    "on_reps_us": data[1],
                }
            result["time"][str(dup)] = row
            print(
                f"[time] dup={dup:.2f} cold off {row['cold']['off_us']:.1f} on {row['cold']['on_us']:.1f} us "
                f"({row['cold']['reduction_pct']:+.2f}%)  warm off {row['warm']['off_us']:.1f} "
                f"on {row['warm']['on_us']:.1f} us ({row['warm']['reduction_pct']:+.2f}%)",
                flush=True,
            )

    text = json.dumps(result, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
