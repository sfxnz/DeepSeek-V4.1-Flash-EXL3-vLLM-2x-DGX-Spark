#!/usr/bin/env python3
"""Round-3 routed-MoE microbench: p2b vs coop variants on real V4.1 layer weights.

GPU only, no serve on this GPU (spark2 via spark2.sh). Inside the serve image, repo at /repo,
pack snapshot at /hf (read-only):

  python3 kernel_study/p2b_coop/bench_r3.py --mode all --variants 0,1 --ms 4,8 \
      --out results/2026-09-25-kernels/coop-moe/<name>.json

Data. One routed layer's 384 experts (gate w1, up w3, down w2: trellis + suh + svh) read from
the pack and TP-sharded exactly like vllm_exl3 (shard_exl3_col / shard_exl3_row) for --tp-rank,
laid out like the serve's RoutedExperts params (w13 [E, 2, 320, 72, 32], w2 [E, 72, 320, 32]).
Routing: the s10 census rows of the SAME layer (6 captures x 511 decode tokens). m <= 4 takes m
consecutive tokens of one capture; m > 4 joins two windows (m//2, m - m//2) from two different
captures (independent sequences, the c=2 verify batch); census_seq keeps m consecutive tokens (an
upper bound on overlap). census_all: the same windows from the census of EVERY routed layer, drawn
layer-stratified (draw i uses layer i mod 40), applied to the loaded layer's weights: the per-step
mix (a verify step runs all 40 routed layers once; the kernel's cost depends on the routing, not on
the weight values). dup0: synth routing with no repeated expert. Activations are synthetic:
RMS-normalized Gaussian rows times the layer's real ffn_norm weight. Routing weights: sqrt-softplus
of Gaussian logits, normalized, x1.5 (routed_scaling_factor).

Variants (= DSV41_P2B_COOP values): 0 p2b (served SORT=0), 1 round-2 coop (SORT=2), 2.. round 3.

check (every variant vs p2b, every m, every source; tolerances fixed before any result):
  onehot   one routing weight 1.0 per slot: the slot sum adds exact zeros, so a variant whose
           per-(row, expert) math equals p2b's must be bit-exact (gate: bitwise).
  full     real routing weights: fixed slot order vs p2b's atomic order, so only a tolerance:
           max|v - p2b| / max|p2b| <= 1e-3 and max ulp reported (gate: rel <= 1e-3).
  repeat   the variant twice on the same inputs is bitwise equal (p2b's own repeat is reported).
  v1       variants >= 2 (dataflow): bitwise equal to round-2 coop (v1) with real routing weights.
  ref      fp64 reference from the decoded weights (exllamav3_ext.reconstruct): error of p2b and
           of the variant vs exact math, max|err| / max|ref| and rms(err) / rms(ref).
capture  per variant and m: one CUDA graph, replays after new ids and weights are copied into
         the static inputs, each replay bitwise equal to an eager call.
time     cold: a >= 2x L2 read-flush before every timed call, a fresh routing per iteration;
         warm: the same call (arm, routing) runs untimed right before the timed one, no flush.
         Arms alternate in rotating order within every
         iteration (same routing for all arms). CUDA events per call; median, p10, p90.
         Bandwidth = unique expert bytes of that call / time; floor = unique bytes / 250 GB/s.
         paired: per-call savings between arms on the same iteration, mean +- 95% CI; mean x 40 is
         the per-step projection (use census_all: iters a multiple of 40 weights layers equally).
phases   the stamped build (bench_r3_ts.cu): %globaltimer before/after every grid.sync per block.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(HERE))
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")
for _k in ("DSV41_P2B_SRC_SORT", "DSV41_P2B_COOP"):
    os.environ.pop(_k, None)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402

from moe_census import routed_layers, synth_routing  # noqa: E402

EXL = "/usr/local/lib/python3.12/dist-packages/exllamav3/exllamav3_ext"
SNAP = Path("/hf/snapshots/2.0bpw-mcg-lmhead-mxfp8")
CENSUS = ROOT / "results/2026-09-24-review/campaign/s10-diag-census-skew/census_npy"
HIDDEN, INTER_FULL, EXPERTS, TOPK, TP = 5120, 2304, 384, 6, 2
INTER = INTER_FULL // TP
SWIGLU_LIMIT = 10.0
ROUTED_SCALE = 1.5
MCG_MARK = -877912083  # 0xCBAC1FED as int32
PEAK = 250e9
REL_TOL = 1e-3
# Bytes one unique expert costs per call on one TP rank: 3 trellis + suh/svh vectors.
TRELLIS_BYTES = 3 * (HIDDEN // 16) * (INTER // 16) * 64
SCALE_BYTES = 3 * (HIDDEN + INTER) * 2
EXPERT_BYTES = TRELLIS_BYTES + SCALE_BYTES


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}Z] {msg}", flush=True)


def build_ext(stamps: bool):
    import make_bench_r3

    make_bench_r3.main()
    name = "bench_r3_ts" if stamps else "bench_r3"
    return load(
        name=f"p2b_{name}",
        sources=[str(make_bench_r3.OUT / f"{name}.cu")],
        extra_include_paths=[EXL, os.path.join(EXL, "quant")],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


# ----------------------------------------------------------------------------- weights


class Layer:
    """One routed layer, TP-sharded for tp_rank, in the serve's parameter layout."""

    def __init__(self, layer: int, tp_rank: int, dev: str) -> None:
        from safetensors import safe_open

        idx = json.loads((SNAP / "model.safetensors.index.json").read_text())["weight_map"]
        pre = f"layers.{layer}.ffn"
        kt, nt = HIDDEN // 16, INTER // 16
        self.layer, self.tp_rank = layer, tp_rank
        self.w13 = torch.empty((EXPERTS, 2, kt, nt, 32), dtype=torch.int16, device=dev)
        self.w2 = torch.empty((EXPERTS, nt, kt, 32), dtype=torch.int16, device=dev)
        self.w13_suh = torch.empty((EXPERTS, 2, HIDDEN), dtype=torch.half, device=dev)
        self.w13_svh = torch.empty((EXPERTS, 2, INTER), dtype=torch.half, device=dev)
        self.w2_suh = torch.empty((EXPERTS, INTER), dtype=torch.half, device=dev)
        self.w2_svh = torch.empty((EXPERTS, HIDDEN), dtype=torch.half, device=dev)
        files = {idx[f"{pre}.experts.{e}.{w}.{s}"] for e in range(EXPERTS) for w in ("w1", "w2", "w3")
                 for s in ("trellis", "suh", "svh", "mcg")}
        r0, i0 = tp_rank * nt, tp_rank * INTER
        marks = set()
        for fname in sorted(files):
            with safe_open(str(SNAP / fname), "pt") as fh:
                keys = set(fh.keys())
                for e in range(EXPERTS):
                    for j, w in ((0, "w1"), (1, "w3")):
                        k = f"{pre}.experts.{e}.{w}"
                        if f"{k}.trellis" not in keys:
                            continue
                        self.w13[e, j].copy_(fh.get_tensor(f"{k}.trellis").narrow(1, r0, nt))
                        self.w13_suh[e, j].copy_(fh.get_tensor(f"{k}.suh"))
                        self.w13_svh[e, j].copy_(fh.get_tensor(f"{k}.svh").narrow(0, i0, INTER))
                        marks.add(int(fh.get_tensor(f"{k}.mcg").item()))
                    k = f"{pre}.experts.{e}.w2"
                    if f"{k}.trellis" in keys:
                        self.w2[e].copy_(fh.get_tensor(f"{k}.trellis").narrow(0, r0, nt))
                        self.w2_suh[e].copy_(fh.get_tensor(f"{k}.suh").narrow(0, i0, INTER))
                        self.w2_svh[e].copy_(fh.get_tensor(f"{k}.svh"))
                        marks.add(int(fh.get_tensor(f"{k}.mcg").item()))
        if marks != {MCG_MARK}:
            raise SystemExit(f"layer {layer}: expected MCG markers only, got {marks}")
        norm_key = f"layers.{layer}.ffn_norm.weight"
        with safe_open(str(SNAP / idx[norm_key]), "pt") as fh:
            self.norm = fh.get_tensor(norm_key).float().to(dev)

        def ptrs(ts):
            return torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device=dev)

        E = range(EXPERTS)
        self.tables = [
            ptrs([self.w13[e, 0] for e in E]), ptrs([self.w13_suh[e, 0] for e in E]), ptrs([self.w13_svh[e, 0] for e in E]),
            ptrs([self.w13[e, 1] for e in E]), ptrs([self.w13_suh[e, 1] for e in E]), ptrs([self.w13_svh[e, 1] for e in E]),
            ptrs([self.w2[e] for e in E]), ptrs([self.w2_suh[e] for e in E]), ptrs([self.w2_svh[e] for e in E]),
        ]


# ----------------------------------------------------------------------------- routing


class Routings:
    def __init__(self, layer: int, rng: random.Random) -> None:
        self.rng = rng
        self.caps: list[np.ndarray] = []
        arrs = [np.load(path, allow_pickle=False) for path in sorted(CENSUS.glob("*.npy"))]
        routed = [set(routed_layers(arr)) for arr in arrs]
        for arr, lay in zip(arrs, routed):
            if layer in lay:
                self.caps.append(arr[:, layer, :].astype(np.int64))
        if len(self.caps) < 2:
            raise SystemExit(f"census: layer {layer} routed in {len(self.caps)} captures (< 2)")
        # census_all: the captures of every layer routed in all of them, drawn layer-stratified
        self.by_layer = {L: [arr[:, L, :].astype(np.int64) for arr in arrs] for L in sorted(set.intersection(*routed))}
        self.next_layer = 0
        self.last_layer = layer

    def _window(self, cap: np.ndarray, n: int) -> list[list[int]]:
        t0 = self.rng.randrange(cap.shape[0] - n + 1)
        return cap[t0 : t0 + n].tolist()

    def _census(self, caps: list[np.ndarray], m: int) -> list[list[int]]:
        if m <= 4:
            return self._window(self.rng.choice(caps), m)
        a, b = self.rng.sample(caps, 2)
        return self._window(a, m // 2) + self._window(b, m - m // 2)

    def draw(self, source: str, m: int) -> list[list[int]]:
        if source == "census":
            return self._census(self.caps, m)
        if source == "census_all":
            # Layer-stratified: successive draws cycle through the routed layers, so n x len(by_layer)
            # consecutive draws weight every layer equally (a step runs each routed layer once).
            layers = list(self.by_layer)
            self.last_layer = layers[self.next_layer % len(layers)]
            self.next_layer += 1
            return self._census(self.by_layer[self.last_layer], m)
        if source == "census_seq":
            return self._window(self.rng.choice(self.caps), m)
        if source.startswith("dup"):
            return synth_routing(m, TOPK, EXPERTS, float(source[3:]), self.rng)
        raise ValueError(source)


def n_unique(rows: list[list[int]]) -> int:
    return len({x for r in rows for x in r})


# ----------------------------------------------------------------------------- bench core


class Bench:
    def __init__(self, ext, layer: Layer, dev: str, seed: int) -> None:
        self.ext, self.L, self.dev = ext, layer, dev
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(seed)

    def x(self, m: int) -> torch.Tensor:
        h = torch.randn(m, HIDDEN, device=self.dev, generator=self.gen)
        h = h / h.pow(2).mean(dim=1, keepdim=True).sqrt()
        return (h * self.L.norm).half()

    def rw(self, m: int) -> torch.Tensor:
        z = torch.randn(m, TOPK, device=self.dev, generator=self.gen)
        s = torch.nn.functional.softplus(z).sqrt()
        return (s / s.sum(dim=1, keepdim=True) * ROUTED_SCALE).half()

    def ids(self, rows) -> torch.Tensor:
        return torch.tensor(rows, dtype=torch.int32, device=self.dev)

    def call(self, v: int, x, ids, rw, out) -> torch.Tensor:
        self.ext.set_variant(v)
        self.ext.p2b_fused_moe(x, out, *self.L.tables, ids, rw, 2, 2, 2, True, INTER, SWIGLU_LIMIT)
        return out

    def once(self, v: int, x, ids, rw) -> torch.Tensor:
        return self.call(v, x, ids, rw, torch.empty_like(x)).clone()


def bitwise(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def ulp_dist(a: torch.Tensor, b: torch.Tensor) -> int:
    def line(t):
        i = t.view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -(i & 0x7FFF), i)

    return int((line(a) - line(b)).abs().max().item())


# ----------------------------------------------------------------------------- fp64 reference

_H128 = None


def had128(v: torch.Tensor) -> torch.Tensor:
    """Blockwise orthonormal Sylvester H128 on the last dim (the kernels' butterfly order)."""
    global _H128
    if _H128 is None or _H128.device != v.device:
        i = torch.arange(128)
        bits = torch.stack([((i[:, None] & i[None, :]) >> b) & 1 for b in range(7)]).sum(0)
        _H128 = ((1 - 2 * (bits % 2)).double() / math.sqrt(128)).to(v.device)
    s = v.shape
    return (v.reshape(*s[:-1], s[-1] // 128, 128) @ _H128).reshape(s)


class Reference:
    """Exact (fp64) MoE from the decoded weights: what p2b approximates in fp16."""

    def __init__(self, layer: Layer) -> None:
        import exllamav3_ext

        self.X, self.L, self.cache = exllamav3_ext, layer, {}

    def _w(self, trellis: torch.Tensor, k: int, n: int) -> torch.Tensor:
        key = trellis.data_ptr()
        if key not in self.cache:
            w = torch.empty((k, n), dtype=torch.half, device=trellis.device)
            self.X.reconstruct(w, trellis, 2, True, False)
            self.cache[key] = w.double()
        return self.cache[key]

    def __call__(self, x: torch.Tensor, rows, rw: torch.Tensor) -> torch.Tensor:
        L = self.L
        out = torch.zeros(x.shape[0], HIDDEN, dtype=torch.float64, device=x.device)
        for r, experts in enumerate(rows):
            xr = x[r].double()
            for s, e in enumerate(experts):
                g = had128(had128(xr * L.w13_suh[e, 0].double()) @ self._w(L.w13[e, 0], HIDDEN, INTER)) * L.w13_svh[e, 0].double()
                u = had128(had128(xr * L.w13_suh[e, 1].double()) @ self._w(L.w13[e, 1], HIDDEN, INTER)) * L.w13_svh[e, 1].double()
                g = g.clamp(max=SWIGLU_LIMIT)
                u = u.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
                a = g / (1 + torch.exp(-g)) * u
                d = had128(had128(a * L.w2_suh[e].double()) @ self._w(L.w2[e], INTER, HIDDEN)) * L.w2_svh[e].double()
                out[r] += rw[r, s].double() * d
        if len(self.cache) > 600:
            self.cache.clear()
        return out


def ref_err(y: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    d = y.double() - ref
    return float(d.abs().max() / ref.abs().max()), float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())


# ----------------------------------------------------------------------------- modes


def check(b: Bench, routes: Routings, variants, ms, sources, n: int, ref: Reference | None) -> dict:
    res: dict = {}
    for m in ms:
        for src in sources:
            if src == "census_seq" and m <= 4:
                continue
            rows_all = [routes.draw(src, m) for _ in range(n)]
            per = {v: {"onehot_bitexact": True, "onehot_max_ulp": 0, "full_max_abs": 0.0, "full_max_rel": 0.0,
                       "full_max_ulp": 0, "full_frac_differ": [], "repeat_bitexact": True, "finite": True,
                       "ref_max_rel": 0.0, "ref_rms_rel": [], "p2b_ref_max_rel": 0.0, "p2b_ref_rms_rel": [],
                       "v1_full_bitexact": True}
                   for v in variants if v != 0}
            p2b_repeat = True
            for rows in rows_all:
                x, ids, rw = b.x(m), b.ids(rows), b.rw(m)
                base = b.once(0, x, ids, rw)
                p2b_repeat &= bitwise(base, b.once(0, x, ids, rw))
                exact = ref(x, rows, rw) if ref is not None else None
                onehots = []
                for slot in range(TOPK):
                    oh = torch.zeros_like(rw)
                    oh[:, slot] = 1.0
                    onehots.append((oh, b.once(0, x, ids, oh)))
                coop1 = b.once(1, x, ids, rw)
                for v, r in per.items():
                    y = b.once(v, x, ids, rw)
                    r["repeat_bitexact"] &= bitwise(y, b.once(v, x, ids, rw))
                    r["v1_full_bitexact"] &= bitwise(y, coop1)
                    r["finite"] &= bool(torch.isfinite(y).all()) and bool(torch.isfinite(base).all())
                    d = (y.float() - base.float()).abs()
                    r["full_max_abs"] = max(r["full_max_abs"], float(d.max()))
                    r["full_max_rel"] = max(r["full_max_rel"], float(d.max() / base.float().abs().max().clamp_min(1e-30)))
                    r["full_max_ulp"] = max(r["full_max_ulp"], ulp_dist(base, y))
                    r["full_frac_differ"].append(float((d > 0).float().mean()))
                    for oh, want in onehots:
                        got = b.once(v, x, ids, oh)
                        r["onehot_bitexact"] &= bitwise(got, want)
                        r["onehot_max_ulp"] = max(r["onehot_max_ulp"], ulp_dist(got, want))
                    if exact is not None:
                        mx, rms = ref_err(y, exact)
                        r["ref_max_rel"] = max(r["ref_max_rel"], mx)
                        r["ref_rms_rel"].append(rms)
                        mx, rms = ref_err(base, exact)
                        r["p2b_ref_max_rel"] = max(r["p2b_ref_max_rel"], mx)
                        r["p2b_ref_rms_rel"].append(rms)
            for v, r in per.items():
                r["full_frac_differ"] = statistics.mean(r["full_frac_differ"])
                for k in ("ref_rms_rel", "p2b_ref_rms_rel"):
                    r[k] = statistics.mean(r[k]) if r[k] else None
                r["unique_ratio"] = statistics.mean(n_unique(rw_) / (m * TOPK) for rw_ in rows_all)
                r["routings"] = n
                r["p2b_repeat_bitexact"] = p2b_repeat
                r["pass"] = bool(r["onehot_bitexact"] and r["full_max_rel"] <= REL_TOL and r["repeat_bitexact"] and r["finite"]
                                 and (v < 2 or r["v1_full_bitexact"]))
                res[f"v{v}/m{m}/{src}"] = r
                log(f"[check] v{v} m={m} {src}: " + json.dumps({k: r[k] for k in (
                    "pass", "onehot_bitexact", "v1_full_bitexact", "full_max_rel", "full_max_ulp", "repeat_bitexact", "ref_max_rel",
                    "p2b_ref_max_rel", "ref_rms_rel", "p2b_ref_rms_rel")}))
    return res


def capture(b: Bench, routes: Routings, variants, ms, replays: int) -> dict:
    res = {}
    for v in variants:
        if v == 0:
            continue
        for m in ms:
            x, ids, rw = b.x(m), b.ids(routes.draw("census", m)), b.rw(m)
            out = torch.empty_like(x)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    b.call(v, x, ids, rw, out)
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            b.ext.set_variant(v)
            with torch.cuda.graph(g):
                b.ext.p2b_fused_moe(x, out, *b.L.tables, ids, rw, 2, 2, 2, True, INTER, SWIGLU_LIMIT)
            ok = True
            for _ in range(replays):
                ids.copy_(b.ids(routes.draw("census", m)))
                rw.copy_(b.rw(m))
                g.replay()
                torch.cuda.synchronize()
                ok &= bitwise(out.clone(), b.once(v, x, ids, rw))
            res[f"v{v}/m{m}"] = {"replays": replays, "replay_equals_eager": ok}
            log(f"[capture] v{v} m={m}: replay_equals_eager={ok} ({replays} replays)")
            del g
    return res


class Flusher:
    def __init__(self, dev: str) -> None:
        l2 = torch.cuda.get_device_properties(0).L2_cache_size
        self.bytes = max(4 * l2, 128 << 20)
        self.buf = torch.ones(self.bytes // 8, dtype=torch.int64, device=dev)

    def __call__(self) -> None:
        self.buf.sum()


def pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


def stats(times_us, bytes_list) -> dict:
    bw = [nb / (t * 1e-6) / 1e9 for nb, t in zip(bytes_list, times_us)]
    floor = [nb / PEAK * 1e6 for nb in bytes_list]
    return {
        "n": len(times_us),
        "median_us": statistics.median(times_us), "p10_us": pct(times_us, 0.1), "p90_us": pct(times_us, 0.9),
        "mean_us": statistics.mean(times_us),
        "gbps_median": statistics.median(bw), "gbps_p10": pct(bw, 0.1), "gbps_p90": pct(bw, 0.9),
        "pct_peak_median": 100.0 * statistics.median(bw) / (PEAK / 1e9),
        "floor_us_median": statistics.median(floor),
        "pct_of_floor_median": 100.0 * statistics.median([f / t for f, t in zip(floor, times_us)]),
        "unique_mean": statistics.mean(bytes_list) / EXPERT_BYTES,
    }


def paired(a_us, c_us) -> dict:
    """Per-call savings a - c of two arms timed on the same iterations: mean, 95% CI of the mean
    (1.96 x standard error), median, and the mean x 40 routed layers in ms/step."""
    d = [x - y for x, y in zip(a_us, c_us)]
    sem = statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else 0.0
    return {"n": len(d), "mean_us": statistics.mean(d), "ci95_us": 1.96 * sem, "median_us": statistics.median(d),
            "ms_per_step_40": statistics.mean(d) * 40 / 1000.0, "ms_per_step_40_ci95": 1.96 * sem * 40 / 1000.0}


def timing(b: Bench, routes: Routings, variants, ms, sources, iters: int, warmup: int, flush: Flusher,
           pair_base=None) -> dict:
    res: dict = {}
    for m in ms:
        for src in sources:
            if src == "census_seq" and m <= 4:
                continue
            x, rw = b.x(m), b.rw(m)
            out = torch.empty_like(x)
            rows, row_layer = [], []
            for _ in range(iters + warmup):
                rows.append(routes.draw(src, m))
                row_layer.append(routes.last_layer if src == "census_all" else None)
            pool = [b.ids(r) for r in rows]
            nbytes = [n_unique(r) * EXPERT_BYTES for r in rows]
            entry = {"dup_frac_mean": 1.0 - statistics.mean(n_unique(r) for r in rows[warmup:]) / (m * TOPK)}
            for kind in ("cold", "warm"):
                evs = {v: [] for v in variants}
                torch.cuda.synchronize()
                t0 = time.time()
                for i in range(iters + warmup):
                    order = variants[i % len(variants):] + variants[: i % len(variants)]
                    for v in order:
                        if kind == "cold":
                            flush()
                        else:
                            b.call(v, x, pool[i], rw, out)  # warm: the same call just ran
                        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        e0.record()
                        b.call(v, x, pool[i], rw, out)
                        e1.record()
                        if i >= warmup:
                            evs[v].append((e0, e1, nbytes[i]))
                torch.cuda.synchronize()
                wall = time.time() - t0
                entry[kind] = {"wall_s": wall}
                for v in variants:
                    ts = [e0.elapsed_time(e1) * 1000.0 for e0, e1, _ in evs[v]]
                    entry[kind][f"v{v}"] = stats(ts, [nb for _, _, nb in evs[v]])
                base = entry[kind]["v0"]["median_us"]
                for v in variants:
                    s = entry[kind][f"v{v}"]
                    s["vs_p2b_pct"] = 100.0 * (base - s["median_us"]) / base
                    s["saving_ms_per_step_40"] = (base - s["median_us"]) * 40 / 1000.0
                # Paired per-call savings (same iteration, same routing): the mean x 40 routed layers is
                # the per-step projection when the routings weight every layer equally (census_all).
                t_us = {v: [e0.elapsed_time(e1) * 1000.0 for e0, e1, _ in evs[v]] for v in variants}
                if pair_base:  # every arm against each base arm
                    pairs = [(a, c) for a in pair_base for c in variants if c != a and not (c in pair_base and c < a)]
                else:
                    pairs = [(0, v) for v in variants if v != 0] + [(a, c) for a, c in zip(variants[1:], variants[2:])]
                entry[kind]["paired"] = {f"v{a}-v{c}": paired(t_us[a], t_us[c]) for a, c in pairs}
                if src == "census_all":
                    lay = row_layer[warmup:]
                    per = {}
                    for L in sorted(set(lay)):
                        idx = [i for i, x in enumerate(lay) if x == L]
                        per[str(L)] = {"n": len(idx),
                                       "dup_frac_mean": 1.0 - statistics.mean(n_unique(rows[warmup + i]) for i in idx) / (m * TOPK),
                                       **{f"v{v}_mean_us": statistics.mean(t_us[v][i] for i in idx) for v in variants}}
                    entry[kind]["per_layer"] = per
                log(f"[time] m={m} {src} {kind}: " + "  ".join(
                    f"v{v} {entry[kind][f'v{v}']['median_us']:.1f}us [{entry[kind][f'v{v}']['p10_us']:.1f},"
                    f"{entry[kind][f'v{v}']['p90_us']:.1f}] mean {entry[kind][f'v{v}']['mean_us']:.1f} "
                    f"{entry[kind][f'v{v}']['gbps_median']:.0f}GB/s "
                    f"{entry[kind][f'v{v}']['pct_of_floor_median']:.0f}%floor" for v in variants)
                    + " | paired mean saving " + "  ".join(
                        f"{k} {p['mean_us']:.1f}+-{p['ci95_us']:.1f}us = {p['ms_per_step_40']:.2f} ms/step"
                        for k, p in entry[kind]["paired"].items()))
            res[f"m{m}/{src}"] = entry
    return res


def stress(b: Bench, routes: Routings, variants, ms, calls: int, flush: Flusher) -> dict:
    """Race hunt for the dataflow synchronization: every variant >= 2 vs round-2 coop (v1, whose
    phases are separated by grid.sync only) on `calls` fresh census routings per m, eager (every
    4th call after an L2 flush, the rest back to back) and as CUDA-graph replays; bitwise."""
    res = {}
    for v in variants:
        if v < 2:
            continue
        for m in ms:
            x, rw = b.x(m), b.rw(m)
            out1, out2 = torch.empty_like(x), torch.empty_like(x)
            ids = b.ids(routes.draw("census", m))
            gout = torch.empty_like(x)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    b.call(v, x, ids, rw, gout)
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            b.ext.set_variant(v)
            with torch.cuda.graph(g):
                b.ext.p2b_fused_moe(x, gout, *b.L.tables, ids, rw, 2, 2, 2, True, INTER, SWIGLU_LIMIT)
            eager_bad = graph_bad = 0
            for i in range(calls):
                ids.copy_(b.ids(routes.draw("census", m)))
                if i % 4 == 0:
                    flush()
                b.call(1, x, ids, rw, out1)
                b.call(v, x, ids, rw, out2)
                g.replay()
                torch.cuda.synchronize()
                eager_bad += not bitwise(out1, out2)
                graph_bad += not bitwise(out1, gout)
            res[f"v{v}/m{m}"] = {"calls": calls, "eager_mismatch": eager_bad, "graph_mismatch": graph_bad}
            log(f"[stress] v{v} m={m}: {calls} calls, eager mismatches {eager_bad}, graph mismatches {graph_bad}")
            del g
    return res


def stream_ceiling(b: Bench, routes: Routings, ms, iters: int, flush: Flusher) -> dict:
    """Read ceiling of the experts' trellis bytes: the coop tile pattern without decode (mode 0 with
    a block barrier per task, mode 1 without), a contiguous read of the same bytes (mode 2), and
    (round 4) a flat 16-B grid-stride read of them at 48 / 96 / 144 blocks (the dense-gemv
    calibration's best pure-read pattern). Cold (flushed) calls, arms rotating per iteration,
    census routing; GB/s over trellis bytes only."""
    res = {}
    sink = torch.zeros(4, dtype=torch.int32, device=b.dev)
    gt, ut, dt = b.L.tables[0], b.L.tables[3], b.L.tables[6]
    arms = {"tile_barrier": lambda uq: b.ext.stream(gt, ut, dt, uq, HIDDEN, INTER, 0, sink),
            "tile_nobarrier": lambda uq: b.ext.stream(gt, ut, dt, uq, HIDDEN, INTER, 1, sink),
            "contiguous": lambda uq: b.ext.stream(gt, ut, dt, uq, HIDDEN, INTER, 2, sink),
            "nsplit_g": lambda uq: b.ext.stream(gt, ut, dt, uq, HIDDEN, INTER, 4, sink),
            "nsplit_k": lambda uq: b.ext.stream(gt, ut, dt, uq, HIDDEN, INTER, 5, sink),
            # the same patterns at the dataflow kernel's own grid (3 blocks/SM)
            "tile_barrier_g144": lambda uq: b.ext.stream(gt, ut, dt, uq, HIDDEN, INTER, 0, sink, 144),
            "nsplit_g_g144": lambda uq: b.ext.stream(gt, ut, dt, uq, HIDDEN, INTER, 4, sink, 144),
            "nsplit_k_g144": lambda uq: b.ext.stream(gt, ut, dt, uq, HIDDEN, INTER, 5, sink, 144)}
    for g in (48, 96, 144):
        arms[f"flat16_g{g}"] = lambda uq, g=g: b.ext.flat(gt, ut, dt, uq, TRELLIS_BYTES // 3, g, sink)
    names = list(arms)
    for m in ms:
        rows = [routes.draw("census", m) for _ in range(iters)]
        uniq = [torch.tensor(sorted({x for r in rr for x in r}), dtype=torch.int32, device=b.dev) for rr in rows]
        evs = {n: [] for n in names}
        for i, uq in enumerate(uniq):
            k = i % len(names)
            for name in names[k:] + names[:k]:
                flush()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                arms[name](uq)
                e1.record()
                evs[name].append((e0, e1, uq.numel() * TRELLIS_BYTES))
        torch.cuda.synchronize()
        entry = {}
        for name in names:
            ts = [e0.elapsed_time(e1) * 1000.0 for e0, e1, _ in evs[name]]
            bw = [nb / (t * 1e-6) / 1e9 for (_, _, nb), t in zip(evs[name], ts)]
            entry[name] = {"median_us": statistics.median(ts), "p10_us": pct(ts, 0.1), "p90_us": pct(ts, 0.9),
                           "mean_us": statistics.mean(ts), "gbps_median": statistics.median(bw),
                           "gbps_p10": pct(bw, 0.1), "gbps_p90": pct(bw, 0.9)}
        res[f"m{m}"] = entry
        log(f"[stream] m={m}: " + "  ".join(f"{k} {v['median_us']:.1f}us {v['gbps_median']:.1f}GB/s [{v['gbps_p10']:.1f},{v['gbps_p90']:.1f}]"
                                          for k, v in entry.items()))
    return res


def bw_probe(b: Bench, iters: int, flush: Flusher) -> dict:
    """Plain contiguous read bandwidth of a 512 MiB buffer: blocks/SM x prefetch depth x load width
    x L2 prefetch distance. Cold (flushed) calls, all configurations alternating per iteration."""
    buf = torch.ones(512 << 18, dtype=torch.int32, device=b.dev)  # 512 MiB
    sink = torch.zeros(4, dtype=torch.int32, device=b.dev)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    cfgs = [(bps, pf, vec, l2d) for bps in (2, 3, 4, 6) for pf in (1, 2, 4, 8) for vec in (1, 4) for l2d in (0, 8)
            if not (vec == 4 and pf == 8)]
    evs = {c: [] for c in cfgs}
    for i in range(iters):
        for c in (cfgs if i % 2 == 0 else cfgs[::-1]):
            flush()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            b.ext.bw(buf, c[0] * sms, c[1], c[2], c[3], sink)
            e1.record()
            evs[c].append((e0, e1))
    torch.cuda.synchronize()
    res = {}
    for c in cfgs:
        blocks = c[0] * sms
        words = (buf.numel() // (blocks * 8)) // 512 * 512 * blocks * 8
        ts = [e0.elapsed_time(e1) * 1e-3 for e0, e1 in evs[c]]
        bw = sorted(words * 4 / t / 1e9 for t in ts)
        res[f"bps{c[0]}_pf{c[1]}_vec{c[2]}_l2d{c[3]}"] = {"gbps_median": statistics.median(bw), "gbps_max": bw[-1]}
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]["gbps_median"]):
        log(f"[bw] {k}: {v['gbps_median']:.1f} GB/s (max {v['gbps_max']:.1f})")
    return res


def ld_probe(b: Bench, iters: int, flush: Flusher) -> dict:
    """Contiguous 512 MiB read with 6 load flavours x blocks/SM {2, 3, 4, 8}: see make_bench_r3 p2b_ld_kernel."""
    buf = torch.ones(512 << 18, dtype=torch.int32, device=b.dev)
    sink = torch.zeros(4, dtype=torch.int32, device=b.dev)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    names = ["ldcs", "ld", "ldg_nc", "ld_L1_noalloc", "cp_async_16B", "tma_bulk_2KB"]
    cfgs = [(bps, k) for bps in (2, 3, 4, 8) for k in range(6)]
    evs = {c: [] for c in cfgs}
    for i in range(iters):
        for c in (cfgs if i % 2 == 0 else cfgs[::-1]):
            flush()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            b.ext.ld(buf, c[0] * sms, c[1], sink)
            e1.record()
            evs[c].append((e0, e1))
    torch.cuda.synchronize()
    res = {}
    for c in cfgs:
        blocks = c[0] * sms
        words = (buf.numel() // (blocks * 8)) // 512 * 512 * blocks * 8
        bw = sorted(words * 4 / (e0.elapsed_time(e1) * 1e-3) / 1e9 for e0, e1 in evs[c])
        res[f"bps{c[0]}_{names[c[1]]}"] = {"gbps_median": statistics.median(bw), "gbps_max": bw[-1]}
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]["gbps_median"]):
        log(f"[ld] {k}: {v['gbps_median']:.1f} GB/s (max {v['gbps_max']:.1f})")
    return res


def gemm_probe(b: Bench, iters: int, flush: Flusher) -> dict:
    """Streaming bandwidth of cuBLAS/cutlass bf16 GEMV-like GEMMs (the serve's lm_head class):
    y = x @ W^T with W [N, 5120] bf16, m in {4, 8}, cold (flushed)."""
    res = {}
    for n in (64640, 16384):
        w = torch.randn(n, HIDDEN, device=b.dev, dtype=torch.bfloat16)
        for m in (4, 8):
            x = torch.randn(m, HIDDEN, device=b.dev, dtype=torch.bfloat16)
            for _ in range(3):
                torch.nn.functional.linear(x, w)
            evs = []
            for _ in range(iters):
                flush()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                torch.nn.functional.linear(x, w)
                e1.record()
                evs.append((e0, e1))
            torch.cuda.synchronize()
            bw = sorted(w.numel() * 2 / (e0.elapsed_time(e1) * 1e-3) / 1e9 for e0, e1 in evs)
            res[f"n{n}_m{m}"] = {"mb": w.numel() * 2 / 1e6, "gbps_median": statistics.median(bw), "gbps_max": bw[-1]}
            log(f"[gemm] N={n} m={m} {w.numel() * 2 / 1e6:.0f} MB: {statistics.median(bw):.1f} GB/s (max {bw[-1]:.1f})")
        del w
    return res


def phases(ext_ts, layer: Layer, routes: Routings, variants, ms, calls: int, flush: Flusher, seed: int) -> dict:
    b = Bench(ext_ts, layer, "cuda", seed)
    res = {}
    for v in variants:
        for m in ms:
            x, rw = b.x(m), b.rw(m)
            out = torch.empty_like(x)
            b.call(v, x, b.ids(routes.draw("census", m)), rw, out)
            grid = ext_ts.last_grid()
            buf = torch.zeros(grid * 64, dtype=torch.int64, device="cuda")
            dfb = torch.zeros(grid * 64, dtype=torch.int64, device="cuda")
            ext_ts.set_ts(buf.data_ptr())
            ext_ts.set_df(dfb.data_ptr())
            per_call = []
            for _ in range(calls):
                buf.zero_()
                dfb.zero_()
                flush()
                rows = routes.draw("census", m)
                b.call(v, x, b.ids(rows), rw, out)
                torch.cuda.synchronize()
                d = dfb.view(grid * 8, 8).cpu().numpy().astype(np.float64)
                t = buf.view(grid, 64).cpu().numpy().astype(np.float64)
                t0 = t[:, 0].min()
                nsync = int((t[:, 1:56] > 0).any(axis=0).sum()) // 2  # slots 56, 57: dataflow prologue stamps
                row = {"launch_skew_us": (t[:, 0].max() - t0) / 1e3, "syncs": [], "unique": n_unique(rows)}
                if (t[:, 56] > 0).all() and (t[:, 57] > 0).all():  # per block, then the median over blocks
                    row["pro"] = {"build_us": float(np.median(t[:, 56] - t[:, 0])) / 1e3,
                                  "prefetch_issue_us": float(np.median(t[:, 57] - t[:, 56])) / 1e3,
                                  "hadamard_us": float(np.median(t[:, 1] - t[:, 57])) / 1e3,
                                  "first_release_us": (t[:, 2].min() - t0) / 1e3}
                if d[:, 6].sum() > 0:  # dataflow accounting, mean per warp (us)
                    fill, loop, red, span = (d[:, k].mean() / 1e3 for k in (3, 4, 5, 7))
                    row["df"] = {"tiles_per_warp": d[:, 6].mean(), "fill_us": fill, "loop_us": loop, "reduce_us": red,
                                 "span_us": span, "between_tiles_us": span - fill - loop - red}
                prev_release = t[:, 0]
                for k in range(nsync):
                    arr, rel = t[:, 1 + 2 * k], t[:, 2 + 2 * k]
                    row["syncs"].append({
                        "work_us": (arr.max() - prev_release.min()) / 1e3,  # first start -> last arrival
                        "imbalance_us": (arr.max() - np.median(arr)) / 1e3,  # last arrival - median arrival
                        "barrier_us": (rel.min() - arr.max()) / 1e3,  # last arrival -> first release
                        "release_spread_us": (rel.max() - rel.min()) / 1e3,
                    })
                    prev_release = rel
                row["tail_us"] = (t[:, 63].max() - prev_release.min()) / 1e3
                row["total_us"] = (t[:, 63].max() - t0) / 1e3
                per_call.append(row)
            ext_ts.set_ts(0)
            ext_ts.set_df(0)
            agg = {"calls": calls, "grid": grid, "unique_mean": statistics.mean(r["unique"] for r in per_call),
                   "launch_skew_us": statistics.median(r["launch_skew_us"] for r in per_call),
                   "tail_us": statistics.median(r["tail_us"] for r in per_call),
                   "total_us": statistics.median(r["total_us"] for r in per_call), "syncs": []}
            for k in range(len(per_call[0]["syncs"])):
                agg["syncs"].append({f: statistics.median(r["syncs"][k][f] for r in per_call)
                                     for f in per_call[0]["syncs"][k]})
            if all("df" in r for r in per_call):
                agg["df"] = {k: statistics.median(r["df"][k] for r in per_call) for k in per_call[0]["df"]}
            if all("pro" in r for r in per_call):
                agg["pro"] = {k: statistics.median(r["pro"][k] for r in per_call) for k in per_call[0]["pro"]}
            res[f"v{v}/m{m}"] = agg
            log(f"[phases] v{v} m={m} grid {grid}: total {agg['total_us']:.1f} us, skew {agg['launch_skew_us']:.1f}, "
                + " | ".join(f"s{k + 1} work {s['work_us']:.1f} imb {s['imbalance_us']:.1f} bar {s['barrier_us']:.1f}"
                             for k, s in enumerate(agg["syncs"])) + f" | tail {agg['tail_us']:.1f} | uniq {agg['unique_mean']:.1f}"
                + ("" if "df" not in agg else " | df " + " ".join(f"{k} {x:.1f}" for k, x in agg["df"].items()))
                + ("" if "pro" not in agg else " | pro " + " ".join(f"{k} {x:.2f}" for k, x in agg["pro"].items())))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", help="comma list of check,capture,time,phases,stress,stream,bw,ld,gemm (all = the first four)")
    ap.add_argument("--variants", default="0,1")
    ap.add_argument("--ms", default="1,3,4,6,8")
    ap.add_argument("--time-ms", default=None, help="m list for timing (default --ms)")
    ap.add_argument("--sources", default="census,dup0")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--tp-rank", type=int, default=0)
    ap.add_argument("--check-routings", type=int, default=6)
    ap.add_argument("--ref", action="store_true", help="fp64 reference errors in check (slow)")
    ap.add_argument("--replays", type=int, default=16)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--phase-calls", type=int, default=30)
    ap.add_argument("--stress-calls", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pair-base", default="", help="time: paired savings of every arm vs each of these arms")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    modes = {"check", "capture", "time", "phases"} if args.mode == "all" else set(args.mode.split(","))
    variants = [int(v) for v in args.variants.split(",")]
    if variants[0] != 0:
        variants = [0] + [v for v in variants if v != 0]
    ms = [int(m) for m in args.ms.split(",")]
    tms = [int(m) for m in (args.time_ms or args.ms).split(",")]
    sources = args.sources.split(",")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dev = "cuda"
    props = torch.cuda.get_device_properties(0)
    ext = build_ext(False)
    t0 = time.time()
    layer = Layer(args.layer, args.tp_rank, dev)
    log(f"loaded layer {args.layer} rank {args.tp_rank} in {time.time() - t0:.1f} s")
    routes = Routings(args.layer, rng)
    b = Bench(ext, layer, dev, args.seed)
    flush = Flusher(dev)
    result: dict = {
        "geometry": {"hidden": HIDDEN, "inter_local": INTER, "experts": EXPERTS, "top_k": TOPK, "k_bits": 2,
                     "mcg": True, "swiglu_limit": SWIGLU_LIMIT, "layer": args.layer, "tp_rank": args.tp_rank,
                     "expert_bytes": EXPERT_BYTES},
        "device": {"name": props.name, "sms": props.multi_processor_count, "l2_bytes": props.L2_cache_size,
                   "flush_bytes": flush.bytes},
        "variants": variants, "census_captures": len(routes.caps), "args": vars(args) | {"out": str(args.out)},
        "tolerance": {"onehot": "bitwise", "full_max_rel": REL_TOL, "repeat": "bitwise", "capture": "bitwise"},
    }
    occ = {}
    for v in variants:
        x = b.x(4)
        b.call(v, x, b.ids(routes.draw("census", 4)), b.rw(4), torch.empty_like(x))
        occ[f"v{v}"] = {"grid": ext.last_grid(), "blocks_per_sm": ext.last_grid() / props.multi_processor_count}
    result["launch"] = occ
    log(f"[setup] {json.dumps(result['device'])} launch {json.dumps(occ)}")
    if "check" in modes:
        ref = Reference(layer) if args.ref else None
        result["check"] = check(b, routes, variants, ms, sources + (["census_seq"] if max(ms) > 4 else []),
                                args.check_routings, ref)
    if "capture" in modes:
        result["capture"] = capture(b, routes, variants, ms, args.replays)
    if "time" in modes:
        result["time"] = timing(b, routes, variants, tms, sources, args.iters, args.warmup, flush,
                                [int(v) for v in args.pair_base.split(",") if v])
    if "gemm" in modes:
        result["gemm"] = gemm_probe(b, max(20, args.iters // 5), flush)
    if "ld" in modes:
        result["ld"] = ld_probe(b, max(10, args.iters // 10), flush)
    if "bw" in modes:
        result["bw"] = bw_probe(b, max(10, args.iters // 10), flush)
    if "stress" in modes:
        result["stress"] = stress(b, routes, variants, ms, args.stress_calls, flush)
    if "stream" in modes:
        result["stream"] = stream_ceiling(b, routes, tms, args.iters, flush)
    if "phases" in modes:
        result["phases"] = phases(build_ext(True), layer, routes, variants, tms, args.phase_calls, flush, args.seed)
    ok = all(r["pass"] for r in result.get("check", {}).values()) and all(
        r["replay_equals_eager"] for r in result.get("capture", {}).values()) and all(
        r["eager_mismatch"] == 0 and r["graph_mismatch"] == 0 for r in result.get("stress", {}).values())
    result["pass"] = ok
    text = json.dumps(result, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    log(f"pass={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
