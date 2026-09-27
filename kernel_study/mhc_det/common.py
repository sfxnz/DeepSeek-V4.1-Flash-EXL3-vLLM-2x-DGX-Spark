"""Shared helpers for the mHC det microbenches (run inside the serve image on spark2).

- real tensors from the local pack snapshot (read-only mount /snaproot)
- input generators (real embeddings, random scales, edge cases)
- CUDA-event timing with a queue-ahead spin kernel, cold-L2 (64 MB flush) and warm modes
"""
from __future__ import annotations

import ctypes
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, "/repo/docker/patch")

SNAP = "/snaproot/snapshots/2.0bpw-mcg-lmhead-mxfp8"
PEAK_GBPS = 250.0
HIDDEN, HC = 5120, 4
K_FULL = HIDDEN * HC


def _index():
    with open(os.path.join(SNAP, "model.safetensors.index.json")) as fh:
        return json.load(fh)["weight_map"]


def load(names: list[str], device="cuda") -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    wm = _index()
    by_file: dict[str, list[str]] = {}
    for n in names:
        by_file.setdefault(wm[n], []).append(n)
    out = {}
    for f, ns in by_file.items():
        with safe_open(os.path.join(SNAP, f), framework="pt", device="cpu") as fh:
            for n in ns:
                out[n] = fh.get_tensor(n).to(device)
    return out


def mhc_names() -> list[str]:
    names = []
    for prefix in [f"layers.{i}" for i in range(40)] + [f"mtp.{i}" for i in range(3)]:
        for sub in ("attn", "ffn"):
            names += [f"{prefix}.hc_{sub}_fn", f"{prefix}.hc_{sub}_scale", f"{prefix}.hc_{sub}_base"]
    return names


def real_fns() -> list[tuple[str, torch.Tensor]]:
    """All 86 decode fn matrices [24, 20480] + the layer-0 broadcast fn [24, 5120]."""
    t = load([n for n in mhc_names() if n.endswith("_fn")])
    fns = [(n, t[n].float().contiguous()) for n in sorted(t, key=_layer_key)]
    l0 = t["layers.0.hc_attn_fn"]
    bc = l0.detach().view(-1, HC, HIDDEN).sum(dim=1).contiguous()  # finalize_mhc_broadcast_weights
    fns.append(("layers.0.hc_attn_fn_broadcast", bc))
    return fns


def _layer_key(n: str):
    parts = n.split(".")
    return (parts[0] != "layers", int(parts[1]), parts[2])


def embeddings(n_rows: int, seed: int = 0) -> torch.Tensor:
    """Real embedding rows (bf16 [n_rows, 5120]) for pseudo-random token ids."""
    from safetensors import safe_open

    wm = _index()
    name = "embed.weight"
    with safe_open(os.path.join(SNAP, wm[name]), framework="pt", device="cpu") as fh:
        sl = fh.get_slice(name)
        vocab = sl.get_shape()[0]
        g = torch.Generator().manual_seed(seed)
        ids = torch.randint(0, vocab, (n_rows,), generator=g).tolist()
        rows = [sl[i : i + 1] for i in ids]
    return torch.cat(rows).to("cuda", torch.bfloat16)


def make_x(kind: str, t: int, k: int, seed: int, emb: torch.Tensor | None = None) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    if kind == "emb":  # layer-0 style: embedding rows (broadcast to 4 streams for K=20480)
        assert emb is not None
        idx = torch.randint(0, emb.shape[0], (t,), device="cuda", generator=g)
        e = emb[idx]
        return (e.repeat(1, k // HIDDEN) if k != HIDDEN else e).contiguous()
    if kind == "randn":
        return torch.randn(t, k, device="cuda", generator=g).bfloat16()
    if kind == "heavy":  # heavy-tailed magnitudes
        z = torch.randn(t, k, device="cuda", generator=g)
        return (z * torch.exp(2.0 * torch.randn(t, k, device="cuda", generator=g))).bfloat16()
    if kind == "big":
        return (torch.randn(t, k, device="cuda", generator=g) * 1000.0).bfloat16()
    if kind == "tiny":
        return (torch.randn(t, k, device="cuda", generator=g) * 1e-30).bfloat16()
    if kind == "sparse":
        x = torch.zeros(t, k, device="cuda")
        idx = torch.randint(0, k, (t, 16), device="cuda", generator=g)
        x.scatter_(1, idx, torch.randn(t, 16, device="cuda", generator=g) * 50)
        return x.bfloat16()
    raise ValueError(kind)


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and bool((a.contiguous().view(torch.int32) == b.contiguous().view(torch.int32)).all())


def maxdiff(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.double(), b.double()
    d = (a - b).abs()
    rel = d / b.abs().clamp_min(1e-30)
    return {"max_abs": float(d.max()), "max_rel": float(rel.max()), "n_diff": int((d != 0).sum())}


_SPIN_SRC = r"""
extern "C" __global__ void spin_ns(unsigned long long ns) {
  unsigned long long t0, t1;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1)); } while (t1 - t0 < ns);
}
"""


class Timer:
    """Per-call CUDA-event timing. Each batch starts with a GPU spin long enough for the host
    to enqueue the whole batch, so event intervals hold GPU time only (no host gaps).
    Cold = a read-only 64 MB sweep before every call (a dirty-L2 flush adds write-back traffic
    that the serve does not have: stream_probe.json rw vs ro). GB10 event resolution ~1 us."""

    def __init__(self, flush_mb: int = 64) -> None:
        from mhc_det_rt import Module

        self.flush = torch.empty(flush_mb * 2**20 // 4, device="cuda")
        self.spin = Module(_SPIN_SRC, "spin.cu").function("spin_ns")

    def _spin(self, us: float) -> None:
        self.spin.launch((1,), (1,), 0, [(int(us * 1000), ctypes.c_ulonglong)])

    def run(self, arms: dict, iters: int = 200, warmup: int = 20, cold: bool = True,
            pre=None, batch: int = 20, spin_us_per_call: float = 250.0) -> dict:
        """arms: name -> callable. pre: callable run before each timed call (after the flush),
        e.g. re-writing the activation so it is L2-hot like in the serve. Arms alternate
        within every iteration and rotate their order each iteration."""
        names = list(arms)
        samples = {n: [] for n in names}
        total = warmup + iters
        it = 0
        while it < total:
            nb = min(batch, total - it)
            self._spin(nb * len(names) * spin_us_per_call)
            recs = []
            for j in range(nb):
                order = names[(it + j) % len(names):] + names[: (it + j) % len(names)]
                for n in order:
                    if cold:
                        self.flush.sum()  # read-only 64 MB sweep: L2 left full of clean lines
                    if pre is not None:
                        pre()
                    a = torch.cuda.Event(enable_timing=True)
                    b = torch.cuda.Event(enable_timing=True)
                    a.record()
                    arms[n]()
                    b.record()
                    recs.append((it + j, n, a, b))
            torch.cuda.synchronize()
            for i, n, a, b in recs:
                if i >= warmup:
                    samples[n].append(a.elapsed_time(b) * 1000.0)
            it += nb
        return {n: summarize(v) for n, v in samples.items()}


def summarize(v: list[float]) -> dict:
    s = sorted(v)
    q = lambda p: s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]  # noqa: E731
    return {"n": len(s), "median_us": round(statistics.median(s), 3), "p10_us": round(q(0.10), 3),
            "p90_us": round(q(0.90), 3), "mean_us": round(statistics.fmean(s), 3), "min_us": round(s[0], 3)}


def graph_time(fn, reps: int, calls: int, warm_replays: int = 3) -> dict:
    """Capture `calls` calls of fn into one graph; time `reps` replays; per-call us stats."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(0)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for c in range(calls):
            fn(c)
    for _ in range(warm_replays):
        g.replay()
    torch.cuda.synchronize()
    per = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        per.append(a.elapsed_time(b) * 1000.0 / calls)
    return summarize(per)


def _capture(body, calls: int):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        body(0)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for c in range(calls):
            body(c)
    return g


def graph_delta_time(arm, base, reps: int, calls: int) -> dict:
    """Per-call us of `arm` inside a serve-like sequence: graph G1 = [base(c); arm(c)] x calls,
    graph G0 = [base(c)] x calls, replayed alternately; per-call = (t(G1) - t(G0)) / calls.
    With a different fn per call (86 > L2), each call streams its weights cold."""
    g1 = _capture(lambda c: (base(c), arm(c)), calls)
    g0 = _capture(base, calls)
    for _ in range(3):
        g1.replay()
        g0.replay()
    torch.cuda.synchronize()
    per, t1s, t0s = [], [], []
    for _ in range(reps):
        ts = []
        for g in (g1, g0):
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b) * 1000.0)
        t1s.append(ts[0] / calls)
        t0s.append(ts[1] / calls)
        per.append((ts[0] - ts[1]) / calls)
    out = summarize(per)
    out["with_base_us"] = round(statistics.median(t1s), 3)
    out["base_only_us"] = round(statistics.median(t0s), 3)
    return out


def gbps(nbytes: float, us: float) -> float:
    return nbytes / (us * 1e-6) / 1e9
