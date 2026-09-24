#!/usr/bin/env python3
"""Expert-overlap census for DSpark verify steps (moe-expert-dedup step 1).

A DSpark-k verify step runs p2b with m = k+1 rows x top-6. Any expert
picked by two of those rows is streamed twice. This measures how often
that happens, per MoE layer, from vLLM's own routing capture, so there is
no counter in the graph.

The live serve does not return routing. It needs a boot with the vLLM flag
(vllm/config/model.py:250 enable_return_routed_experts, forwarded to both
ranks by EXTRA_ARGS):

    EXTRA_ARGS=--enable-return-routed-experts ./run.sh
    python3 tools/moe_census.py --out results/.../census.json --save-npy DIR

Offline re-analysis of saved arrays: python3 tools/moe_census.py --npy DIR/*.npy

The response carries routing for accepted tokens only, so the census uses
sliding windows of m consecutive decode tokens. That approximates the
verify rows (last accepted token + k drafts) when the drafts are accepted.

Saving bound: dup x 22.4 ms p2b/step (trace3). The realistic figure scales
by the streaming share of p2b (~0.78); Hadamard, SwiGLU, reduce, barriers
and dequant do not shrink. A step is ~66 ms.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

N_EXPERTS = 384
TOP_K = 6
P2B_MS_PER_STEP = 22.4
STREAM_SHARE = 0.78
STEP_MS = 66.0
STOP_BELOW = 0.08
PROCEED_AT = 0.10

CODE_PROMPTS = (
    "Write a complete Python module that implements an LRU cache with a TTL per "
    "entry, thread safety, and a stats() method. Include unittest tests. Code only.",
    "Implement in C++17 a lock-free single-producer single-consumer ring buffer "
    "template with push, pop, size and a small benchmark main(). Code only.",
)


def decode_b64(s: str) -> np.ndarray:
    """routed_experts field: base64 .npy of [tokens, layers, top_k]."""
    return np.load(io.BytesIO(base64.b64decode(s)), allow_pickle=False)


def routed_layers(arr: np.ndarray) -> list[int]:
    """Layers whose every row names top_k distinct experts.

    Dense or never-captured layers stay zero-filled and are dropped.
    """
    t, layers, k = arr.shape
    s = np.sort(arr, axis=2)
    distinct = (np.diff(s, axis=2) != 0).all(axis=2) if k > 1 else np.ones((t, layers), bool)
    return [layer for layer in range(layers) if t and distinct[:, layer].all()]


def window_dup(ids: np.ndarray, m: int) -> np.ndarray:
    """1 - unique/(m*k) for each window of m consecutive rows of ids [T, k]."""
    t, k = ids.shape
    if t < m:
        return np.zeros(0)
    out = np.empty(t - m + 1)
    for i in range(t - m + 1):
        out[i] = 1.0 - len(np.unique(ids[i : i + m])) / (m * k)
    return out


def random_dup(m: int, k: int = TOP_K, n_experts: int = N_EXPERTS) -> float:
    """Expected dup fraction for independent uniform routing."""
    unique = n_experts * (1.0 - (1.0 - k / n_experts) ** m)
    return 1.0 - unique / (m * k)


def impact(dup: float) -> dict:
    upper = dup * P2B_MS_PER_STEP
    real = upper * STREAM_SHARE
    return {
        "upper_ms_per_step": upper,
        "realistic_ms_per_step": real,
        "realistic_pct_of_step": 100.0 * real / STEP_MS,
    }


def verdict(dup: float) -> str:
    if dup < STOP_BELOW:
        return f"STOP: dup {dup:.3f} < {STOP_BELOW}; gain below noise, park 2a/2b"
    if dup < PROCEED_AT:
        return f"MARGINAL: dup {dup:.3f}; microbench only, no serve arm"
    return f"PROCEED: dup {dup:.3f} >= {PROCEED_AT}; microbench then serve arm"


def census(arrays: dict[str, np.ndarray], m: int) -> dict:
    """Per-layer and overall window dup over decode routing arrays [T, L, k]."""
    per_layer: dict[int, list[np.ndarray]] = {}
    prompts = {}
    for name, arr in arrays.items():
        layers = routed_layers(arr)
        vals = []
        for layer in layers:
            d = window_dup(arr[:, layer, :], m)
            per_layer.setdefault(layer, []).append(d)
            vals.append(d)
        cat = np.concatenate(vals) if vals else np.zeros(0)
        prompts[name] = {
            "tokens": int(arr.shape[0]),
            "layers": len(layers),
            "dup": float(cat.mean()) if cat.size else None,
        }
    layer_dup = {
        layer: float(np.concatenate(ds).mean())
        for layer, ds in sorted(per_layer.items())
        if sum(d.size for d in ds)
    }
    dup = float(np.mean(list(layer_dup.values()))) if layer_dup else 0.0
    k = next(iter(arrays.values())).shape[2] if arrays else TOP_K
    return {
        "m": m,
        "top_k": k,
        "dup_mean_over_layers": dup,
        "unique_ratio": 1.0 - dup,
        "random_baseline_dup": random_dup(m, k),
        "per_layer_dup": layer_dup,
        "prompts": prompts,
        **impact(dup),
        "verdict": verdict(dup),
    }


def synth_routing(m: int, k: int, n_experts: int, dup: float, rng: random.Random) -> list[list[int]]:
    """[m, k] routing with exactly round(dup*m*k) duplicate slots.

    Duplicates reuse experts of the previous row (adjacent-token locality);
    every row keeps k distinct experts, like a real top-k router.
    """
    d = round(dup * m * k)
    if d > (m - 1) * k:
        raise ValueError(f"dup {dup} needs {d} repeats; at most {(m - 1) * k} fit in m={m}, k={k}")
    per = [0] * m
    for i in range(d):
        per[1 + i % (m - 1)] += 1
    rows: list[list[int]] = []
    used: set[int] = set()
    for r in range(m):
        repeats = rng.sample(rows[r - 1], per[r]) if per[r] else []
        pool = [x for x in range(n_experts) if x not in used]
        row = repeats + rng.sample(pool, k - per[r])
        rng.shuffle(row)
        used.update(row)
        rows.append(row)
    return rows


def request_routing(url: str, model: str, prompt: str, max_tokens: int, temperature: float) -> np.ndarray:
    """Decode-token routing [completion rows, L, k] for one non-streamed chat call."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "ignore_eos": True,
        "temperature": temperature,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False, "thinking": False, "reasoning_effort": "low"},
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        out = json.load(resp)
    enc = out["choices"][0].get("routed_experts")
    if not enc:
        raise SystemExit("no routed_experts in the response: boot with EXTRA_ARGS=--enable-return-routed-experts")
    arr = decode_b64(enc)
    return arr[int(out["usage"]["prompt_tokens"]) :]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    ap.add_argument("--m", type=int, default=4, help="verify rows = NUM_SPECULATIVE_TOKENS + 1")
    ap.add_argument("--runs", type=int, default=2, help="requests per prompt")
    ap.add_argument("--npy", nargs="*", help="analyze saved arrays instead of calling the serve")
    ap.add_argument("--save-npy", type=Path, help="write each decode routing array here")
    ap.add_argument("--out", type=Path, help="write the census JSON here")
    args = ap.parse_args()

    arrays: dict[str, np.ndarray] = {}
    if args.npy:
        for p in args.npy:
            arrays[Path(p).stem] = np.load(p, allow_pickle=False)
    else:
        from measure_lail_prose import LAIL_PROSE, MAX_TOKENS, TEMPERATURE

        prompts = {"lail": LAIL_PROSE, **{f"code{i}": p for i, p in enumerate(CODE_PROMPTS)}}
        for name, prompt in prompts.items():
            for run in range(args.runs):
                key = f"{name}_r{run}"
                arrays[key] = request_routing(args.url, args.model, prompt, MAX_TOKENS, TEMPERATURE)
                print(f"{key}: {arrays[key].shape}", flush=True)
                if args.save_npy:
                    args.save_npy.mkdir(parents=True, exist_ok=True)
                    np.save(args.save_npy / f"{key}.npy", arrays[key])

    result = census(arrays, args.m)
    for group in ("lail", "code"):
        sub = {k: v for k, v in arrays.items() if k.startswith(group)}
        if sub:
            result[f"{group}_dup"] = census(sub, args.m)["dup_mean_over_layers"]
    text = json.dumps(result, indent=1, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
