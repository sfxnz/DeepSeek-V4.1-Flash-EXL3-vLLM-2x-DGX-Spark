#!/usr/bin/env python3
"""Frozen L.A.I.L prose decode harness: c=1, 512 tokens, temperature 0.2.

Matches local-ai-lab packages/serve-engine/app/services/perf.py
(completion_body + decode_tok_per_s = completion_tokens / post-TTFT wall).
Also reports recipe math (completion_tokens-1) and DSpark acceptance.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench_decode import acceptance, decode_rate, spec_counters  # noqa: E402

LAIL_PROSE = (
    "Continue this essay in the same voice. Do not stop.\n\n"
    "Decode throughput and time-to-first-token feel different when a coding "
    "agent shares a long system prompt across tabs on a DGX Spark with unified "
    "memory. The KV cache is the product, not a leftover after util. "
)
MAX_TOKENS = 512
TEMPERATURE = 0.2


def lail_decode_tok_s(completion_tokens: int, decode_s: float) -> float:
    """L.A.I.L decode tok/s: usage completion_tokens over wall after first token."""
    if decode_s <= 0 or completion_tokens <= 0:
        return 0.0
    return completion_tokens / decode_s


def completion_body(model: str) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": LAIL_PROSE}],
        "max_tokens": MAX_TOKENS,
        "min_tokens": MAX_TOKENS,
        "ignore_eos": True,
        "temperature": TEMPERATURE,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {
            "enable_thinking": False,
            "thinking": False,
            "reasoning_effort": "low",
        },
    }


def stream_one(url: str, model: str) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(completion_body(model)).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    first = None
    chunks = 0
    usage: dict = {}
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if ev.get("usage"):
                usage = ev["usage"]
            choices = ev.get("choices") or []
            if not choices:
                continue
            delta = (choices[0].get("delta") or {}).get("content") or ""
            if delta and first is None:
                first = time.perf_counter()
            if delta:
                chunks += 1
    t1 = time.perf_counter()
    if first is None:
        raise RuntimeError("no streamed content tokens")
    completion = int(usage.get("completion_tokens") or 0)
    if completion == 0:
        raise RuntimeError("completion_tokens==0")
    decode_s = t1 - first
    return {
        "ttft_s": first - t0,
        "decode_s": decode_s,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": completion,
        "chunks": chunks,
        "lail_tok_s": lail_decode_tok_s(completion, decode_s),
        "recipe_tok_s": decode_rate(completion, decode_s),
        "tokens_per_chunk": (completion / chunks) if chunks else 0.0,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    p.add_argument("--runs", type=int, default=3)
    args = p.parse_args()
    metrics_url = args.url.split("/v1/", 1)[0] + "/metrics"
    print(
        f"url={args.url} model={args.model} max_tokens={MAX_TOKENS} "
        f"temperature={TEMPERATURE} runs={args.runs} phase=lail_prose c=1",
        flush=True,
    )
    rows = []
    for i in range(args.runs):
        before = spec_counters(metrics_url)
        row = stream_one(args.url, args.model)
        row.update(acceptance(before, spec_counters(metrics_url)))
        rows.append(row)
        print(f"run={i+1} {json.dumps(row)}", flush=True)
    summary = {
        "phase": "lail_prose",
        "concurrency": 1,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "n": len(rows),
        "median_lail_tok_s": statistics.median(r["lail_tok_s"] for r in rows),
        "median_recipe_tok_s": statistics.median(r["recipe_tok_s"] for r in rows),
        "median_ttft_s": statistics.median(r["ttft_s"] for r in rows),
        "median_completion_tokens": statistics.median(
            r["completion_tokens"] for r in rows
        ),
    }
    accs = [r["acceptance_len"] for r in rows if "acceptance_len" in r]
    if accs:
        summary["median_acceptance_len"] = statistics.median(accs)
        summary["median_draft_acceptance_rate"] = statistics.median(
            r["draft_acceptance_rate"] for r in rows if "draft_acceptance_rate" in r
        )
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
