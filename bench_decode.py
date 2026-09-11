#!/usr/bin/env python3
"""Streamed decode bench against a live OpenAI-compatible /v1/chat/completions.

Decode tok/s drops the first completion token (TTFT) and divides remaining
tokens by wall after first token. Aggregate is total decode tokens over the
wave wall minus median TTFT.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


PHASES = {
    "prose": (
        "Write a short paragraph about why sparse attention helps long-context "
        "language models. Keep it around eighty words. No bullet points."
    ),
    "structured": (
        "Count from 1 to 200. Output only the numbers, separated by commas, "
        "with no other text."
    ),
}


def decode_tokens(completion_tokens: int) -> int:
    """Tokens counted toward decode rate: completion minus the first (TTFT)."""
    return max(int(completion_tokens) - 1, 0)


def decode_rate(completion_tokens: int, decode_s: float) -> float:
    """Per-stream decode tok/s from usage completion_tokens and post-TTFT wall."""
    n = decode_tokens(completion_tokens)
    if decode_s <= 0:
        return 0.0
    return n / decode_s


def aggregate_rate(
    completion_tokens_list: list[int], wall_s: float, median_ttft_s: float
) -> float:
    """Wave aggregate tok/s: sum of decode tokens over wall minus median TTFT."""
    n = sum(decode_tokens(c) for c in completion_tokens_list)
    adj = wall_s - median_ttft_s
    if adj <= 0:
        return 0.0
    return n / adj


def stream_one(url: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "ignore_eos": True,
            "chat_template_kwargs": {"thinking": False, "reasoning_effort": "low"},
        }
    ).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    first = None
    chunks = 0
    usage = {}
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
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    decode_s = t1 - first
    return {
        "ttft_s": first - t0,
        "total_s": t1 - t0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion,
        "decode_tok_s": decode_rate(completion, decode_s),
        "chunks": chunks,
    }


def wave(
    url: str, model: str, prompt: str, max_tokens: int, concurrency: int
) -> tuple[list[dict], float, float]:
    t0 = time.perf_counter()
    out = []
    errors: list[BaseException] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [
            pool.submit(stream_one, url, model, prompt, max_tokens)
            for _ in range(concurrency)
        ]
        for fut in as_completed(futs):
            try:
                out.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                print(f"stream failed: {exc}", file=sys.stderr, flush=True)
                errors.append(exc)
    if errors:
        raise RuntimeError(f"{len(errors)} stream(s) in the wave failed")
    if not out:
        raise RuntimeError("every stream in the wave failed")
    if any(int(r["completion_tokens"]) == 0 for r in out):
        raise RuntimeError("a stream returned completion_tokens==0")
    wall = time.perf_counter() - t0
    ttfts = [r["ttft_s"] for r in out]
    agg = aggregate_rate(
        [int(r["completion_tokens"]) for r in out],
        wall,
        statistics.median(ttfts),
    )
    return out, wall, agg


def median_key(rows: list[dict], key: str) -> float:
    return statistics.median(r[key] for r in rows)


SPEC_COUNTERS = ("num_drafts", "num_draft_tokens", "num_accepted_tokens")


def spec_counters(metrics_url: str) -> dict[str, float] | None:
    """Sum vLLM's spec-decode counters across label sets. None when absent."""
    try:
        with urllib.request.urlopen(metrics_url, timeout=10) as resp:
            text = resp.read().decode("utf-8", "replace")
    except (OSError, urllib.error.URLError):
        return None
    out = dict.fromkeys(SPEC_COUNTERS, 0.0)
    seen = False
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith("vllm:spec_decode_"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        for counter in SPEC_COUNTERS:
            if name in (
                f"vllm:spec_decode_{counter}_total",
                f"vllm:spec_decode_{counter}",
            ):
                out[counter] += float(line.rsplit(" ", 1)[1])
                seen = True
    return out if seen else None


def acceptance(before: dict[str, float] | None, after: dict[str, float] | None) -> dict:
    if before is None or after is None:
        return {}
    drafts = after["num_drafts"] - before["num_drafts"]
    draft_tokens = after["num_draft_tokens"] - before["num_draft_tokens"]
    accepted = after["num_accepted_tokens"] - before["num_accepted_tokens"]
    if drafts <= 0:
        return {}
    return {
        "acceptance_len": 1.0 + accepted / drafts,
        "draft_acceptance_rate": (accepted / draft_tokens) if draft_tokens > 0 else 0.0,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 2])
    p.add_argument("--phase", choices=[*PHASES, "both"], default="both")
    args = p.parse_args()

    phases = list(PHASES) if args.phase == "both" else [args.phase]
    print(
        f"url={args.url} model={args.model} max_tokens={args.max_tokens} "
        f"runs={args.runs} concurrency={args.concurrency} phases={phases}",
        flush=True,
    )
    metrics_url = args.url.split("/v1/", 1)[0] + "/metrics"
    summary = []
    for phase in phases:
        prompt = PHASES[phase]
        for c in args.concurrency:
            per_stream = []
            aggs = []
            counters_before = spec_counters(metrics_url)
            for i in range(args.runs):
                rows, wall, agg = wave(args.url, args.model, prompt, args.max_tokens, c)
                per_stream.extend(rows)
                aggs.append(agg)
                dec = ",".join(f"{r['decode_tok_s']:.2f}" for r in rows)
                ttft = ",".join(f"{r['ttft_s']:.3f}" for r in rows)
                print(
                    f"phase={phase} c={c} run={i+1} wall={wall:.2f}s agg={agg:.2f} tok/s "
                    f"per_stream=[{dec}] ttft=[{ttft}]",
                    flush=True,
                )
            summary.append(
                {
                    "phase": phase,
                    "concurrency": c,
                    "median_decode_tok_s": median_key(per_stream, "decode_tok_s"),
                    "median_ttft_s": median_key(per_stream, "ttft_s"),
                    "median_agg_tok_s": statistics.median(aggs),
                    "median_completion_tokens": median_key(per_stream, "completion_tokens"),
                    "n": len(per_stream),
                    **acceptance(counters_before, spec_counters(metrics_url)),
                }
            )
    print("SUMMARY", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
