#!/usr/bin/env python3
"""Warm-prefix cell: send the same >=1k-token prompt twice, nonce at the END.

The doc is fresh per invocation (time seed) and ends with a nonce, so the
first request is cold. The second request is identical and should hit
floor((N-1)/B)*B cached prompt tokens (B = scheduler hit unit, 128 here: lcm
of the 64/128 KV groups). Hits come from /metrics vllm:prefix_cache_hits
deltas; TTFT cold vs warm shows what a shared long system prompt saves.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benches"))

from micro import fresh_doc, stream_once  # noqa: E402
from corpus import char_ratio, make_tokenizer  # noqa: E402


def parse_prefix_counters(text: str) -> dict[str, float] | None:
    """Sum vllm:prefix_cache_{queries,hits} across label sets (not external_)."""
    out = {"queries": 0.0, "hits": 0.0}
    seen = False
    for line in text.splitlines():
        if not line.startswith("vllm:prefix_cache_"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        for key in out:
            if name in (f"vllm:prefix_cache_{key}_total", f"vllm:prefix_cache_{key}"):
                out[key] += float(line.rsplit(" ", 1)[1])
                seen = True
    return out if seen else None


def prefix_counters(metrics_url: str) -> dict[str, float] | None:
    try:
        with urllib.request.urlopen(metrics_url, timeout=10) as resp:
            return parse_prefix_counters(resp.read().decode("utf-8", "replace"))
    except OSError:
        return None


def expected_hits(prompt_tokens: int, block: int) -> int:
    """Cached tokens a full repeat can reuse: whole blocks, last token recomputed."""
    return max(0, (prompt_tokens - 1) // block * block)


def hits_delta(before: dict | None, after: dict | None) -> float | None:
    if before is None or after is None:
        return None
    return after["hits"] - before["hits"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    p.add_argument("--tokens", type=int, default=2048)
    p.add_argument("--hit-block", type=int, default=128)
    p.add_argument("--timeout", type=int, default=900)
    args = p.parse_args()
    if args.tokens < 1024:
        p.error("--tokens must be >= 1024")

    tok = make_tokenizer(args.url.rsplit("/v1", 1)[0], args.model)
    metrics_url = args.url.split("/v1/", 1)[0] + "/metrics"
    seed = (time.time_ns() // 1000) % (1 << 30)
    ratio = char_ratio(tok)
    # trim_to_tokens drops whole paragraphs, so a doc can land well short of
    # the target; reseed until it clears the >=1k floor.
    for attempt in range(5):
        doc, ntok = fresh_doc(args.tokens, ratio, tok, seed + attempt)
        if ntok >= 1024:
            break
    else:
        print(f"doc stayed below 1024 tokens ({ntok}); raise --tokens", file=sys.stderr)
        return 1
    prompt = doc + f"\n\nReference {uuid.uuid4().hex}. Reply with one word."

    runs = []
    for label in ("cold", "warm"):
        before = prefix_counters(metrics_url)
        res = stream_once(args.url, args.model, prompt, 1, args.timeout)
        runs.append({**res, "hits": hits_delta(before, prefix_counters(metrics_url))})
        print(f"{label} ttft={res['ttft_s']:.3f}s prompt_tokens={res['prompt_tokens']} "
              f"hits={runs[-1]['hits']}", flush=True)
    n = runs[1]["prompt_tokens"]
    want = expected_hits(n, args.hit_block)
    got = runs[1]["hits"]
    summary = {
        "cell": "warm_prefix",
        "prompt_tokens": n,
        "hit_block": args.hit_block,
        "ttft_cold_s": runs[0]["ttft_s"],
        "ttft_warm_s": runs[1]["ttft_s"],
        "hits_cold": runs[0]["hits"],
        "hits_warm": got,
        "expected_hits": want,
        "hit_fraction_of_expected": (got / want) if got is not None and want else None,
    }
    print("SUMMARY", json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
