#!/usr/bin/env python3
"""Post-ready warmup: move first-request JIT off the first user's TTFT.

run.sh sends three chats after /v1/models is up (WARMUP=1):
  1. greedy 17*19: prefill metadata, DSpark propose/verify, rejection kernels
  2. temperature 0.7: the sampling kernels (_gumbel_sample, _resample)
  3. a ~3k-token novel prefill: chunked-prefill metadata, indexer, MXFP8 quantize

Every prompt starts with a random nonce, so nothing lands in the prefix cache
that a user request could hit. This is a first-request TTFT fix only: it does
not change steady-state decode, and TileLang mhc still JITs once per new
prefill shape later in the uptime.
"""
from __future__ import annotations

import argparse
import secrets
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from corpus import build_doc  # noqa: E402
from smoke_chat import content_of, post  # noqa: E402

PREFILL_TOKENS = 3000
TOKENS_PER_CHAR = 0.27  # repo text on the V4.1 tokenizer, roughly


def requests(nonce: str) -> list[tuple[str, str, float, int]]:
    """(label, prompt, temperature, max_tokens) for the three warmup chats."""
    doc = build_doc(PREFILL_TOKENS, TOKENS_PER_CHAR, seed=int(nonce, 16))
    return [
        ("greedy", f"[{nonce}] What is 17*19? Return only the integer.", 0.0, 16),
        ("t=0.7", f"[{nonce}] Name one prime number above 100.", 0.7, 16),
        ("prefill", f"[{nonce}]\n{doc}\n\nSummarize the text above in one sentence.", 0.0, 8),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    args = ap.parse_args()
    nonce = secrets.token_hex(8)
    for label, prompt, temperature, max_tokens in requests(nonce):
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"thinking": False, "reasoning_effort": "low"},
        }
        t0 = time.monotonic()
        text = content_of(post(args.url, body))
        print(f"==> warmup {label}: {time.monotonic() - t0:.1f}s {text[:40]!r}")
        if label == "greedy" and "323" not in text:
            print(f"WARNING: warmup 17*19 answered {text!r}, not 323", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
