#!/usr/bin/env python3
"""Capture greedy (temperature 0) completions for fixed prompts from the live
serve, for the lm_head MXFP8 argmax-flip live check.

Saved as JSON: {prompt_hash: {prompt, completion}}. Compare stock vs lm_head
serves token-by-token: flip rate = differing tokens / total (greedy decode,
temp 0, thinking off). Offline numerics (tests/numerics_lmhead_mxfp8.py)
bound the expected flip rate near small single digits percent.
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MAX_TOKENS = 200

PROMPTS = [
    "17 * 19 = ? Step by step, then answer.",
    "Write a Python function to reverse a linked list iteratively.",
    "Explain why the sky is blue in exactly three sentences.",
    "List the first 20 prime numbers, comma separated.",
    "Translate into French: The quick brown fox jumps over the lazy dog.",
    "What is the capital of Australia? One word.",
    "Summarize the plot of Hamlet in five bullet points.",
    "Write a haiku about GPU memory bandwidth.",
    "Solve for x: 3x^2 - 12x + 9 = 0. Show steps.",
    "Name three uses of safetensors in ML infrastructure.",
    "Continue: Once upon a time in a datacenter far away,",
    "What happens if you divide by zero in IEEE 754 floating point?",
]


def gen(prompt: str) -> str:
    body = {
        "model": "deepseek-ai/DeepSeek-V4.1-Flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "chat_template_kwargs": {
            "enable_thinking": False,
            "thinking": False,
            "reasoning_effort": "low",
        },
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.load(r)
    return out["choices"][0]["message"]["content"]


def main() -> int:
    out_path = sys.argv[1]
    tag = sys.argv[2]
    results = {}
    for p in PROMPTS:
        h = hashlib.sha1(p.encode()).hexdigest()[:12]
        results[h] = {"prompt": p, "completion": gen(p)}
        print(f"[{tag}] {h} ok ({len(results[h]['completion'])} chars)", flush=True)
    with open(out_path, "w") as f:
        json.dump({"tag": tag, "results": results}, f, indent=1)
    print(f"wrote {out_path} ({len(results)} prompts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
