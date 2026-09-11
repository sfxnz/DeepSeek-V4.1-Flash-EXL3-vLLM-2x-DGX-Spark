#!/usr/bin/env python3
"""Replay the L.A.I.L prefill that killed EngineCore (81 tokens > 64)."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

# Scheduler dump: prompt_token_ids_len=81, max_tokens=512, min_tokens=512,
# ignore_eos, stream. Prefill crash fires before decode; keep max_tokens small.
PROMPT = " ".join(["benchmark"] * 70)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    p.add_argument("--max-tokens", type=int, default=8)
    args = p.parse_args()
    body = json.dumps(
        {
            "model": args.model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": args.max_tokens,
            "temperature": 0.2,
            "stream": False,
            "chat_template_kwargs": {"thinking": False},
        }
    ).encode()
    req = urllib.request.Request(
        args.url, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(e.read().decode()[:800], file=sys.stderr)
        raise
    text = payload["choices"][0]["message"].get("content") or ""
    if not str(text).strip():
        raise SystemExit("empty content")
    print(text.strip()[:200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
