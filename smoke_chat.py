#!/usr/bin/env python3
"""Thinking-off chat smoke. Fails if content is empty or HTTP is not 2xx."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def post(url: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        raw = resp.read().decode()
        if resp.status < 200 or resp.status >= 300:
            raise SystemExit(f"HTTP {resp.status}: {raw[:500]}")
        return json.loads(raw)


def content_of(payload: dict) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise SystemExit(f"no choices: {payload}")
    msg = choices[0].get("message") or {}
    text = msg.get("content")
    if not isinstance(text, str) or not text.strip():
        raise SystemExit(f"empty content (thinking-on trap?): {payload}")
    return text.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    ap.add_argument("--prompt", default="What is 17*19? Return only the integer.")
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"thinking": False, "reasoning_effort": "low"},
    }
    payload = post(args.url, body)
    text = content_of(payload)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
