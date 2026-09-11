#!/usr/bin/env python3
"""Two concurrent L.A.I.L-style streams. Crashed EngineCore on persistent_topk."""
from __future__ import annotations

import argparse
import json
import sys
import threading
import urllib.error
import urllib.request

# Past index_topk=512 so decode uses persistent_topk, not the short-context fill.
PROMPT = " ".join(["stream"] * 400)


def _one(url: str, model: str, max_tokens: int, tag: str) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": f"{tag} {PROMPT}"}],
            "max_tokens": max_tokens,
            "min_tokens": max_tokens,
            "temperature": 0.2,
            "stream": True,
            "chat_template_kwargs": {"thinking": False},
        }
    ).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    chunks: list[str] = []
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            payload = json.loads(data)
            delta = payload["choices"][0].get("delta") or {}
            piece = delta.get("content") or ""
            if piece:
                chunks.append(piece)
    text = "".join(chunks)
    if not text.strip():
        raise RuntimeError(f"{tag} empty content")
    return text


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    p.add_argument("--max-tokens", type=int, default=32)
    args = p.parse_args()
    err: list[BaseException] = []
    out: dict[str, str] = {}

    def run(tag: str) -> None:
        try:
            out[tag] = _one(args.url, args.model, args.max_tokens, tag)
        except BaseException as e:
            err.append(e)

    t1 = threading.Thread(target=run, args=("A",))
    t2 = threading.Thread(target=run, args=("B",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    if err:
        for e in err:
            if isinstance(e, urllib.error.HTTPError):
                print(e.read().decode()[:800], file=sys.stderr)
            print(repr(e), file=sys.stderr)
        return 1
    print("A", out["A"][:80].replace("\n", " "))
    print("B", out["B"][:80].replace("\n", " "))
    print("TWO_STREAM_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
