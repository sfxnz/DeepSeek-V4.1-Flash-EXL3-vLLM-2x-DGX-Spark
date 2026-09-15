#!/usr/bin/env python3
"""Live OpenAI-compat vision smoke against a running serve.

Fails if the endpoint returns HTTP 400 "is not a multimodal model".
Uses a hardcoded JPEG so the host does not need an image library.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

# Minimal 1x1 JPEG. V4.1's pixel floor may reject this on a live serve;
# that is a processor error, not "is not a multimodal model".
_RED_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEB"
    "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEB"
    "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAAR"
    "CAABAAEDAREAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAA"
    "AAAAAAD/2gAMAwEAAhEDEQA/AKpA/9k="
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    args = parser.parse_args()
    body = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What color is this image? Reply with one word only.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{_RED_JPEG_B64}"},
                    },
                ],
            }
        ],
        "max_tokens": 64,
        "temperature": 0,
        "chat_template_kwargs": {"thinking": False, "reasoning_effort": "low"},
    }
    req = urllib.request.Request(
        args.url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        err = exc.read().decode()
        print(err, file=sys.stderr)
        if "is not a multimodal model" in err:
            print("result=fail reason=not_multimodal", file=sys.stderr)
        return 1
    msg = data["choices"][0]["message"]
    text = (msg.get("content") or "") + (msg.get("reasoning") or "")
    print(
        json.dumps(
            {
                "content": msg.get("content"),
                "finish_reason": data["choices"][0].get("finish_reason"),
            },
            indent=2,
        )
    )
    if "is not a multimodal model" in text:
        print("result=fail reason=not_multimodal", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
