#!/usr/bin/env python3
"""Live OpenAI-compat vision smoke against a running serve.

Sends a 64x64 solid-red PNG. Fails on HTTP 400 "is not a multimodal model"
and when the answer does not contain "red". The PNG is built with stdlib
zlib/struct, so the host does not need an image library.
"""
from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
import urllib.error
import urllib.request
import zlib


def solid_png(width: int = 64, height: int = 64, rgb: tuple = (255, 0, 0)) -> bytes:
    """Solid-colour RGB PNG built with stdlib zlib/struct (no image library)."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    raw = (b"\x00" + bytes(rgb) * width) * height
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))


# 64x64 solid red. The old fixture was a mislabeled 1x1 gray pixel.
RED_PNG_B64 = base64.b64encode(solid_png()).decode()


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
                        "image_url": {"url": f"data:image/png;base64,{RED_PNG_B64}"},
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
    if "red" not in (msg.get("content") or "").lower():
        print("result=fail reason=wrong_color", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
