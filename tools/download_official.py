#!/usr/bin/env python3
"""Resume-safe download of the official DeepSeek-V4.1-Flash snapshot.

The Engram shards are ~95 GiB each. Hugging Face xet must be enabled
(do not set HF_HUB_DISABLE_XET). Requires the ``hf_xet`` package.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.pop("HF_HUB_DISABLE_XET", None)
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

REPO = "deepseek-ai/DeepSeek-V4.1-Flash"
REV = "dba1be0a40aa45a94ad051997016db3960a90277"


def main() -> int:
    from huggingface_hub import snapshot_download

    delay = 15
    while True:
        try:
            path = snapshot_download(REPO, revision=REV)
            print(f"DONE {path}", flush=True)
            return 0
        except Exception as exc:
            print(f"RETRY {type(exc).__name__}: {exc}", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 300)


if __name__ == "__main__":
    sys.exit(main())
