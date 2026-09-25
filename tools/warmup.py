#!/usr/bin/env python3
"""Post-ready warmup: move first-request JIT off the first user's TTFT.

run.sh sends these chats after /v1/models is up (WARMUP=1):
  1. greedy 17*19: prefill metadata, DSpark propose/verify, rejection kernels
  2. temperature 0.7: the sampling kernels (_gumbel_sample, _resample)
  3. a ~3k-token novel prefill: chunked-prefill metadata, indexer, MXFP8 quantize
  4. ~300- and ~1k-token prefills: the mhc_pre_big_fuse_with_norm_tilelang
     n_splits buckets that the boot's dummy runs and graph captures miss
  5. one small image (skipped with --no-vision): the vision path

Why 4: run.sh boots with enable_jit_warmup=false, so the TileLang fused mHC
prenorm kernel compiles lazily, once per n_splits. On GB10 (48 SMs) the stock
split is 16 up to 576 batch tokens, 4 for 577-1536 and 1 above. Graph captures
(<= 64 tokens) and the profile run (MNBT tokens) cover only the ends, and
DSV41_MHC_DECODE_SPLITS=N moves <= 64 tokens off 16 as well. Without 4 the
first 65-1536-token batch after ready (a vision prompt, two batched tool
prompts) paid two ~6 s compiles per rank (Round 34 s5/s6/s7).

Every prompt starts with a random nonce, and each prefill uses its own doc
seed, so no warmup request hits the prefix cache or one a user could hit.
This is a first-request TTFT fix only: it does not change steady-state decode.
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
from smoke_vision import RED_PNG_B64  # noqa: E402

PREFILL_TOKENS = 3000
# One prefill inside each mHC n_splits bucket the boot misses (16: 65-576
# tokens, 4: 577-1536). Docs are cut to target / TOKENS_PER_CHAR chars.
BUCKET_PREFILL_TOKENS = (300, 1000)
TOKENS_PER_CHAR = 0.27  # repo text on the V4.1 tokenizer, roughly


def _prefill(nonce: str, doc: str) -> str:
    return f"[{nonce}]\n{doc}\n\nSummarize the text above in one sentence."


def requests(nonce: str, vision: bool = True) -> list[tuple[str, str | list, float, int]]:
    """(label, content, temperature, max_tokens) for the warmup chats."""
    seed = int(nonce, 16)
    doc = build_doc(PREFILL_TOKENS, TOKENS_PER_CHAR, seed=seed)
    reqs: list[tuple[str, str | list, float, int]] = [
        ("greedy", f"[{nonce}] What is 17*19? Return only the integer.", 0.0, 16),
        ("t=0.7", f"[{nonce}] Name one prime number above 100.", 0.7, 16),
        ("prefill", _prefill(nonce, doc), 0.0, 8),
    ]
    for i, tokens in enumerate(BUCKET_PREFILL_TOKENS, start=1):
        chars = int(tokens / TOKENS_PER_CHAR)
        bucket_doc = build_doc(tokens, TOKENS_PER_CHAR, seed=seed + i)[:chars]
        reqs.append((f"prefill-{tokens}", _prefill(nonce, bucket_doc), 0.0, 8))
    if vision:
        reqs.append(("vision", [
            {"type": "text", "text": f"[{nonce}] What color is this image? Reply with one word only."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{RED_PNG_B64}"}},
        ], 0.0, 8))
    return reqs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    ap.add_argument("--no-vision", action="store_true", help="LANGUAGE_MODEL_ONLY=1 serve")
    args = ap.parse_args()
    nonce = secrets.token_hex(8)
    for label, prompt, temperature, max_tokens in requests(nonce, vision=not args.no_vision):
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
