#!/usr/bin/env python3
"""Deterministic long-context filler shared by tests/correctness and benches.

Filler is built from this repo's own text (README, scripts, patch headers,
python) so token distribution and expert routing look like the real workload
— a coding agent pasting docs and code — instead of one synthetic sentence
bank. Repetition across a long doc is broken by seeded reshuffles per cycle.

Everything is reproducible: same seed and file state, same document.
"""

from __future__ import annotations

import json
import random
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Fallback bank (used only if the repo files are missing, e.g. remote nodes
# with just the patch dir). Same style as before.
BANK = [
    "The scheduler batches decode steps while prefill chunks stream through the same engine core.",
    "Unified memory lets the KV pool spill into host RAM, but bandwidth drops once pages leave the GPU side.",
    "A codebook lookup is cheap next to the dequant epilogue that follows it on every routed expert.",
    "Tensor parallel two means every all-reduce crosses the fabric, so message size dominates step time.",
    "Chunked prefill keeps time-to-first-token bounded even when a request carries a whole repository in context.",
    "The draft model proposes five tokens and the target verifies them in one fused forward pass.",
    "Acceptance length falls off on rare tokens because the draft never saw that distribution in training.",
    "fp8 KV halves the cache footprint but the indexer still keeps its own quantized page format.",
]

_SOURCE_FILES = [
    "README.md", "AGENTS.md", "model-card.md", "flags.md", "run.sh",
    "stop.sh", "serve.sh", "smoke_chat.py", "smoke_vision.py",
    "bench_decode.py", "tools/measure_lail_prose.py",
    "tools/repro_lail_prefill.py", "tools/corpus.py", "tools/mxfp4.py",
    "docker/patch/sitecustomize.py", "kit/render.py",
]

_SEGMENT_RE = re.compile(r"\n\s*\n")

_segments_cache: list[str] | None = None


def _repo_segments() -> list[str]:
    global _segments_cache
    if _segments_cache is not None:
        return _segments_cache
    segs: list[str] = []
    for rel in _SOURCE_FILES:
        path = ROOT / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for chunk in _SEGMENT_RE.split(text):
            chunk = chunk.strip()
            # Keep paragraph-sized segments; long code blocks are split on
            # blank lines too, so segments stay 20-200 tokens.
            if len(chunk) >= 40:
                segs.append(chunk)
    _segments_cache = segs or list(BANK)
    return _segments_cache


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def char_ratio(tokenize) -> float:
    """Calibrate tokens-per-char for the segment pool with one call."""
    sample = "\n\n".join(_repo_segments()[:40])
    return tokenize(sample) / max(1, len(sample))


def build_doc(target_tokens: int, ratio: float, seed: int = 7) -> str:
    """Assemble repo-text segments to ~target_tokens.

    Deterministic in (seed, target, ratio, repo state). Each cycle reshuffles
    with a derived seed so long docs are not one repeated block.
    """
    segs = _repo_segments()
    rng = _rng(seed)
    parts: list[str] = []
    est = 0
    target_chars = target_tokens / ratio
    cycle = 0
    while est < target_chars * 1.05:
        order = list(range(len(segs)))
        random.Random(seed * 1000 + cycle).shuffle(order)
        for i in order:
            parts.append(segs[i])
            est += len(segs[i]) + 2
            if est >= target_chars * 1.05:
                break
        cycle += 1
        if cycle > 1:
            parts.append(f"[End of document bundle {cycle}.]")
    return "\n\n".join(parts[: len(parts)])


def trim_to_tokens(text: str, target_tokens: int, tokenize,
                   ratio: float | None = None,
                   more_segs: list[str] | None = None) -> str:
    """Land the doc in [0.97*target, target] tokens (server-side count).

    Trim with ratio-guided bulk drops; if the build undershot, append
    segments from more_segs before re-measuring.
    """
    paras = [p for p in text.split("\n\n") if p]
    avg_chars = max(1.0, sum(len(p) for p in paras) / max(1, len(paras)))
    cpt = 3.0 if ratio is None else 1.0 / ratio
    take = 0
    while True:
        exact = tokenize("\n\n".join(paras))
        if exact > target_tokens:
            over = exact - target_tokens
            drop = max(1, min(len(paras), int(over * cpt / avg_chars) + 1))
            del paras[-drop:]
            continue
        if exact >= 0.97 * target_tokens or not more_segs:
            return "\n\n".join(paras)
        need = target_tokens - exact
        add = max(1, int(need * cpt / avg_chars) + 1)
        for seg in more_segs[take:take + add]:
            paras.append(seg)
        take += add
        if take >= len(more_segs):
            return "\n\n".join(paras)


def make_tokenizer(url: str = "http://127.0.0.1:8000", model: str = "deepseek-ai/DeepSeek-V4.1-Flash"):
    """Return f(text) -> token count using the live /tokenize endpoint."""

    def count(text: str) -> int:
        body = json.dumps({"model": model, "prompt": text}).encode()
        req = urllib.request.Request(
            f"{url}/tokenize",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return int(json.loads(resp.read().decode())["count"])

    return count


_NEEDLE_TPL = (
    "Maintenance note {n}: the archive passcode is {code}. "
    "Keep it out of the changelog."
)


def needle_prompt(
    total_tokens: int,
    code: str,
    tokenize,
    seed: int = 7,
    pos_frac: float = 0.5,
    ratio: float | None = None,
) -> tuple[str, int]:
    """Document of ~total_tokens with a passcode needle at pos_frac depth.

    Returns (document, actual_token_count). The needle is a maintenance note
    so it blends into the docs instead of standing out as an injected marker.
    """
    if ratio is None:
        ratio = char_ratio(tokenize)
    body = build_doc(total_tokens, ratio, seed=seed)
    body = trim_to_tokens(body, total_tokens - 48, tokenize)
    paras = body.split("\n\n")
    idx = max(1, min(len(paras) - 1, int(len(paras) * pos_frac)))
    n = idx + 3
    note = _NEEDLE_TPL.format(n=n, code=code)
    paras.insert(idx, note)
    doc = "\n\n".join(paras)
    return doc, tokenize(doc)
