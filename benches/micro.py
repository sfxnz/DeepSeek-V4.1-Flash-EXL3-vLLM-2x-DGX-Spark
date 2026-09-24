#!/usr/bin/env python3
"""Isolated prefill (pp) and decode (tg) microbench at fixed context lengths.

For each context length L in CONTEXTS:
  pp_warm@L : one fresh ~L-token repo-text prompt, max_tokens=1, stream.
          Prefill tok/s = prompt_tokens / TTFT. (TTFT includes one verify
          step; constant across configs, so comparisons are clean.) Repo text
          reuses the same n-grams every run, so Engram rows are page-cache
          warm. Historical "pp" rows are this cell.
  pp_novel@L: same, on seeded pseudo-word text (corpus novel=True): fresh
          n-grams every run, so this is the cold-Engram prefill path.
  tg32@L: fresh ~L-token prompt, max_tokens=33, stream, ignore_eos. Decode
          tok/s = (completion_tokens-1) / wall-after-first-token.

Docs are regenerated per run with a fresh seed so prefix caching never
serves a repeat — this measures cold prefill, the number a long-context
user pays on the first turn.

DSpark acceptance is read from /metrics deltas when speculative decoding
is enabled.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from bench_decode import acceptance, decode_rate, spec_counters  # noqa: E402
from corpus import build_doc, char_ratio, make_tokenizer, trim_to_tokens  # noqa: E402

CHAT_KWARGS = {"thinking": False, "reasoning_effort": "low"}
# Above this pp tok/s the run was served from the prefix cache, not the
# kernels (measured cold prefill tops out far lower). Retry with a fresh doc.
PP_CACHE_LIMIT = 3000


def stream_once(url, model, prompt, max_tokens, timeout):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
        "chat_template_kwargs": CHAT_KWARGS,
    }).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    t0 = time.perf_counter()
    first = None
    usage = {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if ev.get("usage"):
                usage = ev["usage"]
            choices = ev.get("choices") or []
            if choices and (choices[0].get("delta") or {}).get("content"):
                if first is None:
                    first = time.perf_counter()
    t1 = time.perf_counter()
    if first is None:
        raise RuntimeError("no content token in stream")
    return {
        "ttft_s": first - t0,
        "wall_s": t1 - t0,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "decode_s": t1 - first,
    }


def fresh_doc(target, ratio, tok, seed, novel=False):
    doc = build_doc(target, ratio, seed=seed, novel=novel)
    doc = trim_to_tokens(doc, target, tok)
    return doc, tok(doc)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    p.add_argument("--contexts", type=int, nargs="+",
                   default=[512, 4096, 16384, 65536])
    p.add_argument("--tg-tokens", type=int, default=32)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--timeout", type=int, default=900)
    args = p.parse_args()

    tok = make_tokenizer(args.url.rsplit("/v1", 1)[0], args.model)
    ratios = {False: char_ratio(tok), True: char_ratio(tok, novel=True)}
    metrics_url = args.url.split("/v1/", 1)[0] + "/metrics"
    rows = []
    # Fresh docs every invocation: the prefix cache makes any repeated
    # prompt a near-free prefill, which would poison pp numbers.
    base_seed = (time.time_ns() // 1000) % (1 << 30)
    seed = base_seed
    for ctx in args.contexts:
        for phase in ("pp_warm", "pp_novel", "tg"):
            novel = phase == "pp_novel"
            per = []
            for r in range(args.runs):
                seed += 1
                doc, ntok = fresh_doc(ctx, ratios[novel], tok, seed, novel)
                before = spec_counters(metrics_url)
                if phase != "tg":
                    res = stream_once(args.url, args.model, doc, 1, args.timeout)
                    rate = res["prompt_tokens"] / res["ttft_s"]
                    for attempt in range(3):
                        if rate <= PP_CACHE_LIMIT or res["prompt_tokens"] < ntok - 64:
                            break
                        print(f"  cache-hit suspected (rate={rate:.0f}); "
                              f"fresh doc, attempt {attempt+2}", flush=True)
                        seed += 1
                        doc, ntok = fresh_doc(ctx, ratios[novel], tok, seed,
                                              novel)
                        res = stream_once(args.url, args.model, doc, 1,
                                          args.timeout)
                        rate = res["prompt_tokens"] / res["ttft_s"]
                else:
                    res = stream_once(args.url, args.model, doc,
                                      args.tg_tokens + 1, args.timeout)
                    rate = decode_rate(res["completion_tokens"], res["decode_s"])
                acc = acceptance(before, spec_counters(metrics_url))
                per.append({**res, "doc_tokens": ntok, "rate": rate, **acc})
                print(f"ctx={ctx} {phase} run={r+1} doc={ntok} "
                      f"ttft={res['ttft_s']:.2f}s rate={rate:.1f} tok/s "
                      f"acc={acc.get('acceptance_len', '-')}", flush=True)
            rows.append({
                "ctx": ctx, "phase": phase,
                "median_rate_tok_s": round(statistics.median(
                    x["rate"] for x in per), 1),
                "median_ttft_s": round(statistics.median(
                    x["ttft_s"] for x in per), 2),
                "doc_tokens": per[0]["doc_tokens"],
                "acceptance_len": round(statistics.median(
                    [x.get("acceptance_len", 0) or 0 for x in per]), 2),
                "runs": len(per),
            })
    print("SUMMARY", json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
