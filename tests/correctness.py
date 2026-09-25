#!/usr/bin/env python3
"""Fixed correctness eval for the DeepSeek-V4.1-Flash EXL3 serve.

Deterministic: temperature 0, thinking off, reasoning_effort low, fixed seed.
Covers math, strict JSON, tool-call arguments, code tracing, long-context
recall at 8k (and 64k with --full), prose anti-collapse, and vision.

Exit code 0 only if every check passes. Compare runs with the same suite;
never change prompts to make a kernel pass.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from corpus import make_tokenizer, needle_prompt  # noqa: E402
from smoke_vision import RED_PNG_B64, says_red  # noqa: E402

CHAT_KWARGS = {"thinking": False, "reasoning_effort": "low"}
SEED = 20260918

CODE_SNIPPET = """```
names = ["alpha", "beta", "gamma"]
short = [n[:2] for n in reversed(names)]
print(":".join(short))
```"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["c", "f"]},
                },
                "required": ["city"],
            },
        },
    }
]


def post(url: str, payload: dict, timeout: int = 600) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()[:400]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def chat(url: str, model: str, messages: list, *, max_tokens: int = 96,
          tools=None, tool_choice=None) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": SEED,
        "chat_template_kwargs": CHAT_KWARGS,
    }
    if tools:
        payload["tools"] = tools
    if tool_choice:
        payload["tool_choice"] = tool_choice
    out = post(url, payload)
    msg = out["choices"][0]["message"]
    return msg.get("content") or "", msg


def type_token_ratio(text: str) -> float:
    words = text.split()
    return len(set(words)) / len(words) if words else 0.0


def ngram_diversity(text: str, n: int = 8) -> float:
    words = text.split()
    if len(words) < 2 * n:
        return 1.0
    windows = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    return len(set(windows)) / len(windows)


def run_suite(url: str, model: str, full: bool) -> tuple[list[dict], bool]:
    results: list[dict] = []
    ok = True
    tok = make_tokenizer(url.rsplit("/v1", 1)[0], model)

    def check(name: str, fn) -> None:
        nonlocal ok
        t0 = time.time()
        try:
            detail = fn()
            passed = True
        except Exception as exc:  # noqa: BLE001
            detail, passed = f"{type(exc).__name__}: {exc}", False
        row = {"check": name, "pass": passed, "s": round(time.time() - t0, 2),
               "detail": str(detail)[:220]}
        results.append(row)
        print(f"{'PASS' if passed else 'FAIL'} {name} ({row['s']}s) {row['detail']}",
              flush=True)
        ok = ok and passed

    def math_small() -> str:
        text, _ = chat(url, model, [{"role": "user",
                      "content": "What is 17*19? Return only the integer."}])
        assert "323" in text, f"want 323 in {text!r}"
        return text.strip()[:40]

    def math_mid() -> str:
        text, _ = chat(url, model, [{"role": "user",
                      "content": "What is 21*12? Return only the integer."}])
        assert "252" in text, f"want 252 in {text!r}"
        return text.strip()[:40]

    def json_strict() -> str:
        text, _ = chat(url, model, [{"role": "user",
                      "content": "Return exactly one JSON object with keys "
                                  "\"model\" and \"params\", values "
                                  "\"deepseek\" and 552. No other text."}])
        m = re.search(r"\{.*\}", text, re.S)
        assert m, f"no JSON in {text!r}"
        obj = json.loads(m.group(0))
        assert obj.get("model") == "deepseek", obj
        assert int(obj.get("params")) == 552, obj
        return json.dumps(obj)

    def tool_call() -> str:
        _, msg = chat(
            url, model,
            [{"role": "user",
              "content": "What is the weather in Paris right now? Call the tool."}],
            tools=TOOLS, tool_choice="auto", max_tokens=128,
        )
        calls = msg.get("tool_calls") or []
        assert calls, f"no tool_calls in message keys {list(msg)}"
        fn = calls[0]["function"]
        assert fn["name"] == "get_weather", fn["name"]
        args = json.loads(fn["arguments"])
        assert "paris" in str(args.get("city", "")).lower(), args
        return f"{fn['name']}({fn['arguments']})"

    def code_trace() -> str:
        text, _ = chat(url, model, [{"role": "user",
                      "content": f"What does this Python program print?\n\n"
                                 f"{CODE_SNIPPET}\n\n"
                                 "Do not show the steps. Return only the "
                                 "final printed value."}],
                       max_tokens=256)
        assert "ga:be:al" in text, f"want ga:be:al in {text!r}"
        return text.strip()[:40]

    def recall_8k() -> str:
        doc, ntok = needle_prompt(8192, "ORCHID-7341", tok, seed=11)
        text, _ = chat(url, model, [
            {"role": "user",
             "content": doc + "\n\nWhat is the archive passcode mentioned in "
                             "the maintenance note? Return only the passcode."},
        ], max_tokens=48)
        assert "ORCHID-7341" in text.upper(), repr(text[:120])
        return f"found in {ntok}-token doc"

    def recall_64k() -> str:
        doc, ntok = needle_prompt(65536, "MAGENTA-5190", tok, seed=13, pos_frac=0.55)
        text, _ = chat(url, model, [
            {"role": "user",
             "content": doc + "\n\nWhat is the archive passcode mentioned in "
                             "the maintenance note? Return only the passcode."},
        ], max_tokens=48)
        assert "MAGENTA-5190" in text.upper(), repr(text[:120])
        return f"found in {ntok}-token doc"

    def prose_sanity() -> str:
        text, _ = chat(url, model, [{"role": "user",
                      "content": "Write one paragraph explaining why chunked "
                                 "prefill bounds time-to-first-token on long "
                                 "prompts. Plain prose, no lists."}],
                       max_tokens=160)
        ttr = type_token_ratio(text)
        div = ngram_diversity(text)
        assert ttr >= 0.30, f"type-token ratio {ttr:.2f} — degenerate loop"
        assert div >= 0.80, f"8-gram diversity {div:.2f} — repetitive collapse"
        return f"ttr={ttr:.2f} div={div:.2f} words={len(text.split())}"

    def vision() -> str:
        text, _ = chat(url, model, [{"role": "user", "content": [
            {"type": "text",
             "text": "What color is this image? Reply with one word only."},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{RED_PNG_B64}"}},
        ]}], max_tokens=16)
        assert says_red(text), f"want red in {text!r}"
        return text.strip()[:40]

    check("math_small", math_small)
    check("math_mid", math_mid)
    check("json_strict", json_strict)
    check("tool_call", tool_call)
    check("code_trace", code_trace)
    check("recall_8k", recall_8k)
    check("prose_sanity", prose_sanity)
    check("vision", vision)
    if full:
        check("recall_64k", recall_64k)
    return results, ok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    p.add_argument("--full", action="store_true",
                   help="include the 64k recall check (slow prefill)")
    args = p.parse_args()
    results, ok = run_suite(args.url, args.model, args.full)
    print("CORRECTNESS", json.dumps({"pass": sum(r["pass"] for r in results),
                                     "total": len(results)}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
