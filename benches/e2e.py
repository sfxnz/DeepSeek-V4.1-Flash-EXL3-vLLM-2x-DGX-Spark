#!/usr/bin/env python3
"""End-to-end bench with realistic prompts — the workload this serve is for.

Phases (all streamed, thinking off, effort low — the shipped defaults):
  coding_agent : ~2k-token agent scaffold + real shell/python code in
                 context, asks for one new guard in run.sh. Graded: must
                 return a bash if-condition and a FORCE_UNSAFE_ mention.
  doc_recall   : 64k-token technical doc with a passcode needle, asks for
                 the passcode, then continues summarizing. Graded on the
                 needle; measures cold long prefill TTFT and decode over a
                 64k KV.
  tool_json    : weather tool call, graded on parsed arguments JSON.

Reports ttft, decode tok/s, and useful tok/s (completion tokens over wall
for phases that pass their quality check). A config that fails a quality
check scores zero useful tokens for that phase — fast garbage is not a win.
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

from bench_decode import decode_rate  # noqa: E402
from corpus import make_tokenizer, needle_prompt  # noqa: E402
from tests.correctness import TOOLS  # noqa: E402

CHAT_KWARGS = {"thinking": False, "reasoning_effort": "low"}

SCAFFOLD = """You are the infrastructure agent for a two-node DGX Spark
serving recipe. You edit the recipe repo directly. Rules you must follow:
- recipe.yaml is the source of truth for generated blocks; regenerate with
  python3 kit/render.py, never hand-edit generated regions.
- Read unified memory with free -h; never read VRAM from nvidia-smi.
- GPUs are exclusive: refuse to start a serve while another --gpus all
  container is up; tell the user to stop it instead.
- Every refuse-guard in run.sh must exit 1 with a FORCE_UNSAFE_ override
  env var, and every new flag needs a measured effect in flags.md before
  it can land on main.
- Prefer one change per experiment loop: name the bottleneck, patch, run
  the correctness suite, then the micro bench that should move, then e2e.
- Do not raise MAX_NUM_SEQS above the occupancy row you measured.
- Do not vendor DeepJIT into the image; it is a kernel JIT library, not a
  serving stack, and the official V4.1 kernels are TileLang.
- The worker node is spark2; SSH must work or the run refuses to start.
- Document rejected experiments the same as accepted ones.

The user will paste a section of run.sh and ask for one new guard or one
small change. Answer with a diff-style patch in a bash block, then one
short paragraph explaining the failure mode the guard protects against.
"""

CODE_CTX = """Existing refuse-guards in run.sh:

```bash
if [[ "$MAX_MODEL_LEN" -gt 1048576 && "$FORCE_UNSAFE_CTX" != 1 ]]; then
  echo "fp8 KV pin cannot hold --max-model-len $MAX_MODEL_LEN. Native window is 1048576. FORCE_UNSAFE_CTX=1 overrides." >&2
  exit 1
fi
if [[ "$MAX_NUM_SEQS" -gt 2 && "$FORCE_UNSAFE_CTX" != 1 ]]; then
  echo "MAX_NUM_SEQS=$MAX_NUM_SEQS exceeds 2 on this pin. FORCE_UNSAFE_CTX=1 overrides." >&2
  exit 1
fi
if [[ "$KV_CACHE_MEMORY" -gt 8589934592 && "$FORCE_UNSAFE_CTX" != 1 ]]; then
  echo "KV_CACHE_MEMORY=$KV_CACHE_MEMORY exceeds 8 GiB pin 8589934592. CSA2 is 890 B/token; 4 GiB holds 1M x 2. FORCE_UNSAFE_CTX=1 overrides." >&2
  exit 1
fi
if [[ "$SPEC" == dspark && $((NUM_SPECULATIVE_TOKENS % 5)) -ne 0 && "$FORCE_UNSAFE_CTX" != 1 ]]; then
  echo "NUM_SPECULATIVE_TOKENS=$NUM_SPECULATIVE_TOKENS is not divisible by 5 (DSpark block size). FORCE_UNSAFE_CTX=1 overrides." >&2
  exit 1
fi
```

Task: add a refuse-guard for a new env var UTIL: refuse when UTIL is above
0.90 unless FORCE_UNSAFE_UTIL=1, because Engram staging needs headroom in
unified memory. Match the style of the guards above.
"""


def post_chat(url, model, messages, *, max_tokens, tools=None,
              stream=True, timeout=900):
    payload = {
        "model": model, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0, "stream": stream, "ignore_eos": True,
        "chat_template_kwargs": CHAT_KWARGS,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    t0 = time.perf_counter()
    first = None
    usage = {}
    content_parts = []
    tool_calls = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload_s = line[5:].strip()
            if payload_s == "[DONE]":
                break
            try:
                ev = json.loads(payload_s)
            except json.JSONDecodeError:
                continue
            if ev.get("usage"):
                usage = ev["usage"]
            choices = ev.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if delta.get("content"):
                if first is None:
                    first = time.perf_counter()
                content_parts.append(delta["content"])
            for tc in delta.get("tool_calls") or []:
                if tc.get("index") is not None:
                    while len(tool_calls) <= tc["index"]:
                        tool_calls.append({"id": "", "function":
                                           {"name": "", "arguments": ""}})
                    slot = tool_calls[tc["index"]]
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["function"]["arguments"] += fn["arguments"]
    t1 = time.perf_counter()
    if first is None:
        first = t1  # tool-only response: no content token
    return {
        "ttft_s": first - t0,
        "wall_s": t1 - t0,
        "decode_s": t1 - first,
        "content": "".join(content_parts),
        "tool_calls": tool_calls,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
    }


def grade_coding(res) -> tuple[bool, str]:
    text = res["content"]
    has_cond = ("UTIL" in text and ("-gt" in text or ">" in text)
                and "0.90" in text)
    has_flag = "FORCE_UNSAFE_UTIL" in text
    has_bash = "```" in text
    ok = has_cond and has_flag and has_bash
    return ok, (f"cond={has_cond} flag={has_flag} block={has_bash} "
                f"words={len(text.split())}")


def grade_recall(res, code) -> tuple[bool, str]:
    ok = code in res["content"].upper()
    return ok, f"needle={'hit' if ok else 'miss'}"


def grade_tool(res) -> tuple[bool, str]:
    if not res["tool_calls"]:
        return False, "no tool_calls"
    fn = res["tool_calls"][0]["function"]
    try:
        args = json.loads(fn["arguments"])
    except json.JSONDecodeError:
        return False, f"unparseable args {fn['arguments']!r}"
    ok = fn["name"] == "get_weather" and "paris" in str(
        args.get("city", "")).lower()
    return ok, f"{fn['name']}({fn['arguments']})"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    p.add_argument("--recall-tokens", type=int, default=65536)
    p.add_argument("--runs", type=int, default=1)
    args = p.parse_args()
    tok = make_tokenizer(args.url.rsplit("/v1", 1)[0], args.model)
    # Fresh docs per invocation: repeated prompts hit the prefix cache and
    # turn the 64k cold-prefill measurement into a near-free lookup.
    seed_base = (time.time_ns() // 1000) % (1 << 30)

    phases = []
    for r in range(args.runs):
        seed = seed_base + r
        # coding_agent
        res = post_chat(args.url, args.model, [
            {"role": "system", "content": SCAFFOLD},
            {"role": "user", "content": CODE_CTX},
        ], max_tokens=448)
        ok, why = grade_coding(res)
        phases.append({"phase": "coding_agent", "run": r + 1, "pass": ok,
                       "detail": why,
                       "ttft_s": round(res["ttft_s"], 2),
                       "decode_s": round(res["decode_s"], 2),
                       "decode_tok_s": round(decode_rate(
                           res["completion_tokens"], res["decode_s"]), 2),
                       "prompt_tokens": res["prompt_tokens"],
                       "completion_tokens": res["completion_tokens"]})
        print(f"coding_agent run={r+1} pass={ok} {why} "
              f"ttft={res['ttft_s']:.2f}s decode="
              f"{phases[-1]['decode_tok_s']}", flush=True)

        # doc_recall (fresh doc per run: cold prefill is the real cost)
        code = f"VERDIGRIS-{4200+r}"
        doc, ntok = needle_prompt(args.recall_tokens, code, tok,
                                  seed=seed, pos_frac=0.5)
        res = post_chat(args.url, args.model, [
            {"role": "user",
             "content": doc + "\n\nFind the archive passcode in the "
                             "maintenance note, then write two sentences on "
                             "why chunked prefill keeps time-to-first-token "
                             "bounded."},
        ], max_tokens=192)
        ok, why = grade_recall(res, code)
        phases.append({"phase": "doc_recall", "run": r + 1, "pass": ok,
                       "detail": why,
                       "ttft_s": round(res["ttft_s"], 2),
                       "decode_s": round(res["decode_s"], 2),
                       "decode_tok_s": round(decode_rate(
                           res["completion_tokens"], res["decode_s"]), 2),
                       "prompt_tokens": res["prompt_tokens"],
                       "completion_tokens": res["completion_tokens"]})
        print(f"doc_recall run={r+1} doc={ntok} pass={ok} {why} "
              f"ttft={res['ttft_s']:.2f}s "
              f"decode={phases[-1]['decode_tok_s']}", flush=True)

        # tool_json
        res = post_chat(args.url, args.model, [
            {"role": "user",
             "content": "What is the weather in Paris right now? "
                        "Call the tool."},
        ], max_tokens=128, tools=TOOLS)
        ok, why = grade_tool(res)
        phases.append({"phase": "tool_json", "run": r + 1, "pass": ok,
                       "detail": why,
                       "ttft_s": round(res["ttft_s"], 2),
                       "decode_s": round(res["decode_s"], 2),
                       "decode_tok_s": round(decode_rate(
                           res["completion_tokens"], res["decode_s"]), 2),
                       "prompt_tokens": res["prompt_tokens"],
                       "completion_tokens": res["completion_tokens"]})
        print(f"tool_json run={r+1} pass={ok} {why}", flush=True)

    useful = [ph for ph in phases if ph["pass"]]
    total_completion = sum(ph["completion_tokens"] for ph in useful)
    total_wall = sum(ph["ttft_s"] + ph["decode_s"] for ph in useful
                     if "decode_s" in ph)
    summary = {
        "phases": phases,
        "useful_tok_s": round(total_completion / total_wall, 2)
        if total_wall else 0.0,
        "pass_rate": f"{len(useful)}/{len(phases)}",
    }
    print("SUMMARY", json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
