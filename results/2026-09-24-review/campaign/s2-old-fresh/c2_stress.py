#!/usr/bin/env python3
"""C2-STRESS: 6 rounds x 2 concurrent chat requests (max_tokens 256, distinct prompts)."""
import json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
PROMPTS = [
    "Explain how a B-tree keeps itself balanced during inserts.",
    "Write a short story about a lighthouse keeper who finds a map.",
    "Compare TCP congestion control algorithms Reno and CUBIC.",
    "Describe the water cycle for a ten-year-old.",
    "List the pros and cons of unified memory on GPUs.",
    "Summarize the causes of the French Revolution.",
    "How does speculative decoding speed up LLM inference?",
    "Write a haiku sequence about autumn in the mountains.",
    "Explain the difference between processes and threads in Linux.",
    "Give a recipe for a vegetarian lentil curry with steps.",
    "What is the role of the Krebs cycle in cellular respiration?",
    "Draft a polite email declining a meeting invitation.",
]
def one(p):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": p}],
                       "max_tokens": 256, "temperature": 0.7,
                       "chat_template_kwargs": {"thinking": False, "reasoning_effort": "low"}}).encode()
    t0 = time.perf_counter()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.load(r)
        return {"ok": True, "s": round(time.perf_counter() - t0, 2),
                "completion_tokens": d["usage"]["completion_tokens"],
                "finish": d["choices"][0]["finish_reason"],
                "nonempty": bool(d["choices"][0]["message"]["content"])}
    except Exception as e:
        return {"ok": False, "s": round(time.perf_counter() - t0, 2), "error": repr(e)[:200]}
for rnd in range(6):
    ps = PROMPTS[2 * rnd: 2 * rnd + 2]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(2) as ex:
        res = list(ex.map(one, ps))
    print(json.dumps({"round": rnd + 1, "wall_s": round(time.perf_counter() - t0, 2), "results": res}), flush=True)
