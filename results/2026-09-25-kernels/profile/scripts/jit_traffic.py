"""Post-ready JIT probe traffic: one ~3k-token novel prompt, then 2 concurrent requests."""
import json, secrets, sys, threading, time, urllib.request
sys.path.insert(0, "/home/sfxnz/projects/ai-lab/recipes/.worktrees/kernels-r3/tools")
from corpus import build_doc
URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
def chat(content, max_tokens, tag, out):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0.0, "chat_template_kwargs": {"thinking": False}}
    t0 = time.time()
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    out[tag] = {"s": round(time.time() - t0, 2), "usage": d["usage"], "text": d["choices"][0]["message"]["content"][:80]}
out = {}
n = secrets.token_hex(6)
doc = build_doc(3000, 0.27, seed=int(n, 16))
chat(f"[{n}]\n{doc}\n\nSummarize the text above in two sentences.", 64, "prompt_3k", out)
ths = []
for i in range(2):
    m = secrets.token_hex(6)
    d2 = build_doc(1000, 0.27, seed=int(m, 16))
    ths.append(threading.Thread(target=chat, args=(f"[{m}]\n{d2}\n\nList three key points of the text above.", 128, f"concurrent_{i}", out)))
for t in ths: t.start()
for t in ths: t.join()
print(json.dumps(out, indent=1))
