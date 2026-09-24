import json, urllib.request, time, random, sys, glob, os
U="http://127.0.0.1:8000/v1/chat/completions"
def pp(text):
    b={"model":"deepseek-ai/DeepSeek-V4.1-Flash","messages":[{"role":"user","content":text+"\n\nSummarize in one word."}],"max_tokens":1,"temperature":0,"stream":True,"stream_options":{"include_usage":True},
       "chat_template_kwargs":{"thinking":False,"reasoning_effort":"low"}}
    r=urllib.request.Request(U,data=json.dumps(b).encode(),headers={"Content-Type":"application/json"})
    t0=time.perf_counter(); first=None; usage={}
    with urllib.request.urlopen(r,timeout=900) as resp:
        for raw in resp:
            l=raw.decode().strip()
            if not l.startswith("data:") or l.endswith("[DONE]"): continue
            ev=json.loads(l[5:])
            if ev.get("usage"): usage=ev["usage"]
            ch=ev.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content") and first is None: first=time.perf_counter()
    first=first or time.perf_counter()
    return usage.get("prompt_tokens"), round(first-t0,2), round(usage.get("prompt_tokens",0)/(first-t0),1)
# novel natural text: python stdlib docs/source from a random seed file set (not in the repo corpus)
files=sorted(glob.glob("/usr/lib/python3.12/**/*.py",recursive=True))
rnd=random.Random(int(sys.argv[1]))
rnd.shuffle(files)
chunks=[];n=0
for f in files:
    try: t=open(f,errors="replace").read()
    except: continue
    chunks.append(t[:4000]); n+=len(chunks[-1])
    if n>int(sys.argv[2])*3.2: break
print("stdlib-src", pp("\n\n".join(chunks)))
