"""Stream a torch profiler trace (gz) and dump NCCL kernels with the previous
kernel on the same stream. Line-based parse (no json.load: R11 OOM)."""
import gzip, json, resource, sys
resource.setrlimit(resource.RLIMIT_AS, (6 << 30, 6 << 30))
f = gzip.open(sys.argv[1], "rt")
cur = {}; ks = []
def val(s):
    return s.split(":", 1)[1].strip().rstrip(",")
for line in f:
    s = line.strip()
    if s.startswith('"cat"'): cur["cat"] = val(s).strip('"')
    elif s.startswith('"name"'): cur["name"] = val(s).strip('"')
    elif s.startswith('"ts"'): cur["ts"] = float(val(s))
    elif s.startswith('"dur"'): cur["dur"] = float(val(s))
    elif s.startswith('"stream"'): cur["stream"] = val(s)
    elif s in ("},", "}") and "cat" in cur:
        if cur.get("cat") == "kernel" and "ts" in cur:
            ks.append((cur["ts"], cur.get("dur", 0.0), cur.get("name", "")[:80], cur.get("stream", "")))
        cur = {}
ks.sort()
last = {}; out = []; mx = 0.0
for ts, dur, name, stream in ks:
    idle = ts - mx
    if "nccl" in name.lower():
        out.append([round(ts, 3), dur, name[:48], (last.get(stream) or "")[:60], round(idle, 3)])
    last[stream] = name
    mx = max(mx, ts + dur)
json.dump({"n_kernels": len(ks), "t0": ks[0][0] if ks else None, "t1": ks[-1][0] if ks else None, "nccl": out}, open(sys.argv[2], "w"))
print("kernels", len(ks), "nccl", len(out))
