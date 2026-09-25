"""Per-kernel-name total device time from a profiler trace (line-streamed)."""
import gzip, json, resource, sys, collections
resource.setrlimit(resource.RLIMIT_AS, (6 << 30, 6 << 30))
cur = {}; agg = collections.defaultdict(lambda: [0, 0.0])
def val(s): return s.split(":", 1)[1].strip().rstrip(",")
for line in gzip.open(sys.argv[1], "rt"):
    s = line.strip()
    if s.startswith('"cat"'): cur["cat"] = val(s).strip('"')
    elif s.startswith('"name"'): cur["name"] = val(s).strip('"')
    elif s.startswith('"dur"'): cur["dur"] = float(val(s))
    elif s.startswith('"grid"'): cur["grid"] = val(s)
    elif s in ("},", "}") and "cat" in cur:
        if cur.get("cat") == "kernel":
            k = cur.get("name", "")[:90] + " " + cur.get("grid", "")
            agg[k][0] += 1; agg[k][1] += cur.get("dur", 0.0)
        cur = {}
json.dump(agg, open(sys.argv[2], "w"))
