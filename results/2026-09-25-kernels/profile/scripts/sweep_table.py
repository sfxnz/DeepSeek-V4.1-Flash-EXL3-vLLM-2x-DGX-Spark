import json, sys
d = json.load(open(sys.argv[1]))
arms = d["arms"]
keep = arms.get("keep")
def cell(a, op, m, mode, f="median_us"):
    for c in arms[a]["cells"]:
        if c["op"] == op and c["rows"] == m and c["mode"] == mode:
            return c.get(f)
order = sorted(arms, key=lambda a: arms[a].get("modeled_ms_per_step", {}).get("c1", 1e9))
print(f"{'arm':<16}{'reps':>5}{'c1 ms':>8}{'d c1':>7}{'c2 ms':>8}{'d c2':>7} | gapped steady us AR m1/3/4/6/8 | AG m3/4/6/8 | startup AR4 | eager AR4")
for a in order:
    e = arms[a]; mm = e.get("modeled_ms_per_step", {})
    dk = lambda w: (mm.get(w, 0) - keep["modeled_ms_per_step"][w]) if keep and w in mm else 0
    ar = "/".join(f"{cell(a,'all_reduce',m,'gapped') or 0:.0f}" for m in (1,3,4,6,8))
    ag = "/".join(f"{cell(a,'all_gather',m,'gapped') or 0:.0f}" for m in (3,4,6,8))
    print(f"{a:<16}{e['reps']:>5}{mm.get('c1',0):>8.2f}{dk('c1'):>+7.2f}{mm.get('c2',0):>8.2f}{dk('c2'):>+7.2f} | {ar:<22} | {ag:<18} | {cell(a,'all_reduce',4,'gapped','startup_us') or 0:>6.0f} | {cell(a,'all_reduce',4,'eager') or 0:.1f}")
