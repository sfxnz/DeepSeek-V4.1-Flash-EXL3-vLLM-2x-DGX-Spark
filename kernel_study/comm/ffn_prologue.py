"""FFN prologue after the attention all-reduce: routing chain vs shared-expert chain, per layer.

Input: a decode_timeline extract pickle (r3 c=1 traces). Usage: python3 ffn_prologue.py c1-rank0.pkl
"""
import pickle, sys, statistics as st, collections, re
d = pickle.load(open(sys.argv[1], "rb"))
names = d["names"]; dev = d["dev"]; rt = d["rt"]
launch = {r[3] for r in rt if names[r[2]] == "cudaGraphLaunch"}
# target graph kernels by correlation
by_corr = collections.defaultdict(list)
for e in dev:
    if e[6] in launch:
        by_corr[e[6]].append(e)
rows = []
for corr, evs in by_corr.items():
    evs.sort()
    if sum(1 for e in evs if names[e[2]].startswith("ncclDevKernel_AllReduce")) < 60:
        continue  # target graph only
    # walk: attn AR -> ... -> p2b
    idx_ar = [i for i, e in enumerate(evs) if names[e[2]].startswith("ncclDevKernel_AllReduce")]
    idx_p2b = [i for i, e in enumerate(evs) if "p2b_moe" in names[e[2]]]
    for pi in idx_p2b:
        # the attn AR right before this p2b
        ars = [i for i in idx_ar if i < pi]
        if not ars:
            continue
        ai = ars[-1]
        ar_end = evs[ai][0] + evs[ai][1]
        seg = evs[ai + 1:pi]
        p2b_start = evs[pi][0]
        gu = [e for e in seg if "dense_blockscaled" in names[e[2]] and tuple(e[3]) == (1, 1, 36)]
        dn = [e for e in seg if "dense_blockscaled" in names[e[2]] and tuple(e[3]) == (1, 1, 48)]
        rg = [e for e in seg if ("cutlass" in names[e[2]] or "Kernel2" in names[e[2]]) and "splitK" not in names[e[2]]]
        if not gu or not dn or not rg:
            continue
        shared_stream = gu[0][5]
        route_stream = rg[0][5]
        shared_end = max(e[0] + e[1] for e in seg if e[5] == shared_stream)
        route_end = max(e[0] + e[1] for e in seg if e[5] == route_stream)
        gate_start = min(rg[0][0], gu[0][0])
        rows.append(dict(C=p2b_start - ar_end, shared_end=shared_end - gate_start, route_end=route_end - gate_start,
                         p2b_after=p2b_start - gate_start, gu=gu[0][1], dn=dn[0][1], rg=rg[0][1],
                         gate_start_after_ar=gate_start - ar_end, same=shared_stream == route_stream))
print("layers", len(rows), "same-stream", sum(r["same"] for r in rows))
for k in ("C", "gate_start_after_ar", "rg", "gu", "dn", "route_end", "shared_end", "p2b_after"):
    xs = [r[k] for r in rows]
    print(f"{k:22s} med {st.median(xs):7.1f}  p10 {sorted(xs)[len(xs)//10]:7.1f}  p90 {sorted(xs)[9*len(xs)//10]:7.1f}")
crit = sum(1 for r in rows if r["shared_end"] >= r["route_end"]) / len(rows)
print(f"shared chain ends last in {crit:.3f} of layers; median slack route-before-shared {st.median(r['shared_end']-r['route_end'] for r in rows):.1f} us")
