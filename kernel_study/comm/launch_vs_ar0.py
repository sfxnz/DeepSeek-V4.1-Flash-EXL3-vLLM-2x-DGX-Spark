"""Per graph replay: host cudaGraphLaunch window vs the GPU start and the first NCCL collective.

Input: a decode_timeline extract pickle (tools/decode_timeline.py; r3 c=1 traces live under
~/projects/data/dsv41-traces/r3-c1-20260925-083142/). Usage: python3 launch_vs_ar0.py c1-rank0.pkl
"""
import pickle, sys, statistics as st, collections
path = sys.argv[1]
d = pickle.load(open(path, "rb"))
names = d["names"]
rt = d["rt"]; dev = d["dev"]
# runtime: (ts, dur, nid, corr, tid)
launch = {r[3]: r for r in rt if names[r[2]] == "cudaGraphLaunch"}
# device: (ts, dur, nid, grid, block, stream, corr, graph_id, node_id, regs, smem, bytes, kind)
by_corr = collections.defaultdict(list)
for e in dev:
    if e[6] in launch:
        by_corr[e[6]].append(e)
rows = []
for corr, evs in by_corr.items():
    evs.sort()
    gid = evs[0][7]
    nccl = [e for e in evs if names[e[2]].startswith("ncclDevKernel")]
    if not nccl:
        continue
    L = launch[corr]
    h0, h1 = L[0], L[0] + L[1]
    g0 = evs[0][0]
    ar0 = nccl[0]
    # device idle right before ar0
    prev_end = max((e[0] + e[1] for e in evs if e[0] + e[1] <= ar0[0] + 1e-3 and e is not ar0), default=g0)
    rows.append(dict(gid=gid, n=len(evs), nccl=len(nccl), h0=h0, h1=h1, g0=g0, ar0s=ar0[0], ar0e=ar0[0] + ar0[1],
                     gap_before_ar0=ar0[0] - prev_end))
by_g = collections.defaultdict(list)
for r in rows: by_g[(r["gid"], r["nccl"])].append(r)
for (gid, nn), rs in sorted(by_g.items(), key=lambda x: -len(x[1]))[:6]:
    f = lambda k: st.median(k(r) for r in rs)
    print(f"graph {gid} nccl/replay {nn} replays {len(rs)} kernels {rs[0]['n']}")
    print(f"  host launch dur med {f(lambda r: r['h1']-r['h0']):.1f} us")
    print(f"  GPU first node - host launch start: med {f(lambda r: r['g0']-r['h0']):.1f} us  (<0: gpu started before?)")
    print(f"  GPU first node - host launch end:   med {f(lambda r: r['g0']-r['h1']):.1f} us")
    print(f"  AR0 start - GPU first node:          med {f(lambda r: r['ar0s']-r['g0']):.1f} us")
    print(f"  AR0 start - host launch end:         med {f(lambda r: r['ar0s']-r['h1']):.1f} us; frac AR0 starts after launch end: {sum(r['ar0s']>r['h1'] for r in rs)/len(rs):.2f}")
    print(f"  idle before AR0 med {f(lambda r: r['gap_before_ar0']):.1f}  AR0 dur med {f(lambda r: r['ar0e']-r['ar0s']):.1f}")
    # correlation: idle-before-AR0 vs (launch end - first node)
    xs = [(r['h1'] - r['g0'], r['gap_before_ar0']) for r in rs]
    xs.sort()
    q = len(xs)//4
    print("  quartiles of (launch_end - first_node) and mean idle-before-AR0 in each:")
    for i in range(4):
        part = xs[i*q:(i+1)*q] if i < 3 else xs[3*q:]
        print(f"    [{part[0][0]:8.1f},{part[-1][0]:8.1f}] idle {st.mean(p[1] for p in part):7.1f}")
