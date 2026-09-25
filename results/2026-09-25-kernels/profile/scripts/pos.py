import pickle, sys, collections, statistics as st
def load(p):
    d = pickle.load(open(p, 'rb')); names = d['names']; dev = sorted(d['dev'])
    rt = {c: names[n] for ts, dur, n, c, tid in d['rt'] if c is not None}
    by = collections.defaultdict(list)
    for i, k in enumerate(dev):
        if rt.get(k[6]) == 'cudaGraphLaunch': by[k[6]].append(i)
    return d, names, dev, by
for r in (0, 1):
    d, names, dev, by = load(sys.argv[1 + r])
    pos = collections.defaultdict(list); allp = []
    for c, idx in by.items():
        if len(idx) < 1500: continue
        idx.sort(key=lambda i: dev[i][0])
        j = 0
        for i in idx:
            if 'p2b_moe' in names[dev[i][2]]:
                pos[j].append(dev[i][1]); allp.append(dev[i][1]); j += 1
    q = sorted(allp)
    print(f"rank{r} p2b n={len(q)} p10 {q[len(q)//10]:.0f} p50 {q[len(q)//2]:.0f} p90 {q[9*len(q)//10]:.0f} p99 {q[99*len(q)//100]:.0f} mean {st.fmean(q):.0f}")
    print('  by layer median/mean:', ' '.join(f"{j}:{st.median(pos[j]):.0f}/{st.fmean(pos[j]):.0f}" for j in (0,1,2,3,5,10,20,30,39)))
