import pickle, sys, collections, statistics as st
for p in sys.argv[1:]:
    d = pickle.load(open(p, 'rb')); names = d['names']; dev = sorted(d['dev'])
    rt = {c: names[n] for ts, dur, n, c, tid in d['rt'] if c is not None}
    by = collections.defaultdict(list)
    for i, k in enumerate(dev):
        if rt.get(k[6]) == 'cudaGraphLaunch': by[k[6]].append(i)
    for kind, cond in (('target', lambda n: n > 1500), ('draft', lambda n: n < 500)):
        pos = collections.defaultdict(list); gapb = collections.defaultdict(list)
        for c, idx in by.items():
            if not cond(len(idx)): continue
            idx.sort(key=lambda i: dev[i][0]); j = 0; pe = None
            for i in idx:
                k = dev[i]
                if 'AllReduce' in names[k[2]]:
                    pos[j].append(k[1]); gapb[j].append(k[0] - pe if pe else 0); j += 1
                pe = max(pe or 0, k[0] + k[1])
        if not pos: continue
        steady = st.median([x for j in pos if j >= 3 for x in pos[j]])
        steady_mean = st.fmean([x for j in pos if j >= 3 for x in pos[j]])
        print(p.split('/')[-1], kind, 'AR0 dur med/mean', round(st.median(pos[0]),1), round(st.fmean(pos[0]),1), 'gap-before med/mean', round(st.median(gapb[0]),1), round(st.fmean(gapb[0]),1),
              '| AR1', round(st.median(pos[1]),1), round(st.fmean(pos[1]),1), '| steady med/mean', round(steady,1), round(steady_mean,1),
              '| startup excess mean (AR0 dur+gap - steady mean - gap steady)', round(st.fmean(pos[0]) + st.fmean(gapb[0]) - steady_mean - st.median(gapb[3]), 1))
