"""Per-phase kernel budget (ms/step): which roles/categories fill each phase."""
import pickle, sys, collections, statistics as st, bisect
sys.path.insert(0, '/home/sfxnz/projects/ai-lab/recipes/.worktrees/kernels-r3/tools')
import decode_timeline as dt
d = pickle.load(open(sys.argv[1], 'rb'))
names = d['names']; dev = sorted(d['dev'])
cat = [dt.classify(names[k[2]]) for k in dev]
rt = {c: names[n] for ts, dur, n, c, tid in d['rt'] if c is not None}
by = collections.defaultdict(list)
for i, k in enumerate(dev):
    if rt.get(k[6]) == 'cudaGraphLaunch': by[k[6]].append(i)
ar = {c: sum(cat[i] == 'nccl_allreduce' for i in idx) for c, idx in by.items()}
mx = max(ar.values())
kind = {c: 'target' if ar[c] >= mx - 1 else ('draft' if ar[c] else 'other') for c in by}
span = {c: (min(dev[i][0] for i in idx), max(dev[i][0] + dev[i][1] for i in idx)) for c, idx in by.items()}
role = {}
for c, idx in by.items():
    seq = [(i, cat[i], dev[i][3], dev[i][1], names[dev[i][2]]) for i in sorted(idx, key=lambda i: dev[i][0])]
    role.update(dt.label_roles(seq, kind[c]))
tg = sorted((span[c][0], c) for c in by if kind[c] == 'target')
steps = [(tg[j][1], tg[j][0], tg[j + 1][0]) for j in range(len(tg) - 1)]
med = st.median(e - s for _, s, e in steps); steps = [x for x in steps if x[2] - x[1] <= 2 * med]
glaunch = {i: c for c, idx in by.items() for i in idx}
eager_seq = [(i, cat[i], dev[i][3], dev[i][1], names[dev[i][2]]) for i in range(len(dev)) if i not in glaunch]
role.update(dt.label_roles(eager_seq, 'eager'))
acc = collections.defaultdict(collections.Counter)
starts = [s for _, s, _ in steps]
for i, k in enumerate(dev):
    j = bisect.bisect_right(starts, k[0]) - 1
    if j < 0 or k[0] >= steps[j][2]: continue
    c0 = steps[j][0]
    ph = kind.get(glaunch.get(i), 'eager')
    if ph == 'eager':
        ph = 'eager_after_target' if k[0] < min((span[c][0] for c in by if kind[c] == 'draft' and span[c][0] > span[c0][0]), default=1e30) else 'eager_before_target'
    acc[ph][role.get(i) or cat[i]] += k[1]
n = len(steps)
for ph, cnt in acc.items():
    tot = sum(cnt.values()) / n / 1e3
    print(f"== {ph}: kernel sum {tot:.3f} ms/step")
    for r, v in cnt.most_common(14):
        print(f"   {r:<28} {v / n / 1e3:7.3f}")
