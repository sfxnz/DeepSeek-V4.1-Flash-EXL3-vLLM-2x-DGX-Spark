"""Per-layer critical-path segments inside the target graph (median over layers x steps)."""
import pickle, sys, collections, statistics as st
sys.path.insert(0, '/home/sfxnz/projects/ai-lab/recipes/.worktrees/kernels-r3/tools')
import decode_timeline as dt
d = pickle.load(open(sys.argv[1], 'rb'))
names = d['names']; dev = sorted(d['dev'])
rt = {c: names[n] for ts, dur, n, c, tid in d['rt'] if c is not None}
by = collections.defaultdict(list)
for i, k in enumerate(dev):
    if rt.get(k[6]) == 'cudaGraphLaunch': by[k[6]].append(i)
seg = collections.defaultdict(list); nl = []
for c, idx in by.items():
    if len(idx) < 1500: continue
    idx.sort(key=lambda i: dev[i][0])
    ks = [(dev[i][0], dev[i][0] + dev[i][1], dt.classify(names[dev[i][2]]), names[dev[i][2]]) for i in idx]
    ars = [k for k in ks if k[2] == 'nccl_allreduce']
    p2bs = [k for k in ks if k[2] == 'p2b_moe']
    pre = [k for k in ks if k[2] == 'mhc_prenorm_gemm']
    if len(ars) != 80 or len(p2bs) != 40: continue
    nl.append(ks[-1][1] - ks[0][0])
    for L in range(40):
        ar_attn, ar_moe, p2b = ars[2 * L], ars[2 * L + 1], p2bs[L]
        prev_end = ars[2 * L - 1][1] if L else ks[0][0]
        seg['A attn: prev MoE AR end -> attn AR start'].append(ar_attn[0] - prev_end)
        seg['B attn AR'].append(ar_attn[1] - ar_attn[0])
        seg['C ffn pre: attn AR end -> p2b start'].append(p2b[0] - ar_attn[1])
        seg['D p2b'].append(p2b[1] - p2b[0])
        seg['E post: p2b end -> MoE AR start'].append(ar_moe[0] - p2b[1])
        seg['F MoE AR'].append(ar_moe[1] - ar_moe[0])
tot = 0
for k in sorted(seg):
    v = seg[k]; tot += st.fmean(v) * 40
    print(f"{k:<44} median {st.median(v):7.1f} us  mean {st.fmean(v):7.1f} us  x40 = {st.fmean(v)*40/1e3:6.2f} ms/step")
print('sum of segment means x40 =', round(tot / 1e3, 2), 'ms; target graph span median', round(st.median(nl) / 1e3, 2), 'ms; replays', len(nl))
