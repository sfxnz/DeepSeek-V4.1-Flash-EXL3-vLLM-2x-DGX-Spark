"""Per-collective arrival skew between TP ranks from two aligned profiler traces.
Collectives complete together, so dur_r0 - dur_r1 = arrival_r1 - arrival_r0
(positive: rank1 arrives later and rank0 waits)."""
import json, statistics as st, sys
a = json.load(open(sys.argv[1]))["nccl"]; b = json.load(open(sys.argv[2]))["nccl"]
assert len(a) == len(b)
names_ok = all(x[2][:20] == y[2][:20] for x, y in zip(a, b))
ends = [(x[0] + x[1]) - (y[0] + y[1]) for x, y in zip(a, b)]
jumps = sorted(abs(ends[k + 1] - ends[k]) for k in range(len(ends) - 1))
ag = [i for i, e in enumerate(a) if "AllGather" in e[2]]
steps = len(ag) / 4.0
def cls(i):
    e = a[i]
    if "AllGather" in e[2]:
        if "flashinfergemm" in e[3]:
            # lm_head gather: target (followed by 8) or draft (followed by 4)
            nxt = next((j for j in ag if j > i), None)
            return "AG_target_logits" if nxt is not None and nxt - i == 8 else "AG_draft_logits"
        return "AG_other"
    if "vocab_parallel_em" in e[3]:
        prv = max((j for j in ag if j < i), default=None)
        if prv is not None and "flashinfergemm" in a[prv][3] and (next((j for j in ag if j > prv), 10**9) - prv) == 8:
            return "AR_draft_start"
        return "AR_target_start"
    if "flashinfergemm" in e[3]:
        return "AR_after_gemm(o_proj)"
    if "elementwise" in e[3]:
        return "AR_after_elementwise(moe_out)"
    return "AR_other"
import collections
g = collections.defaultdict(list)
for i in range(len(a)):
    g[cls(i)].append((a[i][1], b[i][1], a[i][4], b[i][4]))
floor = sorted(min(x[1], y[1]) for x, y in zip(a, b))[len(a) // 10]
out = {"pairs": len(a), "names_match": names_ok, "steps": steps,
       "end_offset_jump_us_p50": round(jumps[len(jumps) // 2], 2), "end_offset_jump_us_p90": round(jumps[int(0.9 * len(jumps))], 2),
       "floor_us_p10_of_min": floor, "classes": {}}
tot_w0 = tot_w1 = 0.0
for k, v in sorted(g.items()):
    sk = [x - y for x, y, _, _ in v]
    w0 = sum(max(0.0, x - floor) for x, _, _, _ in v); w1 = sum(max(0.0, y - floor) for _, y, _, _ in v)
    tot_w0 += w0; tot_w1 += w1
    out["classes"][k] = {
        "n": len(v), "per_step": round(len(v) / steps, 2),
        "skew_mean_us(r0dur-r1dur;+=r1 late)": round(st.mean(sk), 1),
        "skew_median_us": round(st.median(sk), 1),
        "abs_skew_mean_us": round(st.mean(abs(s) for s in sk), 1),
        "frac_r1_late": round(sum(s > 0 for s in sk) / len(sk), 3),
        "r0_dur_mean_us": round(st.mean(x for x, _, _, _ in v), 1), "r1_dur_mean_us": round(st.mean(y for _, y, _, _ in v), 1),
        "r0_wait_ms_per_step": round(w0 / steps / 1e3, 3), "r1_wait_ms_per_step": round(w1 / steps / 1e3, 3),
        "r0_idle_before_mean_us": round(st.mean(i0 for _, _, i0, _ in v), 1), "r1_idle_before_mean_us": round(st.mean(i1 for _, _, _, i1 in v), 1),
    }
out["r0_wait_ms_per_step_total"] = round(tot_w0 / steps / 1e3, 3)
out["r1_wait_ms_per_step_total"] = round(tot_w1 / steps / 1e3, 3)
print(json.dumps(out, indent=1))
