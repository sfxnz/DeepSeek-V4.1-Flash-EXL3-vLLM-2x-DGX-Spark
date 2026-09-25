#!/usr/bin/env python3
"""Decode-step GPU timeline from one rank's torch-profiler trace (CPU only).

  extract  TRACE.pt.trace.json.gz OUT.pkl   stream the trace (ijson, RLIMIT_AS)
  analyze  OUT.pkl --json OUT.json --txt OUT.txt [--label NAME]
  skew     RANK0.pkl RANK1.pkl [--json OUT.json]   per-collective arrival skew

UMA-safe like tools/extract_kernels.py: ijson streaming under an RLIMIT_AS cap,
never json.load; run on an idle host (serve down), one rank at a time.

extract keeps device ops (kernel, memcpy, memset) with grid/block/stream/
graph id/correlation, the CUDA runtime calls (cudaGraphLaunch, launches), the
execute_model annotations, and python_function / cpu_op events on the worker's
main thread longer than --min-host-us (host context for GPU idle gaps).

analyze splits the window into decode steps at the target-model graph launch
(the CUDA graph with the most all-reduces), then reports per step: wall, device
busy (union over streams) and idle, where the idle sits (previous/next kernel
and the host call that launched the next kernel), per-category ms and launches,
every GEMM/GEMV/MoE/lm_head kernel with its shape, bytes and GB/s (model shapes
from MODEL below, TP=2), NCCL calls with bytes, and the small-kernel glue chains.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import gzip
import json
import pickle
import re
import resource
import statistics
import sys
from pathlib import Path

PEAK_GBPS = 250.0  # demonstrated sustained LPDDR5x stream (cutlass lm_head / Markov, kernels.md)

# Model constants (config.json of 2.0bpw-mcg-lmhead-mxfp8), per TP rank where split.
MODEL = {
    "hidden": 5120, "layers": 40, "vocab": 129280, "tp": 2,
    "experts": 384, "topk": 6, "moe_inter": 2304, "q_lora": 1280, "heads": 64, "head_dim": 512,
    "o_groups": 8, "o_lora": 1024, "draft_layers": 3, "markov_rank": 256,
}


# ----------------------------------------------------------------- extract ---

def extract(path: str, min_host_us: float = 20.0) -> dict:
    import ijson

    names: dict[str, int] = {}

    def nid(s: str) -> int:
        return names.setdefault(s[:200], len(names))

    dev, rt, ann, host, meta = [], [], [], [], {}
    with gzip.open(path, "rb") as fh:
        for e in ijson.items(fh, "traceEvents.item", use_float=True):
            cat = e.get("cat")
            ph = e.get("ph")
            if ph == "M":
                if e.get("name") == "process_name":
                    meta.setdefault("process", {})[str(e.get("pid"))] = (e.get("args") or {}).get("name")
                continue
            if ph != "X":
                continue
            a = e.get("args") or {}
            ts, dur = float(e["ts"]), float(e.get("dur", 0.0))
            if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
                dev.append((ts, dur, nid(e["name"]), tuple(a.get("grid") or ()), tuple(a.get("block") or ()),
                            a.get("stream"), a.get("correlation"), a.get("graph id"), a.get("graph node id"),
                            a.get("registers per thread"), a.get("shared memory"), a.get("bytes"),
                            {"kernel": 0, "gpu_memcpy": 1, "gpu_memset": 2}[cat]))
            elif cat in ("cuda_runtime", "cuda_driver"):
                rt.append((ts, dur, nid(e["name"]), a.get("correlation"), e.get("tid")))
            elif cat in ("user_annotation", "gpu_user_annotation"):
                ann.append((ts, dur, nid(e["name"]), cat == "gpu_user_annotation", e.get("tid")))
            elif cat in ("python_function", "cpu_op") and dur >= min_host_us:
                host.append((ts, dur, nid(e["name"]), e.get("tid"), cat == "cpu_op"))
    return {"names": {v: k for k, v in names.items()}, "dev": dev, "rt": rt, "ann": ann, "host": host,
            "meta": meta, "source": path}


# ---------------------------------------------------------------- classify ---

CATEGORIES = [
    ("nccl_allreduce", r"ncclDevKernel_AllReduce"),
    ("nccl_allgather", r"ncclDevKernel_AllGather"),
    ("nccl_other", r"ncclDevKernel|ncclKernel"),
    ("p2b_moe", r"p2b_moe"),
    ("exl3_moe", r"exl3_moe|exl3_gemm|exl3_"),
    ("dense_b12x", r"dense_blockscaled_gemm_sm120_b12x"),
    ("woa_einsum", r"sm120_fp8_fp4_gemm_1d1d_impl<0u, [34]u, 4096u"),
    ("deepgemm_fp8fp4_other", r"sm120_fp8_fp4_gemm_1d1d|fp8_fp4_gemm|m_grouped"),
    ("woa_pack", r"transpose_and_pack_fp32_into_ue8m0"),
    ("mhc_prenorm_gemm", r"hc_prenorm_gemm"),
    ("mhc_pre_fuse", r"mhc_pre_big_fuse|mhc_pre"),
    ("mhc_post", r"mhc_post"),
    ("sparse_mla", r"sparse_mla|flash_mla|mla_decode|merge_attn|MergeStates|merge_state"),
    ("indexer", r"indexer|topKPerRow|radix|mbtopk|sort|topk_mask|fp8_mqa|mqa_logits|paged_mqa"),
    ("router_topk", r"_dsv4_topk|topkGating|topk_softplus|grouped_topk|noaux"),
    ("cutlass_bf16_gemm", r"cutlass_80_wmma|cutlass_80_tensorop|s161616gemm|gemm_bf16|cublasLt|splitKreduce|sm80_xmma|gemv|gemvx|Kernel2"),
    ("act_quant", r"quantize|MXFP8Quantize|fp8_quant|per_token_group_quant|act_quant"),
    ("rope_kv_cache", r"rope|rotary|kv_cache|insert|save_partial|compress|fused_indexer_q|inv_rope"),
    ("norm", r"rms_norm|rmsnorm|layer_norm|norm_kernel"),
    ("sampler_argmax", r"argmax|softmax|sampl|argMax|reduce_kernel"),
    ("memcpy", r"^Memcpy"),
    ("memset", r"^Memset"),
    ("torch_eltwise", r"at::native|elementwise|Functor|copy_kernel|fill|index|gather|scatter|cat_|CatArray|where"),
    ("triton", r"^triton_|_kernel$"),
]
_CAT_RE = [(c, re.compile(p)) for c, p in CATEGORIES]


def classify(name: str) -> str:
    for c, rx in _CAT_RE:
        if rx.search(name):
            return c
    return "other"


def union_busy(intervals: list[tuple[float, float]]) -> float:
    """Total length of the union of [start, end) intervals."""
    tot, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        elif e > cur_e:
            cur_e = e
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def idle_gaps(intervals: list[tuple[float, float, int]], lo: float, hi: float, min_gap: float) -> list[tuple]:
    """Gaps with no device op in [lo, hi): (start, end, idx_before, idx_after)."""
    out = []
    cur_e, cur_i = lo, None
    for s, e, i in sorted(intervals):
        if s >= hi:
            break
        if s - cur_e >= min_gap:
            out.append((cur_e, s, cur_i, i))
        if e > cur_e:
            cur_e, cur_i = e, i
    if hi - cur_e >= min_gap:
        out.append((cur_e, hi, cur_i, None))
    return out


# ------------------------------------------------------------------- roles ---

MB = 1e6
H = MODEL["hidden"]
VOCAB_RANK = MODEL["vocab"] // MODEL["tp"]
EXPERT_BYTES = 3 * H * (MODEL["moe_inter"] // MODEL["tp"]) * 2 / 8  # EXL3 2.0 bpw gate+up+down per rank


def _fp8(n: int, k: int) -> float:
    """fp8 weight [n, k] + ue8m0 scale per 32 along k."""
    return n * k + n * (k // 32)


# role -> (weight bytes per rank, K, N): the streamed bytes of one call at small m.
WEIGHTS = {
    "qkv_a": (_fp8(1280 + 512, H), H, 1792),            # wq_a + wkv fused, replicated
    "wq_b": (_fp8(32768 // 2, 1280), 1280, 16384),
    "indexer_wq_b": (_fp8(4096, 1280), 1280, 4096),     # indexer q proj (compressor layers)
    "wo_a": (_fp8(8192 // 2, 4096), 4096, 4096),        # 4 groups x 1024 o_lora per rank
    "wo_b": (_fp8(H, 8192 // 2), 4096, H),
    "shared_gate_up": (_fp8(2 * 2304 // 2, H), H, 2304),
    "shared_down": (_fp8(H, 2304 // 2), 1152, H),
    "engram_wkv": (_fp8(25600, 6144), 6144, 25600),     # ReplicatedLinear
    "lm_head": (_fp8(VOCAB_RANK, H), H, VOCAB_RANK),    # mxfp8 pack
    "main_proj": (_fp8(H, 3 * H), 3 * H, H),            # draft, replicated
    "draft_ctx_kv": (_fp8(1280 + 512, H), H, 1792),     # draft layers' context kv precompute
    "router_gate": (384 * H * 2, H, 384),               # bf16, replicated
    "draft_router_gate": (128 * H * 2, H, 128),
    "hc_prenorm": (24 * 4 * H * 4, 4 * H, 24),          # fp32 fn
    "markov": (MODEL["vocab"] * MODEL["markov_rank"] * 2, MODEL["markov_rank"], MODEL["vocab"]),
}


def label_roles(seq: list[tuple], where: str) -> dict[int, str]:
    """Role per GEMM-like kernel of one graph replay or eager region.

    seq: [(idx, category, grid, dur_us, name)] in start order; where: target | draft
    | eager. Layer order (both graphs): [AR] mhc_post [engram AG -> wkv]
    prenorm qkv_a(28) wq_b(48) [indexer_wq_b(48)] ... wo_a wo_b(48) AR prenorm
    [router gate || shared gate_up(36) shared_down(48)] MoE.
    """
    roles: dict[int, str] = {}
    seen_wq_b = seen_woa = in_ffn = False
    pending_engram = False
    for idx, cat, grid, dur, name in seq:
        if cat == "mhc_prenorm_gemm":
            roles[idx] = "hc_prenorm"
            if in_ffn or not (seen_wq_b or seen_woa):
                if in_ffn:
                    seen_wq_b = seen_woa = in_ffn = False
            elif seen_woa:
                in_ffn = True
            continue
        if cat == "nccl_allgather" and where == "target":
            pending_engram = True
            continue
        if cat == "woa_einsum":
            roles[idx] = "wo_a"
            seen_woa = True
            continue
        if cat == "p2b_moe":
            roles[idx] = "routed_moe_p2b"
            continue
        if cat == "exl3_moe":
            roles[idx] = "routed_moe_exl3"
            continue
        if cat == "deepgemm_fp8fp4_other":
            roles[idx] = "draft_moe_gate_up" if "2304u, 5120u" in name else (
                "draft_moe_down" if "5120u, 1152u" in name else "deepgemm_other")
            continue
        if cat == "cutlass_bf16_gemm":
            if "splitKreduce" in name:
                roles[idx] = "router_gate_splitk_reduce"
            elif tuple(grid[:2]) == (8, 505):
                roles[idx] = "markov"
            elif where == "draft" and tuple(grid) == (8, 1, 1):
                roles[idx] = "draft_router_gate"
            elif in_ffn and where == "target":
                roles[idx] = "router_gate"
            else:
                roles[idx] = "bf16_gemm_grid" + "x".join(str(v) for v in grid)
            continue
        if cat != "dense_b12x":
            continue
        z = grid[2] if len(grid) > 2 else 0
        if dur >= 1000:
            roles[idx] = "lm_head"
        elif pending_engram:
            roles[idx] = "engram_wkv"
            pending_engram = False
        elif where == "eager" and dur >= 250:
            roles[idx] = "main_proj"
        elif z == 28:
            roles[idx] = "draft_ctx_kv" if where == "eager" else "qkv_a"
        elif z == 36:
            roles[idx] = "shared_gate_up"
        elif in_ffn:
            roles[idx] = "shared_down"
        elif seen_woa:
            roles[idx] = "wo_b"
        elif not seen_wq_b:
            roles[idx] = "wq_b"
            seen_wq_b = True
        else:
            roles[idx] = "indexer_wq_b"
    return roles


# ----------------------------------------------------------------- analyze ---

def _pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))] if s else 0.0


def _st(xs):
    return {"median": round(statistics.median(xs), 2), "p10": round(_pct(xs, .1), 2),
            "p90": round(_pct(xs, .9), 2), "mean": round(statistics.fmean(xs), 2)}


ANCHORS = {"p2b_moe", "exl3_moe", "dense_b12x", "woa_einsum", "deepgemm_fp8fp4_other", "nccl_allreduce",
           "nccl_allgather", "nccl_other", "mhc_prenorm_gemm", "sparse_mla", "cutlass_bf16_gemm"}


def _short(name: str) -> str:
    m = re.search(r"at::native::(?:\(anonymous namespace\)::)?(\w+)<?.*?(?:at::native::(?:\(anonymous namespace\)::)?(\w+))?", name)
    if name.startswith("void at::native") and m:
        inner = re.findall(r"at::native::(?:\(anonymous namespace\)::)?(\w+)", name)
        return "aten:" + "/".join(inner[:2])
    name = re.sub(r"^void ", "", name)
    name = re.sub(r"^kernel_cutlass_kernel_flashinfer", "fi:", name)
    return re.split(r"[<(]", name)[0][:48]


def analyze(d: dict, label: str = "") -> dict:
    names = d["names"]
    dev = sorted(d["dev"])
    cat_of = [classify(names[k[2]]) for k in dev]
    corr_rt = {}
    for ts, dur, n, corr, tid in d["rt"]:
        if corr is not None:
            corr_rt[corr] = (ts, dur, names[n], tid)
    by_launch = collections.defaultdict(list)
    for i, k in enumerate(dev):
        r = corr_rt.get(k[6])
        if r and r[2] == "cudaGraphLaunch":
            by_launch[k[6]].append(i)
    if not by_launch:
        raise ValueError("no cudaGraphLaunch-correlated kernels in trace")
    graph_of = {c: dev[idx[0]][7] for c, idx in by_launch.items()}
    ar_of = {c: sum(cat_of[i] == "nccl_allreduce" for i in idx) for c, idx in by_launch.items()}
    max_ar = max(ar_of.values())
    kind = {c: ("target" if ar_of[c] >= max(2, max_ar - 1) else ("draft" if ar_of[c] > 0 else "graph_other"))
            for c in by_launch}
    span = {c: (min(dev[i][0] for i in idx), max(dev[i][0] + dev[i][1] for i in idx)) for c, idx in by_launch.items()}
    targets = sorted((span[c][0], c) for c in by_launch if kind[c] == "target")
    raw_steps = [(targets[j][1], targets[j][0], targets[j + 1][0]) for j in range(len(targets) - 1)]
    walls = [e - s for _, s, e in raw_steps]
    medw = statistics.median(walls)
    steps = [(c, s, e) for c, s, e in raw_steps if e - s <= 2.0 * medw]
    dropped = len(raw_steps) - len(steps)
    kernel_launch = {i: c for c, idx in by_launch.items() for i in idx}
    step_starts = [s for _, s, _ in steps]

    def step_of(t):
        j = bisect.bisect_right(step_starts, t) - 1
        return j if j >= 0 and t < steps[j][2] else None

    # group steps by target graph id (capture size)
    groups = collections.defaultdict(list)
    for j, (c, s, e) in enumerate(steps):
        groups[graph_of[c]].append(j)
    step_dev = [[] for _ in steps]
    for i, k in enumerate(dev):
        j = step_of(k[0])
        if j is not None:
            step_dev[j].append(i)
    # roles per graph replay and per eager region
    role = {}
    for c, idx in by_launch.items():
        seq = [(i, cat_of[i], dev[i][3], dev[i][1], names[dev[i][2]]) for i in sorted(idx, key=lambda i: dev[i][0])]
        role.update(label_roles(seq, kind[c]))
    eager_seq = [(i, cat_of[i], dev[i][3], dev[i][1], names[dev[i][2]])
                 for j in range(len(steps)) for i in step_dev[j] if i not in kernel_launch]
    role.update(label_roles(eager_seq, "eager"))

    main_tid = None
    for pid, pname in (d.get("meta", {}).get("process") or {}).items():
        if pname and "Worker" in pname and pid not in ("0",) and int(pid) > 20:
            main_tid = int(pid)
    host = sorted((h for h in d.get("host", []) if main_tid is None or h[3] == main_tid), key=lambda h: h[0])
    host_starts = [h[0] for h in host]

    out = {"label": label, "source": d.get("source"), "steps_total": len(steps), "steps_dropped_outliers": dropped,
           "peak_gbps": PEAK_GBPS, "groups": {}}
    for gid, js in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        n = len(js)
        g = {"target_graph_id": gid, "steps": n}
        # verify rows m: the deep_gemm wo_a einsum template carries M (<0u, M, 4096u ...)
        m_t = None
        for i in by_launch[steps[js[0]][0]]:
            mm = re.search(r"gemm_1d1d_impl<0u, (\d+)u, 4096u", names[dev[i][2]])
            if mm:
                m_t = int(mm.group(1))
                break
        g["verify_rows_m"] = m_t
        wall = [steps[j][2] - steps[j][1] for j in js]
        busy, phase = [], collections.defaultdict(list)
        cat_ms, cat_n = collections.Counter(), collections.Counter()
        role_d = collections.defaultdict(list)
        nccl = collections.defaultdict(list)
        glue = collections.defaultdict(lambda: [0, 0.0, 0.0, collections.Counter(), 0, 0.0])
        gaps = collections.defaultdict(lambda: [0, 0.0, collections.Counter(), 0])
        for j in js:
            c0, s0, e0 = steps[j]
            ivs = [(dev[i][0], dev[i][0] + dev[i][1], i) for i in step_dev[j]]
            busy.append(union_busy([(a, b) for a, b, _ in ivs]))
            # phases: target graph span, draft graph span(s), eager before/after
            tspan = span[c0]
            dspans = [span[c] for c in by_launch if kind[c] == "draft" and s0 <= span[c][0] < e0]
            phase["target_graph_ms"].append((tspan[1] - tspan[0]) / 1e3)
            phase["draft_graph_ms"].append(sum(b - a for a, b in dspans) / 1e3)
            if dspans:
                phase["eager_after_target_ms"].append((min(a for a, _ in dspans) - tspan[1]) / 1e3)
                phase["eager_before_next_target_ms"].append((e0 - max(b for _, b in dspans)) / 1e3)
            for i in step_dev[j]:
                k = dev[i]
                cat_ms[cat_of[i]] += k[1]
                cat_n[cat_of[i]] += 1
                r = role.get(i)
                if r:
                    role_d[r].append(k[1])
                if cat_of[i] in ("nccl_allreduce", "nccl_allgather"):
                    where = kind.get(kernel_launch.get(i), "eager")
                    nccl[(cat_of[i], where, tuple(k[3]))].append(k[1])
            # idle gaps with context
            for g0, g1, ib, ia in idle_gaps(ivs, s0, e0, 3.0):
                wb = kind.get(kernel_launch.get(ib), "eager") if ib is not None else "-"
                wa = kind.get(kernel_launch.get(ia), "eager") if ia is not None else "-"
                nb = (role.get(ib) or _short(names[dev[ib][2]])) if ib is not None else "<start>"
                na = (role.get(ia) or _short(names[dev[ia][2]])) if ia is not None else "<end>"
                lr = corr_rt.get(dev[ia][6]) if ia is not None else None
                how = lr[2] if lr else "?"
                hb = bool(lr and lr[0] + lr[1] > g0)
                key = (f"{wb}:{nb}", f"{wa}:{na}", how)
                a = gaps[key]
                a[0] += 1
                a[1] += g1 - g0
                a[3] += hb
                if g1 - g0 >= 50 and host:
                    lo = bisect.bisect_left(host_starts, g0 - 200000)
                    for h in host[lo:]:
                        if h[0] > g1:
                            break
                        ov = min(g1, h[0] + h[1]) - max(g0, h[0])
                        if ov > 0:
                            a[2][names[h[2]][:90]] += ov
            # glue chains between anchors, in start order over all streams
            order = sorted(step_dev[j], key=lambda i: dev[i][0])
            run, prev_anchor, anchor_end = [], "<start>", s0
            for i in order + [None]:
                is_anchor = i is None or cat_of[i] in ANCHORS
                if is_anchor:
                    if run:
                        nxt = dev[i][0] if i is not None else e0
                        key = (prev_anchor, (role.get(i) or cat_of[i]) if i is not None else "<end>")
                        st, en = dev[run[0]][0], max(dev[q][0] + dev[q][1] for q in run)
                        a = glue[key]
                        a[0] += 1
                        a[1] += sum(dev[q][1] for q in run)
                        a[2] += en - st
                        a[3][tuple(_short(names[dev[q][2]]) for q in run)] += 1
                        a[4] += len(run)
                        a[5] += max(0.0, nxt - anchor_end)  # no anchor running: glue + gaps on the path
                    run = []
                    if i is not None:
                        prev_anchor = role.get(i) or cat_of[i]
                        anchor_end = max(anchor_end, dev[i][0] + dev[i][1])
                else:
                    run.append(i)
        g["step_wall_ms"] = _st([w / 1e3 for w in wall])
        g["device_busy_ms"] = round(statistics.fmean(busy) / 1e3, 3)
        g["device_idle_ms"] = round(statistics.fmean([w - b for w, b in zip(wall, busy)]) / 1e3, 3)
        g["kernel_sum_ms"] = round(sum(cat_ms.values()) / n / 1e3, 3)
        g["phases_ms"] = {k: _st(v) for k, v in phase.items()}
        g["categories"] = sorted(({"category": c, "ms_per_step": round(cat_ms[c] / n / 1e3, 3),
                                   "launches_per_step": round(cat_n[c] / n, 2)} for c in cat_ms),
                                 key=lambda r: -r["ms_per_step"])
        rows = []
        for r, durs in role_d.items():
            row = {"role": r, "calls_per_step": round(len(durs) / n, 2), "us": _st(durs),
                   "ms_per_step": round(sum(durs) / n / 1e3, 3)}
            wb = None
            if r in WEIGHTS:
                wb = WEIGHTS[r][0]
            elif r == "routed_moe_p2b" and m_t:
                wb = m_t * MODEL["topk"] * EXPERT_BYTES  # p2b streams every (row, expert) pair
                row["pairs"] = m_t * MODEL["topk"]
            if wb:
                row["weight_MB"] = round(wb / MB, 2)
                row["GBps_at_median"] = round(wb / (statistics.median(durs) * 1e-6) / 1e9, 1)
                row["pct_of_peak"] = round(100 * row["GBps_at_median"] / PEAK_GBPS, 1)
                row["roofline_us"] = round(wb / (PEAK_GBPS * 1e9) * 1e6, 1)
                row["recoverable_ms_per_step"] = round(max(0.0, sum(durs) / n - len(durs) / n * row["roofline_us"]) / 1e3, 3)
            rows.append(row)
        g["gemm_roles"] = sorted(rows, key=lambda r: -r["ms_per_step"])
        g["nccl"] = sorted(({"op": op, "where": w, "grid": list(gr), "calls_per_step": round(len(v) / n, 2),
                             "us": _st(v), "ms_per_step": round(sum(v) / n / 1e3, 3)}
                            for (op, w, gr), v in nccl.items()), key=lambda r: -r["ms_per_step"])
        gl = []
        for (nb, na, how), (cnt, tot, frames, hb) in gaps.items():
            e = {"before": nb, "after": na, "next_launched_by": how, "count_per_step": round(cnt / n, 2),
                 "ms_per_step": round(tot / n / 1e3, 3), "us_mean": round(tot / cnt, 1),
                 "host_bound_frac": round(hb / cnt, 2)}
            if frames:
                e["host_frames_top"] = [[f, round(100 * v / tot, 1)] for f, v in frames.most_common(40)]
            gl.append(e)
        g["idle_gaps"] = sorted(gl, key=lambda r: -r["ms_per_step"])[:30]
        chains = []
        for (pa, na), (cnt, dsum, spn, variants, nk, expo) in glue.items():
            ks, vn = variants.most_common(1)[0]
            chains.append({"after": pa, "before": na, "kernels": list(ks), "variants": len(variants),
                           "top_variant_share": round(vn / cnt, 2), "count_per_step": round(cnt / n, 2),
                           "kernel_ms_per_step": round(dsum / n / 1e3, 3), "span_ms_per_step": round(spn / n / 1e3, 3),
                           "exposed_ms_per_step": round(expo / n / 1e3, 3), "launches_per_step": round(nk / n, 1)})
        g["glue_chains"] = sorted(chains, key=lambda r: -r["exposed_ms_per_step"])[:25]
        g["glue_total"] = {"span_ms_per_step": round(sum(c[2] for c in glue.values()) / n / 1e3, 3),
                           "kernel_ms_per_step": round(sum(c[1] for c in glue.values()) / n / 1e3, 3),
                           "exposed_ms_per_step": round(sum(c[5] for c in glue.values()) / n / 1e3, 3),
                           "launches_per_step": round(sum(c[4] for c in glue.values()) / n, 1)}
        gl_host = collections.defaultdict(list)
        for c, (a, b) in span.items():
            if step_of(a) in set(js) and c in corr_rt:
                gl_host[kind[c]].append(corr_rt[c][1])
        g["graph_launch_host_us"] = {k: _st(v) for k, v in gl_host.items()}
        out["groups"][str(gid)] = g
    return out


def render(res: dict) -> str:
    """Readable per-group tables from analyze() output."""
    L = []
    L.append(f"# {res.get('label', '')}  source={res.get('source')}")
    L.append(f"steps={res['steps_total']} (dropped outliers {res['steps_dropped_outliers']}), peak={res['peak_gbps']} GB/s")
    for gid, g in res["groups"].items():
        L.append("")
        L.append(f"## target graph {gid}: m={g['verify_rows_m']} steps={g['steps']}")
        w = g["step_wall_ms"]
        L.append(f"step wall ms median {w['median']} p10 {w['p10']} p90 {w['p90']} mean {w['mean']}; "
                 f"device busy {g['device_busy_ms']} idle {g['device_idle_ms']}; kernel sum {g['kernel_sum_ms']}")
        L.append("phases (ms/step median): " + ", ".join(f"{k} {v['median']}" for k, v in g["phases_ms"].items()))
        L.append("cudaGraphLaunch host us (median): " + ", ".join(f"{k} {v['median']}" for k, v in g["graph_launch_host_us"].items()))
        L.append("")
        L.append(f"{'category':<24}{'ms/step':>9}{'launch/step':>13}")
        for c in g["categories"]:
            L.append(f"{c['category']:<24}{c['ms_per_step']:>9.3f}{c['launches_per_step']:>13.1f}")
        L.append("")
        L.append(f"{'GEMM/GEMV/MoE role':<28}{'n/step':>7}{'us med':>9}{'p10':>8}{'p90':>8}{'ms/step':>9}{'MB':>9}{'GB/s':>8}{'%peak':>7}{'roof us':>9}{'recov ms':>9}")
        for r in g["gemm_roles"]:
            L.append(f"{r['role']:<28}{r['calls_per_step']:>7.1f}{r['us']['median']:>9.1f}{r['us']['p10']:>8.1f}{r['us']['p90']:>8.1f}"
                     f"{r['ms_per_step']:>9.3f}{r.get('weight_MB', ''):>9}{r.get('GBps_at_median', ''):>8}{r.get('pct_of_peak', ''):>7}"
                     f"{r.get('roofline_us', ''):>9}{r.get('recoverable_ms_per_step', ''):>9}")
        L.append("")
        L.append(f"{'NCCL op':<16}{'where':<8}{'grid':<12}{'n/step':>7}{'us med':>9}{'p10':>8}{'p90':>8}{'mean':>8}{'ms/step':>9}")
        for r in g["nccl"]:
            L.append(f"{r['op']:<16}{r['where']:<8}{str(r['grid']):<12}{r['calls_per_step']:>7.1f}{r['us']['median']:>9.1f}"
                     f"{r['us']['p10']:>8.1f}{r['us']['p90']:>8.1f}{r['us']['mean']:>8.1f}{r['ms_per_step']:>9.3f}")
        L.append("")
        L.append("idle gaps (device idle, all streams), top by ms/step:")
        for r in g["idle_gaps"][:15]:
            L.append(f"  {r['ms_per_step']:7.3f} ms/step  {r['count_per_step']:6.2f}/step x {r['us_mean']:8.1f} us  "
                     f"{r['before']} -> {r['after']}  (next via {r['next_launched_by']}, host-bound {r['host_bound_frac']})")
            generic = ("decorate_context", "_execute_worker_rpc", "worker_base.py", "gpu_worker.py", "prefill_empty_cache")
            fr = [f for f in r.get("host_frames_top", []) if f[1] >= 20 and not any(g in f[0] for g in generic)]
            if fr:
                L.append("           host (main thread, % of gap): " + "; ".join(f"{n} {p}%" for n, p in fr[:8]))
        L.append("")
        gt = g["glue_total"]
        L.append(f"glue (non-anchor small kernels): {gt['launches_per_step']} launches/step, kernel {gt['kernel_ms_per_step']} ms, "
                 f"span {gt['span_ms_per_step']} ms, exposed (no anchor kernel running) {gt['exposed_ms_per_step']} ms per step; "
                 "top chains by exposed time:")
        for r in g["glue_chains"][:20]:
            ks = r["kernels"]
            L.append(f"  {r['exposed_ms_per_step']:6.3f} ms exposed {r['span_ms_per_step']:6.3f} span {r['kernel_ms_per_step']:6.3f} kern {r['count_per_step']:6.2f}/step "
                     f"{r['launches_per_step']:6.1f} launches [{r['after']} -> {r['before']}] ({r['variants']} variants) "
                     f"e.g. {len(ks)}: {', '.join(ks[:10])}{' ...' if len(ks) > 10 else ''}")
    return "\n".join(L) + "\n"


def nccl_skew(d0: dict, d1: dict) -> dict:
    """Arrival skew per collective between two ranks' traces of the same window.

    Collectives finish together, so dur_r0 - dur_r1 = arrival_r1 - arrival_r0
    (positive: rank 1 arrives later and rank 0 waits). Matched by order.
    """
    def coll(d):
        names = d["names"]
        corr = {c: names[n] for ts, dur, n, c, tid in d["rt"] if c is not None}
        out = []
        for k in sorted(d["dev"]):
            nm = names[k[2]]
            if nm.startswith("ncclDevKernel"):
                where = "graph" if corr.get(k[6]) == "cudaGraphLaunch" else "eager"
                out.append((k[1], nm.split("(")[0].replace("ncclDevKernel_", ""), where, tuple(k[3])))
        return out
    a, b = coll(d0), coll(d1)
    n = min(len(a), len(b))
    mism = sum(1 for x, y in zip(a[:n], b[:n]) if x[1] != y[1] or x[3] != y[3])
    floor = sorted(min(x[0], y[0]) for x, y in zip(a[:n], b[:n]))[n // 10] if n else 0.0
    cls = collections.defaultdict(list)
    for x, y in zip(a[:n], b[:n]):
        cls[(x[1], x[2], x[3])].append((x[0], y[0]))
    rows = []
    for (op, where, grid), v in cls.items():
        sk = [p - q for p, q in v]
        rows.append({"op": op, "where": where, "grid": list(grid), "n": len(v),
                     "skew_median_us(+=rank1 late)": round(statistics.median(sk), 1),
                     "abs_skew_mean_us": round(statistics.fmean(abs(x) for x in sk), 1),
                     "frac_rank1_late": round(sum(x > 0 for x in sk) / len(sk), 3),
                     "r0_dur_median_us": round(statistics.median(p for p, _ in v), 1),
                     "r1_dur_median_us": round(statistics.median(q for _, q in v), 1),
                     "r0_wait_over_floor_ms": round(sum(max(0.0, p - floor) for p, _ in v) / 1e3, 3),
                     "r1_wait_over_floor_ms": round(sum(max(0.0, q - floor) for _, q in v) / 1e3, 3)})
    rows.sort(key=lambda r: -r["n"])
    return {"collectives": [len(a), len(b)], "matched": n, "op_or_grid_mismatch": mism,
            "floor_us_p10_of_min": round(floor, 2), "classes": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    x = sub.add_parser("extract")
    x.add_argument("trace")
    x.add_argument("out")
    x.add_argument("--rlimit-gib", type=int, default=6)
    x.add_argument("--min-host-us", type=float, default=20.0)
    a = sub.add_parser("analyze")
    a.add_argument("pkl")
    a.add_argument("--json")
    a.add_argument("--txt")
    a.add_argument("--label", default="")
    k = sub.add_parser("skew")
    k.add_argument("rank0_pkl")
    k.add_argument("rank1_pkl")
    k.add_argument("--json")
    args = ap.parse_args(argv)
    if args.cmd == "skew":
        with open(args.rank0_pkl, "rb") as f0, open(args.rank1_pkl, "rb") as f1:
            res = nccl_skew(pickle.load(f0), pickle.load(f1))
        text = json.dumps(res, indent=1)
        if args.json:
            Path(args.json).write_text(text + "\n")
        print(text)
        return 0
    if args.cmd == "extract":
        cap = args.rlimit_gib << 30
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        d = extract(args.trace, args.min_host_us)
        with open(args.out, "wb") as fh:
            pickle.dump(d, fh)
        print(f"dev {len(d['dev'])} rt {len(d['rt'])} ann {len(d['ann'])} host {len(d['host'])} names {len(d['names'])}")
        return 0
    with open(args.pkl, "rb") as fh:
        d = pickle.load(fh)
    res = analyze(d, args.label)
    text = json.dumps(res, indent=1)
    if args.json:
        Path(args.json).write_text(text + "\n")
    if args.txt:
        Path(args.txt).write_text(render(res))
    if not (args.json or args.txt):
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
