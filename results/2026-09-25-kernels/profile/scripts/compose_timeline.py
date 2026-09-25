#!/usr/bin/env python3
"""Compose timeline-r3.json from the per-rank analyzer outputs (c1/, c2/), the
reference benches and the NCCL sweep. Every opportunity number is computed
here from those inputs; the formula sits next to it.

  python3 scripts/compose_timeline.py  (run from results/2026-09-25-kernels/profile)
"""
import json
import re
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
PEAK = 250.0
EXPERT_MB = 3 * 5120 * 1152 * 2 / 8 / 1e6  # EXL3 2.0 bpw gate+up+down per rank
DUP = 0.2987  # s10 census: duplicate (row, expert) fraction per layer at m=4


def load(p):
    return json.loads((HERE / p).read_text())


def group(doc, m=None):
    gs = list(doc["groups"].values())
    if m is not None:
        gs = [g for g in gs if g["verify_rows_m"] == m]
    return max(gs, key=lambda g: g["steps"])


def roles(g):
    return {r["role"]: r for r in g["gemm_roles"]}


def lail_summary(path):
    txt = (HERE / path).read_text()
    s = json.loads(txt[txt.index("SUMMARY ") + 8:])
    runs = [json.loads(ln.split(" ", 1)[1]) for ln in txt.splitlines() if ln.startswith("run=")]
    return {"median_lail_tok_s": round(s["median_lail_tok_s"], 2), "median_acceptance_len": round(s["median_acceptance_len"], 3),
            "ms_per_step": round(1000 * s["median_acceptance_len"] / s["median_lail_tok_s"], 2),
            "runs_tok_s": [round(r["lail_tok_s"], 2) for r in runs], "n": s["n"], "median_ttft_s": round(s["median_ttft_s"], 3)}


def bench_summary(path):
    txt = (HERE / path).read_text()
    arr = json.loads(txt[txt.rfind("[\n  {"):])
    keep = ("phase", "concurrency", "median_decode_tok_s", "median_agg_tok_s", "acceptance_len", "median_ms_per_step",
            "inter_chunk_ms_p50", "inter_chunk_ms_p90", "n")
    return [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in a.items() if k in keep} for a in arr]


def smi_summary(prefix):
    out = {}
    for host in ("spark1", "spark2"):
        rows = [ln.strip().split(", ") for ln in open(HERE / f"{prefix}.{host}.csv") if ln.strip()]
        busy = [r for r in rows if len(r) > 5 and r[5].rstrip(" %").isdigit() and int(r[5].rstrip(" %")) > 50]
        sm = [int(r[1].split()[0]) for r in busy]
        out[host] = {"samples": len(rows), "busy_samples": len(busy), "sm_mhz_median": st.median(sm), "sm_mhz_min": min(sm),
                     "power_w_median": round(st.median(float(r[3].split()[0]) for r in busy), 2),
                     "temp_c_max": max(int(r[4]) for r in busy), "mem_clock": rows[0][2]}
    return out


def main():
    c1 = [load(f"c1/c1-rank{r}.json") for r in (0, 1)]
    c2 = [load(f"c2/c2-rank{r}.json") for r in (0, 1)]
    g1 = [group(d, 4) for d in c1]
    g2 = [group(d, 8) for d in c2]
    r0, r1 = roles(g1[0]), roles(g1[1])
    sk1 = load("c1/c1-skew.json")
    nccl = load("nccl-sweep.json")
    steps1 = g1[0]["steps"]

    iso = {"qkv_a": 43.7, "wq_b": 95.6, "wo_b": 96.1, "shared_gate_up": 53.5, "shared_down": 27.7,
           "main_proj": 360.6, "routed_moe_p2b": 512.3}
    serve_vs_iso = {}
    for rank, rr in (("rank0", r0), ("rank1", r1)):
        rows = {}
        for k, v in iso.items():
            x = rr[k]
            rows[k] = {"serve_us_median": x["us"]["median"], "serve_us_mean": x["us"]["mean"], "isolated_us": v,
                       "pct_median": round(100 * (x["us"]["median"] / v - 1), 1), "pct_mean": round(100 * (x["us"]["mean"] / v - 1), 1),
                       "excess_ms_per_step_mean": round(x["calls_per_step"] * (x["us"]["mean"] - v) / 1e3, 3)}
        rows["total_excess_ms_per_step"] = round(sum(v["excess_ms_per_step_mean"] for v in rows.values()), 2)
        serve_vs_iso[rank] = rows

    skew_ps = {}
    for c in sk1["classes"]:
        k = f"{c['op'].split('_')[0]}_{c['where']}_grid{c['grid'][0]}"
        skew_ps[k] = {"per_step": round(c["n"] / steps1, 2), "rank0_wait_ms_per_step": round(c["r0_wait_over_floor_ms"] / steps1, 3),
                      "rank1_wait_ms_per_step": round(c["r1_wait_over_floor_ms"] / steps1, 3),
                      "skew_median_us_pos_rank1_late": c["skew_median_us(+=rank1 late)"], "frac_rank1_late": c["frac_rank1_late"]}

    p2b = r0["routed_moe_p2b"]
    pairs = p2b["pairs"]
    uniq = pairs * (1 - DUP)
    dense_roles = ["wq_b", "wo_b", "qkv_a", "shared_gate_up", "shared_down", "engram_wkv", "lm_head", "main_proj",
                   "indexer_wq_b", "draft_ctx_kv"]
    dense_ms = sum(r0[k]["ms_per_step"] for k in dense_roles)
    dense_recov = sum(r0[k]["recoverable_ms_per_step"] for k in dense_roles)
    glue = {(c["after"], c["before"]): c for c in g1[0]["glue_chains"]}
    gl = lambda a, b: glue.get((a, b), {}).get("exposed_ms_per_step", 0.0)  # noqa: E731
    gaps = g1[0]["idle_gaps"]
    engram_gap = next(x for x in gaps if "Memcpy DtoH" in x["before"] and "Memcpy HtoD" in x["after"])
    cat = {c["category"]: c for c in g1[0]["categories"]}
    mhc_ms = cat["mhc_prenorm_gemm"]["ms_per_step"] + cat["mhc_pre_fuse"]["ms_per_step"] + cat["mhc_post"]["ms_per_step"]
    bm = nccl["best_vs_keep"]["clean_modeled_ms_per_step"]
    startup = (HERE / "c1/c1_graph_startup.txt").read_text()
    st_t = float(re.search(r"c1-rank0.pkl target .*startup excess mean .*? ([\d.]+)$", startup, re.M).group(1))
    st_d = float(re.search(r"c1-rank0.pkl draft .*startup excess mean .*? ([\d.]+)$", startup, re.M).group(1))
    dense_facts = ", ".join(f"{k} {r0[k]['us']['median']} us {r0[k]['GBps_at_median']} GB/s" for k in dense_roles[:6])
    r2 = roles(g2[1])  # c=2 rank1: spark2's GPU was not shared, kernel times are clean there

    opp = [
        {"rank": 1, "item": "Routed MoE (p2b) streaming and expert dedup",
         "now_ms_per_step": p2b["ms_per_step"],
         "facts": f"40 x {p2b['us']['median']} us median ({p2b['us']['mean']} mean), {pairs} (row, expert) pairs x {EXPERT_MB:.2f} MB = {p2b['weight_MB']} MB/call, {p2b['GBps_at_median']} GB/s = {p2b['pct_of_peak']}% of 250; c=2: {r2['routed_moe_p2b']['us']['median']} us/call at m=8 (48 pairs, {r2['routed_moe_p2b']['GBps_at_median']} GB/s), {r2['routed_moe_p2b']['ms_per_step']} ms/step = ~49% of a clean c=2 step",
         "recoverable_ms_per_step": {
             "roofline_all_pairs": round(p2b["ms_per_step"] - 40 * pairs * EXPERT_MB / PEAK, 2),
             "dedup_at_today_GBps": round(p2b["ms_per_step"] * DUP, 2),
             "dedup_at_roofline": round(p2b["ms_per_step"] - 40 * uniq * EXPERT_MB / PEAK, 2),
             "round2_coop_estimate": 4.9},
         "formula": "roofline_all_pairs = ms - 40*pairs*MB/250; dedup = ms*dup (dup 0.2987, s10 census); dedup_at_roofline = ms - 40*pairs*(1-dup)*MB/250",
         "note": "rank0 tail (mean - median) is 1.3 ms/step of this; the coop kernel (DSV41_P2B_COOP, round 2) exists but was never measured on GPU"},
        {"rank": 2, "item": "In-graph NCCL: graph-mixing machinery + graph startup",
         "now_ms_per_step": bm["keep_c1"],  # sweep model of the step's collective critical path under KEEP
         "facts": f"target in-layer AR kernels {g1[0]['nccl'][0]['ms_per_step']} ms/step on rank0 ({g1[0]['nccl'][0]['us']['median']} us median, {g1[0]['nccl'][0]['us']['mean']} mean); sweep (clean): keep AR m=4 gapped steady {nccl['best_vs_keep']['per_call_clean']['ar_m4_gapped_steady_us'][0]} us vs mixing-off {nccl['best_vs_keep']['per_call_clean']['ar_m4_gapped_steady_us'][1]} us; first AR per replay: target +{st_t:.0f} us, draft +{st_d:.0f} us",
         "recoverable_ms_per_step": {"mixing_off_modeled_c1": round(bm["keep_c1"] - bm["mix0_c1"], 2),
                                     "mixing_off_modeled_c2": round(bm["keep_c2"] - bm["mix0_c2"], 2),
                                     "graph_startup_c1": round((st_t + st_d) / 1e3, 2)},
         "note": "NCCL_GRAPH_MIXING_SUPPORT=0 is unsafe alone (eager AG/AR launched during the outstanding target graph in 100% of steps); needs eager collectives on a second communicator; serve untested"},
        {"rank": 3, "item": "Dense MXFP8 GEMMs (b12x): streaming efficiency + serving tax",
         "now_ms_per_step": round(dense_ms, 2),
         "facts": dense_facts + f"; in-serve vs isolated cold microbench +{serve_vs_iso['rank0']['qkv_a']['pct_median']}..{serve_vs_iso['rank0']['wq_b']['pct_median']}% (median)",
         "recoverable_ms_per_step": {"roofline": round(dense_recov, 2),
                                     "serving_tax_rank0_mean_basis": round(sum(v["excess_ms_per_step_mean"] for k, v in serve_vs_iso["rank0"].items() if k not in ("routed_moe_p2b", "total_excess_ms_per_step")), 2),
                                     "engram_wkv_tp_shard": round(r0["engram_wkv"]["ms_per_step"] / 2, 2)},
         "note": "engram_wkv is a ReplicatedLinear [25600, 6144] fp8 = 162 MB per rank per engram layer; half of it per rank + a 205 KB all_gather"},
        {"rank": 4, "item": "Host Engram stage on the critical path (CPU gather + dequant)",
         "now_ms_per_step": engram_gap["ms_per_step"],
         "facts": f"GPU idle {engram_gap['us_mean']} us/step between the draft's DtoH and the next step's HtoD; main thread in engram.py(1747) stage waiting on 16 _fast_stage_one futures (_gather_dequant_v2 ~20% of the gap per thread); census read 0.6-1.3 ms/call, pf_hit 100%",
         "recoverable_ms_per_step": {"upper": engram_gap["ms_per_step"]},
         "note": "not a GPU kernel, but the largest single idle block; GPU-side dequant of raw fp8 rows, fewer preadv calls, or starting the accepted-prefix rows before the draft finishes"},
        {"rank": 5, "item": "mHC hyper-connection kernels (prenorm GEMM + pre fuse + post)",
         "now_ms_per_step": round(mhc_ms, 2),
         "facts": f"hc_prenorm {r0['hc_prenorm']['us']['median']} us x 86 = {r0['hc_prenorm']['ms_per_step']} ms at {r0['hc_prenorm']['pct_of_peak']}% of peak; exposed glue AR->prenorm {gl('nccl_allreduce', 'hc_prenorm')} + prenorm->qkv_a {gl('hc_prenorm', 'qkv_a')} + prenorm->router {gl('hc_prenorm', 'router_gate')} ms/step",
         "recoverable_ms_per_step": {"prenorm_roofline": r0["hc_prenorm"]["recoverable_ms_per_step"],
                                     "fuse_post_prenorm_pre_exposed": round(gl("nccl_allreduce", "hc_prenorm") + gl("hc_prenorm", "qkv_a") + gl("hc_prenorm", "router_gate"), 2)},
         "note": "DSV41_MHC_DECODE_SPLITS=40 was rejected in R34; a fused post+prenorm+pre kernel is the untried path"},
        {"rank": 6, "item": "Eager next-step prep chain (host-launch bound)",
         "now_ms_per_step": max((c["exposed_ms_per_step"] for c in g1[0]["glue_chains"] if c["launches_per_step"] > 200 and c["count_per_step"] <= 1.1), default=0.0),
         "facts": "~258 tiny eager kernels per step at ~4 us host cadence after the draft graph (sparse-Markov gather/argmax, attention metadata, SWA indices, cub scans) before the target embed AR",
         "recoverable_ms_per_step": {"upper": 1.0}, "note": "capture into the draft graph or fuse"},
        {"rank": 7, "item": "wo_a fp8 einsum", "now_ms_per_step": r0["wo_a"]["ms_per_step"],
         "facts": f"{r0['wo_a']['us']['median']} us x 43 at {r0['wo_a']['pct_of_peak']}%",
         "recoverable_ms_per_step": {"roofline": r0["wo_a"]["recoverable_ms_per_step"]}},
        {"rank": 8, "item": "In-graph glue to fold into neighbours", "now_ms_per_step": round(gl("routed_moe_p2b", "nccl_allreduce") + gl("indexer_wq_b", "sparse_mla"), 2),
         "facts": f"p2b -> MoE AR combine (copy, bf16 copy, add; 3 launches/layer) {gl('routed_moe_p2b', 'nccl_allreduce')} ms exposed; indexer chain {gl('indexer_wq_b', 'sparse_mla')} ms exposed",
         "recoverable_ms_per_step": {"upper": round(gl("routed_moe_p2b", "nccl_allreduce") + gl("indexer_wq_b", "sparse_mla"), 2)}},
        {"rank": 9, "item": "Draft path GEMV/GEMMs", "now_ms_per_step": round(r0["draft_router_gate"]["ms_per_step"] + r0["draft_moe_gate_up"]["ms_per_step"] + r0["draft_moe_down"]["ms_per_step"], 2),
         "facts": f"draft router gate cutlass {r0['draft_router_gate']['us']['median']} us at {r0['draft_router_gate']['GBps_at_median']} GB/s (1.3 MB bf16), on the draft critical path; draft MoE deepgemm {r0['draft_moe_gate_up']['us']['median']}+{r0['draft_moe_down']['us']['median']} us",
         "recoverable_ms_per_step": {"router_gate_gemv": r0["draft_router_gate"]["recoverable_ms_per_step"]}},
        {"rank": 10, "item": "Rank straggler (spark1 = head, rank 0)",
         "now_ms_per_step": skew_ps.get("AllReduce_graph_grid5", {}).get("rank1_wait_ms_per_step"),
         "facts": f"rank1 waits {skew_ps.get('AllReduce_graph_grid5', {}).get('rank1_wait_ms_per_step')} ms/step in target ARs vs rank0 {skew_ps.get('AllReduce_graph_grid5', {}).get('rank0_wait_ms_per_step')}; rank0 non-NCCL kernel time {round(g1[0]['kernel_sum_ms'] - cat['nccl_allreduce']['ms_per_step'], 2)} vs rank1 {round(g1[1]['kernel_sum_ms'] - [c for c in g1[1]['categories'] if c['category'] == 'nccl_allreduce'][0]['ms_per_step'], 2)} ms/step; p2b p99 1170 vs 724 us; SM 2171 vs 2190 MHz",
         "recoverable_ms_per_step": {"upper": round(serve_vs_iso["rank0"]["total_excess_ms_per_step"] - serve_vs_iso["rank1"]["total_excess_ms_per_step"], 2)},
         "note": "environmental (head-node host load, and from 07:27 UTC another session's GPU use); keep spark1 free of other GPU/CPU work during serving"},
    ]

    out = {
        "stage": "k3 profile: fresh ground truth for kernel work",
        "git_head": (HERE / "git_head.txt").read_text().strip(),
        "serve": {"worktree": "/home/sfxnz/projects/ai-lab/recipes/.worktrees/kernels-r3", "cmd": "AUDIT=strict EXTRA_ARGS='--profiler-config {\"profiler\":\"torch\",\"torch_profiler_dir\":\"/tmp/dsv41-traces\"}' ./run.sh",
                  "image": "dsv41-flash-exl3-sm121:canonical-e13 sha256:c81762335a12 (both ranks)", "pack": "2.0bpw-mcg-lmhead-mxfp8",
                  "coop": "off (DSV41_P2B_COOP unset)", "audit": "strict: audit ok on head and worker, both boots",
                  "boot1": "run.sh start 08:14:57Z, exit 08:25:47Z (rc 0)", "boot2": "start 08:38:46Z, exit 08:49:04Z (rc 0), for the c=2 window after boot1's profiler kept ~20 GB of host memory"},
        "reference_profiler_idle_boot1": {"lail_prose_n10": lail_summary("04-lail-prose-n10.log"),
                                          "bench_decode_c1_n9": bench_summary("05-bench-decode-c1-n9.log"),
                                          "clocks_during_lail": smi_summary("04-smi-lail"), "clocks_during_bench": smi_summary("05-smi-bench")},
        "jit_after_ready": {"file": "03-jit-check.txt",
                            "tilelang_after_ready": 0, "triton_after_ready_per_rank": 3,
                            "kernels": ["BuildPrefillChunkMetadataKernel.kernel (first smoke request)", "_ring_slot_mapping_kernel (first smoke request)",
                                        "_compute_global_topk_indices_and_lens_kernel (2 concurrent requests)"],
                            "note": "the monitor logs warning_once per kernel name, so counts are distinct kernels; the round-2 mHC n_splits fix holds (TileLang compiled only inside run.sh's warmup)"},
        "c1_lail_window": {"trace_dir": (HERE / "06-trace_out_dir_c1.txt").read_text().strip(), "per_rank": {
            f"rank{r}": {k: g1[r][k] for k in ("steps", "verify_rows_m", "step_wall_ms", "device_busy_ms", "device_idle_ms", "kernel_sum_ms", "phases_ms", "graph_launch_host_us")}
            for r in (0, 1)}, "full_analysis": ["c1/c1-rank0.json", "c1/c1-rank1.json"],
            "profiled_vs_unprofiled_ms_per_step": [g1[0]["step_wall_ms"]["median"], lail_summary("04-lail-prose-n10.log")["ms_per_step"]],
            "collective_skew_per_step": skew_ps, "serve_vs_isolated_microbench": serve_vs_iso,
            "layer_budget": ["c1/c1_layer_budget_r0.txt", "c1/c1_layer_budget_r1.txt"], "phase_budget": ["c1/c1_phase_budget_r0.txt", "c1/c1_phase_budget_r1.txt"],
            "graph_startup": "c1/c1_graph_startup.txt", "p2b_distribution": "c1/c1_p2b_dist.txt"},
        "c2_pair_window": {"trace_dir": (HERE / "11-trace_out_dir_c2.txt").read_text().strip(),
                           "contamination": "spark1's GB10 was shared with another session's Playwright chromium GPU processes (11-hostload-during-c2.log); rank0 p2b median 1524 us (p90 2810) vs rank1 1032 us (p90 1090): rank1 (spark2) kernel times are clean, step-level numbers are not",
                           "max_tokens": 200, "rank1_clean_kernels_m8": {k: {"us_median": v["us"]["median"], "ms_per_step": v["ms_per_step"], "GBps": v.get("GBps_at_median")} for k, v in r2.items()},
                           "full_analysis": ["c2/c2-rank0.json", "c2/c2-rank1.json"]},
        "boot2_contaminated_benches": {"lail_n3": "10-boot2-lail-n3.log (17.15 tok/s)", "bench_c1_n3": "10-boot2-bench-c1-n3.log (106.9 ms/step)",
                                       "bench_c2_n9": "10-bench-decode-c2-n9.log (prose 20.1 tok/s/stream, 128.3 ms/step; structured 31.0, 127.9 ms/step)",
                                       "why_invalid": "same GPU sharing as above; boot-1 numbers match the recipe band, boot-2 numbers are 1.7x slower"},
        "nccl": {"file": "nccl-sweep.json", "best_vs_keep": nccl["best_vs_keep"]["clean_modeled_ms_per_step"], "per_call_clean": nccl["best_vs_keep"]["per_call_clean"]},
        "opportunities_ranked": opp,
        "weight_stream_roofline_c1_rank0": {"gemm_moe_ms": round(sum(v["ms_per_step"] for v in r0.values() if "roofline_us" in v), 2),
                                            "roofline_ms": round(sum(v["calls_per_step"] * v["roofline_us"] / 1e3 for v in r0.values() if "roofline_us" in v), 2)},
    }
    (HERE / "timeline-r3.json").write_text(json.dumps(out, indent=1) + "\n")
    for o in opp:
        print(o["rank"], o["item"], o["now_ms_per_step"], o["recoverable_ms_per_step"])
    print(out["weight_stream_roofline_c1_rank0"])


if __name__ == "__main__":
    main()
