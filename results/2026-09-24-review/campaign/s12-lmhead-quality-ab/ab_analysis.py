#!/usr/bin/env python3
"""Offline lm_head A/B analysis for s12 (no serve traffic).

Reads saved quality_eval JSONs and writes ab_analysis.json next to this file:
  - NLL per arm and paired per-passage deltas (stock bf16 head vs mxfp8 head)
  - greedy selfcons cross-boot hazard: stock-vs-mxfp8 pairs (A/B) against
    mxfp8-vs-mxfp8 pairs (control) and each boot's own A/A
  - GSM8K / MMLU / tools paired deltas vs the mxfp8 full baseline
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REV = HERE.parents[1]  # results/2026-09-24-review
sys.path.insert(0, str(REV.parents[1] / "tests"))
from quality_eval import hazard, wilson  # noqa: E402

MXFP8 = {  # label -> (path, bit-exact decode path vs defaults?)
    "baseline-quick": (REV / "quality-baseline/quick.json", True),
    "baseline-full": (REV / "quality-baseline/full.json", True),
    "s2": (REV / "campaign/s2-old-fresh/quality.json", True),
    "s3": (REV / "campaign/s3-B1-new-defaults/quality.json", True),
    "s5": (REV / "campaign/s5-C1-decode-bundle/quality.json", False),  # MHC=40 not bit-exact
    "s6": (REV / "campaign/s6-B2-defaults-e13/quality.json", True),
    "s7": (REV / "campaign/s7-C2-decode-bundle/quality.json", False),  # MHC=40 not bit-exact
    "s8": (REV / "campaign/s8-S-sparse-markov/quality.json", True),
}
STOCK = {
    "s12-quick": HERE / "quality_quick.json",
    "s12-full": HERE / "stock_full.json",
}


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def hz(pairs) -> dict:
    """quality_eval.hazard plus its at-risk token count (same definition)."""
    h = hazard(pairs)
    h["events_at_risk"] = sum(len(a) if d is None else d + 1
                              for (a, _), d in zip(pairs, h["first_div"]))
    return h


def cross(a: dict, b: dict) -> dict:
    ra, rb = a["components"]["selfcons"]["runs"], b["components"]["selfcons"]["runs"]
    pairs = [(x[i], y[i]) for x in ra for y in rb for i in range(len(x))]
    return hz(pairs)


def pooled(rows):
    ev = sum(r["diverged"] for r in rows)
    risk = sum(r["events_at_risk"] for r in rows)
    return {"diverged": ev, "at_risk_tokens": risk, "hazard": round(ev / risk, 5) if risk else None}


def paired_binary(cur_miss: set, base_miss: set, n: int) -> dict:
    """McNemar-style discordant counts and a Newcombe-ish CI for the paired delta."""
    b = len(cur_miss - base_miss)  # base ok, cur wrong
    c = len(base_miss - cur_miss)  # base wrong, cur ok
    d = (c - b) / n  # cur_acc - base_acc
    # Wald CI on paired difference (small-sample: report alongside exact sign test)
    se = math.sqrt(max((b + c) / n - d * d, 0.0) / n) if n else 0.0
    k, m = min(b, c), b + c
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m) if m else 1.0
    return {"n": n, "cur_only_wrong": b, "base_only_wrong": c, "delta_acc": round(d, 4),
            "delta_ci95_wald": [round(d - 1.96 * se, 4), round(d + 1.96 * se, 4)],
            "mcnemar_exact_p": round(p, 4)}


def main() -> int:
    mx = {k: load(p) for k, (p, _) in MXFP8.items()}
    st = {k: load(p) for k, p in STOCK.items()}
    mx = {k: v for k, v in mx.items() if v}
    st = {k: v for k, v in st.items() if v}
    out: dict = {"arms": {}, "nll": {}, "selfcons": {}, "rates": {}}

    for k, v in {**mx, **st}.items():
        c = v["components"]
        out["arms"][k] = {
            "head": "bf16-stock" if k.startswith("s12") else "mxfp8",
            "image": v["provenance"].get("image"), "flags_sha256": v["provenance"].get("flags_sha256"),
            "mode": v["mode"], "nll": c.get("nll", {}).get("mean_nll"),
            "nll_run_means": c.get("nll", {}).get("run_means"),
            "decode_median_dlp": c.get("decode", {}).get("median_abs_dlogprob"),
            "decode_gen_nll_prefill": c.get("decode", {}).get("gen_nll_prefill"),
            "aa_hazard": c.get("selfcons", {}).get("aa", {}).get("hazard"),
            "selfcons_identical": c.get("selfcons", {}).get("identical"),
            "tools_exact": c.get("tools", {}).get("exact_args", {}).get("k"),
        }

    # NLL: every per_passage run, paired per passage.
    def pp(v):
        return v["components"]["nll"]["per_passage"]
    mx_nll = {k: v["components"]["nll"]["run_means"] for k, v in mx.items() if "nll" in v["components"]}
    st_nll = {k: v["components"]["nll"]["run_means"] for k, v in st.items() if "nll" in v["components"]}
    mx_all = [m for ms in mx_nll.values() for m in ms]
    st_all = [m for ms in st_nll.values() for m in ms]
    out["nll"]["mxfp8_run_means"] = mx_nll
    out["nll"]["stock_run_means"] = st_nll
    out["nll"]["mxfp8_mean_of_runs"] = round(sum(mx_all) / len(mx_all), 6)
    out["nll"]["mxfp8_range"] = [min(mx_all), max(mx_all)]
    out["nll"]["stock_mean_of_runs"] = round(sum(st_all) / len(st_all), 6) if st_all else None
    out["nll"]["stock_range"] = [min(st_all), max(st_all)] if st_all else None
    if st_all:
        out["nll"]["delta_stock_minus_mxfp8"] = round(out["nll"]["stock_mean_of_runs"] - out["nll"]["mxfp8_mean_of_runs"], 6)
        same = "s8" if "s8" in mx else None
        if same and "s12-quick" in st:
            a, b = pp(st["s12-quick"]), pp(mx[same])
            ds = [a[k] - b[k] for k in a]
            out["nll"]["paired_per_passage_s12quick_minus_s8"] = {
                "mean_passage_delta": round(sum(ds) / len(ds), 6),
                "stock_lower": sum(d < 0 for d in ds), "stock_higher": sum(d > 0 for d in ds),
                "max_abs": round(max(abs(d) for d in ds), 5)}
        # control: passage deltas between two mxfp8 boots (same config family)
        if "s3" in mx and "s8" in mx:
            a, b = pp(mx["s3"]), pp(mx["s8"])
            ds = [a[k] - b[k] for k in a]
            out["nll"]["control_per_passage_s3_minus_s8"] = {
                "mean_passage_delta": round(sum(ds) / len(ds), 6),
                "max_abs": round(max(abs(d) for d in ds), 5)}

    # Selfcons cross-boot hazards.
    ab, ctrl, ctrl_exact = [], [], []
    names = list(mx) + list(st)
    allj = {**mx, **st}
    matrix = {}
    for x, y in itertools.combinations(names, 2):
        if "selfcons" not in allj[x]["components"] or "selfcons" not in allj[y]["components"]:
            continue
        h = cross(allj[x], allj[y])
        row = {"pair": f"{x}|{y}", "hazard": h["hazard"], "diverged": h["diverged"],
               "pairs": h["pairs"], "events_at_risk": h["events_at_risk"]}
        matrix[row["pair"]] = row
        xs, ys = x.startswith("s12"), y.startswith("s12")
        if xs and ys:
            row["kind"] = "stock-vs-stock"
        elif xs or ys:
            row["kind"] = "A/B stock-vs-mxfp8"
            ab.append(row)
        else:
            row["kind"] = "control mxfp8-vs-mxfp8"
            ctrl.append(row)
            if MXFP8[x][1] and MXFP8[y][1]:
                ctrl_exact.append(row)
    aa_rows = []
    for k, v in allj.items():
        runs = v["components"].get("selfcons", {}).get("runs")
        if runs:
            aa = hz(list(zip(runs[0], runs[1])))
            aa_rows.append({"arm": k, "hazard": aa["hazard"], "diverged": aa["diverged"],
                            "events_at_risk": aa["events_at_risk"]})
    out["selfcons"] = {
        "within_boot_aa": aa_rows,
        "pooled_aa_mxfp8": pooled([r for r in aa_rows if not r["arm"].startswith("s12")]),
        "pooled_aa_stock": pooled([r for r in aa_rows if r["arm"].startswith("s12")]),
        "pooled_cross_control_mxfp8": pooled(ctrl),
        "pooled_cross_control_mxfp8_excl_mhc40": pooled(ctrl_exact),
        "pooled_cross_ab_stock_vs_mxfp8": pooled(ab),
        "ab_vs_s8_same_config": [r for r in ab if "s8" in r["pair"].split("|")],
        "matrix": matrix,
    }

    # Paired rate deltas vs the mxfp8 full baseline.
    bf = mx.get("baseline-full")
    sf = st.get("s12-full")
    if bf and sf:
        for comp in ("gsm8k", "gsm8k_think"):
            if comp in sf["components"] and "acc" in sf["components"][comp]:
                n = sf["components"][comp]["acc"]["n"]
                cm = {m["idx"] for m in sf["components"][comp]["misses"]}
                bm = {m["idx"] for m in bf["components"][comp]["misses"]}
                r = paired_binary(cm, bm, n)
                r.update({"stock": sf["components"][comp]["acc"], "mxfp8": bf["components"][comp]["acc"],
                          "stock_misses": sorted(cm), "mxfp8_misses": sorted(bm)})
                out["rates"][comp] = r
        if "mmlu" in sf["components"] and "acc" in sf["components"]["mmlu"]:
            n = sf["components"]["mmlu"]["acc"]["n"]
            cm = {m.rsplit(":", 1)[0] for m in sf["components"]["mmlu"]["misses"]}
            bm = {m.rsplit(":", 1)[0] for m in bf["components"]["mmlu"]["misses"]}
            r = paired_binary(cm, bm, n)
            r.update({"stock": sf["components"]["mmlu"]["acc"], "mxfp8": bf["components"]["mmlu"]["acc"]})
            out["rates"]["mmlu"] = r
    tools = {}
    for k, v in allj.items():
        t = v["components"].get("tools", {})
        if "items" in t:
            tools[k] = {"exact_k": t["exact_args"]["k"], "n": t["exact_args"]["n"],
                        "misses": sorted(i.get("id") for i in t["items"] if i.get("exact") is False)}
    out["rates"]["tools_exact_by_arm"] = tools

    (HERE / "ab_analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    brief = {k: out["selfcons"][k] for k in out["selfcons"] if k.startswith("pooled")}
    print(json.dumps({"nll": {k: v for k, v in out["nll"].items() if "run_means" not in k},
                      "selfcons": brief,
                      "rates": {k: {kk: vv for kk, vv in v.items() if "misses" not in kk}
                                for k, v in out["rates"].items() if k != "tools_exact_by_arm"},
                      "tools": {k: (v["exact_k"], v["misses"]) for k, v in tools.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
