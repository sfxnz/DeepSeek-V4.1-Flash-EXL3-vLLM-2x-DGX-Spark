#!/usr/bin/env python3
"""Parse four-numbers capture logs into one JSON record.

Usage: four_numbers_parse.py ARM TS OUTDIR MODE   # MODE = complete|partial
       four_numbers_parse.py filter-env < Config.Env JSON > filtered JSON

Called by tools/four_numbers.sh both on success and from the EXIT trap, so a
failed capture still writes partial JSON (whatever logs exist get parsed).
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ENV_KEEP = re.compile(r"^(NCCL|DSV41|VLLM)_")
# Anchored at the end: "PASS" / "TOKEN" mid-name are real levers
# (DSV41_DSPARK_REFINE_PASS, DSV41_PREFILL_EMPTY_CACHE_TOKENS).
ENV_SECRET = re.compile(r"(TOKEN|KEY|SECRET|PASSWORD|PASSWD)$")


def filter_env(env: list[str]) -> list[str]:
    """Serve levers only (NCCL_/DSV41_/VLLM_), secrets dropped, sorted."""
    return sorted(
        e for e in env
        if ENV_KEEP.match(e) and not ENV_SECRET.search(e.split("=", 1)[0])
    )


def env_digest(env: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(env)).encode()).hexdigest()


# Legitimately different on each rank (each node's own fabric address).
ENV_PER_RANK = {"VLLM_HOST_IP"}


def env_rank_diff(a: list[str], b: list[str]) -> list[str]:
    """Names whose values differ between two ranks, per-rank names ignored."""
    da = dict(e.split("=", 1) for e in a)
    db = dict(e.split("=", 1) for e in b)
    return sorted(k for k in set(da) | set(db)
                  if k not in ENV_PER_RANK and da.get(k) != db.get(k))


def parse_time(s: str) -> datetime | None:
    """RFC3339 from docker (ns precision) or the header ts; None if unparseable."""
    m = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(\.\d+)?Z", s.strip())
    if not m:
        return None
    frac = (m.group(2) or ".0")[:7]
    return datetime.fromisoformat(m.group(1) + frac + "+00:00")


def parse_host_state(text: str) -> dict:
    """00-host-state.log -> {node: {uptime_s, <meminfo key>_gib}}."""
    out: dict = {}
    node = None
    for ln in text.splitlines():
        m = re.match(r"^== (spark\d)", ln)
        if m:
            node = m.group(1)
            out[node] = {}
            continue
        if node is None:
            continue
        m = re.match(r"^uptime_s ([0-9.]+)", ln)
        if m:
            out[node]["uptime_s"] = float(m.group(1))
            continue
        m = re.match(r"^([A-Za-z()]+):\s+(\d+) kB", ln)
        if m:
            key = m.group(1).replace("(", "_").replace(")", "")
            out[node][f"{key}_gib"] = round(int(m.group(2)) / 1024 ** 2, 2)
    return out


def parse(arm: str, ts: str, out: Path, mode: str) -> dict:
    res = {
        "arm": arm, "ts": ts, "host": "", "serve_image": None,
        "serve_cmd_digest": None, "prose_median_tok_s": None,
        "prose_runs": [], "acceptance": None, "prefill_8k_tok_s": None,
        "prefill_32k_tok_s": None, "lail_prose_median_tok_s": None,
        "lail_runs": [], "lail_post_eos_fraction": None,
        "memavail_after_prefill_gib_spark1": None,
        "memavail_after_prefill_gib_spark2": None,
        "prefill_novel_8k_tok_s": None, "prefill_novel_32k_tok_s": None,
        "prose_cell": None, "prose_post_eos_fraction": None,
        "prose_median_ms_per_step": None, "prose_long_cells": [],
        "warm_prefix": None, "serve_env": {}, "serve_env_digest": {},
        "serve_env_ranks_match": None, "serve_env_rank_diff": None,
        "serve_started_at": {},
        "serve_uptime_s": None, "host_state": {},
        "notes": "", "partial": mode == "partial",
    }

    def log(name: str) -> str:
        p = out / name
        return p.read_text(errors="replace") if p.exists() else ""

    def last_summary(text: str):
        """JSON value after the last 'SUMMARY ' marker (indented, multiline)."""
        i = text.rfind("SUMMARY ")
        if i < 0:
            return None
        try:
            return json.JSONDecoder().raw_decode(text[i + 8:].strip())[0]
        except json.JSONDecodeError:
            return None

    header = log("00-header.log")
    m = re.search(r"host=(\S+)", header)
    if m:
        res["host"] = m.group(1)

    res["serve_image"] = log("serve_image.txt").strip() or None
    try:
        cmd = json.loads(log("serve_cmd.json") or "[]")
        res["serve_cmd_digest"] = hashlib.sha256(
            json.dumps(cmd).encode()).hexdigest()
    except (json.JSONDecodeError, TypeError):
        pass

    # provenance: filtered env per rank, container start, host state
    for node in ("spark1", "spark2"):
        try:
            env = json.loads(log(f"serve_env_{node}.json") or "null")
        except json.JSONDecodeError:
            env = None
        if env:
            res["serve_env"][node] = env
            res["serve_env_digest"][node] = env_digest(env)
        started = parse_time(log(f"serve_started_{node}.txt"))
        if started:
            res["serve_started_at"][node] = started.isoformat()
    if len(res["serve_env"]) == 2:
        diff = env_rank_diff(res["serve_env"]["spark1"], res["serve_env"]["spark2"])
        res["serve_env_rank_diff"] = diff
        res["serve_env_ranks_match"] = not diff
    t0, s1 = parse_time(ts), res["serve_started_at"].get("spark1")
    if t0 and s1:
        res["serve_uptime_s"] = round(
            (t0 - datetime.fromisoformat(s1)).total_seconds(), 1)
    res["host_state"] = parse_host_state(log("00-host-state.log"))

    # 1) prose decode 9-run median + per-run rates
    m = last_summary(log("01-prose.log"))
    if isinstance(m, list):
        e = next((x for x in m if x.get("phase") == "prose"
                  and x.get("concurrency") == 1), None)
        if e:
            res["prose_median_tok_s"] = round(e["median_decode_tok_s"], 2)
            if "acceptance_len" in e:
                res["acceptance"] = round(e["acceptance_len"], 3)
            res["prose_cell"] = e
            res["prose_post_eos_fraction"] = e.get("post_eos_fraction")
            res["prose_median_ms_per_step"] = e.get("median_ms_per_step")
    for vals in re.findall(
            r"phase=prose c=1 run=\d+.*?per_stream=\[([0-9.,]+)\]",
            log("01-prose.log")):
        res["prose_runs"].extend(float(v) for v in vals.split(",") if v)

    # 2) cold prefill medians (pp phase rows)
    m = last_summary(log("02-micro.log"))
    if isinstance(m, list):
        for row in m:
            # "pp" is the pre-2026-09-24 name of pp_warm
            kind = {"pp": "", "pp_warm": "", "pp_novel": "novel_"}.get(
                row.get("phase"))
            if kind is None:
                continue
            ctx = {8192: "8k", 32768: "32k"}.get(row.get("ctx"))
            if ctx:
                res[f"prefill_{kind}{ctx}_tok_s"] = row["median_rate_tok_s"]

    # 4) MemAvailable GiB — last 'Mem:' line (free -b) per host section
    sections = re.split(r"^== (spark\d).*?$", log("03-mem.log"),
                        flags=re.MULTILINE)
    for i in range(1, len(sections) - 1, 2):
        host, body = sections[i], sections[i + 1]
        mems = [ln for ln in body.splitlines() if ln.startswith("Mem:")]
        if not mems:
            continue
        toks = mems[-1].split()  # free -b: total used free shared buff cache available
        try:
            avail_b = int(toks[-1])
        except ValueError:
            continue
        gib = round(avail_b / 1024 ** 3, 2)
        if host == "spark1":
            res["memavail_after_prefill_gib_spark1"] = gib
        elif host == "spark2":
            res["memavail_after_prefill_gib_spark2"] = gib

    # L.A.I.L prose median + per-run rates; acceptance fallback
    m = last_summary(log("04-lail.log"))
    if isinstance(m, dict):
        res["lail_prose_median_tok_s"] = round(m["median_lail_tok_s"], 2)
        res["lail_post_eos_fraction"] = m.get("post_eos_fraction")
        if res["acceptance"] is None and "median_acceptance_len" in m:
            res["acceptance"] = round(m["median_acceptance_len"], 3)
    for payload in re.findall(r"run=\d+ (\{.*\})", log("04-lail.log")):
        try:
            res["lail_runs"].append(
                round(json.loads(payload)["lail_tok_s"], 2))
        except (json.JSONDecodeError, KeyError):
            pass

    # prose_long c=1/c=2 and warm-prefix cells
    m = last_summary(log("06-prose-long.log"))
    if isinstance(m, list):
        res["prose_long_cells"] = [x for x in m if x.get("phase") == "prose_long"]
    m = last_summary(log("07-warm-prefix.log"))
    if isinstance(m, dict):
        res["warm_prefix"] = m

    res["notes"] = (
        "number 3 (MoE/attention ms per layer at shipped chunk) NOT captured: "
        "no live probe exists; documented fallback = boot knobs "
        "DSV41_STEP_CENSUS=1 / DSV41_ENGRAM_CENSUS=1 (decode-side only; E0 "
        "prefill-flush caveat), real profiling is a separate gated step "
        "(TODO). serve_cmd_digest = sha256(json.dumps(Config.Cmd list)); "
        "serve_env_digest = sha256(json.dumps(sorted NCCL_/DSV41_/VLLM_ env, "
        "secrets dropped)) per rank; serve_env_ranks_match ignores "
        "VLLM_HOST_IP. host_state is read before the benches. "
        "MemAvailable read on both nodes immediately after the 32k prefill, "
        "before the L.A.I.L runs."
    )
    return res


def main() -> int:
    if sys.argv[1:] == ["filter-env"]:
        try:
            env = json.loads(sys.stdin.read() or "[]")
        except json.JSONDecodeError:
            env = []
        print(json.dumps(filter_env(env or []), indent=1))
        return 0
    if len(sys.argv) != 5 or sys.argv[4] not in ("complete", "partial"):
        print(f"usage: {sys.argv[0]} ARM TS OUTDIR MODE", file=sys.stderr)
        return 2
    arm, ts, out, mode = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4]
    res = parse(arm, ts, out, mode)
    dest = out / "four_numbers.json"
    dest.write_text(json.dumps(res, indent=2) + "\n")
    print(f"wrote {dest} (mode={mode})")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
