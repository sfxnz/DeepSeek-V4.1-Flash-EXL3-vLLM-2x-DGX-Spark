#!/usr/bin/env python3
"""Output-quality eval for the DeepSeek-V4.1-Flash EXL3 serve (HTTP only).

Stdlib only. It talks to the OpenAI-compatible serve, prints a JSON summary
line (`QUALITY {...}`), writes the full result with --out, and exits 1 when a
gate fails. Gates are listed in GATE_RULES and applied by gates().

Components (vendored data in tests/quality/, see tests/quality/README.md):
  nll        teacher-forced NLL on 40 fixed ~512-token public-domain passages
             via /v1/completions prompt_logprobs. BOS is prepended and the first
             16 passage tokens are not scored. --full scores the set twice
             and records the repeat noise (repeat_abs_delta); --quick scores
             it once to stay near 8 min. Prefill path only.
  decode     decode-path probe: greedy 128 tokens with logprobs on 8 passage
             prefixes, then the same tokens re-scored through prompt_logprobs.
             Reports the decode-vs-prefill |dlogprob| distribution.
  tools      tools30.json: 22 tool calls with varied schemas and 8 no-tool
             negatives. Reports JSON-valid, exact-args and correct-no-call rates.
  needle     passcode recall at 8k/32k (plus 128k in --full) x depth
             0.1/0.5/0.9. Filler is generated here from a seeded RNG, not taken
             from repo text, and a per-run header busts the prefix cache.
  selfcons   greedy control: 12 prompts x 2 runs. Reports the A/A flip rate
             and, with --baseline, first divergence vs the baseline's run A.
  c2         two concurrent requests that must both be right.
  vision     64x64 solid-red PNG must be answered "red".
  gsm8k      GSM8K-100, temperature 0, thinking off            (--full only)
  gsm8k_think  first 40 GSM8K items, thinking on (effort high) (--full only)
  mmlu       MMLU 4 x 57 subjects                               (--full only)

Chat requests use chat_template_kwargs {thinking: false, reasoning_effort:
low} except gsm8k_think. Concurrency never exceeds 2 (MAX_NUM_SEQS=2); nll,
decode, selfcons and needle run at c=1. Never run this next to a bench.

  python3 tests/quality_eval.py --quick --out q.json
  python3 tests/quality_eval.py --quick --baseline results/2026-09-24-review/quality-baseline/quick.json
  python3 tests/quality_eval.py --result armB.json --baseline armA.json   # offline re-gate
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "quality"
sys.path.insert(0, str(ROOT))

from smoke_vision import RED_PNG_B64  # noqa: E402

BOS_TEXT = "<｜begin▁of▁sentence｜>"
CHAT_KWARGS = {"thinking": False, "reasoning_effort": "low"}
THINK_KWARGS = {"thinking": True, "reasoning_effort": "high"}
NLL_SKIP = 16          # passage tokens after BOS that are never scored
DECODE_PREFIX = 48     # passage tokens fed to the decode probe
DECODE_TOKENS = 128
SELFCONS_TOKENS = 128
DEPTHS = (0.1, 0.5, 0.9)

# Same 12 prompts as results/2026-09-22-lmhead/greedy_capture.py (R30).
SELFCONS_PROMPTS = [
    "17 * 19 = ? Step by step, then answer.",
    "Write a Python function to reverse a linked list iteratively.",
    "Explain why the sky is blue in exactly three sentences.",
    "List the first 20 prime numbers, comma separated.",
    "Translate into French: The quick brown fox jumps over the lazy dog.",
    "What is the capital of Australia? One word.",
    "Summarize the plot of Hamlet in five bullet points.",
    "Write a haiku about GPU memory bandwidth.",
    "Solve for x: 3x^2 - 12x + 9 = 0. Show steps.",
    "Name three uses of safetensors in ML infrastructure.",
    "Continue: Once upon a time in a datacenter far away,",
    "What happens if you divide by zero in IEEE 754 floating point?",
]

QUICK = ["nll", "decode", "tools", "needle", "selfcons", "c2", "vision"]
FULL = QUICK + ["gsm8k", "gsm8k_think", "mmlu"]
NEEDLE_LENGTHS = {"quick": (8192, 32768), "full": (8192, 32768, 131072)}

GATE_RULES = {
    "error": "every component ran without an exception",
    "vision": "answer contains 'red'",
    "c2": "both concurrent answers correct",
    "nll": "mean_nll <= base + max(0.01, 3 * base.repeat_abs_delta) nats",
    "decode.median": "median |dlogprob| <= base + 0.05",
    "decode.gen_nll": "prefill NLL of greedy text <= base + 0.15",
    "selfcons": "golden hazard <= 2 * max(base aa_hazard, 0.005)",
    "selfcons.aa": "A/A hazard <= 2 * max(base aa_hazard, 0.005)",
    "rate": "Wilson 95% upper bound of (k+1)/n >= base rate (one item of slack)",
    "needle": "found count >= base found count on shared cells",
}


# ----------------------------------------------------------------- pure logic

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes out of n."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else 0.0,
            "ci95": [round(lo, 4), round(hi, 4)]}


def token_logprobs(prompt_logprobs: list, ids: list[int], start: int) -> list[float]:
    """Logprob of each actual prompt token from position start on."""
    out = []
    for i in range(start, len(ids)):
        entry = prompt_logprobs[i]
        out.append(float(entry[str(ids[i])]["logprob"]))
    return out


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    pos = (len(s) - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def extract_gsm_answer(text: str) -> str | None:
    """Number after the last 'Answer:', else the last number in the text."""
    num = r"-?\d[\d,]*(?:\.\d+)?"
    hits = re.findall(r"answer\s*[:：]\s*\**\s*\$?\s*(" + num + ")", text, re.I)
    if not hits:
        hits = re.findall(num, text)
    if not hits:
        return None
    return hits[-1].replace(",", "")


def gsm_correct(text: str, answer: str) -> bool:
    got = extract_gsm_answer(text)
    if got is None:
        return False
    try:
        return abs(float(got) - float(answer)) < 1e-6
    except ValueError:
        return False


def extract_letter(text: str) -> str | None:
    m = re.search(r"\b([ABCD])\b", text.strip())
    return m.group(1) if m else None


def values_equal(got, want) -> bool:
    if isinstance(want, bool) or isinstance(got, bool):
        return isinstance(got, bool) and isinstance(want, bool) and got == want
    if isinstance(want, (int, float)):
        return isinstance(got, (int, float)) and abs(got - want) < 1e-9
    if isinstance(want, str):
        return isinstance(got, str) and got.strip().casefold() == want.strip().casefold()
    if isinstance(want, list):
        return (isinstance(got, list) and len(got) == len(want)
                and all(values_equal(g, w) for g, w in zip(got, want)))
    if isinstance(want, dict):
        return isinstance(got, dict) and args_match(got, want)
    return got == want


def args_match(got: dict, want: dict, ignore=()) -> bool:
    """Same keys (minus ignore) and equal values. Strings compare casefolded."""
    got = {k: v for k, v in got.items() if k not in ignore}
    return set(got) == set(want) and all(values_equal(got[k], want[k]) for k in want)


def score_tool_item(item: dict, msg: dict) -> dict:
    calls = msg.get("tool_calls") or []
    row = {"id": item["id"], "n_calls": len(calls)}
    expect = item["expect"]
    if calls:
        fn = calls[0].get("function") or {}
        row["name"] = fn.get("name")
        try:
            args = json.loads(fn.get("arguments") or "")
            row["json_valid"] = isinstance(args, dict)
        except (TypeError, ValueError):
            args, row["json_valid"] = None, False
        row["args"] = args if row["json_valid"] else fn.get("arguments")
    if expect is None:
        row["no_call_ok"] = not calls
        return row
    row["exact"] = bool(calls and row["json_valid"]
                        and row["name"] == expect["name"]
                        and args_match(row["args"], expect["args"], item.get("ignore", ())))
    return row


def summarize_tools(rows: list[dict]) -> dict:
    called = [r for r in rows if r["n_calls"]]
    pos = [r for r in rows if "exact" in r]
    neg = [r for r in rows if "no_call_ok" in r]
    return {
        "json_valid": rate(sum(r["json_valid"] for r in called), len(called)),
        "exact_args": rate(sum(r["exact"] for r in pos), len(pos)),
        "no_call": rate(sum(r["no_call_ok"] for r in neg), len(neg)),
        "items": rows,
    }


def first_divergence(a: list, b: list) -> int | None:
    """First index where a and b differ; None when identical."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def hazard(pairs: list[tuple[list, list]]) -> dict:
    """Per-token divergence hazard: diverged pairs / tokens at risk."""
    events, at_risk, divs = 0, 0, []
    for a, b in pairs:
        d = first_divergence(a, b)
        divs.append(d)
        if d is None:
            at_risk += len(a)
        else:
            events += 1
            at_risk += d + 1
    return {"hazard": round(events / at_risk, 5) if at_risk else 0.0,
            "diverged": events, "pairs": len(pairs), "first_div": divs}


# --------------------------------------------------------- novel filler text

_ADJ = ("amber brittle copper dusky eager feral gilded hollow ivory jagged "
        "knotted languid mossy narrow ochre pallid quiet russet silent tawny "
        "umber velvet wary woven yellowed zealous ashen bleak crimson dim").split()
_NOUN = ("lantern orchard ledger kettle bridge harbor quarry meadow chimney "
         "anvil cellar compass lighthouse granary loom mill parcel quill ridge "
         "saddle tannery thicket vault wagon well workshop barge bellows "
         "cistern ferry").split()
_VERB = ("mended carried weighed painted counted guarded sealed traded "
         "repaired measured polished hid copied lifted buried sketched "
         "hauled borrowed wrapped tended").split()
_ADV = ("slowly", "quietly", "twice", "again", "carefully", "briskly",
        "at dawn", "before supper", "after the rain", "in secret")
_SYL = ("ka lo mir ven tas ob rune fel dra is quo zen hal por ith gam "
        "sel bri vo tur ney ash cor").split()


def _name(rng: random.Random) -> str:
    return " ".join("".join(rng.choice(_SYL) for _ in range(rng.randint(2, 3))).capitalize()
                    for _ in range(2))


def _sentence(rng: random.Random) -> str:
    t = rng.randrange(4)
    a, n, v = rng.choice(_ADJ), rng.choice(_NOUN), rng.choice(_VERB)
    if t == 0:
        return f"{_name(rng)} {v} the {a} {n} {rng.choice(_ADV)}."
    if t == 1:
        return (f"In {_name(rng).split()[0]}, the {a} {n} was {v} by "
                f"{rng.randint(2, 97)} workers near the {rng.choice(_NOUN)}.")
    if t == 2:
        return (f"Nobody recalled why the {n} beside the {rng.choice(_ADJ)} "
                f"{rng.choice(_NOUN)} had been {v}.")
    return (f"The ledger lists {rng.randint(3, 999)} {n}s, each {a}, and "
            f"{_name(rng)} {v} them {rng.choice(_ADV)}.")


def filler_paragraphs(seed: int, count: int) -> list[str]:
    """Deterministic nonsense-but-grammatical paragraphs (not repo text)."""
    rng = random.Random(seed)
    return [" ".join(_sentence(rng) for _ in range(rng.randint(4, 7)))
            for _ in range(count)]


def needle_code(seed: int) -> tuple[str, str]:
    rng = random.Random(seed * 7919 + 1)
    letters = "BCDFGHJKLMNPQRSTVWXZ"
    code = (f"{''.join(rng.choice(letters) for _ in range(3))}-"
            f"{rng.randint(1000, 9999)}-{''.join(rng.choice(letters) for _ in range(2))}")
    return _name(rng), code


def needle_doc(paras: list[str], depth: float, name: str, code: str,
               header: str) -> str:
    idx = max(1, min(len(paras) - 1, int(len(paras) * depth)))
    note = f"The vault code assigned to {name} is {code}. It was never written down again."
    return "\n\n".join([header] + paras[:idx] + [note] + paras[idx:])


# -------------------------------------------------------------------- gates

def _get(d: dict | None, path: str):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def gates(cur: dict, base: dict | None) -> list[dict]:
    """Gate rows for a run. Relative gates need --baseline."""
    rows = []

    def add(gate, ok, value, limit, rule):
        rows.append({"gate": gate, "pass": bool(ok), "value": value,
                     "limit": limit, "rule": GATE_RULES[rule]})

    comps = cur.get("components", {})
    for name, comp in comps.items():
        if "error" in comp:
            add(f"{name}.error", False, comp["error"][:200], None, "error")
    if "pass" in comps.get("vision", {}):
        add("vision", comps["vision"]["pass"], comps["vision"].get("content"), "red", "vision")
    if "pass" in comps.get("c2", {}):
        add("c2", comps["c2"]["pass"], comps["c2"].get("answers"), None, "c2")
    if not base:
        return rows
    bc = base.get("components", {})

    v, b = _get(comps, "nll.mean_nll"), _get(bc, "nll.mean_nll")
    if v is not None and b is not None:
        lim = b + max(0.01, 3 * (_get(bc, "nll.repeat_abs_delta") or 0.0))
        add("nll.mean_nll", v <= lim, v, round(lim, 5), "nll")
    for key, slack, rule in (("median_abs_dlogprob", 0.05, "decode.median"),
                             ("gen_nll_prefill", 0.15, "decode.gen_nll")):
        v, b = _get(comps, f"decode.{key}"), _get(bc, f"decode.{key}")
        if v is not None and b is not None:
            add(f"decode.{key}", v <= b + slack, v, round(b + slack, 5), rule)
    # Floor from the baseline only: a candidate that adds nondeterminism must
    # not raise its own limit.
    b_aa = _get(bc, "selfcons.aa.hazard")
    lim = 2 * max(b_aa or 0.0, 0.005)
    v = _get(comps, "selfcons.golden.hazard")
    if v is not None:
        add("selfcons.golden_hazard", v <= lim, v, round(lim, 5), "selfcons")
    v = _get(comps, "selfcons.aa.hazard")
    if v is not None and b_aa is not None:
        add("selfcons.aa_hazard", v <= lim, v, round(lim, 5), "selfcons.aa")
    for path in ("tools.json_valid", "tools.exact_args", "tools.no_call",
                 "gsm8k.acc", "gsm8k_think.acc", "mmlu.acc"):
        v, b = _get(comps, path), _get(bc, path)
        if v and b and v.get("n") and b.get("n"):
            # One item of slack: at a 100% baseline the plain upper bound
            # fails on a single miss, and one greedy flip is A/A noise.
            hi = wilson(min(v["k"] + 1, v["n"]), v["n"])[1]
            add(path, hi >= b["rate"], v["rate"], b["rate"], "rate")
    vc, bcells = _get(comps, "needle.cells"), _get(bc, "needle.cells")
    if vc and bcells:
        shared = sorted(set(vc) & set(bcells))
        if shared:
            got = sum(vc[k]["found"] for k in shared)
            want = sum(bcells[k]["found"] for k in shared)
            add("needle.found", got >= want, got, want, "needle")
    return rows


# ------------------------------------------------------------------- client

class Client:
    def __init__(self, url: str, model: str | None, timeout: int = 1800):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.models = self.get("/v1/models")
        self.model = model or self.models["data"][0]["id"]

    def get(self, path: str):
        with urllib.request.urlopen(self.url + path, timeout=60) as r:
            body = r.read().decode()
        return body if path == "/metrics" else json.loads(body)

    def post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            self.url + path, data=json.dumps({"model": self.model, **body}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode()[:300]}") from exc

    def chat(self, content, *, max_tokens: int, kwargs=CHAT_KWARGS, **extra) -> dict:
        return self.post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": kwargs, **extra})

    def count(self, text: str) -> int:
        return int(self.post("/tokenize", {"prompt": text})["count"])

    def prefix_hits(self) -> float | None:
        try:
            m = re.search(r"^vllm:prefix_cache_hits_total\{[^}]*\} ([0-9.e+]+)$",
                          self.get("/metrics"), re.M)
            return float(m.group(1)) if m else None
        except Exception:  # noqa: BLE001
            return None


def _pmap(fn, items, workers: int = 2):
    with cf.ThreadPoolExecutor(max_workers=min(2, workers)) as ex:
        return list(ex.map(fn, items))


def _jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / name).read_text().splitlines() if line]


# --------------------------------------------------------------- components

def run_nll(c: Client, repeats: int) -> dict:
    passages = _jsonl("nll_passages.jsonl")
    hits0 = c.prefix_hits()
    runs = []
    for _ in range(repeats):
        per = []
        for p in passages:
            out = c.post("/v1/completions", {"prompt": BOS_TEXT + p["text"], "max_tokens": 1,
                                             "temperature": 0, "prompt_logprobs": 0,
                                             "return_token_ids": True})
            ch = out["choices"][0]
            ids = ch["prompt_token_ids"]
            lps = token_logprobs(ch["prompt_logprobs"], ids, 1 + NLL_SKIP)
            per.append({"id": p["id"], "n": len(lps), "nll": -sum(lps)})
        runs.append(per)
    first = runs[0]
    tok = sum(r["n"] for r in first)
    means = [sum(r["nll"] for r in run) / tok for run in runs]
    res = {"passages": len(first), "tokens_scored": tok, "bos_id": ids[0],
           "mean_nll": round(means[0], 6), "ppl": round(math.exp(means[0]), 4),
           "run_means": [round(m, 6) for m in means],
           "per_passage": {r["id"]: round(r["nll"] / r["n"], 5) for r in first}}
    if repeats > 1:
        res["repeat_abs_delta"] = round(max(abs(m - means[0]) for m in means), 6)
        res["repeat_max_passage_delta"] = round(max(
            abs(a["nll"] / a["n"] - b["nll"] / b["n"])
            for run in runs[1:] for a, b in zip(first, run)), 6)
    hits1 = c.prefix_hits()
    if hits0 is not None and hits1 is not None:
        res["prefix_cache_hits_delta"] = hits1 - hits0
    return res


def run_decode(c: Client) -> dict:
    passages = _jsonl("nll_passages.jsonl")[::5]
    deltas, gen_pref, gen_dec, per = [], [], [], []
    for p in passages:
        ids = c.post("/tokenize", {"prompt": BOS_TEXT + p["text"]})["tokens"][:1 + DECODE_PREFIX]
        out = c.post("/v1/completions", {"prompt": ids, "max_tokens": DECODE_TOKENS,
                                         "temperature": 0, "logprobs": 0,
                                         "return_token_ids": True})
        ch = out["choices"][0]
        gen = ch["token_ids"]
        dec = [float(x) for x in ch["logprobs"]["token_logprobs"]]
        n = min(len(gen), len(dec))
        gen, dec = gen[:n], dec[:n]
        re_out = c.post("/v1/completions", {"prompt": ids + gen, "max_tokens": 1,
                                            "temperature": 0, "prompt_logprobs": 0,
                                            "return_token_ids": True})
        rc = re_out["choices"][0]
        pre = token_logprobs(rc["prompt_logprobs"], rc["prompt_token_ids"], len(ids))
        d = [abs(a - b) for a, b in zip(dec, pre)]
        deltas += d
        gen_pref += pre
        gen_dec += dec
        per.append({"id": p["id"], "tokens": n, "max_abs": round(max(d, default=0.0), 4)})
    return {"prompts": len(passages), "tokens": len(deltas),
            "median_abs_dlogprob": round(percentile(deltas, 0.5), 5),
            "p99_abs_dlogprob": round(percentile(deltas, 0.99), 5),
            "mean_abs_dlogprob": round(sum(deltas) / max(1, len(deltas)), 5),
            "gen_nll_prefill": round(-sum(gen_pref) / max(1, len(gen_pref)), 5),
            "gen_nll_decode": round(-sum(gen_dec) / max(1, len(gen_dec)), 5),
            "per_prompt": per}


def run_tools(c: Client) -> dict:
    spec = json.loads((DATA / "tools30.json").read_text())

    def one(item):
        out = c.chat(item["prompt"], max_tokens=256, tool_choice="auto",
                     tools=[spec["tools"][t] for t in item["tools"]])
        return score_tool_item(item, out["choices"][0]["message"])

    return summarize_tools(_pmap(one, spec["items"]))


def run_needle(c: Client, lengths, nonce: str) -> dict:
    cells = {}
    ratio = None
    for li, length in enumerate(lengths):
        for di, depth in enumerate(DEPTHS):
            seed = 1000 * (li + 1) + di
            name, code = needle_code(seed)
            header = f"Archive {nonce}-{seed}. Field notes follow."
            if ratio is None:
                sample = "\n\n".join(filler_paragraphs(1, 40))
                ratio = c.count(sample) / len(sample)
            budget = length - 64
            n = max(4, int(budget / ratio / 480))
            paras = filler_paragraphs(seed, int(n * 1.3))
            for _ in range(6):   # land in [0.97, 1.0] x budget
                ntok = c.count(needle_doc(paras[:n], depth, name, code, header))
                if 0.97 * budget <= ntok <= budget:
                    break
                n = max(4, int(n * budget / ntok * (0.985 if ntok > budget else 1.0)))
                if n > len(paras):
                    paras = filler_paragraphs(seed, int(n * 1.3))
            doc = needle_doc(paras[:n], depth, name, code, header)
            q = (f"{doc}\n\nWhat is the vault code assigned to {name}? "
                 "Reply with only the code.")
            t0 = time.time()
            out = c.chat(q, max_tokens=32)
            text = out["choices"][0]["message"].get("content") or ""
            cells[f"{length}@{depth}"] = {
                "found": code in text.upper(), "prompt_tokens": out["usage"]["prompt_tokens"],
                "s": round(time.time() - t0, 1), "answer": text.strip()[:60], "code": code}
    return {"found": sum(v["found"] for v in cells.values()), "total": len(cells),
            "cells": cells}


def run_selfcons(c: Client) -> dict:
    runs = []
    for _ in range(2):
        seqs = []
        for p in SELFCONS_PROMPTS:
            out = c.chat(p, max_tokens=SELFCONS_TOKENS, return_token_ids=True)
            seqs.append(out["choices"][0]["token_ids"])
        runs.append(seqs)
    aa = hazard(list(zip(runs[0], runs[1])))
    return {"prompts": len(SELFCONS_PROMPTS), "max_tokens": SELFCONS_TOKENS,
            "identical": aa["pairs"] - aa["diverged"], "aa": aa, "runs": runs}


def add_golden(selfcons: dict, base: dict | None) -> None:
    golden = _get(base, "components.selfcons.runs")
    if not golden or "runs" not in selfcons:
        return
    g = golden[0]
    pairs = [(s, g[i]) for run in selfcons["runs"] for i, s in enumerate(run)]
    selfcons["golden"] = hazard(pairs)


def run_c2(c: Client) -> dict:
    qs = [("What is 17*19? Return only the integer.", "323"),
          ("What is 21*12? Return only the integer.", "252")]
    barrier = threading.Barrier(2)

    def one(q):
        barrier.wait()
        out = c.chat(q[0], max_tokens=32)
        return out["choices"][0]["message"].get("content") or ""

    answers = _pmap(one, qs)
    return {"pass": all(want in a for a, (_, want) in zip(answers, qs)),
            "answers": [a.strip()[:40] for a in answers]}


def run_vision(c: Client) -> dict:
    out = c.chat([
        {"type": "text", "text": "What color is this image? Reply with one word only."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{RED_PNG_B64}"}},
    ], max_tokens=16)
    text = out["choices"][0]["message"].get("content") or ""
    return {"pass": "red" in text.lower(), "content": text.strip()[:40]}


GSM_SUFFIX = ("\n\nSolve the problem step by step. End with a final line of the "
              "form 'Answer: <integer>'.")


def run_gsm(c: Client, n: int, think: bool) -> dict:
    items = _jsonl("gsm8k_100.jsonl")[:n]

    def one(it):
        out = c.chat(it["question"] + GSM_SUFFIX, max_tokens=4096 if think else 1024,
                     kwargs=THINK_KWARGS if think else CHAT_KWARGS)
        text = out["choices"][0]["message"].get("content") or ""
        return {"idx": it["idx"], "ok": gsm_correct(text, it["answer"]),
                "got": extract_gsm_answer(text), "want": it["answer"],
                "completion_tokens": out["usage"]["completion_tokens"]}

    rows = _pmap(one, items)
    return {"acc": rate(sum(r["ok"] for r in rows), len(rows)),
            "mean_completion_tokens": round(sum(r["completion_tokens"] for r in rows) / len(rows), 1),
            "misses": [r for r in rows if not r["ok"]]}


def run_mmlu(c: Client) -> dict:
    items = _jsonl("mmlu_228.jsonl")

    def one(it):
        opts = "\n".join(f"{l}. {t}" for l, t in zip("ABCD", it["choices"]))
        q = (f"The following is a multiple choice question about "
             f"{it['subject'].replace('_', ' ')}.\n\n{it['question']}\n{opts}\n\n"
             "Answer with only the letter A, B, C or D.")
        out = c.chat(q, max_tokens=8)
        got = extract_letter(out["choices"][0]["message"].get("content") or "")
        return {"subject": it["subject"], "row": it["row"], "ok": got == it["answer"], "got": got}

    rows = _pmap(one, items)
    return {"acc": rate(sum(r["ok"] for r in rows), len(rows)),
            "misses": [f"{r['subject']}/{r['row']}:{r['got']}" for r in rows if not r["ok"]]}


# --------------------------------------------------------------------- main

def provenance(c: Client) -> dict:
    info = {"model": c.model, "root": c.models["data"][0].get("root"),
            "max_model_len": c.models["data"][0].get("max_model_len")}
    try:
        raw = subprocess.run(
            ["docker", "inspect", "dsv41-flash-exl3", "--format",
             "{{.Config.Image}}\t{{json .Config.Env}}\t{{json .Config.Cmd}}"],
            capture_output=True, text=True, timeout=20, check=True).stdout.strip()
        image, env, cmd = raw.split("\t")
        keep = sorted(e for e in json.loads(env) if e.split("=")[0].startswith(("DSV41_", "VLLM_", "NCCL_")))
        info["image"] = image
        info["flags_sha256"] = hashlib.sha256(json.dumps([keep, json.loads(cmd)]).encode()).hexdigest()[:16]
    except Exception as exc:  # noqa: BLE001
        info["image"] = f"unavailable ({type(exc).__name__})"
    return info


def run_components(args, comps: list[str], mode_name: str) -> dict:
    c = Client(args.url, args.model)
    nonce = f"{time.time_ns() % 10**9:09d}"
    runners = {
        "nll": lambda: run_nll(c, repeats=2 if args.full else 1),
        "decode": lambda: run_decode(c),
        "tools": lambda: run_tools(c),
        "needle": lambda: run_needle(c, NEEDLE_LENGTHS[mode_name], nonce),
        "selfcons": lambda: run_selfcons(c),
        "c2": lambda: run_c2(c),
        "vision": lambda: run_vision(c),
        "gsm8k": lambda: run_gsm(c, 100, think=False),
        "gsm8k_think": lambda: run_gsm(c, 40, think=True),
        "mmlu": lambda: run_mmlu(c),
    }
    result = {"schema": 1, "mode": mode_name, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "url": args.url, "provenance": provenance(c), "nonce": nonce,
              "baseline": str(args.baseline) if args.baseline else None, "components": {}}
    t_all = time.time()
    for name in comps:
        t0 = time.time()
        try:
            res = runners[name]()
        except Exception as exc:  # noqa: BLE001
            res = {"error": f"{type(exc).__name__}: {exc}"}
        res["s"] = round(time.time() - t0, 1)
        result["components"][name] = res
        print(f"[{name}] {res['s']}s {_brief(name, res)}", flush=True)
    result["elapsed_s"] = round(time.time() - t_all, 1)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="default mode (~8 min)")
    mode.add_argument("--full", action="store_true", help="adds GSM8K, MMLU, 128k needle")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None, help="default: first id in /v1/models")
    ap.add_argument("--baseline", type=Path, help="JSON from an earlier run to gate against")
    ap.add_argument("--out", type=Path, help="write the full result JSON here")
    ap.add_argument("--only", help="comma list of components to run (debugging)")
    ap.add_argument("--result", type=Path,
                    help="gate this saved result JSON against --baseline; sends no traffic")
    args = ap.parse_args(argv)

    mode_name = "full" if args.full else "quick"
    comps = FULL if args.full else QUICK
    if args.only:
        comps = [x for x in args.only.split(",") if x]
        unknown = set(comps) - set(FULL)
        if unknown:
            ap.error(f"unknown components: {sorted(unknown)}")
    base = json.loads(args.baseline.read_text()) if args.baseline else None
    if args.result:
        result = json.loads(args.result.read_text())
        result["baseline"] = str(args.baseline) if args.baseline else None
    else:
        result = run_components(args, comps, mode_name)
    sc = result["components"].get("selfcons")
    if sc:
        sc.pop("golden", None)
        add_golden(sc, base)
    result["gates"] = gates(result, base)
    result["pass"] = all(g["pass"] for g in result["gates"])
    for g in result["gates"]:
        print(f"{'PASS' if g['pass'] else 'FAIL'} {g['gate']} value={g['value']} limit={g['limit']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + "\n")
    print("QUALITY", json.dumps(headline(result)))
    return 0 if result["pass"] else 1


def headline(result: dict) -> dict:
    c = result["components"]
    h = {"mode": result["mode"], "pass": result["pass"], "elapsed_s": result["elapsed_s"]}
    for key, path in (("nll", "nll.mean_nll"), ("nll_repeat_delta", "nll.repeat_abs_delta"),
                      ("decode_median_dlp", "decode.median_abs_dlogprob"),
                      ("decode_gen_nll", "decode.gen_nll_prefill"),
                      ("tools_exact", "tools.exact_args.rate"), ("tools_json", "tools.json_valid.rate"),
                      ("tools_nocall", "tools.no_call.rate"), ("needle", "needle.found"),
                      ("selfcons_identical", "selfcons.identical"), ("selfcons_aa_hazard", "selfcons.aa.hazard"),
                      ("selfcons_golden_hazard", "selfcons.golden.hazard"),
                      ("gsm8k", "gsm8k.acc.rate"), ("gsm8k_think", "gsm8k_think.acc.rate"),
                      ("mmlu", "mmlu.acc.rate"), ("c2", "c2.pass"), ("vision", "vision.pass")):
        v = _get(c, path)
        if v is not None:
            h[key] = v
    return h


def _brief(name: str, res: dict) -> str:
    if "error" in res:
        return f"ERROR {res['error'][:200]}"
    skip = {"per_passage", "items", "runs", "cells", "misses", "per_prompt", "aa"}
    return json.dumps({k: v for k, v in res.items() if k not in skip and k != "s"})[:300]


if __name__ == "__main__":
    sys.exit(main())
