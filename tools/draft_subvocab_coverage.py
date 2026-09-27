#!/usr/bin/env python3
"""CPU-only feasibility probe for a reduced DSpark draft vocabulary.

Ranks token ids by frequency on a --freq corpus, then reports how much of each
--eval corpus falls inside the top-N ids. Two layouts are scored:

- global: one replicated top-N head (no draft AllGather).
- tp2: each TP rank keeps its top N/2 ids inside its own half of the vocab,
  so each draft shard is a row subset of the local target shard.

The live serve drafts greedily (map_draft_to_target(argmax)), so a target token
outside the sub-vocab can never be proposed and truncates the accepted chain.
accept_len() turns coverage into an acceptance-length estimate under the
independent-position model (conservative: rare tokens are also the ones the
drafter misses anyway).

Sources are "kind:path[:fields]":
  rows:<datasets-server jsonl>:<field,field>   text fields of each row
  ids:<essay_corpus.json>                       token ids, prompt_len stripped
  completions:<results json>                    results.*.completion text
  files:<glob>                                  raw text/code files

Tokenizing needs `tokenizers`; run it inside the recipe image (CPU only,
no --gpus). See results/2026-09-24-review/research/draft-subvocab.md.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

VOCAB = 129280


def rank_ids(counts: np.ndarray) -> np.ndarray:
    """Ids by count, descending; ties broken by lower id (BPE merge order)."""
    return np.lexsort((np.arange(counts.size), -counts))


def topn_global(counts: np.ndarray, n: int) -> np.ndarray:
    return rank_ids(counts)[:n]


def topn_tp(counts: np.ndarray, n: int, tp: int = 2) -> np.ndarray:
    """Top n/tp ids inside each rank's contiguous vocab shard."""
    shard = counts.size // tp
    per = n // tp
    out = []
    for r in range(tp):
        lo = r * shard
        out.append(lo + rank_ids(counts[lo : lo + shard])[:per])
    return np.concatenate(out)


def coverage(ids: np.ndarray, keep: np.ndarray, vocab: int = VOCAB) -> float:
    if ids.size == 0:
        return float("nan")
    mask = np.zeros(vocab, dtype=bool)
    mask[keep] = True
    return float(mask[ids].mean())


def accept_len(alpha: float, k: int, c: float = 1.0) -> float:
    """Tokens per verify step: bonus token + sum_i (alpha*c)^i over k drafts."""
    a = alpha * c
    return 1.0 + sum(a**i for i in range(1, k + 1))


def alpha_for(tau: float, k: int) -> float:
    """Per-position acceptance that gives tau tokens/step at coverage 1."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if accept_len(mid, k) < tau:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _texts(kind: str, path: str, fields: str | None) -> list[str]:
    if kind == "rows":
        keys = (fields or "text").split(",")
        out = []
        for line in Path(path).read_text(errors="replace").splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue  # datasets-server error pages / blank lines
            for r in d.get("rows", []) if isinstance(d, dict) else []:
                out.extend(str(r["row"][k]) for k in keys if r["row"].get(k))
        return out
    if kind == "completions":
        d = json.loads(Path(path).read_text())
        return [v["completion"] for v in d["results"].values() if v.get("completion")]
    if kind == "files":
        return [Path(p).read_text(errors="replace") for p in sorted(glob.glob(path, recursive=True))]
    raise SystemExit(f"unknown source kind {kind!r}")


def load_ids(spec: str, tok) -> np.ndarray:
    kind, _, rest = spec.partition(":")
    path, _, fields = rest.partition(":") if kind == "rows" else (rest, "", "")
    if kind == "ids":
        d = json.loads(Path(path).read_text())
        skip = int(d.get("prompt_len", 0))
        seqs = [s[skip:] for s in d["seqs"]]
        return np.fromiter((t for s in seqs for t in s), dtype=np.int64)
    texts = _texts(kind, path, fields or None)
    encs = tok.encode_batch(texts, add_special_tokens=False)
    return np.fromiter((t for e in encs for t in e.ids), dtype=np.int64)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json from the model snapshot")
    ap.add_argument("--freq", action="append", default=[], help="source spec for the ranking corpus")
    ap.add_argument("--id-order", action="store_true", help="rank by token id only (BPE merge order baseline)")
    ap.add_argument("--eval", action="append", required=True, help="label=source spec (repeatable)")
    ap.add_argument("--n", default="16384,32768,49152,65536")
    ap.add_argument("--tau", type=float, default=2.3, help="current tokens/verify step")
    ap.add_argument("--k", type=int, default=3, help="draft tokens per step (DSpark-3)")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer)
    counts = np.zeros(VOCAB, dtype=np.int64)
    for spec in [] if args.id_order else args.freq:
        ids = load_ids(spec, tok)
        np.add.at(counts, ids[ids < VOCAB], 1)
    ns = [int(x) for x in args.n.split(",")]
    alpha = alpha_for(args.tau, args.k)
    report = {"freq_tokens": int(counts.sum()), "tau": args.tau, "k": args.k, "alpha": alpha, "eval": {}}
    print(f"freq tokens={counts.sum()} alpha={alpha:.4f} (tau={args.tau}, k={args.k})")
    print(f"{'eval':12} {'tokens':>8} {'N':>6} {'cov_global':>10} {'cov_tp2':>8} {'tau_loss_global%':>16}")
    for item in args.eval:
        label, _, spec = item.partition("=")
        ids = load_ids(spec, tok)
        rows = []
        for n in ns:
            cg = coverage(ids, topn_global(counts, n))
            ct = coverage(ids, topn_tp(counts, n))
            loss = 100 * (1 - accept_len(alpha, args.k, cg) / accept_len(alpha, args.k))
            rows.append({"n": n, "cov_global": cg, "cov_tp2": ct, "tau_loss_pct_global": loss})
            print(f"{label:12} {ids.size:8d} {n:6d} {cg:10.4f} {ct:8.4f} {loss:16.2f}")
        report["eval"][label] = {"tokens": int(ids.size), "rows": rows}
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
