#!/usr/bin/env python3
"""Post-ready engagement audit: did the default-on patches engage on every rank?

run.sh saves `docker logs` of the head (and the worker over ssh) after
/v1/models is up and runs:

    env <FORWARD_ENVS values> python3 tools/engagement_audit.py --mode warn \
        head=.run-state/audit-head-<ts>.log worker=.run-state/audit-worker-<ts>.log

The marker strings are the LOG_ENGAGED / LOG_DISARMED constants next to each
patch in docker/patch (read with ast, nothing is imported). A marker is
expected only when its env gate is on. Any disarm line is reported.
--mode warn prints WARNING lines and exits 0; --mode strict exits 1.
"""
from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path

PATCH_DIR = Path(__file__).resolve().parents[1] / "docker" / "patch"

# patch file -> env values that must all hold for LOG_ENGAGED to be expected.
PATCHES = {
    "lmhead_mxfp8.py": {"DSV41_LMHEAD_MXFP8": "1"},
    "engram_prefetch_v3.py": {"DSV41_ENGRAM_PREFETCH": "1"},
    "engram_gather_v2.py": {"DSV41_ENGRAM_GATHER_V2": "1"},
    "fix_o_proj_woa_fp8.py": {},  # baked into the image at build time
    "drop_page_cache.py": {"DSV41_DROP_PAGE_CACHE": "1"},
    "sitecustomize.py": None,  # disarm markers only
    "decode_levers.py": None,  # disarm markers only
}
# vLLM's own line (vllm/compilation/breakable_cudagraph.py:290 in the pinned image).
BREAKABLE_CUDAGRAPH = "Breakable CUDA graph enabled"
BREAKABLE_GATE = {"VLLM_USE_BREAKABLE_CUDAGRAPH": "1", "ENFORCE_EAGER": "0"}


def markers(path: Path) -> dict[str, tuple[str, ...]]:
    """Module-level LOG_ENGAGED / LOG_DISARMED string (or tuple) constants."""
    out: dict[str, tuple[str, ...]] = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", "")
            if name in ("LOG_ENGAGED", "LOG_DISARMED"):
                value = ast.literal_eval(node.value)
                out[name] = (value,) if isinstance(value, str) else tuple(value)
    return out


def expectations(env) -> tuple[list[tuple[str, str]], list[str]]:
    """([(label, engaged marker)] expected under env, [disarm markers])."""
    expected: list[tuple[str, str]] = []
    disarm: list[str] = []
    for name, gate in PATCHES.items():
        found = markers(PATCH_DIR / name)
        disarm.extend(found.get("LOG_DISARMED", ()))
        if gate is not None and all(env.get(k, "") == v for k, v in gate.items()):
            expected.extend((name, m) for m in found["LOG_ENGAGED"])
    if all(env.get(k, "") == v for k, v in BREAKABLE_GATE.items()):
        expected.append(("vllm", BREAKABLE_CUDAGRAPH))
    return expected, disarm


def audit(logs: dict[str, str], env) -> list[str]:
    """Problems found in {rank: log text}; empty means every rank engaged."""
    expected, disarm = expectations(env)
    problems = []
    for rank, text in logs.items():
        for label, marker in expected:
            if marker not in text:
                problems.append(f"{rank}: missing {marker!r} ({label})")
        seen = set()
        for line in text.splitlines():
            if line not in seen and any(m in line for m in disarm):
                seen.add(line)
                problems.append(f"{rank}: {line.strip()[:240]}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("warn", "strict", "off"), default="warn")
    ap.add_argument("logs", nargs="+", metavar="RANK=LOGFILE")
    args = ap.parse_args(argv)
    if args.mode == "off":
        return 0
    logs = {}
    for item in args.logs:
        rank, _, path = item.partition("=")
        try:
            logs[rank] = Path(path).read_text(errors="replace")
        except OSError as exc:
            logs[rank] = ""
            print(f"WARNING audit {rank}: cannot read {path}: {exc}", file=sys.stderr)
    problems = audit(logs, os.environ)
    for p in problems:
        print(f"WARNING audit {p}", file=sys.stderr)
    if not problems:
        print(f"==> audit ok: every expected patch engaged on {', '.join(logs)}")
    return 1 if problems and args.mode == "strict" else 0


if __name__ == "__main__":
    sys.exit(main())
