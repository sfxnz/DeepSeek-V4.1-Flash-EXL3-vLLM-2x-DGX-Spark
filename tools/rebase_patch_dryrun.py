#!/usr/bin/env python3
"""Phase A of the vLLM rebase probe: dry-run docker/patch against an image.

Runs INSIDE a throwaway container of the image under test, CPU only:

  docker run --rm --network none -e NVIDIA_VISIBLE_DEVICES=void \
    -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages \
    -v $PWD/docker/patch:/opt/dsv41-patch:ro -v $PWD/tools:/probe:ro -v $OUT:/out \
    --entrypoint python3 IMAGE -S /probe/rebase_patch_dryrun.py --json /out/X.json

`-S` keeps a baked sitecustomize from patching the tree before we look.
The container is --rm, so the writes below never outlive the run.

Steps mirror the build (docker/Dockerfile, Dockerfile.e11) and the runtime
file patches in sitecustomize.py, in the same order, with each failure
recorded instead of swallowed. --retarget links models/deepseek_v4_1 to
models/deepseek_v41 when only the new name exists (v0.30 layout).

Outcomes per step: changed | unchanged | error:<msg> | skipped:<msg>.
On a pristine tree "unchanged" means the anchors did not land.
Then a static check: every `from vllm/flashinfer.X import Y` and every
`Imported.attr = ...` monkeypatch in docker/patch must resolve in the tree.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import json
import re
import subprocess
import sys
from pathlib import Path

SITE = Path("/usr/local/lib/python3.12/dist-packages")
PATCH = Path("/opt/dsv41-patch")
V41 = "models/deepseek_v4_1"

# (label, argv relative to SITE) — docker/Dockerfile + Dockerfile.e11 order.
BUILD_STEPS = [
    ("apply_engram_disk", ["--engram", f"vllm/{V41}/common/engram.py",
                           "--weights", "vllm/model_executor/model_loader/weight_utils.py"]),
    ("apply_engram_prestage", ["--engram", f"vllm/{V41}/common/engram.py",
                               "--model-state", f"vllm/{V41}/nvidia/model_state.py"]),
    ("widen_b12x_smalls", ["flashinfer/gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py"]),
    ("probe_wo_a", ["vllm/models/deepseek_v4/nvidia/ops/o_proj.py"]),
    ("fix_o_proj_woa_fp8", ["vllm/models/deepseek_v4/nvidia/ops/o_proj.py"]),
]

# (label, module, function, args relative to SITE, kind) — sitecustomize.py order.
# kind "apply": call f(*paths); kind "src": file.write_text(f(file.read_text())).
RUNTIME_STEPS = [
    ("prefer_b12x_mxfp8", "prefer_b12x_mxfp8", "apply", ["vllm"], "apply"),
    ("indexer_workspace", "indexer_workspace", "apply", ["vllm"], "apply"),
    ("widen_mla_io2 (off by default)", "widen_mla_io2", "apply", ["flashinfer"], "apply"),
    ("widen_mla_tile32", "widen_mla_tile32", "apply", ["flashinfer"], "apply"),
    ("persistent_topk", "sm120_page", "patch_persistent_topk_source",
     ["vllm/model_executor/layers/sparse_attn_indexer.py"], "src"),
    ("kpool_persistent_topk", "sm120_page", "patch_kpool_persistent_topk_source",
     ["vllm/model_executor/layers/sparse_attn_indexer_kpool.py"], "src"),
    ("swa_prefill_image_width", "sm120_page", "patch_swa_prefill_image_width_source",
     ["vllm/v1/attention/backends/mla/sparse_swa.py"], "src"),
    ("attention_image_width", "sm120_page", "patch_attention_image_width_source",
     [f"vllm/{V41}/attention.py"], "src"),
    ("native_indexer_decode", "sm120_page", "patch_native_indexer_decode_source",
     ["vllm/v1/attention/backends/mla/indexer.py"], "src"),
    ("indexer_adaptive", "sm120_page", "patch_indexer_adaptive_source",
     ["vllm/v1/attention/backends/mla/indexer.py"], "src"),
    ("indexer_short_context", "sm120_page", "patch_indexer_short_context_source",
     [f"vllm/{V41}/attention.py"], "src"),
    ("engram_stage_census", "engram_stage_census", "apply", ["vllm"], "apply"),
    ("engram_stage_fast", "engram_stage_fast", "apply", [f"vllm/{V41}"], "apply"),
    ("engram_prefetch_v3", "engram_prefetch_v3", "apply",
     [f"vllm/{V41}", "vllm/v1/worker/gpu/model_runner.py", "vllm"], "apply"),
    ("engram_cpu_hash", "engram_cpu_hash", "apply",
     [f"vllm/{V41}", "vllm/v1/worker/gpu/model_runner.py"], "apply"),
    ("engram_gather_v2", "engram_gather_v2", "apply", ["vllm"], "apply"),
    ("engram_defer", "engram_defer", "apply",
     [f"vllm/{V41}", "vllm/v1/worker/gpu/model_runner.py", f"vllm/{V41}/nvidia/model_state.py"], "apply"),
]

SKIP_WORDS = ("skip", "missing", "not found", "not present")

# vllm/flashinfer-targeting patches kept in docker/patch but not wired by default.
UNWIRED = ["prefer_b12x_bmm", "widen_mla_kv_buf", "c1_graph_safe_adaptive", "sm120_wo_a", "g8_stream_feed"]
ANCHOR_NAME = re.compile(r"(^OLD|_OLD$|_OLD_|ANCHOR|_TAIL$|^GD_DEF$)")


def tree_digest(root: Path) -> dict[str, str]:
    out = {}
    for sub in ("vllm", "flashinfer"):
        for p in (root / sub).rglob("*"):
            if p.is_file() and p.suffix in (".py", ".cuh", ".cu", ".h"):
                out[str(p.relative_to(root))] = hashlib.md5(p.read_bytes()).hexdigest()
    return out


def outcome(before: dict, after: dict, err: str | None, printed: str) -> tuple[str, list[str]]:
    touched = sorted(k for k in after if before.get(k) != after[k])
    if err:
        return f"error:{err}", touched
    low = printed.lower()
    if not touched and any(w in low for w in SKIP_WORDS):
        return "skipped:" + printed.strip().splitlines()[-1][:200], touched
    return ("changed" if touched else "unchanged"), touched


def run_build(root: Path) -> list[dict]:
    rows = []
    for label, argv in BUILD_STEPS:
        args = [a if a.startswith("--") else str(root / a) for a in argv]
        before = tree_digest(root)
        cp = subprocess.run([sys.executable, "-S", str(PATCH / f"{label}.py"), *args],
                            capture_output=True, text=True)
        err = None if cp.returncode == 0 else (cp.stderr.strip().splitlines() or ["rc!=0"])[-1][:200]
        res, touched = outcome(before, tree_digest(root), err, cp.stdout)
        rows.append({"step": label, "phase": "build", "outcome": res, "touched": touched})
    return rows


def run_runtime(root: Path) -> list[dict]:
    sys.path.insert(0, str(PATCH))
    rows = []
    for label, mod, fn, rel, kind in RUNTIME_STEPS:
        before = tree_digest(root)
        buf = io.StringIO()
        err = None
        try:
            with contextlib.redirect_stdout(buf):
                f = getattr(__import__(mod), fn)
                paths = [root / r for r in rel]
                if kind == "src":
                    if not paths[0].is_file():
                        raise FileNotFoundError(f"{rel[0]} missing")
                    paths[0].write_text(f(paths[0].read_text()))
                else:
                    f(*paths)
        except BaseException as e:  # SystemExit is how most patches refuse
            err = f"{type(e).__name__}: {e}"[:200]
        res, touched = outcome(before, tree_digest(root), err, buf.getvalue())
        rows.append({"step": label, "phase": "runtime", "outcome": res, "touched": touched})
    return rows


def anchors(patch_dir: Path, modules: list[str]) -> list[tuple[str, str, str]]:
    """(module, constant, text) for module-level str anchors (OLD / ANCHOR names)."""
    out = []
    for mod in modules:
        for node in ast.parse((patch_dir / f"{mod}.py").read_text()).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name) and ANCHOR_NAME.search(node.targets[0].id):
                try:
                    val = ast.literal_eval(node.value)
                except (ValueError, TypeError, SyntaxError):  # built from other names
                    continue
                if isinstance(val, str) and val.strip():
                    out.append((mod, node.targets[0].id, val))
    return out


def anchor_scan(root: Path, found_anchors, retarget: bool) -> list[dict]:
    """Where each anchor occurs in the pristine tree (before any patch runs)."""
    files = {}
    for sub in ("vllm", "flashinfer"):
        for p in (root / sub).rglob("*"):
            if p.is_file() and p.suffix in (".py", ".cuh", ".cu", ".h"):
                files[str(p.relative_to(root))] = p.read_text(errors="replace")
    blob = "\0".join(files.values())
    rows = []
    for mod, name, text in found_anchors:
        if retarget:
            text = text.replace("deepseek_v4_1", "deepseek_v41")
        hits = [f for f, t in files.items() if text in t] if text in blob else []
        rows.append({"patch": mod, "anchor": name, "n_files": len(hits), "files": hits[:5]})
    return rows


def module_file(root: Path, dotted: str) -> Path | None:
    base = root.joinpath(*dotted.split("."))
    for cand in (base.with_suffix(".py"), base / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def defined_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(errors="replace"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            names.add(node.attr)
    if (path.parent / "__init__.py") == path:  # package: submodules count
        names.update(p.stem for p in path.parent.glob("*.py"))
        names.update(p.name for p in path.parent.iterdir() if (p / "__init__.py").is_file())
    return names


def static_refs(patch_dir: Path) -> list[tuple[str, str, str]]:
    """(patch file, module, name) for vllm/flashinfer imports and monkeypatched attrs."""
    refs = []
    for p in sorted(patch_dir.glob("*.py")):
        tree = ast.parse(p.read_text())
        alias: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 \
                    and node.module.split(".")[0] in ("vllm", "flashinfer"):
                for a in node.names:
                    refs.append((p.name, node.module, a.name))
                    alias[a.asname or a.name] = f"{node.module}.{a.name}"
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] in ("vllm", "flashinfer"):
                        refs.append((p.name, a.name, ""))
                        if a.asname:
                            alias[a.asname] = a.name
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) \
                            and t.value.id in alias:
                        refs.append((p.name, alias[t.value.id], "." + t.attr))
    return refs


def check_refs(root: Path, refs, retarget: bool) -> list[dict]:
    out = []
    for fname, mod, name in sorted(set(refs)):
        m = mod.replace("deepseek_v4_1", "deepseek_v41") if retarget else mod
        f = module_file(root, m)
        attr = name.startswith(".")
        if f is None and not attr and name:  # `from pkg import submodule`
            f = module_file(root, f"{m}.{name}")
            if f is not None:
                continue
        if f is None and attr:  # alias points at a class: module.Class
            owner, _, cls = m.rpartition(".")
            f = module_file(root, owner)
            status = "ok" if f is not None and name[1:] in defined_names(f) else "missing-attr"
        elif f is None:
            status = "missing-module"
        elif name and not attr and name not in defined_names(f):
            status = "missing-name"
        elif attr and name[1:] not in defined_names(f):
            status = "missing-attr"
        else:
            status = "ok"
        if status != "ok":
            out.append({"patch": fname, "module": m, "name": name, "status": status})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", type=Path, default=SITE)
    ap.add_argument("--retarget", action="store_true",
                    help="symlink models/deepseek_v4_1 -> deepseek_v41 when only the new name exists")
    ap.add_argument("--json", type=Path, required=True)
    args = ap.parse_args(argv)
    root = args.site
    models = root / "vllm/models"
    retargeted = False
    if args.retarget and not (models / "deepseek_v4_1").exists() and (models / "deepseek_v41").is_dir():
        (models / "deepseek_v4_1").symlink_to("deepseek_v41")
        retargeted = True
    # Static checks first: they must see the pristine tree.
    refs = check_refs(root, static_refs(PATCH), args.retarget)
    mods = sorted({b[0] for b in BUILD_STEPS} | {r[1] for r in RUNTIME_STEPS} | set(UNWIRED))
    scan = anchor_scan(root, anchors(PATCH, mods), args.retarget)
    rows = run_build(root) + run_runtime(root)
    report = {"site": str(root), "retargeted": retargeted, "steps": rows,
              "unresolved_refs": refs, "anchors": scan}
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    for r in rows:
        print(f"{r['phase']:7} {r['step']:32} {r['outcome'][:110]}")
    miss = [a for a in scan if not a["files"]]
    print(f"anchors: {len(scan)} scanned, {len(miss)} not found in the pristine tree")
    for a in miss:
        print(f"  miss {a['patch']:24} {a['anchor']}")
    print(f"unresolved vllm/flashinfer refs: {len(refs)}")
    for r in refs:
        print(f"  {r['status']:14} {r['patch']:24} {r['module']} {r['name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
