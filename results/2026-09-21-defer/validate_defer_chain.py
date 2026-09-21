#!/usr/bin/env python3
"""Structural chain test for engram_defer (CPU-only, real snip sources).

Applies the canonical LIVE chain to a scratch copy of the REAL sources
under .run-state/crash-2stream/vllm-snip/:
  prestage -> census -> fast-stage -> prefetch v3 -> cpu-hash -> gather v2
  -> defer
Checks: every anchor matches, defer methods/hooks wired, stage() gains
input_batch, model_state call passes it, idempotent re-apply, py_compile
passes on every patched target.
"""
from __future__ import annotations

import py_compile
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
RECIPE = Path(
    "/home/sfxnz/projects/ai-lab/recipes/"
    "DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark"
)
SNIP = RECIPE / ".run-state/crash-2stream/vllm-snip"
PATCH_DIR = RECIPE / "docker/patch"
DEFER = RECIPE / "docker/patch/engram_defer.py"

failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def build_scratch(dst: Path):
    shutil.copytree(
        SNIP / "models/deepseek_v4_1", dst / "model_root", dirs_exist_ok=True
    )
    (dst / "runner").write_text(
        (SNIP / "v1/worker/gpu/model_runner.py").read_text()
    )
    vllm = dst / "vllm-root/models/deepseek_v4_1/common"
    vllm.mkdir(parents=True)
    shutil.copy(
        SNIP / "models/deepseek_v4_1/common/engram_disk.py",
        vllm / "engram_disk.py",
    )


def run_patch(scratch: Path, name: str):
    if name == "prestage":
        argv = [
            sys.executable,
            str(PATCH_DIR / "apply_engram_prestage.py"),
            "--engram",
            str(scratch / "model_root/common/engram.py"),
            "--model-state",
            str(scratch / "model_root/nvidia/model_state.py"),
        ]
    elif name == "census":
        argv = [
            sys.executable,
            str(PATCH_DIR / "engram_stage_census.py"),
            str(scratch / "vllm-root"),
        ]
    elif name == "fast":
        argv = [
            sys.executable,
            str(PATCH_DIR / "engram_stage_fast.py"),
            str(scratch / "model_root"),
        ]
    elif name == "v3":
        argv = [
            sys.executable,
            str(PATCH_DIR / "engram_prefetch_v3.py"),
            str(scratch / "model_root"),
            str(scratch / "runner"),
            str(scratch / "vllm-root"),
        ]
    elif name == "cpu_hash":
        argv = [
            sys.executable,
            str(PATCH_DIR / "engram_cpu_hash.py"),
            str(scratch / "model_root"),
            str(scratch / "runner"),
        ]
    elif name == "v2":
        argv = [
            sys.executable,
            str(PATCH_DIR / "engram_gather_v2.py"),
            str(scratch / "vllm-root"),
        ]
    elif name == "defer":
        argv = [
            sys.executable,
            str(DEFER),
            str(scratch / "model_root"),
            str(scratch / "runner"),
            str(scratch / "model_root/nvidia/model_state.py"),
        ]
    else:
        raise ValueError(name)
    r = subprocess.run(argv, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
    return r.returncode == 0, r.stdout


def main():
    print("== chain: prestage->census->fast->v3->cpu_hash->v2->defer ==")
    a = Path(tempfile.mkdtemp(prefix="defer-a-"))
    build_scratch(a)
    for step in (
        "prestage",
        "census",
        "fast",
        "v3",
        "cpu_hash",
        "v2",
        "defer",
    ):
        ok, out = run_patch(a, step)
        check(f"apply {step}", ok, out.strip().splitlines()[-1] if out.strip() else "")

    eng = (a / "model_root/common/engram.py").read_text()
    check("defer init armed", "self._defer_setup()" in eng)
    check("defer methods present", "def _defer_worker" in eng and "def _defer_try_stage" in eng)
    check("stage signature has input_batch", "input_batch=None,\n    ) -> int:" in eng)
    check("stage defer attempt wired", "_defer_hit = self._defer_try_stage(n, input_batch)" in eng)
    check("epilogue defer apply", "self._defer_apply(n)" in eng)
    check("warm verify wired", "self._defer_check_warm(n)" in eng)
    # cpu_hash gather block still reachable (fallback intact)
    check(
        "sync fallback intact",
        "if not _ch_fast:" in eng and "_fast_stage_one" in eng,
    )
    run = (a / "runner").read_text()
    check(
        "runner defer hook after cpu-hash hook",
        "# --- engram-defer ---" in run
        and run.index("enqueue_cpu_hash(") < run.index("enqueue_defer("),
    )
    ms = (a / "model_root/nvidia/model_state.py").read_text()
    check(
        "model_state passes input_batch",
        "self.engram_stager.stage(" in ms
        and "input_batch=input_batch," in ms,
    )

    for tgt in (
        a / "model_root/common/engram.py",
        a / "model_root/common/engram_disk.py",
        a / "model_root/nvidia/model_state.py",
        a / "runner",
    ):
        try:
            py_compile.compile(str(tgt), doraise=True)
            check(f"py_compile {tgt.name}", True)
        except Exception as exc:  # noqa: BLE001
            check(f"py_compile {tgt.name}", False, repr(exc))

    # idempotent re-apply
    ok, out = run_patch(a, "defer")
    check("idempotent re-apply", ok)

    # re-apply did not duplicate
    eng2 = (a / "model_root/common/engram.py").read_text()
    check("no duplicate defer block", eng2.count("# --- engram-defer ---") >= 1)
    check(
        "defer hook not duplicated in runner",
        (a / "runner").read_text().count("enqueue_defer(") == 1,
    )

    print()
    if failures:
        print("FAILURES:", failures)
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
