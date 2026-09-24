#!/usr/bin/env python3
"""Run docker/patch/sitecustomize.py on the host with stub patch modules.

The host has no vLLM, so every real patch block fails its import and is
skipped. Stubs placed first on sys.path record which gated installs ran.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docker/patch/sitecustomize.py"

STUB = '''
def _hit(*a, **k):
    print("STUB {name}", flush=True)
    return True
apply = install = _hit
'''


def run_site(stubs: list[str], **env: str) -> str:
    """Exec sitecustomize with only `env` set; return stdout."""
    with tempfile.TemporaryDirectory() as d:
        for name in stubs:
            (Path(d) / f"{name}.py").write_text(STUB.format(name=name))
        code = (
            "import runpy, sys; sys.path.insert(0, sys.argv[1]); "
            "runpy.run_path(sys.argv[2], run_name='sitecustomize')"
        )
        r = subprocess.run(
            [sys.executable, "-S", "-c", code, d, str(SITE)],
            env={"PATH": os.environ.get("PATH", ""), **env},
            capture_output=True,
            text=True,
            timeout=60,
        )
    if r.returncode != 0:
        raise AssertionError(f"sitecustomize exited {r.returncode}: {r.stderr[-2000:]}")
    return r.stdout


ENGRAM = ["engram_cpu_hash", "engram_defer"]


def setUpModule() -> None:
    # Inside the image the real blocks would rewrite dist-packages files.
    if Path("/usr/local/lib/python3.12/dist-packages/vllm").exists():
        raise unittest.SkipTest("vLLM install present; sitecustomize would patch it")


class EngramCpuHashDeferGateTests(unittest.TestCase):
    """R20/R24 reverted levers: no text install unless one is enabled."""

    def test_off_by_default(self) -> None:
        out = run_site(ENGRAM)
        self.assertNotIn("STUB engram_cpu_hash", out)
        self.assertNotIn("STUB engram_defer", out)

    def test_either_flag_installs_both(self) -> None:
        for flag in ("DSV41_ENGRAM_CPU_HASH", "DSV41_ENGRAM_DEFER"):
            with self.subTest(flag=flag):
                out = run_site(ENGRAM, **{flag: "1"})
                self.assertIn("STUB engram_cpu_hash", out)
                self.assertIn("STUB engram_defer", out)
                self.assertLess(out.index("STUB engram_cpu_hash"), out.index("STUB engram_defer"))

    def test_zero_is_off(self) -> None:
        out = run_site(ENGRAM, DSV41_ENGRAM_CPU_HASH="0", DSV41_ENGRAM_DEFER="0")
        self.assertNotIn("STUB engram", out)

    def test_other_engram_chain_still_installs(self) -> None:
        chain = ["engram_stage_census", "engram_stage_fast", "engram_prefetch_v3", "engram_gather_v2"]
        out = run_site(chain + ENGRAM)
        for name in chain:
            self.assertIn(f"STUB {name}", out)


if __name__ == "__main__":
    unittest.main()
