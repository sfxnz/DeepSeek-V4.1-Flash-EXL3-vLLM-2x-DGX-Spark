#!/usr/bin/env python3
"""tools/engagement_audit.py on canned boot logs (CPU only)."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import engagement_audit as ea  # noqa: E402

DEFAULT_ENV = {
    "DSV41_LMHEAD_MXFP8": "1",
    "DSV41_ENGRAM_PREFETCH": "1",
    "DSV41_ENGRAM_GATHER_V2": "1",
    "DSV41_DROP_PAGE_CACHE": "1",
    "VLLM_USE_BREAKABLE_CUDAGRAPH": "1",
    "ENFORCE_EAGER": "0",
}
# Lines as they appear in the 2026-09-22 live boot logs (both ranks).
ENGAGED = """\
(Worker_TP0 pid=618) dsv41: lm_head mxfp8 enabled (b12x, (64640, 5120))
(Worker_TP0 pid=618) dsv41: engram prefetch v3 armed (ngram=4 heads=8 depth=3 layers=[0, 1])
(Worker_TP0 pid=618) [dsv41-drop-page-cache] dropped page cache of 48 shard files: MemFree 15.78GiB -> 37.55GiB
(Worker_TP0 pid=618) [woa-requant] fp8 einsum engaged: (4, 1024, 4096)
(Worker_TP0 pid=618) [woa-requant] fp8 einsum engaged: (4, 1024, 4096)
(Worker_TP0 pid=618) INFO 09-22 12:03:56 [breakable_cudagraph.py:290] Breakable CUDA graph enabled
(Worker_TP0 pid=618) dsv41: engram gather v2 self-check bit-exact (r=120)
[dsv41-prefill-empty-cache] skipped #1 (longest seq 9000, MemAvailable 3.1 GiB)
"""


class MarkerSourceTests(unittest.TestCase):
    def test_every_marker_is_printed_by_its_patch(self) -> None:
        for name in ea.PATCHES:
            src = (ea.PATCH_DIR / name).read_text()
            found = ea.markers(ea.PATCH_DIR / name)
            self.assertIn("LOG_DISARMED", found, name)
            for marker in found.get("LOG_ENGAGED", ()) + found["LOG_DISARMED"]:
                # once in the constant, at least once where it is printed
                self.assertGreaterEqual(src.count(marker), 2, f"{name}: {marker!r}")

    def test_gated_patches_define_an_engaged_marker(self) -> None:
        for name, gate in ea.PATCHES.items():
            if gate is not None:
                self.assertIn("LOG_ENGAGED", ea.markers(ea.PATCH_DIR / name), name)

    def test_gates_are_forwarded_env_names(self) -> None:
        run = (ROOT / "run.sh").read_text()
        for gate in list(ea.PATCHES.values()) + [ea.BREAKABLE_GATE]:
            for env in gate or {}:
                self.assertIn(f"{env}=", run, env)


class AuditTests(unittest.TestCase):
    def test_live_like_logs_pass_on_both_ranks(self) -> None:
        self.assertEqual(ea.audit({"head": ENGAGED, "worker": ENGAGED}, DEFAULT_ENV), [])

    def test_benign_prefill_skip_is_not_a_disarm(self) -> None:
        self.assertEqual(ea.audit({"head": ENGAGED}, DEFAULT_ENV), [])

    def test_missing_marker_names_rank_and_patch(self) -> None:
        log = ENGAGED.replace("dsv41: lm_head mxfp8 enabled", "")
        problems = ea.audit({"head": ENGAGED, "worker": log}, DEFAULT_ENV)
        self.assertEqual(len(problems), 1, problems)
        self.assertTrue(problems[0].startswith("worker: missing"))
        self.assertIn("lmhead_mxfp8.py", problems[0])

    def test_disarm_lines_are_reported_once(self) -> None:
        bad = ENGAGED + (
            "dsv41: lm_head mxfp8 self-disarmed (no lm_head.weight_scale under /snap)\n"
            "(Worker_TP1 pid=1) dsv41: engram prefetch disabled after 3 errors: IndexError()\n"
            "dsv41: engram gather v2 DISABLED -> stock path: OSError()\n"
            "dsv41-patch FAIL swa_page_coerce: ImportError()\n"
            "dsv41-patch FAIL swa_page_coerce: ImportError()\n"
            "dsv41: drop-page-cache install skipped: ImportError()\n"
        )
        problems = ea.audit({"head": bad}, DEFAULT_ENV)
        self.assertEqual(len(problems), 5, problems)

    def test_decode_lever_off_lines_are_disarms(self) -> None:
        # Printed by docker/patch/decode_levers.py (image dry-run, scenario B).
        bad = ENGAGED + (
            "dsv41: decode lever sparse-markov FAILED, lever is OFF: ValueError('x')\n"
            "dsv41: WARNING DSV41_WOA_PREPACK=1 but this image's o_proj.py has no "
            "prepack stage; the lever is OFF. Build docker/Dockerfile.woa-prepack.\n"
        )
        problems = ea.audit({"head": bad}, DEFAULT_ENV)
        self.assertEqual(len(problems), 2, problems)

    def test_gates_off_expect_nothing(self) -> None:
        env = dict(DEFAULT_ENV, DSV41_LMHEAD_MXFP8="0", ENFORCE_EAGER="1")
        log = ENGAGED.replace("dsv41: lm_head mxfp8 enabled", "").replace("Breakable CUDA graph enabled", "")
        self.assertEqual(ea.audit({"head": log}, env), [])
        expected, _ = ea.expectations({})
        self.assertEqual([label for label, _ in expected], ["fix_o_proj_woa_fp8.py"])

    def test_cli_warn_exits_0_strict_exits_1(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "head.log"
            log.write_text("dsv41: lm_head mxfp8 self-disarmed (no key)\n")
            argv = [sys.executable, str(ROOT / "tools/engagement_audit.py")]
            env = dict(DEFAULT_ENV, PATH="/usr/bin:/bin")
            for mode, rc in (("warn", 0), ("strict", 1), ("off", 0)):
                proc = subprocess.run(argv + ["--mode", mode, f"head={log}"], env=env, capture_output=True, text=True)
                self.assertEqual(proc.returncode, rc, (mode, proc.stderr))
                if mode != "off":
                    self.assertIn("WARNING audit head: dsv41: lm_head mxfp8 self-disarmed", proc.stderr)
            log.write_text(ENGAGED)
            proc = subprocess.run(argv + ["--mode", "strict", f"head={log}"], env=env, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("audit ok", proc.stdout)
            proc = subprocess.run(argv + ["--mode", "strict", f"worker={d}/missing.log"], env=env, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 1)
            self.assertIn("cannot read", proc.stderr)


if __name__ == "__main__":
    unittest.main()
