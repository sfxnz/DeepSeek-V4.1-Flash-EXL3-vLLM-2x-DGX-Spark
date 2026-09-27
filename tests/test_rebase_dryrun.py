#!/usr/bin/env python3
"""CPU tests for tools/rebase_patch_dryrun.py."""
from __future__ import annotations

import ast
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import rebase_patch_dryrun as dry  # noqa: E402


class RebaseDryrunTests(unittest.TestCase):
    def test_every_step_names_a_real_patch_and_function(self) -> None:
        patch = ROOT / "docker/patch"
        for label, _ in dry.BUILD_STEPS:
            self.assertTrue((patch / f"{label}.py").is_file(), label)
        for _, mod, fn, _, _ in dry.RUNTIME_STEPS:
            tree = ast.parse((patch / f"{mod}.py").read_text())
            defs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
            self.assertIn(fn, defs, mod)
        for mod in dry.UNWIRED:
            self.assertTrue((patch / f"{mod}.py").is_file(), mod)

    def test_anchor_scan_finds_known_anchors(self) -> None:
        found = {(m, n) for m, n, _ in dry.anchors(ROOT / "docker/patch", ["sm120_page", "apply_engram_disk"])}
        self.assertIn(("sm120_page", "PERSISTENT_TOPK_OLD"), found)
        self.assertIn(("apply_engram_disk", "SIG_OLD"), found)
        self.assertNotIn(("apply_engram_disk", "SIG_NEW"), found)

    def test_outcome(self) -> None:
        self.assertEqual(dry.outcome({"a": "1"}, {"a": "2"}, None, "")[0], "changed")
        self.assertEqual(dry.outcome({"a": "1"}, {"a": "1"}, None, "")[0], "unchanged")
        self.assertTrue(dry.outcome({"a": "1"}, {"a": "1"}, None, "x: missing; skipped\n")[0].startswith("skipped:"))
        self.assertTrue(dry.outcome({}, {}, "SystemExit: anchor", "")[0].startswith("error:"))

    def test_check_refs_on_fake_tree(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            mod = root / "vllm/models/deepseek_v41"
            mod.mkdir(parents=True)
            (root / "vllm/__init__.py").write_text("")
            (mod / "attention.py").write_text("class Attn:\n    def forward(self):\n        pass\n")
            refs = [("p.py", "vllm.models.deepseek_v4_1.attention", "Attn"),
                    ("p.py", "vllm.models.deepseek_v4_1.attention.Attn", ".forward"),
                    ("p.py", "vllm.models.deepseek_v4_1.attention", "Gone")]
            out = dry.check_refs(root, refs, retarget=True)
            self.assertEqual([(r["name"], r["status"]) for r in out], [("Gone", "missing-name")])
            self.assertEqual(len(dry.check_refs(root, refs, retarget=False)), 3)


if __name__ == "__main__":
    unittest.main()
