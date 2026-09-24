#!/usr/bin/env python3
"""sitecustomize.py patch manifest: _patch / _rewrite helpers and wiring (CPU only).

The helpers are lifted out of sitecustomize.py with ast and run in a child
python, so nothing touches vLLM and os._exit cannot kill the test runner.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docker/patch/sitecustomize.py"
HELPERS = ("_PatchSkip", "_patch", "_rewrite", "_sm120_rewrite", "_PATCH_STRICT", "_VLLM")
REQUIRED = {
    "persistent_topk",
    "kpool_persistent_topk",
    "exl3_weight_block_size",
    "warmup_stubs",
    "swa_page_coerce",
    "kernel_block_sizes",
    "extra_page_bump",
    "native_indexer_decode",
    "indexer_adaptive",
}
REQUIRED_UNLESS_LM_ONLY = {"swa_image_width", "attention_image_width"}
sys.path.insert(0, str(ROOT / "docker/patch"))
from sm120_page import NATIVE_DECODE_OLD  # noqa: E402

OLD_INDEXER = "def _supports_native_decode(next_n):\n" + NATIVE_DECODE_OLD + "\n"


def _helper_source() -> str:
    tree = ast.parse(SITE.read_text())
    keep = []
    for node in tree.body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign):
            name = getattr(node.targets[0], "id", None)
        if name in HELPERS:
            keep.append(ast.unparse(node))
    return "import os\nimport sys\n" + "\n\n".join(keep) + "\n"


def _run(body: str, strict: str = "0") -> subprocess.CompletedProcess[str]:
    # -S: no site import, so docker/patch/sitecustomize.py itself never runs here.
    code = f"import sys\nsys.path.insert(0, {str(ROOT / 'docker/patch')!r})\n" + _helper_source() + textwrap.dedent(body)
    env = dict(os.environ, DSV41_PATCH_STRICT=strict)
    return subprocess.run([sys.executable, "-S", "-c", code], env=env, capture_output=True, text=True)


def _patch_calls() -> dict[str, ast.Call]:
    calls = {}
    for node in ast.parse(SITE.read_text()).body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if getattr(call.func, "id", None) == "_patch":
                calls[call.args[0].value] = call
    return calls


class PatchHelperTests(unittest.TestCase):
    def test_ok_skip_fail_lines(self) -> None:
        proc = _run(
            """
            def ok(): return "rewritten"
            def skip(): raise _PatchSkip("DSV41_X off")
            def fail(): raise ImportError("no vllm")
            _patch("a", ok)
            _patch("b", skip)
            _patch("c", fail)
            _patch("d", fail, required=True)
            print("reached end")
            """
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        lines = proc.stderr.splitlines()
        self.assertEqual(lines[0], "dsv41-patch ok a: rewritten")
        self.assertEqual(lines[1], "dsv41-patch skip b: DSV41_X off")
        self.assertEqual(lines[2], "dsv41-patch FAIL c: ImportError('no vllm')")
        self.assertEqual(lines[3], "dsv41-patch FAIL d: ImportError('no vllm')")
        self.assertIn("reached end", proc.stdout, "non-strict: a required FAIL only reports")

    def test_required_fail_exits_when_strict(self) -> None:
        proc = _run(
            """
            _patch("opt", lambda: 1 / 0)
            print("after optional", flush=True)
            _patch("req", lambda: 1 / 0, required=True)
            print("after required")
            """,
            strict="1",
        )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("after optional", proc.stdout)
        self.assertNotIn("after required", proc.stdout)
        self.assertIn("dsv41-patch FAIL req: ZeroDivisionError", proc.stderr)

    def test_system_exit_is_labelled_then_kept_loud(self) -> None:
        proc = _run(
            """
            def miss(): raise SystemExit("prefer_b12x_mxfp8: backend=auto not present")
            _patch("prefer_b12x_mxfp8", miss)
            print("not reached")
            """
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("dsv41-patch FAIL prefer_b12x_mxfp8: SystemExit(", proc.stderr)
        self.assertNotIn("not reached", proc.stdout)

    def test_rewrite_writes_once_atomically_and_keeps_mode(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "indexer.py"
            target.write_text(OLD_INDEXER)
            target.chmod(0o640)
            proc = _run(
                f"""
                import sm120_page
                _patch("native_indexer_decode", lambda: _rewrite({str(target)!r}, sm120_page.patch_native_indexer_decode_source), required=True)
                _patch("native_indexer_decode", lambda: _rewrite({str(target)!r}, sm120_page.patch_native_indexer_decode_source), required=True)
                _patch("absent", lambda: _rewrite({str(target) + ".nope"!r}, sm120_page.patch_native_indexer_decode_source), required=True)
                """,
                strict="1",
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            lines = proc.stderr.splitlines()
            self.assertEqual(lines[0], "dsv41-patch ok native_indexer_decode: rewritten")
            self.assertEqual(lines[1], "dsv41-patch ok native_indexer_decode: already applied")
            self.assertTrue(lines[2].startswith("dsv41-patch skip absent: "), lines[2])
            self.assertIn("is_device_capability_family(120)", target.read_text())
            self.assertEqual(target.stat().st_mode & 0o777, 0o640)
            self.assertEqual(sorted(p.name for p in Path(d).iterdir()), ["indexer.py"], "no tmp file left")

    def test_anchor_miss_fails_required_and_leaves_file(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "indexer.py"
            target.write_text("def unrelated():\n    pass\n")
            proc = _run(
                f"""
                import sm120_page
                _patch("native_indexer_decode", lambda: _rewrite({str(target)!r}, sm120_page.patch_native_indexer_decode_source), required=True)
                """,
                strict="1",
            )
            self.assertEqual(proc.returncode, 1)
            self.assertIn("dsv41-patch FAIL native_indexer_decode: ValueError('native indexer decode dispatch not found')", proc.stderr)
            self.assertEqual(target.read_text(), "def unrelated():\n    pass\n")

    def test_one_anchor_miss_does_not_skip_the_next_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            bad, good = Path(d) / "attention.py", Path(d) / "indexer.py"
            bad.write_text("no anchors here\n")
            good.write_text(OLD_INDEXER)
            proc = _run(
                f"""
                import sm120_page
                _patch("attention_image_width", lambda: _rewrite({str(bad)!r}, sm120_page.patch_attention_image_width_source))
                _patch("native_indexer_decode", lambda: _rewrite({str(good)!r}, sm120_page.patch_native_indexer_decode_source))
                """
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("dsv41-patch FAIL attention_image_width", proc.stderr)
            self.assertIn("dsv41-patch ok native_indexer_decode: rewritten", proc.stderr)


class SitecustomizeWiringTests(unittest.TestCase):
    def test_no_top_level_bare_except_pass(self) -> None:
        for node in ast.parse(SITE.read_text()).body:
            if isinstance(node, ast.Try):
                for h in node.handlers:
                    bare = len(h.body) == 1 and isinstance(h.body[0], ast.Pass)
                    exc = getattr(h.type, "id", None)
                    self.assertFalse(bare and exc == "Exception", f"bare pass at line {h.lineno}")

    def test_required_set(self) -> None:
        required, lm_gated = set(), set()
        for name, call in _patch_calls().items():
            for kw in call.keywords:
                if kw.arg == "required":
                    if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        required.add(name)
                    elif ast.unparse(kw.value) == "not _lm_only":
                        lm_gated.add(name)
        self.assertEqual(required, REQUIRED)
        self.assertEqual(lm_gated, REQUIRED_UNLESS_LM_ONLY)

    def test_each_sm120_rewrite_is_its_own_patch(self) -> None:
        src = SITE.read_text()
        rewrites = [
            "patch_persistent_topk_source",
            "patch_kpool_persistent_topk_source",
            "patch_swa_prefill_image_width_source",
            "patch_attention_image_width_source",
            "patch_native_indexer_decode_source",
            "patch_indexer_adaptive_source",
            "patch_indexer_short_context_source",
        ]
        wired = {}
        for name, call in _patch_calls().items():
            body = call.args[1]
            if isinstance(body, ast.Call) and getattr(body.func, "id", None) == "_sm120_rewrite":
                wired[body.args[1].value] = name
        self.assertEqual(sorted(wired), sorted(rewrites))
        self.assertEqual(len(set(wired.values())), len(rewrites))
        self.assertNotIn(".write_text(patch_", src, "sm120 rewrites go through _rewrite")

    def test_fat_threshold_env_set_before_plugins_load(self) -> None:
        src = SITE.read_text()
        fat = src.index('_os_fat.environ["VLLM_EXL3_FAT_THRESHOLD"] = str(2**30)')
        self.assertLess(fat, src.index("load_general_plugins()"))
        self.assertLess(fat, src.index("from vllm_exl3.exl3 import Exl3Config"))

    def test_manifest_goes_to_stderr_not_stdout(self) -> None:
        tree = ast.parse(SITE.read_text())
        fn = next(n for n in tree.body if getattr(n, "name", None) == "_patch")
        for call in (n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "print"):
            self.assertIn("file=sys.stderr", ast.unparse(call))


if __name__ == "__main__":
    unittest.main()
