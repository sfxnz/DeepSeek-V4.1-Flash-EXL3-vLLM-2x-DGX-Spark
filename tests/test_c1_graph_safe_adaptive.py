"""Drive c1_graph_safe_adaptive snap + source patches."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "docker/patch/attic/c1_graph_safe_adaptive.py"


def _load():
    spec = importlib.util.spec_from_file_location("c1_graph_safe_adaptive", PATCHER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestC1GraphSafeAdaptive(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load()

    def test_snap_uses_captured_query_sizes_only(self) -> None:
        snap = self.mod.snap_draft_budget
        self.assertEqual(self.mod.CAPTURED_DRAFTS, (2,))
        self.assertEqual(self.mod.PINNED_DRAFT_BUDGET, 2)
        self.assertEqual(snap(5), 2)
        self.assertEqual(snap(4), 2)
        self.assertEqual(snap(3), 2)
        self.assertEqual(snap(2), 2)
        self.assertEqual(snap(1), 2)
        self.assertEqual(snap(0), 2)
        for draft in self.mod.CAPTURED_DRAFTS:
            self.assertIn(1 + draft, self.mod.CAPTURE_SIZES)

    def test_patch_adaptive_skips_sm120_gates_and_pins_budget(self) -> None:
        src = (
            "    if not enable_adaptive_verification:\n"
            "        return None\n"
            "\n"
            "    # The selector rejects unsupported backends\n"
            "        draft_budget = int(np.argmax(num_tokens_to_estimated_accepted_tokens / costs))\n"
            "        self._batch_budget = (\n"
        )
        out = self.mod.patch_adaptive(src)
        self.assertIn("max_num_seqs", out)
        self.assertIn("if int(scheduled_drafts.sum()) > 0:", out)
        self.assertIn("draft_budget = 2", out)
        self.assertIn("The selector rejects unsupported backends", out)

    def test_patch_runner_disables_varlen_decode(self) -> None:
        src = "            varlen_decode=self.adaptive_verification is not None,\n"
        out = self.mod.patch_runner(src)
        self.assertIn("varlen_decode=False,", out)
        self.assertNotIn("self.adaptive_verification is not None", out)

    def test_apply_rewrites_tree(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            adaptive = root / "v1/worker/gpu/spec_decode/adaptive_verification.py"
            runner = root / "v1/worker/gpu/model_runner.py"
            adaptive.parent.mkdir(parents=True)
            runner.parent.mkdir(parents=True, exist_ok=True)
            adaptive.write_text(
                "    if not enable_adaptive_verification:\n"
                "        return None\n"
                "\n"
                "    # The selector rejects unsupported backends\n"
                "        draft_budget = int(np.argmax(num_tokens_to_estimated_accepted_tokens / costs))\n"
                "        self._batch_budget = (\n"
            )
            runner.write_text(
                "            varlen_decode=self.adaptive_verification is not None,\n"
            )
            self.assertTrue(self.mod.apply(root))
            patched = adaptive.read_text()
            self.assertIn("if int(scheduled_drafts.sum()) > 0:", patched)
            self.assertIn("draft_budget = 2", patched)
            self.assertIn("varlen_decode=False,", runner.read_text())
            self.assertFalse(self.mod.apply(root))


if __name__ == "__main__":
    unittest.main()
