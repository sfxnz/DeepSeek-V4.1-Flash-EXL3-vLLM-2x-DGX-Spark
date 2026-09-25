"""moe_prep_fused (DSV41_MOE_PREP_FUSED): source rewrite and wiring, host only.

The GPU proof (576 sweep cases incl. invalid ids / NaN / inf / fp16 overflow,
100 graph replays, the rewritten function vs the stock one with a stand-in
p2b: all bit-exact; 14 -> 1 kernels) is kernel_study/fusion_host/
moe_prep_check.py. The function it rewrites is pinned from canonical-e13 in
tests/fixtures/exl3_apply_native_e13.pin.py.
"""

from __future__ import annotations

import ast
import sys
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "docker" / "patch"
FIX = ROOT / "tests" / "fixtures" / "exl3_apply_native_e13.pin.py"
sys.path.insert(0, str(PATCH))
import decode_levers as dl  # noqa: E402
import moe_prep_fused as mpf  # noqa: E402


def pinned_function(name: str = "_apply_native_fused_moe") -> str:
    src = FIX.read_text()
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(src, fn)


class RewriteTests(unittest.TestCase):
    def test_only_the_four_anchors_change(self) -> None:
        src = pinned_function()
        out = mpf.patch_source(src)
        ast.parse(out)
        undo = (
            out.replace(mpf.NEW_BLOCK, mpf.OLD_BLOCK)
            .replace(mpf.SIG_NEW, mpf.SIG_OLD)
            .replace(mpf.RET_NEW, mpf.RET_OLD)
            .replace(mpf.HEAD_NEW, mpf.HEAD_OLD)
        )
        self.assertEqual(undo, src)
        self.assertNotIn("map_topk_to_local(", out)
        self.assertEqual(out.count("_dsv41_moe_prep(ids, weights, x2d, n_exp, expert_map)"), 1)
        self.assertIn("out_dtype: torch.dtype | None = None,", out)
        self.assertIn("return native_out.to(dtype=torch.float32 if out_dtype is None else out_dtype)", out)
        # the eager self-test runs before the dimension check (prefill calls return there)
        self.assertLess(out.index("_dsv41_moe_prep.selftest_once(x2d.device)"), out.index("_native_moe_dimensions_supported("))

    def test_experts_call_passes_the_model_dtype(self) -> None:
        src = pinned_function("apply_exl3_experts")
        out = mpf.patch_experts_source(src)
        ast.parse(out)
        self.assertEqual(out.replace(mpf.CALL_NEW, mpf.CALL_OLD), src)
        self.assertIn("x2d, ids, weights, layer, inners, expert_map, limit, out_dtype=x.dtype", out)
        # the caller's own cast then sees x.dtype already: no kernel
        self.assertIn("return native_out.to(dtype=x.dtype)", out)

    def test_other_callers_keep_the_fp32_contract(self) -> None:
        # apply_exl3_fused_moe's native attempt (not rewritten) calls without
        # out_dtype and keeps getting fp32.
        out = mpf.patch_source(pinned_function())
        self.assertIn("torch.float32 if out_dtype is None", out)

    def test_names_the_block_defined_are_still_bound(self) -> None:
        # The rest of the function uses safe_ids, safe_weights and xh only.
        src = pinned_function()
        tail = src.partition(mpf.OLD_BLOCK)[2]
        for name in ("safe_ids", "safe_weights", "xh"):
            self.assertIn(name, tail)
        for name in ("local", "valid", "topk"):
            self.assertIsNone(__import__("re").search(rf"\b{name}\b", tail), name)

    def test_drifted_source_is_refused(self) -> None:
        src = pinned_function().replace("local.clamp(min=0", "local.clamp(min=1")
        with self.assertRaises(ValueError):
            mpf.patch_source(src)
        with self.assertRaises(ValueError):
            mpf.patch_source(mpf.OLD_BLOCK * 2)

    def test_stock_replica_matches_the_block(self) -> None:
        # MoePrep.stock is the fallback and the verify reference: same ops.
        tree = ast.parse((PATCH / "moe_prep_fused.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MoePrep")
        stock = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "stock")
        body = ast.unparse(stock).replace("self.map_topk_to_local", "map_topk_to_local")
        block = ast.unparse(ast.parse("def f():\n" + mpf.OLD_BLOCK))
        for stmt in (
            "local = map_topk_to_local(ids, n_exp, expert_map).reshape(ids.shape)",
            "safe_ids = local.clamp(min=0, max=n_exp - 1).to(dtype=torch.int32).contiguous()",
            "valid = (local >= 0) & (local < n_exp)",
            "safe_weights = weights.reshape_as(local).to(dtype=torch.float16).mul(valid.to(dtype=torch.float16)).contiguous()",
            "xh = x2d.to(dtype=torch.float16).contiguous()",
        ):
            self.assertIn(stmt, block, stmt)
            self.assertIn(stmt, body, stmt)


class WiringTests(unittest.TestCase):
    def test_env(self) -> None:
        self.assertFalse(mpf.enabled({}))
        self.assertTrue(mpf.enabled({"DSV41_MOE_PREP_FUSED": "1"}))
        self.assertEqual(mpf.verify_calls({}), 16)
        self.assertEqual(mpf.verify_calls({"DSV41_MOE_PREP_VERIFY": "0"}), 1)

    def test_decode_levers_step(self) -> None:
        self.assertIn('("moe-prep-fused", _install_moe_prep_fused)', (PATCH / "decode_levers.py").read_text())
        calls = []
        fake = types.ModuleType("moe_prep_fused")
        fake.install = lambda: calls.append(1) or "fused"
        with mock.patch.dict(sys.modules, {"moe_prep_fused": fake}), redirect_stdout(StringIO()) as out:
            dl._install_moe_prep_fused({})
            dl._install_moe_prep_fused({"DSV41_MOE_PREP_FUSED": "0"})
            self.assertEqual(calls, [])
            dl._install_moe_prep_fused({"DSV41_MOE_PREP_FUSED": "1"})
        self.assertEqual(calls, [1])
        self.assertIn("moe prep fused: fused", out.getvalue())

    def test_top_level_imports_are_stdlib_only(self) -> None:
        tree = ast.parse((PATCH / "moe_prep_fused.py").read_text())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        mods = {a.name for n in top if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module for n in top if isinstance(n, ast.ImportFrom)}
        self.assertEqual(mods, {"__future__", "os"})


if __name__ == "__main__":
    unittest.main()
