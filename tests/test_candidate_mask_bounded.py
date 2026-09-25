"""candidate_mask_bounded (DSV41_CANDIDATE_MASK_BOUNDED): routing, contract, wiring.

Host only. The GPU proof (54 mask cases bit-exact below each row's end, 40
decode top-k cases with NaN/inf garbage past the end giving the same index
sets, graph replay 95.2 -> 7.1 us at the serve's 1M width) is
kernel_study/fusion_host/candidate_mask_check.py. The stock kernels and the
decode call site are pinned in tests/fixtures/candidate_mask_e13.pin.py.
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
FIX = ROOT / "tests" / "fixtures" / "candidate_mask_e13.pin.py"
sys.path.insert(0, str(PATCH))
import candidate_mask_bounded as cmb  # noqa: E402
import decode_levers as dl  # noqa: E402


class Kernel:
    def __init__(self):
        self.grids = []

    def __getitem__(self, grid):
        def run(*args, **kw):
            self.grids.append(grid)

        return run


class Logits:
    is_cuda = True

    def __init__(self, rows, width):
        self.shape = (rows, width)
        self.device = "cuda:0"

    def stride(self):
        return (self.shape[1], 1)


class Ids:
    shape = (4, 2048)

    def stride(self):
        return (2048, 1)

    def stride0(self):
        return 1


def fake_torch(capturing=True):
    t = types.SimpleNamespace(uint8="uint8", empty=lambda *a, **k: object())
    t.cuda = types.SimpleNamespace(is_current_stream_capturing=lambda: capturing)
    return t


class RowKe:
    def stride(self, i):
        return 1


class RoutingTests(unittest.TestCase):
    def setUp(self):
        cmb._STATE.update(armed=True, verify_left=0, verified=0)
        self.stock_calls = []
        self.fk, self.mk = Kernel(), Kernel()
        triton = types.SimpleNamespace(cdiv=lambda a, b: -(-a // b))
        self.apply = cmb.make_apply(fake_torch(), triton, lambda *a: self.stock_calls.append(a), self.fk, self.mk)

    def test_decode_launches_the_bounded_kernels(self):
        self.apply(Logits(4, 1 << 20), None, RowKe(), Ids(), 8, 1)
        self.assertEqual(self.stock_calls, [])
        self.assertEqual(self.fk.grids, [(4,)])
        self.assertEqual(self.mk.grids, [(4, 1024)])  # same grid as stock: capture-safe

    def test_prefill_and_disarmed_take_the_stock_mask(self):
        self.apply(Logits(4, 4096), object(), RowKe(), Ids(), 8, 1)
        cmb._STATE["armed"] = False
        self.apply(Logits(4, 4096), None, RowKe(), Ids(), 8, 1)
        self.assertEqual(len(self.stock_calls), 2)
        self.assertEqual((self.fk.grids, self.mk.grids), ([], []))

    def test_empty_logits_do_nothing(self):
        self.apply(Logits(0, 4096), None, RowKe(), Ids(), 8, 1)
        self.apply(Logits(4, 0), None, RowKe(), Ids(), 8, 1)
        self.assertEqual((self.stock_calls, self.fk.grids, self.mk.grids), ([], [], []))


class ContractTests(unittest.TestCase):
    """The bounded kernels repeat the stock per-column logic below the end."""

    def test_mask_logic_matches_the_stock_kernel(self):
        stock = FIX.read_text()
        mine = (PATCH / "candidate_mask_bounded.py").read_text()
        for stmt in (
            "valid = (cols >= start) & (cols < end) & (cols < width)",
            "block = (cols - start) // BLOCK_SIZE",
        ):
            self.assertIn(stmt, stock, stmt)
            self.assertIn(stmt.replace("start", "0").replace("(cols - 0)", "cols"), mine, stmt)
        for stmt in (
            "keep = (keep != 0) | ((cols == width - 1) & (edge != 0))",
            "(cols < width) & ~(valid & keep),",
            "tl.store(flags + row * (nblocks + 1) + block, 1, (cols < K) & (block >= 0))",
            "tl.debug_barrier()",
        ):
            self.assertIn(stmt, stock, stmt)
            self.assertIn(stmt, mine, stmt)
        self.assertIn("block = tl.where(start + block * BLOCK_SIZE >= width, nblocks, block)", stock)
        self.assertIn("block = tl.where(block * BLOCK_SIZE >= width, nblocks, block)", mine)
        self.assertIn("_mask_candidates_kernel[(rows, triton.cdiv(width, 1024))](", stock)
        self.assertEqual(cmb.TILE, 1024)

    def test_decode_call_site_passes_no_row_starts_and_wide_logits(self):
        site = FIX.read_text()
        call = site[site.index("#                 _apply_candidate_mask(") :]
        self.assertIn("#                     None,", call.splitlines()[2])
        self.assertIn("max_model_len=max_model_len,", site)
        self.assertIn("clean_logits=False,", site)
        self.assertIn("ops.top_k_per_row_decode(", site)


class WiringTests(unittest.TestCase):
    def test_env(self):
        self.assertFalse(cmb.enabled({}))
        self.assertTrue(cmb.enabled({"DSV41_CANDIDATE_MASK_BOUNDED": "1"}))
        self.assertEqual(cmb.verify_calls({}), 4)
        self.assertEqual(cmb.verify_calls({"DSV41_CANDIDATE_MASK_VERIFY": "0"}), 1)

    def test_decode_levers_step(self):
        self.assertIn(
            '("candidate-mask-bounded", _install_candidate_mask_bounded)',
            (PATCH / "decode_levers.py").read_text(),
        )
        calls = []
        fake = types.ModuleType("candidate_mask_bounded")
        fake.install = lambda: calls.append(1)
        with mock.patch.dict(sys.modules, {"candidate_mask_bounded": fake}), redirect_stdout(StringIO()):
            dl._install_candidate_mask_bounded({})
            dl._install_candidate_mask_bounded({"DSV41_CANDIDATE_MASK_BOUNDED": "0"})
            self.assertEqual(calls, [])
            dl._install_candidate_mask_bounded({"DSV41_CANDIDATE_MASK_BOUNDED": "1"})
        self.assertEqual(calls, [1])

    def test_top_level_imports_are_stdlib_only(self):
        tree = ast.parse((PATCH / "candidate_mask_bounded.py").read_text())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        mods = {a.name for n in top if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module for n in top if isinstance(n, ast.ImportFrom)}
        self.assertEqual(mods, {"__future__", "os"})


if __name__ == "__main__":
    unittest.main()
