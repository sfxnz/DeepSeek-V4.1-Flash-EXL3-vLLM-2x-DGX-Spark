import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docker" / "patch"))

import widen_b12x_smalls as w

STOCK = """def _select_default_mma_tiler_mn(
    m, n, sm_count, *, is_mxfp8=False, expected_m=None, k=None,
):
    coarse_tile = (128, 128)
    if is_mxfp8 and n > 1536:
        if expected_m is not None:
            if expected_m == 1:
                return (16, 64)
            if expected_m <= 8:
                return (16, 128)
            if expected_m <= 128:
                return (32, 128)
            return (64, 128)
        if m == 1:
            return (16, 64)
        if m <= 8:
            return (16, 128)
        return (64, 128)
    return (64, 64)
"""


class TestWidenB12xSmalls(unittest.TestCase):
    def test_patch_switches_small_m_branches_only(self):
        out = w.patch(STOCK)
        self.assertEqual(out.count("widen_b12x_smalls"), 1)
        # both <=8 branches now conditional on n
        self.assertIn("return (16, 64) if n <= 8192 else (16, 128)\n            if expected_m <= 128:", out)
        # untouched branches
        self.assertIn("if expected_m <= 128:\n                return (32, 128)", out)
        self.assertIn("if expected_m == 1:\n                return (16, 64)", out)
        # idempotent
        self.assertEqual(w.patch(out), out)

    def test_missing_anchor_raises(self):
        with self.assertRaises(SystemExit):
            w.patch("def nothing(): pass")


if __name__ == "__main__":
    unittest.main()
