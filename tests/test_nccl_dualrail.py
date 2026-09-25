#!/usr/bin/env python3
"""CPU tests for tools/nccl_allreduce_sweep.py and tools/nccl_dualrail.sh."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import nccl_allreduce_sweep as nccl  # noqa: E402


class NcclSweepTests(unittest.TestCase):
    def test_sizes_span_8b_to_256mib(self) -> None:
        s = nccl.sizes(8, 256 << 20)
        self.assertEqual((s[0], s[-1], len(s)), (8, 256 << 20, 26))

    def test_busbw_two_ranks_equals_algbw(self) -> None:
        self.assertAlmostEqual(nccl.busbw(10**9, 1.0, 2), 1.0)
        self.assertAlmostEqual(nccl.busbw(10**9, 1.0, 4), 1.5)

    def _rows(self, bw: float, lat_us: float) -> dict:
        rows = [{"bytes": n, "us": lat_us, "busbw_GBps": 0.0} for n in nccl.LAT_SIZES]
        rows += [{"bytes": n, "us": 0.0, "busbw_GBps": bw} for n in nccl.BW_SIZES]
        return {"world": 2, "rows": rows}

    def test_gate(self) -> None:
        base = self._rows(12.0, 40.0)
        self.assertTrue(nccl.compare(base, self._rows(20.0, 41.0))["pass"])
        self.assertFalse(nccl.compare(base, self._rows(12.5, 40.0))["pass"])  # bw not up enough
        self.assertFalse(nccl.compare(base, self._rows(20.0, 44.0))["pass"])  # decode AR slower

    def test_script_arms_pin_active_rails_and_cross_nic(self) -> None:
        sh = (ROOT / "tools/nccl_dualrail.sh").read_text()
        self.assertNotIn("rocep1s0f0", sh)
        self.assertNotIn("roceP2p1s0f0", sh)
        for arm in ("dual)", "dual_keep)", "dual_keep_nomerge)"):
            line = next(ln for ln in sh.splitlines() if ln.strip().startswith(arm))
            self.assertIn("NCCL_CROSS_NIC=0", line, arm)
        self.assertIn("refusing: GPU holders up", sh)


if __name__ == "__main__":
    unittest.main()
