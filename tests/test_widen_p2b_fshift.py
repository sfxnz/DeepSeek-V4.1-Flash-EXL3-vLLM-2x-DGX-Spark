import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docker" / "patch"))

import widen_p2b_fshift as m


PATCHED_TILE = '''#include <cooperative_groups.h>
namespace cg = cooperative_groups;

template <int bits, int cb, int CFG>
__device__ __forceinline__ void run_gemv_tile(
    const uint32_t* __restrict__ B32)
{
    if constexpr (bits == 2) {
        int i1 = lane >> 1;
        x_src_b = i1;
        x_src_a = (i1 + 15) & 15;
    }
    uint32_t bw[LOADS];
    #pragma unroll
    for (int t = 0; t < WNT; ++t) {
        FragB f0, f1;
        if constexpr (bits == 4) {
            uint32_t aw = __shfl_sync(0xffffffffu, bw[t], (lane + 31) & 31);
            exl3_gemv_ns::dq8_regs_4bits<cb>(aw, bw[t], f0, f1);
        } else if constexpr (bits == 2) {
            const uint32_t w = bw[t >> 1];
            const int base = (t & 1) << 4;
                    uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                    uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    exl3_gemv_ns::dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);
        }
    }
}
'''


class TestWidenP2bFshift(unittest.TestCase):
    def _apply(self, src: str) -> str:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "csrc").mkdir()
            (root / "csrc" / "p2b_moe.cu").write_text(src)
            m.apply(root)
            return (root / "csrc" / "p2b_moe.cu").read_text()

    def test_patch_applies_and_is_idempotent(self):
        out = self._apply(PATCHED_TILE)
        self.assertIn("bench_fshift::dq8_regs_2bits_fs<cb>(awv, bwv, lane << 3, f0, f1)", out)
        self.assertIn("__funnelshift_r(b, a, shift)", out)
        self.assertIn("if constexpr (cb == 1)", out)
        # cb != 1 keeps stock decode
        self.assertIn("exl3_gemv_ns::dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);", out)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "csrc").mkdir()
            (root / "csrc" / "p2b_moe.cu").write_text(out)
            m.apply(root)  # idempotent
            again = (root / "csrc" / "p2b_moe.cu").read_text()
        self.assertEqual(out, again)

    def test_missing_call_raises(self):
        bad = PATCHED_TILE.replace(OLD := m.OLD_CALL, "void nothing() {}") if False else PATCHED_TILE.replace(
            "exl3_gemv_ns::dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);",
            "// gone",
        )
        with self.assertRaises(SystemExit):
            self._apply(bad)


if __name__ == "__main__":
    unittest.main()
