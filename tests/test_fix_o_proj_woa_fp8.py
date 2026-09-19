import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docker" / "patch"))

import fix_o_proj_woa_fp8 as f

STOCK = '''def deep_gemm_fp8_o_proj(
    o, positions, cos_sin_cache, wo_a, wo_b, *, n_groups=4,
    heads_per_group=1, nope_dim=1, rope_dim=1, o_lora_rank=1,
    einsum_recipe=(1, 1, 128), tma_aligned_scales=True,
):
    use_fp8 = wo_a.weight.dtype == torch.float8_e4m3fn
    # --- probe_wo_a (diagnostic, no behavior change) ---
    print("x")
    # --- end probe_wo_a ---
    o_proj_input, o_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
    )
    z = torch.empty(0)
    if use_fp8:
        weight_scale = wo_a.weight_scale
        fp8_einsum("bhr,hdr->bhd", (o_proj_input, o_scale), (wo_a.weight, weight_scale), z, recipe=einsum_recipe)
    else:
        torch.bmm(o_proj_input.transpose(0, 1), wo_a.weight.view(1, 1, -1).transpose(1, 2), out=z.transpose(0, 1))
    return wo_b(z.flatten(1))
'''


class TestFixOProjWoaFp8(unittest.TestCase):
    def _apply(self, src):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "o_proj.py"
            p.write_text(src)
            f.apply(p)
            return p.read_text()

    def test_patch_inserts_guard_and_helper(self):
        out = self._apply(STOCK)
        self.assertIn("if not use_fp8:\n        use_fp8 = _woa_try_requant(wo_a, n_groups, o_lora_rank)", out)
        self.assertIn("def _woa_try_requant(", out)
        self.assertIn("is_current_stream_capturing", out)
        self.assertIn("_woa_fp8fail", out)
        # helper placed before the entry function
        self.assertLess(out.index("def _woa_try_requant"), out.index("def deep_gemm_fp8_o_proj"))
        # idempotent
        self.assertEqual(f.patch(out), out)

    def test_missing_anchor_raises(self):
        with self.assertRaises(SystemExit):
            f.patch("def unrelated(): pass")


if __name__ == "__main__":
    unittest.main()
