"""Drive sm120_wo_a patches on attention, FlashInfer scales, and o_proj."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "docker/patch/sm120_wo_a.py"


def _load():
    spec = importlib.util.spec_from_file_location("sm120_wo_a", PATCHER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestSm120WoA(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load()

    def test_patch_attention_clears_is_bmm(self) -> None:
        src = "        self.wo_a.is_bmm = True\n        self.wo_a.bmm_batch_size = self.n_local_groups\n"
        out = self.mod.patch_attention(src)
        self.assertIn("self.wo_a.is_bmm = False", out)
        self.assertNotIn("self.wo_a.is_bmm = True", out)

    def test_patch_flashinfer_keeps_2d_scales(self) -> None:
        src = (
            "        weight_scale_2d = layer.weight_scale.data[:N, :scale_k].contiguous()\n"
            "        weight_scale_swizzled = swizzle_mxfp8_scale(weight_scale_2d, M=N, K=K)\n"
        )
        out = self.mod.patch_flashinfer(src)
        self.assertIn("layer.weight_scale_2d = Parameter(weight_scale_2d", out)
        self.assertIn("swizzle_mxfp8_scale(weight_scale_2d, M=N, K=K)", out)

    def test_patch_o_proj_uses_grouped_mm_on_sm120(self) -> None:
        src = (
            "def deep_gemm_fp8_o_proj(\n"
            "    o, positions, cos_sin_cache, wo_a, wo_b, *,\n"
            "    n_groups, heads_per_group, nope_dim, rope_dim, o_lora_rank,\n"
            "    einsum_recipe, tma_aligned_scales,\n"
            "):\n"
            "    use_fp8 = wo_a.weight.dtype == torch.float8_e4m3fn\n"
            "    o_proj_input, o_scale = fused_inv_rope_fp8_quant(\n"
            "        o,\n"
            "        positions,\n"
            "        cos_sin_cache,\n"
            "        n_groups=n_groups,\n"
            "        heads_per_group=heads_per_group,\n"
            "    )\n"
        )
        out = self.mod.patch_o_proj(src)
        self.assertIn("_sm120_grouped_wo_a", out)
        self.assertIn("cap.major >= 12", out)
        self.assertIn("wo_a_dense_gemm_mxfp8", out)
        self.assertIn("quantize_wo_a_input_mxfp8", out)
        self.assertIn("use_fp8 = wo_a.weight.dtype == torch.float8_e4m3fn", out)

    def test_apply_rewrites_tree(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            attn = root / "models/deepseek_v4_1/attention.py"
            fi = root / "model_executor/kernels/linear/mxfp8/flashinfer.py"
            oproj = root / "models/deepseek_v4/nvidia/ops/o_proj.py"
            attn.parent.mkdir(parents=True)
            fi.parent.mkdir(parents=True)
            oproj.parent.mkdir(parents=True)
            attn.write_text("        self.wo_a.is_bmm = True\n")
            fi.write_text(
                "        weight_scale_2d = layer.weight_scale.data[:N, :scale_k].contiguous()\n"
                "        weight_scale_swizzled = swizzle_mxfp8_scale(weight_scale_2d, M=N, K=K)\n"
            )
            oproj.write_text(
                "def deep_gemm_fp8_o_proj(\n"
                "    o,\n"
                "):\n"
                "    use_fp8 = wo_a.weight.dtype == torch.float8_e4m3fn\n"
                "    o_proj_input, o_scale = fused_inv_rope_fp8_quant(\n"
                "        o,\n"
                "        positions,\n"
                "        cos_sin_cache,\n"
                "        n_groups=n_groups,\n"
                "        heads_per_group=heads_per_group,\n"
                "    )\n"
            )
            self.assertTrue(self.mod.apply(root))
            self.assertIn("is_bmm = False", attn.read_text())
            self.assertIn("weight_scale_2d = Parameter", fi.read_text())
            self.assertIn("_sm120_grouped_wo_a", oproj.read_text())
            self.assertFalse(self.mod.apply(root))


if __name__ == "__main__":
    unittest.main()
