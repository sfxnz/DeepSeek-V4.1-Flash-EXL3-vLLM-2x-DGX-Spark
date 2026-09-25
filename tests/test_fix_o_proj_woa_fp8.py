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
        weight_scale = (
            wo_a.weight_scale
            if hasattr(wo_a, "weight_scale")
            else wo_a.weight_scale_inv
        )
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

    def test_prepack_is_env_gated_and_after_scale_pick(self):
        out = self._apply(STOCK)
        self.assertIn('_WOA_PREPACK = _woa_os.environ.get("DSV41_WOA_PREPACK", "0") == "1"', out)
        body = out[out.index("def deep_gemm_fp8_o_proj(") :]
        self.assertLess(body.index("wo_a.weight_scale_inv"), body.index("if _WOA_PREPACK:"))
        self.assertLess(body.index("if _WOA_PREPACK:"), body.index("fp8_einsum("))
        self.assertLess(out.index("def _woa_prepacked_scale("), out.index("def deep_gemm_fp8_o_proj("))

    def test_prepack_helper_contract(self):
        helper = f.PREPACK_HELPER
        # vLLM's own packer, recipe from the layer, is_sfa=False, stored as-is.
        self.assertIn("_tsf(weight_scale, o_lora_rank, k, einsum_recipe, n_groups, False)", helper)
        self.assertNotIn(".contiguous()", helper)
        self.assertNotIn("(1, 1, 32)", helper)
        self.assertIn("is_current_stream_capturing", helper)
        self.assertIn("_torch.equal(z_ref, z_new)", helper)
        self.assertIn("float8_e4m3fn", helper)
        self.assertIn("_torch.exp2(", helper)
        self.assertIn("_woa_sf_packfail = True", helper)
        self.assertIn("REJECTED", helper)
        self.assertIn("return weight_scale", helper)

    def test_pinned_e12_o_proj_gets_prepack_only(self):
        # canonical-e12 already carries probe + requant; stage 2 must apply alone.
        pin = (Path(__file__).resolve().parent / "fixtures" / "o_proj_e12.pin.py").read_text()
        self.assertEqual(pin.count("def _woa_try_requant("), 1)
        out = self._apply(pin)
        self.assertEqual(out.count("def _woa_try_requant("), 1)
        self.assertEqual(out.count("def _woa_prepacked_scale("), 1)
        self.assertEqual(out.count("if _WOA_PREPACK:"), 1)
        self.assertEqual(f.patch(out), out)
        compile(out, "o_proj.py", "exec")
        # Only additions: removing the stage-2 text gives the pin back.
        back = out.replace(f.PREPACK_HELPER, "", 1).replace(f.PREPACK_NEW, f.PREPACK_OLD, 1)
        self.assertEqual(back, pin)


class TestWoaPrepackLogic(unittest.TestCase):
    """Control flow of the stage-2 helper on CPU with a fake deep_gemm."""

    def setUp(self):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("torch not installed")
        import types
        from unittest import mock

        self.torch = torch
        self.calls = {"tsf": 0}
        self.scale_factor = 1.0

        def tsf(sf, mn, k, recipe, num_groups, is_sfa):
            self.calls["tsf"] += 1
            self.assertEqual((mn, k, recipe, num_groups, is_sfa), (8, 64, (1, 1, 32), 2, False))
            return (sf * self.scale_factor).transpose(-1, -2).contiguous().transpose(-1, -2)  # MN-major-like

        def fp8_einsum(eq, a, b, out, recipe):
            x, xs = a
            w, ws = b
            xd = x.float() * xs.repeat_interleave(recipe[2], dim=-1)
            wd = w.float() * ws.float().repeat_interleave(recipe[2], dim=-1)
            out.copy_(torch.einsum("bhr,hdr->bhd", xd, wd).to(out.dtype))

        dg = types.ModuleType("vllm.utils.deep_gemm")
        dg.fp8_einsum = fp8_einsum
        dg.transform_sf_into_required_layout = tsf
        mods = {"vllm": types.ModuleType("vllm"), "vllm.utils": types.ModuleType("vllm.utils"),
                "vllm.utils.deep_gemm": dg}
        self.capturing = False
        for p in (
            mock.patch.dict("sys.modules", mods),
            mock.patch.object(torch.cuda, "synchronize", lambda: None),
            mock.patch.object(torch.cuda, "is_current_stream_capturing", lambda: self.capturing),
        ):
            p.start()
            self.addCleanup(p.stop)
        ns = {}
        exec(f.PREPACK_HELPER, ns)
        self.fn = ns["_woa_prepacked_scale"]

    def _layer(self):
        t = self.torch
        wo_a = t.nn.Module()
        wo_a.weight = t.nn.Parameter(t.randn(2, 8, 64).to(t.float8_e4m3fn), requires_grad=False)
        ws = t.exp2(t.randint(-3, 3, (2, 8, 2)).float())
        return wo_a, ws

    def test_packs_once_and_caches(self):
        wo_a, ws = self._layer()
        out = self.fn(wo_a, ws, 2, 8, (1, 1, 32))
        self.assertIs(out, wo_a._woa_sf_packed)
        self.assertFalse(out.is_contiguous())
        self.assertIs(self.fn(wo_a, ws, 2, 8, (1, 1, 32)), out)
        self.assertEqual(self.calls["tsf"], 1)

    def test_mismatch_keeps_fp32_scale(self):
        self.scale_factor = 2.0
        wo_a, ws = self._layer()
        self.assertIs(self.fn(wo_a, ws, 2, 8, (1, 1, 32)), ws)
        self.assertTrue(wo_a._woa_sf_packfail)
        self.assertFalse(hasattr(wo_a, "_woa_sf_packed"))
        self.assertIs(self.fn(wo_a, ws, 2, 8, (1, 1, 32)), ws)
        self.assertEqual(self.calls["tsf"], 1)

    def test_skips_mid_capture_and_non_fp32(self):
        wo_a, ws = self._layer()
        self.capturing = True
        self.assertIs(self.fn(wo_a, ws, 2, 8, (1, 1, 32)), ws)
        self.capturing = False
        packed_int = ws.to(self.torch.int32)
        self.assertIs(self.fn(wo_a, packed_int, 2, 8, (1, 1, 32)), packed_int)
        self.assertEqual(self.calls["tsf"], 0)


if __name__ == "__main__":
    unittest.main()
