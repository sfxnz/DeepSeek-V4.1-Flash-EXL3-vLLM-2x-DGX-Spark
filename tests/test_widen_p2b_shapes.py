#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STOCK_CU = """
at::Tensor p2b_fused_moe_cuda(const at::Tensor& x, at::Tensor& out,
    const at::Tensor& gt, const at::Tensor& gu, const at::Tensor& gv,
    const at::Tensor& ut, const at::Tensor& uu, const at::Tensor& uv,
    const at::Tensor& dt, const at::Tensor& du, const at::Tensor& dv,
    const at::Tensor& ids, const at::Tensor& rw, int64_t kg, int64_t ku,
    int64_t kd, bool mcg, int64_t intermediate_size, float swiglu_limit) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "fused MoE requires CUDA fp16 input");
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1 && x.size(1) == 4096,
                "fused MoE requires one input row with hidden width 4096");
    TORCH_CHECK(out.sizes() == x.sizes(), "fused MoE output shape must match input");
    TORCH_CHECK(intermediate_size == 1024 || intermediate_size == 2048,
                "fused MoE local intermediate width must be 1024 or 2048");
    const int e = static_cast<int>(ids.numel());
    constexpr int m = 1, hidden = 4096;
    const int inter = static_cast<int>(intermediate_size);
    auto gate = at::empty({e, m, inter}, x.options());
    return out;
}
"""

STOCK_PY = '''
def _native_moe_dimensions_supported(
    x2d: torch.Tensor,
    layer: torch.nn.Module,
    inners: list[dict[str, Any]],
    limit: float | None = None,
) -> bool:
    hidden_meta = int(getattr(layer, "_exl3_hidden_size", x2d.shape[1]))
    inter_meta = int(getattr(layer, "_exl3_intermediate_local", 2048))
    rows = int(x2d.shape[0])
    bits = int(getattr(layer, "_exl3_k", getattr(layer, "_exl3_bits", -1)))
    if not (
        rows >= 1
        and int(x2d.shape[1]) == hidden_meta == 4096
        and inter_meta in (1024, 2048)
        and bits in (2, 3, 4)
        and len(inners) > 0
    ):
        return False
    return rows <= _native_moe_max_rows(bits)
'''


def _load():
    path = ROOT / "docker/patch/widen_p2b_shapes.py"
    spec = importlib.util.spec_from_file_location("widen_p2b_shapes", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class WidenP2bShapesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _load()

    def test_patch_cu_widens_hidden_and_inter_and_keeps_m1(self) -> None:
        patched = self.mod.patch_cu(STOCK_CU)
        self.assertNotEqual(patched, STOCK_CU)
        self.assertNotIn("x.size(1) == 4096", patched)
        self.assertNotIn("hidden = 4096", patched)
        self.assertNotIn("intermediate_size == 1024 || intermediate_size == 2048", patched)
        self.assertIn("x.size(0) == 1", patched)
        self.assertIn("x.size(1) % 128 == 0", patched)
        self.assertIn("intermediate_size > 0 && intermediate_size % 128 == 0", patched)
        self.assertIn("constexpr int m = 1;", patched)
        self.assertIn("const int hidden = static_cast<int>(x.size(1));", patched)
        self.assertIn("{e, m, inter}", patched)

    def test_patch_cu_is_idempotent(self) -> None:
        once = self.mod.patch_cu(STOCK_CU)
        self.assertEqual(self.mod.patch_cu(once), once)

    def test_patch_cu_raises_on_unrelated_source(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            self.mod.patch_cu("int main() { return 0; }")
        self.assertIn("widen_p2b_shapes", str(ctx.exception))

    def test_patch_py_drops_glm_only_geometry_gate(self) -> None:
        patched = self.mod.patch_py(STOCK_PY)
        self.assertNotEqual(patched, STOCK_PY)
        self.assertNotIn("hidden_meta == 4096", patched)
        self.assertNotIn("inter_meta in (1024, 2048)", patched)
        self.assertIn("int(x2d.shape[1]) == hidden_meta", patched)
        self.assertIn("hidden_meta % 128 == 0", patched)
        self.assertIn("inter_meta % 128 == 0", patched)
        self.assertIn("rows >= 1", patched)

    def test_patch_py_is_idempotent(self) -> None:
        once = self.mod.patch_py(STOCK_PY)
        self.assertEqual(self.mod.patch_py(once), once)

    def test_apply_rewrites_pin_layout_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cu = root / "csrc" / "p2b_moe.cu"
            py = root / "src" / "vllm_exl3" / "exl3.py"
            cu.parent.mkdir(parents=True)
            py.parent.mkdir(parents=True)
            cu.write_text(STOCK_CU)
            py.write_text(STOCK_PY)
            self.mod.apply(root)
            self.assertIn("x.size(1) % 128 == 0", cu.read_text())
            self.assertIn("constexpr int m = 1;", cu.read_text())
            self.assertIn("hidden_meta % 128 == 0", py.read_text())
