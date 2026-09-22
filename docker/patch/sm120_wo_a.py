#!/usr/bin/env python3
"""SM120 MLA wo_a: MXFP8 grouped GEMM instead of BF16 torch.bmm.

Stock marks wo_a is_bmm=True. DeepGEMM BMM is SM100-only, so GB10 uses
EmulationMxfp8 (dequant to BF16 at load). Then deep_gemm_fp8_o_proj sees
BF16 weights and runs torch.bmm every decode step.

is_bmm=False selects FlashInferCutlass (same path as Q/O KEEP). o_proj on
SM120 then runs one mm_mxfp8 per local group (G=4, N=1024, K=4096) with
backend=auto/b12x instead of fp8_einsum (DeepGEMM layout crash).
"""

from __future__ import annotations

import sys
from pathlib import Path

IS_BMM_OLD = "        self.wo_a.is_bmm = True"
IS_BMM_NEW = "        self.wo_a.is_bmm = False"

SCALE_OLD = """        weight_scale_2d = layer.weight_scale.data[:N, :scale_k].contiguous()
        weight_scale_swizzled = swizzle_mxfp8_scale(weight_scale_2d, M=N, K=K)"""

SCALE_NEW = """        weight_scale_2d = layer.weight_scale.data[:N, :scale_k].contiguous()
        layer.weight_scale_2d = Parameter(weight_scale_2d, requires_grad=False)
        weight_scale_swizzled = swizzle_mxfp8_scale(weight_scale_2d, M=N, K=K)"""

OPROJ_OLD = """    use_fp8 = wo_a.weight.dtype == torch.float8_e4m3fn
    o_proj_input, o_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,"""

OPROJ_NEW = '''    cap = current_platform.get_device_capability()
    if (
        cap is not None
        and cap.major >= 12
        and wo_a.weight.dtype == torch.float8_e4m3fn
    ):
        o_proj_input, _ = fused_inv_rope_fp8_quant(
            o,
            positions,
            cos_sin_cache,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            quant_group_size=32,
            tma_aligned_scales=False,
            quantize=False,
        )
        z = _sm120_grouped_wo_a(o_proj_input, wo_a, n_groups, o_lora_rank)
        return wo_b(z.flatten(1))
    use_fp8 = wo_a.weight.dtype == torch.float8_e4m3fn
    o_proj_input, o_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,'''

HELPER = '''
def _sm120_grouped_wo_a(
    o_proj_input: torch.Tensor,
    wo_a: nn.Module,
    n_groups: int,
    o_lora_rank: int,
) -> torch.Tensor:
    """Grouped wo_a via b12x wo_a_dense_gemm_mxfp8. o_proj_input is [B, G, K]."""
    import torch
    from b12x.gemm._shared.wo_mxfp8 import (
        MXFP8Rows,
        pack_mxfp8_scales_for_dense_gemm,
        quantize_wo_a_input_mxfp8,
        wo_a_dense_gemm_mxfp8,
    )

    tokens, groups, k = o_proj_input.shape
    if groups != n_groups:
        raise ValueError(f"wo_a groups {groups} != n_groups {n_groups}")
    n_out = int(o_lora_rank)
    scale2d = getattr(wo_a, "weight_scale_2d", None)
    if scale2d is None:
        raise RuntimeError("sm120 wo_a missing weight_scale_2d")
    packed = getattr(wo_a, "_sm120_wo_a_packed", None)
    if packed is None:
        values = wo_a.weight.contiguous().view(groups, n_out, k).permute(1, 2, 0)
        scale_rows = scale2d.reshape(groups, n_out, k // 32)
        if scale_rows.dtype == torch.uint8:
            scale_rows = scale_rows.view(torch.float8_e8m0fnu)
        scale_mma = pack_mxfp8_scales_for_dense_gemm(
            scale2d, m=n_out, k=k, num_groups=groups
        )
        values_tiled = None
        if (groups, n_out, k) == (4, 1024, 4096):
            values_tiled = (
                values.permute(2, 0, 1)
                .reshape(groups, n_out // 64, 64, k // 128, 128)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
            )
        packed = MXFP8Rows(
            values=values,
            scale_rows=scale_rows,
            scale_mma=scale_mma,
            values_tiled=values_tiled,
        )
        wo_a._sm120_wo_a_packed = packed
    x_q = quantize_wo_a_input_mxfp8(o_proj_input)
    tmp = wo_a_dense_gemm_mxfp8(x_q, packed, expected_m=tokens)
    return tmp.permute(0, 2, 1).contiguous()


'''


def patch_attention(src: str) -> str:
    out = src
    if IS_BMM_OLD in out:
        out = out.replace(IS_BMM_OLD, IS_BMM_NEW)
    if IS_BMM_NEW not in out:
        raise SystemExit("sm120_wo_a: is_bmm=False not present")
    if IS_BMM_OLD in out:
        raise SystemExit("sm120_wo_a: is_bmm=True still present")
    return out


def patch_flashinfer(src: str) -> str:
    out = src
    if SCALE_OLD in out:
        out = out.replace(SCALE_OLD, SCALE_NEW, 1)
    if "weight_scale_2d = Parameter" not in out:
        raise SystemExit("sm120_wo_a: weight_scale_2d not present")
    return out


def patch_o_proj(src: str) -> str:
    out = src
    if "_sm120_grouped_wo_a" not in out:
        needle = "def deep_gemm_fp8_o_proj("
        if needle not in out:
            raise SystemExit("sm120_wo_a: deep_gemm_fp8_o_proj not present")
        out = out.replace(needle, HELPER + needle, 1)
    if "cap.major >= 12" not in out:
        if OPROJ_OLD not in out:
            raise SystemExit("sm120_wo_a: o_proj fp8_einsum prelude not present")
        out = out.replace(OPROJ_OLD, OPROJ_NEW, 1)
    if "_sm120_grouped_wo_a" not in out:
        raise SystemExit("sm120_wo_a: grouped wo_a helper not present")
    if "cap.major >= 12" not in out:
        raise SystemExit("sm120_wo_a: SM120 o_proj branch not present")
    return out


def apply(tree: Path) -> bool:
    changed = False
    for path in tree.rglob("attention.py"):
        text = path.read_text()
        if IS_BMM_OLD not in text and IS_BMM_NEW not in text:
            continue
        out = patch_attention(text)
        if out != text:
            path.write_text(out)
            print(f"patched {path}")
            changed = True
    for path in tree.rglob("mxfp8/flashinfer.py"):
        text = path.read_text()
        if SCALE_OLD not in text and "weight_scale_2d = Parameter" not in text:
            continue
        out = patch_flashinfer(text)
        if out != text:
            path.write_text(out)
            print(f"patched {path}")
            changed = True
    for path in tree.rglob("nvidia/ops/o_proj.py"):
        text = path.read_text()
        if "def deep_gemm_fp8_o_proj(" not in text:
            continue
        out = patch_o_proj(text)
        if out != text:
            path.write_text(out)
            print(f"patched {path}")
            changed = True
    return changed


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: sm120_wo_a.py TREE", file=sys.stderr)
        return 2
    apply(Path(argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
