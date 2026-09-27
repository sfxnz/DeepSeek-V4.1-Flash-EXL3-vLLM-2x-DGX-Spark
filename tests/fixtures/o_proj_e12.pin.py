# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_einsum


def compute_fp8_einsum_recipe(
    block_size: int = 128,
) -> tuple[tuple[int, int, int], bool]:
    """fp8_einsum recipe + scale layout for the current GPU arch.

    SM90 keeps block-row FP32 scales. SM100 uses packed per-row E8M0 scales.

    Returns ``(einsum_recipe, tma_aligned_scales)`` for ``deep_gemm_fp8_o_proj``.
    """
    cap = current_platform.get_device_capability()
    assert cap is not None, "DeepseekV4 attention requires a CUDA device"
    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, block_size)
    tma_aligned_scales = cap.major >= 10
    return einsum_recipe, tma_aligned_scales



# --- fix_o_proj_woa_fp8: exact requant of emulated-dequant wo_a ---
def _woa_try_requant(wo_a, n_groups, o_lora_rank):
    """One-time BF16 -> e4m3 requant using the layer's retained e8m0 scales.

    Returns True when wo_a now carries fp8 weights and the fp8 einsum path
    may run. Self-tests the einsum on a dummy row first; any failure leaves
    the layer untouched (stock bf16 bmm fallback) and is logged once.
    """
    import torch as _torch

    if getattr(wo_a, "_woa_fp8ok", False):
        return True
    if getattr(wo_a, "_woa_fp8fail", False):
        return False
    try:
        if _torch.cuda.is_current_stream_capturing():
            return False  # resolve during an eager pass, never mid-capture
        w = wo_a.weight
        if w.dtype != _torch.bfloat16 or w.dim() != 2:
            wo_a._woa_fp8fail = True
            return False
        scale = getattr(wo_a, "weight_scale", None)
        if scale is None:
            scale = getattr(wo_a, "weight_scale_inv", None)
        if scale is None or scale.dim() != 2:
            wo_a._woa_fp8fail = True
            return False
        n, k = int(w.shape[0]), int(w.shape[1])
        if n != n_groups * o_lora_rank or k % 32 != 0 or int(scale.shape[1]) != k // 32:
            wo_a._woa_fp8fail = True
            return False
        if scale.dtype == _torch.float32:
            s = scale
        elif scale.dtype in (_torch.uint8,):
            s = scale.view(_torch.float8_e8m0fnu).to(_torch.float32)
        elif scale.dtype == _torch.float8_e8m0fnu:
            s = scale.to(_torch.float32)
        else:
            wo_a._woa_fp8fail = True
            return False
        if int(s.shape[0]) != n:
            wo_a._woa_fp8fail = True
            return False
        wf = w.float().view(n, k // 32, 32)
        w8 = (wf / s.unsqueeze(-1)).to(_torch.float8_e4m3fn).view(n, k)
        w8_3d = w8.view(n_groups, o_lora_rank, k)
        s_3d = s.view(n_groups, o_lora_rank, k // 32)
        from vllm.utils.deep_gemm import fp8_einsum as _fp8_einsum

        x8 = _torch.ones(1, n_groups, k, device=w.device, dtype=_torch.float8_e4m3fn)
        xs = _torch.ones(1, n_groups, k // 32, device=w.device, dtype=_torch.float32)
        z = _torch.empty(1, n_groups, o_lora_rank, device=w.device, dtype=_torch.bfloat16)
        _fp8_einsum("bhr,hdr->bhd", (x8, xs), (w8_3d, s_3d), z, recipe=(1, 1, 32))
        _torch.cuda.synchronize()
        wo_a.weight = _torch.nn.Parameter(w8_3d.contiguous(), requires_grad=False)
        wo_a.weight_scale = _torch.nn.Parameter(s_3d.contiguous(), requires_grad=False)
        wo_a._woa_fp8ok = True
        print(
            f"[woa-requant] fp8 einsum engaged: {tuple(w8_3d.shape)}",
            flush=True,
        )
        return True
    except Exception as exc:  # noqa: BLE001 - diagnostic fallback by design
        wo_a._woa_fp8fail = True
        print(f"[woa-requant] fp8 path rejected ({exc!r}); bf16 bmm stays", flush=True)
        return False

def deep_gemm_fp8_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: tuple[int, int, int],
    tma_aligned_scales: bool,
) -> torch.Tensor:
    """O projection: inverse RoPE + grouped wo_a + wo_b.

    Shared by the FlashMLA and FlashInfer CUDA backends. The attention
    layer selects the recipe at initialization.
    """
    use_fp8 = wo_a.weight.dtype == torch.float8_e4m3fn
    # --- probe_wo_a (diagnostic, no behavior change) ---
    import os as _probe_os
    _every = _probe_os.environ.get("DSV41_PROBE_WO_A_EVERY", "0") == "1"
    if _every or not globals().get("_PROBE_WO_A_LOGGED", False):
        globals()["_PROBE_WO_A_LOGGED"] = True
        print(
            f"[wo_a-probe] wo_a.dtype={wo_a.weight.dtype} "
            f"wo_a.shape={tuple(wo_a.weight.shape)} use_fp8={use_fp8} "
            f"wo_b.dtype={getattr(wo_b.weight, 'dtype', None)} "
            f"n_groups={n_groups}",
            flush=True,
        )
    # --- end probe_wo_a ---
    if not use_fp8:
        use_fp8 = _woa_try_requant(wo_a, n_groups, o_lora_rank)
    o_proj_input, o_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        quant_group_size=einsum_recipe[2],
        tma_aligned_scales=tma_aligned_scales,
        quantize=use_fp8,
    )
    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )
    if use_fp8:
        weight_scale = (
            wo_a.weight_scale
            if hasattr(wo_a, "weight_scale")
            else wo_a.weight_scale_inv
        )
        fp8_einsum(
            "bhr,hdr->bhd",
            (o_proj_input, o_scale),
            (wo_a.weight, weight_scale),
            z,
            recipe=einsum_recipe,
        )
    else:
        grouped_weight = wo_a.weight.view(n_groups, o_lora_rank, -1)
        torch.bmm(
            o_proj_input.transpose(0, 1),
            grouped_weight.transpose(1, 2),
            out=z.transpose(0, 1),
        )
    return wo_b(z.flatten(1))
