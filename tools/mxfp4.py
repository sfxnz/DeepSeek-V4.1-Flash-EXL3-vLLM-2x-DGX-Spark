#!/usr/bin/env python3
"""Dequant official DeepSeek MXFP4 expert weights (E2M1 x UE8M0 per 32)."""
from __future__ import annotations

import torch

# E2M1 codes 0..7: 0, 0.5, 1, 1.5, 2, 3, 4, 6. High nibble is the second value.
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """packed uint8 [..., in/2] -> float32 [..., in] with sign."""
    low = packed & 0x0F
    high = packed >> 4
    table = _E2M1.to(device=packed.device, dtype=torch.float32)
    mag_lo = table[(low & 7).reshape(-1).long()].reshape(low.shape)
    mag_hi = table[(high & 7).reshape(-1).long()].reshape(high.shape)
    sign_lo = torch.where((low >> 3) != 0, -1.0, 1.0)
    sign_hi = torch.where((high >> 3) != 0, -1.0, 1.0)
    return torch.stack((mag_lo * sign_lo, mag_hi * sign_hi), dim=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 2
    )



def dequant_mxfp4(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block: int = 32,
) -> torch.Tensor:
    """weight: uint8 [out, in/2] or float4 packed; scale: uint8 [out, in/block] ue8m0.

    Returns bf16 [out, in].
    """
    w = weight.view(torch.uint8)
    s = scale.view(torch.uint8)
    vals = unpack_e2m1(w).float()
    out, inn = vals.shape
    if inn % block:
        raise ValueError(f"K={inn} not divisible by block={block}")
    if s.shape != (out, inn // block):
        raise ValueError(f"scale shape {tuple(s.shape)} != {(out, inn // block)}")
    scale_f = (s.to(torch.int32) << 23).view(torch.float32)
    vals = vals.view(out, inn // block, block) * scale_f.unsqueeze(-1)
    return vals.reshape(out, inn).to(torch.bfloat16)
