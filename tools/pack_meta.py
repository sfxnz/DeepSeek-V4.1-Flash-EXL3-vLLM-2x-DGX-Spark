#!/usr/bin/env python3
"""EXL3 mixed-pack metadata for DeepSeek-V4.1-Flash.

Routed experts are EXL3 trellis. Everything else stays in the official
MXFP8/MXFP4/BF16 layout, including Engram tables (read from disk at serve).
DSpark draft experts stay source-format from backbone layer 40.

Codebook is a named pair (marker suffix, p2b cb). MCG is cb=1. MUL1 is cb=2.
The published Hub pin is still 2.0bpw-mcg. Rebuild tools default to MUL1
and p2b cb=2 exists so a later pack can use it. spark1+spark2 measured
MUL1 + cb=2 at 23.52 vs 27.98 prose decode. Do not switch the serve pin.
A pack-only swap without cb=2 drops native p2b onto generic exl3_moe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

BACKBONE_LAYERS = 40
DEFAULT_BITS = 2


@dataclass(frozen=True)
class Codebook:
    name: str
    cb: int
    suffix: str
    quant_key: str


CODEBOOKS = {
    "mcg": Codebook("mcg", 1, "mcg", "mcg"),
    "mul1": Codebook("mul1", 2, "mul1", "mul1"),
}
DEFAULT_CODEBOOK = "mul1"
SERVE_REVISION = "2.0bpw-mcg-lmhead-mxfp8"


def get_codebook(name: str) -> Codebook:
    key = str(name).strip().lower()
    found = CODEBOOKS.get(key)
    if found is None:
        raise ValueError(f"unsupported codebook={name}")
    return found


def revision_for(bits: int = DEFAULT_BITS, codebook: str = DEFAULT_CODEBOOK) -> str:
    get_codebook(codebook)
    return f"{int(bits)}.0bpw-{codebook}"


def is_routed_expert_tensor(name: str) -> bool:
    """True for backbone routed-expert w1/w2/w3 weight or scale tensors."""
    if ".ffn.shared_experts." in name or ".engram." in name:
        return False
    if ".ffn.experts." not in name:
        return False
    if name.startswith("mtp.") or ".mtp." in name:
        return False
    return name.endswith(
        (".w1.weight", ".w2.weight", ".w3.weight", ".w1.scale", ".w2.scale", ".w3.scale")
    )


def is_routed_expert_weight(name: str) -> bool:
    return is_routed_expert_tensor(name) and name.endswith(".weight")


def build_quantization_config(
    *,
    bits: int = DEFAULT_BITS,
    codebook: str = DEFAULT_CODEBOOK,
    layer_bits: dict[str, int] | None = None,
) -> dict[str, Any]:
    if bits not in (2, 3, 4, 5, 6):
        raise ValueError(f"unsupported EXL3 bits={bits}")
    cb = get_codebook(codebook)
    cfg: dict[str, Any] = {
        "quant_method": "exl3",
        "bits": int(bits),
        "codebook": cb.name,
        "mtp_experts": "source",
        "mtp_experts_start_layer": BACKBONE_LAYERS,
        # Copied onto Exl3Config so DSV4.1's scale mapper uses MXFP8
        # ``weight_scale`` (not block-FP8 ``weight_scale_inv``).
        "weight_block_size": [32, 32],
        # Not bf16_as_stored: attention/shared-expert tensors stay official
        # MXFP8 (ue8m0, 32x32). vllm-exl3 delegates those to DeepseekV4FP8Config.
        "non_routed_quantization": {
            "quant_method": "deepseek_v4_fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "scale_fmt": "ue8m0",
            "weight_block_size": [32, 32],
            "expert_dtype": "fp4",
        },
    }
    if layer_bits:
        cfg["layer_bits"] = {str(k): int(v) for k, v in layer_bits.items()}
    return cfg


def apply_pack_config(config: dict[str, Any], quant: dict[str, Any]) -> dict[str, Any]:
    """Write EXL3 quantization_config onto a DeepSeek-V4.1 config.json object."""
    out = dict(config)
    out["quantization_config"] = quant
    # Keep nested text_config's architecture fields; do not flatten.
    return out
