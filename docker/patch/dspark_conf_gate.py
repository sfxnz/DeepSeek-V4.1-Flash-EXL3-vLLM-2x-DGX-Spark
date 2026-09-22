#!/usr/bin/env python3
"""Gate DSpark Markov bias by the unused confidence head.

compute_confidence already returns sigmoid P(accept) in [0, 1]. Full Markov
is KEEP. Scale 0 (backbone only) lost. Per-position gate is
logits = base + conf * bias so a low-confidence tail does not get the full
Markov kick. Not an id overlay. Official TileLang decode is FP4 TP=8 and
does not fit 121 GiB UMA.
"""

from __future__ import annotations


def dspark_conf_gate_from_env(env_flag: int) -> bool:
    return int(env_flag) == 1


def gate_markov_logits_list(
    base: list[list[float]],
    bias: list[list[float]],
    confidence: list[float],
) -> list[list[float]]:
    """CPU path. out[r][v] = base[r][v] + confidence[r] * bias[r][v]."""
    out: list[list[float]] = []
    for row_base, row_bias, conf in zip(base, bias, confidence, strict=True):
        c = float(conf)
        out.append([float(b) + c * float(u) for b, u in zip(row_base, row_bias, strict=True)])
    return out


def apply_confidence_gate(base_i, bias, confidence):
    """base_i and bias are [R, V]; confidence is [R] in [0, 1] (already sigmoid)."""
    return base_i + confidence.unsqueeze(-1).to(dtype=base_i.dtype) * bias
