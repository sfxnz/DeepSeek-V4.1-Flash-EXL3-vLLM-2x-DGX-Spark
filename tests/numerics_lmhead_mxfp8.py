#!/usr/bin/env python3
"""Offline numerical validation for the lm_head MXFP8 patch (CPU-only).

Real pack dims from 2.0bpw-mcg config.json: vocab=129280, hidden=5120.
Synthetic bf16 lm_head with realistic row-norm spread. Quantizes the weight
with the pack's dense recipe (per-32 e8m0 blocks, identical math to
docker/patch/lmhead_mxfp8.quantize_weight_mxfp8 and vLLM's
_mxfp8_e4m3_quantize_torch), dequantizes, and compares logits vs the bf16
GEMM on random hidden states:
  - relerr (Frobenius and max-abs) over 1024 states
  - top-1 flip rate over 512 states
  - greedy-decode sanity: 200-step Markov chain over a 256-state subset
    (hidden states drawn near the anchor manifold so argmax margins are
    realistic, not uniform-random worst case)
  - input-side quantization term isolated (activation mxfp8 adds on top).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docker" / "patch"))
from lmhead_mxfp8 import quantize_weight_mxfp8  # noqa: E402

torch.manual_seed(20260921)

V, H = 129280, 5120  # real dims (config.json text_config)


def quantize_activation_mxfp8(x: torch.Tensor):
    """mxfp8_e4m3_quantize math (per-32 blocks along K, swizzle irrelevant)."""
    M, K = x.shape
    xb = x.to(torch.float32).view(M, K // 32, 32)
    amax = xb.abs().amax(dim=-1).clamp(min=torch.finfo(torch.float32).tiny)
    biased = (torch.ceil(torch.log2(amax / 448.0)) + 127.0).clamp(0, 254)
    scales = biased.to(torch.uint8)
    xq = (xb / torch.exp2(biased - 127.0).unsqueeze(-1)).view(M, K).to(
        torch.float8_e4m3fn
    )
    return xq, scales


def dequant_w(xq: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    N, K = xq.shape
    descale = torch.exp2(scales.to(torch.float32) - 127.0)
    x = xq.to(torch.float32).view(N, K // 32, 32) * descale.unsqueeze(-1)
    return x.view(N, K).to(torch.bfloat16)


def dequant_x(xq: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return dequant_w(xq, scales)


def main() -> int:
    dev = torch.device("cpu")
    # Weight: bf16 with row-norm spread similar to a trained head.
    row_scale = torch.empty(V, 1).exponential_(1.0).log()  # ~ lognormal(0, 1)
    w = (torch.randn(V, H, dtype=torch.float32) * row_scale).to(torch.bfloat16)
    wq, ws = quantize_weight_mxfp8(w)
    w_deq = dequant_w(wq, ws)

    # Hidden states: unit-ish scale with correlated structure (one dominant
    # direction family per "token cluster", 64 clusters).
    anchors = torch.randn(64, H, dtype=torch.float32)
    n_states = 1024
    h = (
        anchors[torch.randint(0, 64, (n_states,))] * 1.5
        + torch.randn(n_states, H) * 0.5
    ).to(torch.bfloat16)

    # Reference: bf16 GEMM in fp32 accumulation (what the current path does).
    logits_bf16 = (h.to(torch.float32) @ w.to(torch.float32).t()).to(torch.bfloat16)

    # Weight-only mxfp8 (dequantized weight, bf16 activation).
    logits_w = (h.to(torch.float32) @ w_deq.to(torch.float32).t()).to(torch.bfloat16)

    # Full patch path: activation mxfp8 quantize + weight mxfp8.
    hq, hs = quantize_activation_mxfp8(h.to(torch.float32))
    h_deq = dequant_x(hq, hs)
    logits_full = (h_deq.to(torch.float32) @ w_deq.to(torch.float32).t()).to(
        torch.bfloat16
    )

    def relerr(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
        # normwise Frobenius relerr + max-abs error over logit RMS (the
        # elementwise ratio is meaningless near zero-crossing logits).
        af, bf = a.to(torch.float32), b.to(torch.float32)
        norm = float((af - bf).norm() / bf.norm())
        rms = float(bf.pow(2).mean().sqrt())
        max_abs = float((af - bf).abs().max()) / rms
        return norm, max_abs

    def flips(a: torch.Tensor, b: torch.Tensor) -> int:
        return int((a.argmax(-1) != b.argmax(-1)).sum())

    r_w = relerr(logits_w, logits_bf16)
    r_full = relerr(logits_full, logits_bf16)
    f_w = flips(logits_w[:512], logits_bf16[:512])
    f_full = flips(logits_full[:512], logits_bf16[:512])

    # Greedy decode: 200-step chain where the next hidden state is drawn near
    # the anchor of the sampled token (mild drift), tracking token equality
    # and top-1 margin distribution.
    margins = []
    agree = 0
    steps = 200
    cluster = torch.randint(0, 64, (1,)).item()
    for _ in range(steps):
        hh = (anchors[cluster] * 1.5 + torch.randn(H) * 0.5).to(torch.bfloat16)
        lb = (hh.to(torch.float32) @ w.to(torch.float32).t())
        hq1, hs1 = quantize_activation_mxfp8(hh.unsqueeze(0).to(torch.float32))
        hd = dequant_x(hq1, hs1)
        lf = (hd.to(torch.float32) @ w_deq.to(torch.float32).t()).squeeze(0)
        tb, tf = lb.argmax(), lf.argmax()
        top2 = lb.topk(2).values
        margins.append(float(top2[0] - top2[1]))
        agree += int(tb == tf)
        cluster = int(tb) % 64  # drift the cluster with the chosen token

    margins_t = torch.tensor(margins)
    print(f"dims V={V} H={H}  states={n_states}")
    print(f"weight-only  relerr mean={r_w[0]:.3e} max={r_w[1]:.3e}  "
          f"top1 flips {f_w}/512")
    print(f"full (w+act) relerr mean={r_full[0]:.3e} max={r_full[1]:.3e}  "
          f"top1 flips {f_full}/512")
    print(f"greedy 200-step token agreement: {agree}/200 "
          f"({100.0 * agree / steps:.1f}%)")
    print(f"logit top1-top2 margin: median={margins_t.median():.3f} "
          f"p10={margins_t.quantile(0.1):.3f}")
    # Sanity: scale coverage of the e8m0 scales actually produced.
    print(f"weight scale bytes used: min={int(ws.min())} max={int(ws.max())} "
          f"(127=1.0 scale)")
    ok = r_full[0] < 5e-2 and f_full / 512 < 0.05 and agree / steps > 0.90
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
