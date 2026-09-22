#!/usr/bin/env python3
"""Route o_proj wo_a back onto the fp8 einsum when emulation dequantized it.

On GB10 the ModelOpt MXFP8 BMM path selects the emulation kernel, which
dequantizes wo_a to BF16 at load (VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD,
default on). deep_gemm_fp8_o_proj then falls back to a strided bf16
torch.bmm that cublas maps to a cutlass_80 WMMA kernel (~176 us/layer at
decode, ~0.9 TFLOPS; measured in the live profile).

wo_a's e8m0 block scales survive on the layer, so requantizing the bf16
weight with THOSE scales is a bit-exact roundtrip of the original e4m3
weights. The patch converts once (capture-guarded, self-tested): it runs a
dummy fp8_einsum with the requantized weight; on any failure it reverts to
the stock bf16 path and logs, so the worst case is the status quo. When it
holds, o_proj runs the same deep_gemm fp8 einsum the native path uses
(~45 us vs ~157 us at m=5, 3.5x, isolated bench).

Apply AFTER probe_wo_a.py (anchors on the probe-inserted block).
Idempotent. No new numerics: identical bytes into the intended fp8 path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

HELPER = r'''
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

'''

ANCHOR_OLD = """    o_proj_input, o_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,"""

ANCHOR_NEW = """    if not use_fp8:
        use_fp8 = _woa_try_requant(wo_a, n_groups, o_lora_rank)
    o_proj_input, o_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,"""

HELPER_ANCHOR = "def deep_gemm_fp8_o_proj("


def patch(src: str) -> str:
    if "_woa_try_requant" in src:
        return src
    if src.count(HELPER_ANCHOR) != 1:
        raise SystemExit("fix_o_proj_woa_fp8: function anchor not found")
    if src.count(ANCHOR_OLD) != 1:
        raise SystemExit(
            f"fix_o_proj_woa_fp8: inv-rope call anchor not found (count={src.count(ANCHOR_OLD)}; apply after probe_wo_a.py)"
        )
    out = src.replace(HELPER_ANCHOR, HELPER + HELPER_ANCHOR, 1)
    out = out.replace(ANCHOR_OLD, ANCHOR_NEW, 1)
    if "_woa_try_requant(wo_a, n_groups, o_lora_rank)" not in out:
        raise SystemExit("fix_o_proj_woa_fp8: rewrite failed")
    return out


def apply(path: Path) -> None:
    src = path.read_text()
    out = patch(src)
    if out != src:
        path.write_text(out)
    print(f"patched {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", type=Path)
    args = ap.parse_args()
    apply(args.target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
