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

Stage 2 (woa-scale-prepack, DSV41_WOA_PREPACK=1, default off): the fp32
wo_a weight scale makes fp8_einsum run transpose_and_pack_fp32_into_ue8m0
on every call (~43/step, target and draft layers). The first eager call per
layer packs it once with vLLM's transform_sf_into_required_layout and keeps
the packed tensor as-is (MN-major, TMA-aligned; .contiguous() would break
it). A bitwise self-test (random e4m3 input, fp32 vs packed scale) must pass,
else the layer keeps the fp32 scale and logs. The stage also applies on its
own to an already requant-patched o_proj.py (Dockerfile.woa-prepack).
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Boot-log markers for tools/engagement_audit.py (run.sh post-ready audit).
LOG_ENGAGED = "[woa-requant] fp8 einsum engaged"
LOG_DISARMED = ("[woa-requant] fp8 path rejected", "[woa-prepack] REJECTED")

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


PREPACK_HELPER = r'''
# --- woa_prepack: pack the fp32 wo_a scale once (DSV41_WOA_PREPACK=1) ---
import os as _woa_os

_WOA_PREPACK = _woa_os.environ.get("DSV41_WOA_PREPACK", "0") == "1"


def _woa_prepacked_scale(wo_a, weight_scale, n_groups, o_lora_rank, einsum_recipe):
    """fp32 wo_a scale -> DeepGEMM packed UE8M0 layout, once per layer.

    Returns the scale to hand to fp8_einsum. The packed tensor is cached on
    wo_a as-is. Bitwise self-test first; any mismatch or error keeps the fp32
    scale for this layer and is logged once.
    """
    import torch as _torch

    packed = getattr(wo_a, "_woa_sf_packed", None)
    if packed is not None:
        return packed
    if getattr(wo_a, "_woa_sf_packfail", False) or weight_scale.dtype != _torch.float32:
        return weight_scale
    if _torch.cuda.is_current_stream_capturing():
        return weight_scale  # pack during an eager pass, never mid-capture
    try:
        from vllm.utils.deep_gemm import fp8_einsum as _fp8_einsum
        from vllm.utils.deep_gemm import transform_sf_into_required_layout as _tsf

        w = wo_a.weight
        k = int(w.shape[-1])
        sp = _tsf(weight_scale, o_lora_rank, k, einsum_recipe, n_groups, False)
        gen = _torch.Generator(device=w.device)
        gen.manual_seed(0)
        m, kb = 4, k // int(einsum_recipe[2])
        x8 = _torch.randn(m, n_groups, k, device=w.device, generator=gen)
        x8 = x8.to(_torch.float8_e4m3fn)
        xs = _torch.randint(-8, 8, (m, n_groups, kb), device=w.device, generator=gen)
        xs = _torch.exp2(xs.float())
        z_ref = _torch.empty(m, n_groups, o_lora_rank, device=w.device, dtype=_torch.bfloat16)
        z_new = _torch.empty_like(z_ref)
        _fp8_einsum("bhr,hdr->bhd", (x8, xs), (w, weight_scale), z_ref, recipe=einsum_recipe)
        _fp8_einsum("bhr,hdr->bhd", (x8, xs), (w, sp), z_new, recipe=einsum_recipe)
        _torch.cuda.synchronize()
        if not _torch.equal(z_ref, z_new):
            diff = (z_ref.float() - z_new.float()).abs().max().item()
            raise ValueError(f"self-test not bitwise equal (maxabs {diff})")
        wo_a._woa_sf_packed = sp
        print(
            f"[woa-prepack] packed ue8m0 scale engaged: {tuple(sp.shape)} "
            f"stride={tuple(sp.stride())} dtype={sp.dtype}",
            flush=True,
        )
        return sp
    except Exception as exc:  # noqa: BLE001 - fallback by design, logged loudly
        wo_a._woa_sf_packfail = True
        print(
            f"[woa-prepack] REJECTED ({exc!r}); this layer keeps the fp32 scale "
            "(per-call repack)",
            flush=True,
        )
        return weight_scale

'''

PREPACK_OLD = """        weight_scale = (
            wo_a.weight_scale
            if hasattr(wo_a, "weight_scale")
            else wo_a.weight_scale_inv
        )
        fp8_einsum("""

PREPACK_NEW = """        weight_scale = (
            wo_a.weight_scale
            if hasattr(wo_a, "weight_scale")
            else wo_a.weight_scale_inv
        )
        if _WOA_PREPACK:
            weight_scale = _woa_prepacked_scale(
                wo_a, weight_scale, n_groups, o_lora_rank, einsum_recipe
            )
        fp8_einsum("""


def patch_prepack(src: str) -> str:
    if "_woa_prepacked_scale" in src:
        return src
    if src.count(HELPER_ANCHOR) != 1 or src.count(PREPACK_OLD) != 1:
        raise SystemExit(
            f"fix_o_proj_woa_fp8: prepack anchor not found (count={src.count(PREPACK_OLD)})"
        )
    out = src.replace(HELPER_ANCHOR, PREPACK_HELPER + HELPER_ANCHOR, 1)
    return out.replace(PREPACK_OLD, PREPACK_NEW, 1)


def patch(src: str) -> str:
    return patch_prepack(patch_requant(src))


def patch_requant(src: str) -> str:
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
