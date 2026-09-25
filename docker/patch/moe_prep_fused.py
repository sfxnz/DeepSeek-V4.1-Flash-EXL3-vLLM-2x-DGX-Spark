"""One kernel for the native p2b MoE's input prep (DSV41_MOE_PREP_FUSED=1, default off).

vllm_exl3's _apply_native_fused_moe turns the router output into p2b inputs
with ~13 single-block torch kernels per MoE layer (map_topk_to_local: two
compares, or, fill, where; clamp, int32 cast; valid: two compares, and;
weights -> fp16, valid -> fp16, mul; x -> fp16). In the r3 trace they run on
the routing stream between _dsv4_topk_kernel and p2b, beside the shared
expert's GEMMs, and take ~68 us per layer (tiny kernels queued behind a
bandwidth-saturating GEMM): the routing path ends ~7 us after the shared
expert in ~98% of layers, so p2b waits for it.

With the lever, one Triton kernel writes the same three tensors (int32 ids,
fp16 weights, fp16 x) bit for bit, for the TP layout (no expert_map). Anything
else (an expert_map, non-contiguous or unexpected dtypes) runs the stock ops.
The function body is the image's own source with the glue block swapped for
one call: install() rewrites that block only when it matches verbatim (else
one LOG_DISARMED line and the stock function stays). The first
DSV41_MOE_PREP_VERIFY eager calls (default 16, minimum 1; never during CUDA
graph capture) also run the stock ops and compare raw bits; a mismatch
disarms to the stock ops for good.

Top-level imports are stdlib only. decode_levers.install() calls install()
when the env is on.
"""

from __future__ import annotations

import os

# Boot-log markers for tools/engagement_audit.py.
LOG_ENGAGED = "dsv41: moe prep fused self-check bit-exact"
LOG_DISARMED = "dsv41: moe prep fused DISABLED ->"

DEFAULT_VERIFY = 16
IDS_BLOCK = 64
X_BLOCK = 1024

OLD_BLOCK = """    n_exp = len(inners)
    local = map_topk_to_local(ids, n_exp, expert_map).reshape(ids.shape)
    topk = int(local.shape[-1])
    if topk < 1:
        return None
    # p2b_fused_moe reads int32 IDs and fp16 routing weights.  Clamp before
    # conversion so the invalid sentinel cannot wrap into a large int32 value.
    safe_ids = local.clamp(min=0, max=n_exp - 1).to(dtype=torch.int32).contiguous()
    valid = (local >= 0) & (local < n_exp)
    safe_weights = (
        weights.reshape_as(local)
        .to(dtype=torch.float16)
        .mul(valid.to(dtype=torch.float16))
        .contiguous()
    )
    xh = x2d.to(dtype=torch.float16).contiguous()
"""

NEW_BLOCK = """    n_exp = len(inners)
    if int(ids.shape[-1]) < 1:
        return None
    # dsv41 moe_prep_fused: the stock ops above as one kernel (bit-exact).
    safe_ids, safe_weights, xh = _dsv41_moe_prep(ids, weights, x2d, n_exp, expert_map)
"""

_STATE = {"armed": True, "verify_left": DEFAULT_VERIFY, "engaged": False}


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return env.get("DSV41_MOE_PREP_FUSED", "0") == "1"


def verify_calls(env=None) -> int:
    env = os.environ if env is None else env
    return max(1, int(env.get("DSV41_MOE_PREP_VERIFY", "") or DEFAULT_VERIFY))


def patch_source(src: str) -> str:
    """The image's _apply_native_fused_moe with the glue block replaced."""
    if src.count(OLD_BLOCK) != 1:
        raise ValueError("glue block not found verbatim in _apply_native_fused_moe")
    return src.replace(OLD_BLOCK, NEW_BLOCK)


def _disarm(exc: BaseException) -> None:
    if _STATE["armed"]:
        _STATE["armed"] = False
        print("dsv41: moe prep fused DISABLED -> stock ops: %r" % (exc,), flush=True)


def _build_kernel(tl, triton):
    @triton.jit
    def _moe_prep_kernel(
        ids_ptr, w_ptr, x_ptr, sid_ptr, sw_ptr, xh_ptr,
        n_ids, n_x, n_exp, x_programs,
        IDS_BLOCK: tl.constexpr, X_BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid < x_programs:
            x_offs = pid * X_BLOCK + tl.arange(0, X_BLOCK)
            x_mask = x_offs < n_x
            x = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)
            tl.store(xh_ptr + x_offs, x.to(tl.float32).to(tl.float16), mask=x_mask)
        else:
            i_offs = (pid - x_programs) * IDS_BLOCK + tl.arange(0, IDS_BLOCK)
            i_mask = i_offs < n_ids
            ids = tl.load(ids_ptr + i_offs, mask=i_mask, other=0).to(tl.int64)
            invalid = (ids < 0) | (ids >= n_exp)
            local = tl.where(invalid, n_exp, ids)
            safe = tl.minimum(tl.maximum(local, 0), n_exp - 1)
            valid = (local >= 0) & (local < n_exp)
            w = tl.load(w_ptr + i_offs, mask=i_mask, other=0.0).to(tl.float32).to(tl.float16)
            # half * half runs in float on CUDA and rounds once (torch opmath).
            wv = (w.to(tl.float32) * valid.to(tl.float32)).to(tl.float16)
            tl.store(sid_ptr + i_offs, safe.to(tl.int32), mask=i_mask)
            tl.store(sw_ptr + i_offs, wv, mask=i_mask)

    return _moe_prep_kernel


class MoePrep:
    def __init__(self, torch, kernel, triton, map_topk_to_local):
        self.torch = torch
        self.kernel = kernel
        self.triton = triton
        self.map_topk_to_local = map_topk_to_local

    def stock(self, ids, weights, x2d, n_exp, expert_map):
        """The image's glue ops, verbatim."""
        torch = self.torch
        local = self.map_topk_to_local(ids, n_exp, expert_map).reshape(ids.shape)
        safe_ids = local.clamp(min=0, max=n_exp - 1).to(dtype=torch.int32).contiguous()
        valid = (local >= 0) & (local < n_exp)
        safe_weights = (
            weights.reshape_as(local)
            .to(dtype=torch.float16)
            .mul(valid.to(dtype=torch.float16))
            .contiguous()
        )
        xh = x2d.to(dtype=torch.float16).contiguous()
        return safe_ids, safe_weights, xh

    def fused(self, ids, weights, x2d, n_exp):
        torch = self.torch
        safe_ids = torch.empty(ids.shape, dtype=torch.int32, device=ids.device)
        safe_weights = torch.empty(ids.shape, dtype=torch.float16, device=ids.device)
        xh = torch.empty(x2d.shape, dtype=torch.float16, device=x2d.device)
        n_ids, n_x = ids.numel(), x2d.numel()
        x_programs = self.triton.cdiv(n_x, X_BLOCK)
        grid = (x_programs + self.triton.cdiv(n_ids, IDS_BLOCK),)
        self.kernel[grid](
            ids, weights, x2d, safe_ids, safe_weights, xh,
            n_ids, n_x, n_exp, x_programs,
            IDS_BLOCK=IDS_BLOCK, X_BLOCK=X_BLOCK, num_warps=4,
        )
        return safe_ids, safe_weights, xh

    def usable(self, ids, weights, x2d, expert_map) -> bool:
        torch = self.torch
        return (
            _STATE["armed"]
            and expert_map is None
            and ids.is_cuda
            and ids.dtype in (torch.int64, torch.int32)
            and weights.dtype in (torch.float32, torch.bfloat16, torch.float16)
            and x2d.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and weights.shape == ids.shape
            and ids.is_contiguous()
            and weights.is_contiguous()
            and x2d.is_contiguous()
        )

    def __call__(self, ids, weights, x2d, n_exp, expert_map):
        if not self.usable(ids, weights, x2d, expert_map):
            return self.stock(ids, weights, x2d, n_exp, expert_map)
        out = self.fused(ids, weights, x2d, n_exp)
        torch = self.torch
        if _STATE["verify_left"] > 0 and not torch.cuda.is_current_stream_capturing():
            ref = self.stock(ids, weights, x2d, n_exp, expert_map)
            same = (
                torch.equal(out[0], ref[0])
                and torch.equal(out[1].view(torch.int16), ref[1].view(torch.int16))
                and torch.equal(out[2].view(torch.int16), ref[2].view(torch.int16))
            )
            if not same:
                _disarm(RuntimeError(f"fused != stock (ids {tuple(ids.shape)}, x {tuple(x2d.shape)})"))
                return ref
            _STATE["verify_left"] -= 1
            if not _STATE["engaged"]:
                _STATE["engaged"] = True
                print(
                    "dsv41: moe prep fused self-check bit-exact (ids %s, x %s)"
                    % (tuple(ids.shape), tuple(x2d.shape)),
                    flush=True,
                )
        return out


def install() -> str:
    import inspect

    import torch
    import vllm_exl3.exl3 as exl3
    from vllm.triton_utils import tl, triton

    fn = exl3._apply_native_fused_moe
    if getattr(fn, "_dsv41_moe_prep", False):
        return "already installed"
    try:
        new_src = patch_source(inspect.getsource(fn))
    except (OSError, ValueError) as exc:
        _disarm(exc)
        return "not installed"
    _STATE["verify_left"] = verify_calls()
    ns = exl3.__dict__
    ns["_dsv41_moe_prep"] = MoePrep(torch, _build_kernel(tl, triton), triton, exl3.map_topk_to_local)
    exec(compile(new_src, f"{exl3.__file__} [dsv41 moe_prep_fused]", "exec"), ns)
    ns["_apply_native_fused_moe"]._dsv41_moe_prep = True
    return f"vllm_exl3 _apply_native_fused_moe glue fused (verify={_STATE['verify_left']})"
