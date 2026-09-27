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
disarms to the stock ops for good. Decode batches replay CUDA graphs, so eager
decode calls may never come: the first eager call of the rewritten function
(any size; the profile run's returns before the glue) also self-tests the
kernel on synthetic routings (m 1..8, invalid ids, NaN/inf weights and x)
against the stock ops, and prints LOG_ENGAGED or disarms.

The same lever drops one of the two conversions after p2b: the native path
returns p2b's fp16 output as fp32 and apply_exl3_experts casts that to the
model dtype (two kernels between p2b and the MoE all-reduce, 40 a step).
_apply_native_fused_moe gains out_dtype (default fp32, the stock contract for
every other caller) and apply_exl3_experts passes x.dtype: fp16 -> bf16 in one
cast is bit-identical to fp16 -> fp32 -> bf16 (the fp32 step is exact).

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

# p2b output: one cast to the model dtype (apply_exl3_experts only).
SIG_OLD = """    expert_map: torch.Tensor | None,
    limit: float | None = None,
) -> torch.Tensor | None:
    \"\"\"Run the native cooperative kernel for decode rows when it is safe."""
SIG_NEW = """    expert_map: torch.Tensor | None,
    limit: float | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    \"\"\"Run the native cooperative kernel for decode rows when it is safe."""
HEAD_OLD = """    module = _load_native_exl3_ext()
    if module is None or not _native_moe_dimensions_supported("""
HEAD_NEW = """    _dsv41_moe_prep.selftest_once(x2d.device)
    module = _load_native_exl3_ext()
    if module is None or not _native_moe_dimensions_supported("""
RET_OLD = """    return native_out.to(dtype=torch.float32)"""
RET_NEW = """    # dsv41 moe_prep_fused: one cast when the caller names its dtype (bit-exact).
    return native_out.to(dtype=torch.float32 if out_dtype is None else out_dtype)"""
CALL_OLD = """            native_out = _apply_native_fused_moe(
                x2d, ids, weights, layer, inners, expert_map, limit
            )
        except Exception as exc:
            native_out = None
            layer._exl3_native_error = repr(exc)
            getattr(logger, "warning_once", logger.warning)("""
CALL_NEW = """            native_out = _apply_native_fused_moe(
                x2d, ids, weights, layer, inners, expert_map, limit, out_dtype=x.dtype
            )
        except Exception as exc:
            native_out = None
            layer._exl3_native_error = repr(exc)
            getattr(logger, "warning_once", logger.warning)("""

_STATE = {"armed": True, "verify_left": DEFAULT_VERIFY, "engaged": False}


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return env.get("DSV41_MOE_PREP_FUSED", "0") == "1"


def verify_calls(env=None) -> int:
    env = os.environ if env is None else env
    return max(1, int(env.get("DSV41_MOE_PREP_VERIFY", "") or DEFAULT_VERIFY))


def _replace_once(src: str, old: str, new: str, what: str) -> str:
    if src.count(old) != 1:
        raise ValueError(f"{what} not found verbatim")
    return src.replace(old, new)


def patch_source(src: str) -> str:
    """The image's _apply_native_fused_moe: glue block, out_dtype parameter."""
    src = _replace_once(src, HEAD_OLD, HEAD_NEW, "head of _apply_native_fused_moe")
    src = _replace_once(src, OLD_BLOCK, NEW_BLOCK, "glue block of _apply_native_fused_moe")
    src = _replace_once(src, SIG_OLD, SIG_NEW, "signature of _apply_native_fused_moe")
    return _replace_once(src, RET_OLD, RET_NEW, "return of _apply_native_fused_moe")


def patch_experts_source(src: str) -> str:
    """The image's apply_exl3_experts: pass the model dtype to the native path."""
    return _replace_once(src, CALL_OLD, CALL_NEW, "native call of apply_exl3_experts")


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

    def selftest_once(self, device) -> None:
        """Fused vs stock on synthetic routings, once, outside graph capture."""
        torch = self.torch
        if not _STATE["armed"] or _STATE["engaged"] or torch.cuda.is_current_stream_capturing():
            return
        try:
            g = torch.Generator(device=device).manual_seed(0)
            for m in range(1, 9):
                ids = torch.randint(0, 384, (m, 6), device=device, generator=g)
                ids.view(-1)[0] = -1
                ids.view(-1)[-1] = 384
                w = torch.rand(m, 6, device=device, generator=g)
                w.view(-1)[1] = float("nan")
                w.view(-1)[2] = float("inf")
                x = (torch.randn(m, 5120, device=device, generator=g) * 3).to(torch.bfloat16)
                x.view(-1)[5] = float("inf")
                out = self.fused(ids, w, x, 384)
                ref = self.stock(ids, w, x, 384, None)
                if not (
                    torch.equal(out[0], ref[0])
                    and torch.equal(out[1].view(torch.int16), ref[1].view(torch.int16))
                    and torch.equal(out[2].view(torch.int16), ref[2].view(torch.int16))
                ):
                    raise RuntimeError(f"fused != stock in the self-test (m={m})")
        except Exception as exc:  # noqa: BLE001
            _disarm(exc)
            return
        _STATE["engaged"] = True
        print("dsv41: moe prep fused self-check bit-exact (synthetic, m=1..8)", flush=True)

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
        experts_src = patch_experts_source(inspect.getsource(exl3.apply_exl3_experts))
    except (OSError, ValueError) as exc:
        _disarm(exc)
        return "not installed"
    _STATE["verify_left"] = verify_calls()
    ns = exl3.__dict__
    ns["_dsv41_moe_prep"] = MoePrep(torch, _build_kernel(tl, triton), triton, exl3.map_topk_to_local)
    exec(compile(new_src, f"{exl3.__file__} [dsv41 moe_prep_fused]", "exec"), ns)
    exec(compile(experts_src, f"{exl3.__file__} [dsv41 moe_prep_fused]", "exec"), ns)
    ns["_apply_native_fused_moe"]._dsv41_moe_prep = True
    return (
        "vllm_exl3 _apply_native_fused_moe glue fused, p2b output cast once "
        f"(verify={_STATE['verify_left']})"
    )
