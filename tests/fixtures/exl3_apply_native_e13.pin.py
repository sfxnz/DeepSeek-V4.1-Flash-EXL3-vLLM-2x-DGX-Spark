# Pinned from image dsv41-flash-exl3-sm121:canonical-e13 (sha256:c81762335a12),
# /usr/local/lib/python3.12/dist-packages/vllm_exl3/exl3.py (after the image's
# build-time widen_p2b_* patches): _apply_native_fused_moe and apply_exl3_experts.
# docker/patch/moe_prep_fused.py (DSV41_MOE_PREP_FUSED) rewrites both.


def _apply_native_fused_moe(
    x2d: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    layer: torch.nn.Module,
    inners: list[dict[str, Any]],
    expert_map: torch.Tensor | None,
    limit: float | None = None,
) -> torch.Tensor | None:
    """Run the native cooperative kernel for decode rows when it is safe.

    The native ABI consumes one input row and one routing list per launch.  A
    decode batch is therefore submitted as row views into one preallocated
    output tensor.  No per-token pointer/weight tensors are allocated; the only
    conversion is one contiguous int32 routing table for the complete batch.
    Invalid/non-local IDs are clamped to a valid pointer and receive zero
    routing weight, preventing an out-of-bounds read while preserving fallback
    semantics.
    """
    module = _load_native_exl3_ext()
    if module is None or not _native_moe_dimensions_supported(
        x2d, layer, inners, limit
    ):
        return None
    intermediate = int(getattr(layer, "_exl3_intermediate_local", 2048))
    clamp_limit = float(limit) if limit is not None else 0.0
    extended_abi = getattr(module, "P2B_MOE_ABI_VERSION", 1) >= 2
    if not extended_abi and (intermediate != 2048 or clamp_limit > 0):
        # A stale .so still accepts the legacy arguments but would interpret TP2
        # pointer tables as 2048-wide weights or silently omit required clipping.
        reason = "local intermediate width/clipping requires native MoE ABI 2; rebuild vllm_exl3_c"
        layer._exl3_native_error = reason
        getattr(logger, "warning_once", logger.warning)(
            "Native EXL3 MoE fallback: %s (intermediate=%s, limit=%s)",
            reason, intermediate, clamp_limit,
        )
        return None
    ptrs = getattr(layer, "_exl3_ptrs", None)
    if not isinstance(ptrs, dict):
        return None
    required = (
        "gate_trellis",
        "gate_suh",
        "gate_svh",
        "up_trellis",
        "up_suh",
        "up_svh",
        "down_trellis",
        "down_suh",
        "down_svh",
    )
    if any(key not in ptrs for key in required):
        return None

    n_exp = len(inners)
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
    native_out = torch.empty_like(xh)
    k = int(getattr(layer, "_exl3_k", getattr(layer, "_exl3_bits", 4)))
    fn = module.p2b_fused_moe
    flags = getattr(layer, "_exl3_codebook_flags", (True, False, True, False, True, False))
    mcg = bool(flags[0])
    mul1 = bool(flags[1])
    if mcg == mul1:
        return None
    extra_args = (intermediate, clamp_limit) if extended_abi else ()
    result = fn(
        xh,
        native_out,
        ptrs["gate_trellis"],
        ptrs["gate_suh"],
        ptrs["gate_svh"],
        ptrs["up_trellis"],
        ptrs["up_suh"],
        ptrs["up_svh"],
        ptrs["down_trellis"],
        ptrs["down_suh"],
        ptrs["down_svh"],
        safe_ids,
        safe_weights,
        k,
        k,
        k,
        mcg,
        *extra_args,
    )
    # pybind returns the same output tensor, while lightweight test doubles
    # may return a fresh tensor.  Accommodate both without synchronizing.
    if isinstance(result, torch.Tensor) and result is not native_out:
        native_out.copy_(result.reshape_as(native_out))
    return native_out.to(dtype=torch.float32)


def apply_exl3_experts(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    layer: torch.nn.Module,
    *,
    limit: float | None = None,
    fused: bool | None = None,
) -> torch.Tensor:
    """Shipped routed-expert apply. `fused=None` honors EXL3_FUSED_MOE."""
    if _EXL3_PREFILL_SYNC:
        _prefill_sync(int(x.numel() // x.shape[-1]))
    inners = getattr(layer, "_exl3_inners", None)
    if not inners:
        raise RuntimeError("EXL3 experts were not built after weight load")
    tokens, hidden = x.shape[-2], x.shape[-1]
    x2d = x.reshape(tokens, hidden)
    ids = topk_ids.reshape(tokens, -1).to(torch.long)
    weights = topk_weights.reshape(tokens, -1)
    expert_map = pin_exl3_expert_map(layer, x2d.device)

    # Native p2b is a decode-only path.  It is selected explicitly with
    # ``native`` or automatically when the optional extension is installed;
    # unsupported shapes and launch failures fall through to the established
    # ExLlamaV3/Python implementations below.
    backend = get_moe_kernel_backend()
    if backend == "native" and (fused is not False):
        try:
            native_out = _apply_native_fused_moe(
                x2d, ids, weights, layer, inners, expert_map, limit
            )
        except Exception as exc:
            native_out = None
            layer._exl3_native_error = repr(exc)
            getattr(logger, "warning_once", logger.warning)(
                "Native EXL3 MoE dispatch failed; falling back to %s: %s",
                "ExLlamaV3" if _exllamav3_moe_available() else "Python loop",
                exc,
            )
        if native_out is not None:
            layer._exl3_last_apply = "native"
            return native_out.to(dtype=x.dtype)

    have_ptrs = bool(getattr(layer, "_exl3_ptrs", None))
    if fused is True and not have_ptrs:
        raise RuntimeError("EXL3 fused apply requested but pointer tables are missing")
    use_fused = (fused_moe_enabled() if fused is None else bool(fused)) and have_ptrs
    if use_fused:
        try:
            import exllamav3_ext

            use_fused = hasattr(exllamav3_ext, "exl3_moe")
        except Exception:
            use_fused = False
    if use_fused:
        out = apply_exl3_fused_moe(x2d, ids, weights, layer, inners, expert_map, limit)
        layer._exl3_last_apply = "fused"
    else:
        out = apply_exl3_python_loop(x2d, ids, weights, inners, expert_map, limit)
        layer._exl3_last_apply = "loop"
    return out.to(dtype=x.dtype)
