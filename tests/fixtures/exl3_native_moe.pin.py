# Minimal pin of vllm-exl3 native MoE python loop for widen_p2b_mrow.patch_py.
    extra_args = (intermediate, clamp_limit) if extended_abi else ()
    for row in range(int(x2d.shape[0])):
        result = fn(
            xh[row : row + 1],
            native_out[row : row + 1],
            ptrs["gate_trellis"],
            ptrs["gate_suh"],
            ptrs["gate_svh"],
            ptrs["up_trellis"],
            ptrs["up_suh"],
            ptrs["up_svh"],
            ptrs["down_trellis"],
            ptrs["down_suh"],
            ptrs["down_svh"],
            safe_ids[row],
            safe_weights[row],
            k,
            k,
            k,
            True,
            *extra_args,
        )
        # pybind returns the same output tensor, while lightweight test doubles
        # may return a fresh tensor.  Accommodate both without synchronizing.
        if isinstance(result, torch.Tensor) and result is not native_out:
            native_out[row : row + 1].copy_(result.reshape(1, -1))
    return native_out.to(dtype=torch.float32)
