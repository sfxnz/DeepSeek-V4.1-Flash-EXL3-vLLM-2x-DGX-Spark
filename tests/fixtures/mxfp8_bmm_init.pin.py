def init_mxfp8_linear_kernel(*, bmm_batch_size: int | None = None) -> Mxfp8LinearKernel:
    """Select and instantiate the best MXFP8 linear kernel for the
    current platform."""
    config = Mxfp8LinearLayerConfig(bmm_batch_size=bmm_batch_size)

    platform = current_platform._enum
    possible: list[type[Mxfp8LinearKernel]]
    if bmm_batch_size is not None:
        possible = (
            [DeepGemmMxfp8BmmLinearKernel, EmulationMxfp8LinearKernel]
            if current_platform.is_cuda()
            else []
        )
    else:
        possible = list(_POSSIBLE_MXFP8_KERNELS.get(platform, []))
