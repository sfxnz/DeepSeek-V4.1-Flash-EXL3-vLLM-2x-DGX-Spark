"""NVRTC compile + CUDA driver launch for the mHC det kernels.

cuda.bindings (cuda-python, in the serve image) only: no torch extension build, no nvcc.
A Module compiles one CUDA source for this GPU's sm_XYa and loads the cubin into the
current (torch primary) context. Function.launch uses cuLaunchKernelEx on torch's
current stream, so it is captured by torch.cuda.graph like any other kernel, and can
carry the programmatic-stream-serialization (PDL) attribute.
"""

from __future__ import annotations

import ctypes

from cuda.bindings import driver as cu
from cuda.bindings import nvrtc


def _ok(res, what: str):
    err = res[0]
    if isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"{what}: {err}")
    elif err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{what}: {err}")
    return res[1] if len(res) == 2 else res[1:]


def device_arch() -> str:
    import torch

    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}a"


def ensure_context() -> None:
    """Make torch's primary context current on this thread (module loads need one)."""
    ctx = _ok(cu.cuCtxGetCurrent(), "cuCtxGetCurrent")
    if int(ctx) != 0:
        return
    import torch

    torch.empty(1, device="cuda")  # torch creates and binds the primary context
    ctx = _ok(cu.cuCtxGetCurrent(), "cuCtxGetCurrent")
    if int(ctx) == 0:
        dev = _ok(cu.cuDeviceGet(torch.cuda.current_device()), "cuDeviceGet")
        _ok(cu.cuCtxSetCurrent(_ok(cu.cuDevicePrimaryCtxRetain(dev), "retain")), "cuCtxSetCurrent")


class Function:
    def __init__(self, handle, name: str) -> None:
        self.handle = handle
        self.name = name
        self._max_smem = 48 * 1024

    def reserve_smem(self, nbytes: int) -> None:
        if nbytes > self._max_smem:
            attr = cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
            _ok(cu.cuFuncSetAttribute(self.handle, attr, int(nbytes)), f"{self.name} smem {nbytes}")
            self._max_smem = int(nbytes)

    def launch(self, grid, block, smem: int, args, stream: int | None = None, pdl: bool = False) -> None:
        """args: [(value, ctypes type)]; stream: raw cudaStream_t (default torch current)."""
        if stream is None:
            import torch

            stream = torch.cuda.current_stream().cuda_stream
        self.reserve_smem(smem)
        cfg = cu.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = (tuple(grid) + (1, 1))[:3]
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = (tuple(block) + (1, 1))[:3]
        cfg.sharedMemBytes = int(smem)
        cfg.hStream = cu.CUstream(int(stream))
        if pdl:
            attr = cu.CUlaunchAttribute()
            attr.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
            attr.value.programmaticStreamSerializationAllowed = 1
            cfg.attrs = [attr]
            cfg.numAttrs = 1
        else:
            cfg.numAttrs = 0
        values = tuple(v for v, _ in args)
        types = tuple(t for _, t in args)
        _ok(cu.cuLaunchKernelEx(cfg, self.handle, (values, types), 0), f"launch {self.name}")


class Module:
    def __init__(self, src: str, name: str = "mhc_det.cu", opts=(), arch: str | None = None) -> None:
        arch = arch or device_arch()
        prog = _ok(nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], []), "nvrtcCreateProgram")
        options = [f"--gpu-architecture={arch}".encode(), b"-std=c++17", b"-default-device"]
        options += [o.encode() if isinstance(o, str) else o for o in opts]
        res = nvrtc.nvrtcCompileProgram(prog, len(options), options)
        size = _ok(nvrtc.nvrtcGetProgramLogSize(prog), "log size")
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        self.log = log.decode(errors="replace").rstrip("\x00 \n")
        if res[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"nvrtc compile {name} ({arch}) failed: {res[0]}\n{self.log}")
        size = _ok(nvrtc.nvrtcGetCUBINSize(prog), "cubin size")
        cubin = b" " * size
        _ok(nvrtc.nvrtcGetCUBIN(prog, cubin), "cubin")
        nvrtc.nvrtcDestroyProgram(prog)
        self.cubin = cubin
        self.arch = arch
        ensure_context()
        self.handle = _ok(cu.cuModuleLoadData(cubin), "cuModuleLoadData")
        self._fns: dict[str, Function] = {}

    def function(self, name: str) -> Function:
        fn = self._fns.get(name)
        if fn is None:
            fn = Function(_ok(cu.cuModuleGetFunction(self.handle, name.encode()), f"get {name}"), name)
            self._fns[name] = fn
        return fn


__all__ = ["Module", "Function", "device_arch", "ctypes"]
