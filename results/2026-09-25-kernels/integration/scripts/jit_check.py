#!/usr/bin/env python3
"""CPU-only (no GPU): the image's toolchain compiles every round-3 first-use kernel for sm_121a,
through the same entry the serve uses where that entry needs no device, and the kernels are in
the output. Writes one JSON object to argv[1].

- dense_gemv_kernel.cu: dense_gemv._ext() (torch cpp_extension + nvcc, TORCH_CUDA_ARCH_LIST as
  run.sh sets it), then the extension's functions and gemv_kernel symbols in the .so.
- mhc_det.cu: NVRTC with mhc_det_rt.Module's options for sm_121a (Module itself also loads the
  cubin into a CUDA context, which needs a GPU), then the six extern "C" kernel names in the cubin.
- engram_native.c: engram_native_stage.build() + load() (gcc, ctypes), ABI version check.
The Triton kernels (l2pf, moe prep, swa meta, candidate mask, wp GEMV) compile on the GPU at
first call; not covered here."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

PATCH = Path("/opt/dsv41-patch")
MHC_KERNELS = ("mhc_det_post", "mhc_det_norm", "mhc_det_norm_li", "mhc_det_norm_coef",
               "mhc_det_gemm_t8", "mhc_det_gemm_t16")


def dense_gemv() -> dict:
    import dense_gemv as dg

    t = time.time()
    ext = dg._ext()
    so = Path(ext.__file__).read_bytes()
    return {"ok": all(hasattr(ext, f) for f in ("gemv", "plan_grid", "gemv_grouped")) and b"gemv_kernel" in so,
            "functions": sorted(n for n in dir(ext) if not n.startswith("_")),
            "so": ext.__file__, "so_bytes": len(so), "gemv_kernel_refs": so.count(b"gemv_kernel"),
            "sm_121a": so.count(b"sm_121a"), "seconds": round(time.time() - t, 1)}


def mhc_det() -> dict:
    from cuda.bindings import nvrtc
    from mhc_det_rt import _ok

    src = (PATCH / "mhc_det.cu").read_text()
    t = time.time()
    prog = _ok(nvrtc.nvrtcCreateProgram(src.encode(), b"mhc_det.cu", 0, [], []), "create")
    opts = [b"--gpu-architecture=sm_121a", b"-std=c++17", b"-default-device"]  # = mhc_det_rt.Module
    res = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    size = _ok(nvrtc.nvrtcGetProgramLogSize(prog), "log size")
    log = b" " * size
    nvrtc.nvrtcGetProgramLog(prog, log)
    if res[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        return {"ok": False, "error": str(res[0]), "log": log.decode(errors="replace")[-2000:]}
    cubin = b" " * _ok(nvrtc.nvrtcGetCUBINSize(prog), "cubin size")
    _ok(nvrtc.nvrtcGetCUBIN(prog, cubin), "cubin")
    found = {k: cubin.count(k.encode() + b"\0") > 0 for k in MHC_KERNELS}
    return {"ok": all(found.values()), "kernels": found, "cubin_bytes": len(cubin),
            "log": log.decode(errors="replace").rstrip("\x00 \n")[-500:], "seconds": round(time.time() - t, 1)}


def engram_native() -> dict:
    import engram_native_stage as ens

    with tempfile.TemporaryDirectory() as d:
        so = ens.build(cache_dir=Path(d))
        lib = ens.load(so)
        return {"ok": lib.eng_abi_version() == ens.ABI_VERSION, "abi": lib.eng_abi_version(),
                "so": so.name, "cmd": " ".join(ens.compile_cmd(Path("engram_native.c"), Path(so.name)))}


def main() -> int:
    out = {"env": {k: os.environ.get(k, "") for k in ("TORCH_CUDA_ARCH_LIST", "CUDA_HOME")},
           "gpu_devices": len([p for p in os.listdir("/dev") if p.startswith("nvidia")])}
    for name, fn in (("dense_gemv", dense_gemv), ("mhc_det", mhc_det), ("engram_native", engram_native)):
        try:
            out[name] = fn()
        except Exception as exc:  # noqa: BLE001
            out[name] = {"ok": False, "error": repr(exc)[:1500]}
    out["pass"] = all(out[n]["ok"] for n in ("dense_gemv", "mhc_det", "engram_native"))
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({n: out[n].get("ok") for n in ("dense_gemv", "mhc_det", "engram_native")} | {"pass": out["pass"]}))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
