#!/usr/bin/env python3
"""Generate the p2b coop microbench sources from the committed pin (CPU only).

Writes build/:
  chain_srcsort.cu  fixture + the docker/Dockerfile.e13 p2b chain (the live canonical-e13 code)
  chain_coop.cu     chain_srcsort + widen_p2b_coop.py (what docker/Dockerfile.e14 compiles)
  bench_coop.cu     chain_coop with a runtime coop switch and a pybind module

The bench replaces the DSV41_P2B_COOP getenv (read once per process) with
set_coop(0|1), so one process can A/B p2b (SORT=0) vs coop (SORT=2) on
identical data. DSV41_P2B_SRC_SORT stays unset, so the baseline is the
served SORT=0 kernel.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PIN = ROOT / "tests/fixtures/p2b_moe.pin.cu"
CHAIN = ("shapes", "mrow", "cfg1", "codebook", "fshift", "srcsort")

BENCH = r'''

// ---------------------------------------------------------------------------
// kernel_study/p2b_coop bench bindings
// ---------------------------------------------------------------------------
#include <pybind11/pybind11.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod)
{
    mod.def("p2b_fused_moe", &p2b_fused_moe_cuda, "p2b fused MoE (chain + srcsort + coop)");
    mod.def("set_coop", [](int on) { g_bench_coop = on; });
    mod.def("get_coop", []() { return g_bench_coop; });
    mod.def("occupancy", [](int sort) {
        void* k = sort == 2 ? (void*) p2b_moe_batched_kernel<2, 1, 2>
                : sort == 1 ? (void*) p2b_moe_batched_kernel<2, 1, 1>
                            : (void*) p2b_moe_batched_kernel<2, 1, 0>;
        int resident = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, k, 256, 0);
        return resident;
    }, "resident 256-thread blocks per SM for p2b_moe_batched_kernel<2, 1, sort>");
}
'''


def _patcher(name: str):
    path = ROOT / "docker/patch" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sub1(src: str, old: str, new: str) -> str:
    if src.count(old) != 1:
        raise SystemExit(f"make_bench: expected one {old!r}, found {src.count(old)}")
    return src.replace(old, new)


def build() -> dict[str, str]:
    base = PIN.read_text()
    for name in CHAIN:
        base = _patcher(f"widen_p2b_{name}").patch_cu(base)
    coop = _patcher("widen_p2b_coop").patch_cu(base)
    bench = _sub1(coop, "constexpr int P2B_COOP_ROWS = 8;\n",
                  "constexpr int P2B_COOP_ROWS = 8;\nstatic int g_bench_coop = 0;\n")
    bench = _sub1(bench, "p2b_coop_enabled() && m * e <= P2B_SORT_CAP",
                  "g_bench_coop && m * e <= P2B_SORT_CAP")
    return {"chain_srcsort.cu": base, "chain_coop.cu": coop, "bench_coop.cu": bench + BENCH}


def main() -> int:
    out = HERE / "build"
    out.mkdir(exist_ok=True)
    for name, text in build().items():
        (out / name).write_text(text)
        print(f"wrote {out / name} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
