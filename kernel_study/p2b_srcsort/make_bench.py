#!/usr/bin/env python3
"""Generate the p2b src-sort microbench sources from the committed pin (CPU only).

Writes build/:
  chain_base.cu     fixture + the docker/Dockerfile p2b chain (shapes..fshift)
  chain_srcsort.cu  chain_base + widen_p2b_srcsort.py (what the image compiles)
  bench_srcsort.cu  chain_srcsort with a runtime sort switch and a pybind module

The bench replaces the DSV41_P2B_SRC_SORT getenv (read once per process)
with set_sort(0|1), so one process can A/B off vs on on identical data.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PIN = ROOT / "tests/fixtures/p2b_moe.pin.cu"
CHAIN = ("shapes", "mrow", "cfg1", "codebook", "fshift")

BENCH = r'''

// ---------------------------------------------------------------------------
// kernel_study/p2b_srcsort bench bindings
// ---------------------------------------------------------------------------
#include <pybind11/pybind11.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod)
{
    mod.def("p2b_fused_moe", &p2b_fused_moe_cuda, "p2b fused MoE (chain + srcsort)");
    mod.def("set_sort", [](int on) { g_bench_sort = on; });
    mod.def("get_sort", []() { return g_bench_sort; });
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
    srcsort = _patcher("widen_p2b_srcsort").patch_cu(base)
    bench = _sub1(srcsort, "constexpr int P2B_SORT_CAP = 64;\n",
                  "constexpr int P2B_SORT_CAP = 64;\nstatic int g_bench_sort = 0;\n")
    bench = _sub1(bench, "p2b_src_sort_enabled() && m * e <= P2B_SORT_CAP",
                  "g_bench_sort && m * e <= P2B_SORT_CAP")
    return {"chain_base.cu": base, "chain_srcsort.cu": srcsort, "bench_srcsort.cu": bench + BENCH}


def main() -> int:
    out = HERE / "build"
    out.mkdir(exist_ok=True)
    for name, text in build().items():
        (out / name).write_text(text)
        print(f"wrote {out / name} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
