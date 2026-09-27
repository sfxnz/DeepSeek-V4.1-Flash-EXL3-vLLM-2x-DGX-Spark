#!/usr/bin/env python3
"""p2b with an in-kernel L2 prefetch prologue: bench build only (k3 comm fix pass).

A separate prefetch kernel before p2b costs its own run time (p2b_prefetch_probe.py: p2b is
a cooperative launch that fills every SM's register file, so it cannot overlap anything, and
a 1-CTA TMA prefetch stays resident ~as long as its transfer). Here p2b issues the prefetch
itself: thread 0 of every block issues cp.async.bulk.prefetch.L2 for its share of up to four
byte ranges (16 KiB chunks c = blockIdx.x, blockIdx.x + gridDim.x, ...) before the kernel's
first phase. The CTAs stay resident for the whole p2b (150-1000 us), so the TMA units have
that long to take the requests, and the lines (normal priority) survive p2b's evict-first
trellis stream (measured). Nothing else in the kernel changes.

Source = kernel_study/p2b_coop/make_bench.build()["bench_coop.cu"] (the canonical-e13 chain
+ the coop switch) with exact substitutions (each must match once):
  kernel params  + const char* pf0..pf3, long long pf0_bytes..pf3_bytes
  prologue       the chunked prefetch, before the SORT builds and the accum zeroing
  launch args    the four ranges from host globals, read at launch (so a CUDA graph captures
                 the ranges current at capture time)
  bindings       set_prefetch([ptr, ...], [bytes, ...]), at most 4; [] [] = no prefetch
Writes kernel_study/comm/.p2b_pf_src/p2b_pf_bench.cu; build() returns the text.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / ".p2b_pf_src" / "p2b_pf_bench.cu"

SIG_OLD = "    int inter,\n    float swiglu_limit)\n{\n    auto grid = cg::this_grid();"
SIG_NEW = ("    int inter,\n    float swiglu_limit,\n"
           "    const char* pf0, long long pf0_bytes, const char* pf1, long long pf1_bytes,\n"
           "    const char* pf2, long long pf2_bytes, const char* pf3, long long pf3_bytes)\n{\n"
           "    auto grid = cg::this_grid();")
PRO_OLD = "    __shared__ float sh_red[8][1][64];\n"
PRO_NEW = PRO_OLD + r"""    // k3 comm bench: L2 prefetch of the next layer's weights, issued by this block's thread 0
    // (16 KiB TMA bulk prefetches, chunks strided over the grid; fire and forget).
    if (threadIdx.x == 0 && (pf0_bytes | pf1_bytes | pf2_bytes | pf3_bytes) > 0) {
        const char* base[4] = {pf0, pf1, pf2, pf3};
        const long long bytes[4] = {pf0_bytes, pf1_bytes, pf2_bytes, pf3_bytes};
        long long first = 0;  // global chunk index of range r's chunk 0
        for (int r = 0; r < 4; r++) {
            const long long n = (bytes[r] + 16383) / 16384;
            long long c = blockIdx.x - first;  // this block's first chunk of range r
            if (c < 0) c += ((-c + gridDim.x - 1) / gridDim.x) * (long long) gridDim.x;
            for (; c < n; c += gridDim.x) {
                const long long off = c * 16384, rem = bytes[r] - off;
                const unsigned sz = (unsigned) (rem < 16384 ? rem : 16384) & ~15u;
                if (sz) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                                     :: "l"(base[r] + off), "r"(sz) : "memory");
            }
            first += n;
        }
    }
"""
ARGS_OLD = "        (void*)&e, (void*)&m, (void*)&hidden, (void*)&inter, (void*)&swiglu_limit\n    };"
ARGS_NEW = ("        (void*)&e, (void*)&m, (void*)&hidden, (void*)&inter, (void*)&swiglu_limit,\n"
            "        (void*)&g_pf[0], (void*)&g_pf_bytes[0], (void*)&g_pf[1], (void*)&g_pf_bytes[1],\n"
            "        (void*)&g_pf[2], (void*)&g_pf_bytes[2], (void*)&g_pf[3], (void*)&g_pf_bytes[3]\n    };")
GLOBALS_OLD = "static int g_bench_coop = 0;\n"
GLOBALS_NEW = GLOBALS_OLD + "static const char* g_pf[4] = {};\nstatic long long g_pf_bytes[4] = {};\n"
BIND_OLD = '    mod.def("get_coop", []() { return g_bench_coop; });\n'
BIND_NEW = BIND_OLD + ('    mod.def("set_prefetch", [](std::vector<int64_t> ptrs, std::vector<int64_t> nbytes) {\n'
                       '        TORCH_CHECK(ptrs.size() == nbytes.size() && ptrs.size() <= 4, "set_prefetch: <= 4 ranges");\n'
                       '        for (int r = 0; r < 4; r++) {\n'
                       '            const bool on = r < (int) ptrs.size();\n'
                       '            g_pf[r] = on ? reinterpret_cast<const char*>(ptrs[r]) : nullptr;\n'
                       '            g_pf_bytes[r] = on ? nbytes[r] : 0;\n'
                       '        }\n'
                       '    }, "L2 prefetch ranges for the next p2b launches (read at launch; [] = off)");\n')
INC_OLD = "#include <pybind11/pybind11.h>\n"
INC_NEW = INC_OLD + "#include <pybind11/stl.h>\n"


def _sub1(src: str, old: str, new: str) -> str:
    if src.count(old) != 1:
        raise SystemExit(f"p2b_pf_bench: expected one {old[:60]!r}, found {src.count(old)}")
    return src.replace(old, new)


def build() -> str:
    sys.path.insert(0, str(ROOT / "kernel_study" / "p2b_coop"))
    import make_bench

    src = make_bench.build()["bench_coop.cu"]
    for old, new in ((SIG_OLD, SIG_NEW), (PRO_OLD, PRO_NEW), (ARGS_OLD, ARGS_NEW), (GLOBALS_OLD, GLOBALS_NEW),
                     (INC_OLD, INC_NEW), (BIND_OLD, BIND_NEW)):
        src = _sub1(src, old, new)
    return src


def main() -> int:
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(build())
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
