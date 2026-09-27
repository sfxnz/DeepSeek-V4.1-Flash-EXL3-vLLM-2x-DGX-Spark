#!/usr/bin/env python3
"""Split cooperative p2b fused MoE into 9 non-cooperative phase launches.

Live CFG=1 kernel is one cudaLaunchCooperativeKernel with eight grid.sync.
A kernel launch is already a grid-wide barrier. Replay of 9 graph nodes
drops this_grid() waits and keeps CFG=1 tiles, m-row work lists, and
launch_bounds(256, 4). Same occupancy grid. Not p2b_gemv_batched.

Apply after widen_p2b_cfg1.py. Idempotent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

N_PHASES = 9

SIG_OLD = """    int hidden,
    int inter,
    float swiglu_limit)
{
    auto grid = cg::this_grid();
"""

SIG_NEW = """    int hidden,
    int inter,
    float swiglu_limit,
    int phase)
{
"""

LAUNCH_OLD = """    void* args[] = {
        (void*)&xp, (void*)&gtp, (void*)&gup, (void*)&gvp,
        (void*)&utp, (void*)&uup, (void*)&uvp,
        (void*)&dtp, (void*)&dup, (void*)&dvp,
        (void*)&idp, (void*)&rwp,
        (void*)&gp, (void*)&up_p, (void*)&dp, (void*)&op,
        (void*)&hg_p, (void*)&hu_p, (void*)&hd_p, (void*)&accp,
        (void*)&e, (void*)&m, (void*)&hidden, (void*)&inter, (void*)&swiglu_limit
    };

    cuda_check(cudaLaunchCooperativeKernel(kernel, dim3(grid), dim3(256), args, 0, stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
"""

LAUNCH_NEW = """    int phase = 0;
    void* args[] = {
        (void*)&xp, (void*)&gtp, (void*)&gup, (void*)&gvp,
        (void*)&utp, (void*)&uup, (void*)&uvp,
        (void*)&dtp, (void*)&dup, (void*)&dvp,
        (void*)&idp, (void*)&rwp,
        (void*)&gp, (void*)&up_p, (void*)&dp, (void*)&op,
        (void*)&hg_p, (void*)&hu_p, (void*)&hd_p, (void*)&accp,
        (void*)&e, (void*)&m, (void*)&hidden, (void*)&inter, (void*)&swiglu_limit,
        (void*)&phase
    };

    for (phase = 0; phase < 9; ++phase)
        cuda_check(cudaLaunchKernel(kernel, dim3(grid), dim3(256), args, 0, stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
"""

MARKERS = (
    "    // Zero accum\n",
    "    // Phase 2: Batched Gate & Up GEMV across all active experts\n",
    "    // Epilogue Hadamard on Gate and Up\n",
    "    // Phase 3: SwiGLU activation + Down input Hadamard across all active experts\n",
    "        // Down input Hadamard on had_down\n",
    "    // Phase 4: Batched Down GEMV across all active experts\n",
    "    // Down output Hadamard and atomic accumulation into accum\n",
    "        // Weighted reduction into accum\n",
    "    // Write back to out\n",
)

DONE_MARKERS = (
    "for (phase = 0; phase < 9; ++phase)",
    "cudaLaunchKernel(kernel, dim3(grid), dim3(256)",
    "int phase)",
    "if (phase == 0)",
    "if (phase == 8)",
)

LEFTOVERS = (
    "grid.sync()",
    "cudaLaunchCooperativeKernel",
    "cg::this_grid()",
)


def _already(src: str) -> bool:
    return all(marker in src for marker in DONE_MARKERS) and not any(
        old in src for old in LEFTOVERS
    )


def _replace_one(src: str, old: str, new: str, label: str) -> str:
    if new in src and old not in src:
        return src
    if old not in src:
        raise SystemExit(f"widen_p2b_nocoop: {label} not found")
    return src.replace(old, new, 1)


def _balance(s: str) -> int:
    return s.count("{") - s.count("}")


def _gate_phases(src: str) -> str:
    start = src.find(MARKERS[0])
    if start < 0:
        raise SystemExit("widen_p2b_nocoop: Zero accum marker missing")
    kclose = src.find("template <int BITS>\nstatic void launch_moe_batched", start)
    if kclose < 0:
        raise SystemExit("widen_p2b_nocoop: launch_moe_batched missing")
    body_end = src.rfind("\n}\n", start, kclose)
    if body_end < 0:
        raise SystemExit("widen_p2b_nocoop: kernel close missing")
    inner = src[start:body_end]
    if inner.count("grid.sync()") != 8:
        raise SystemExit(
            f"widen_p2b_nocoop: expected 8 grid.sync in body, got {inner.count('grid.sync()')}"
        )
    chunks: list[str] = []
    for i, marker in enumerate(MARKERS):
        a = inner.find(marker)
        if a < 0:
            raise SystemExit(f"widen_p2b_nocoop: marker {i} missing")
        b = inner.find(MARKERS[i + 1]) if i + 1 < len(MARKERS) else len(inner)
        if b <= a:
            raise SystemExit(f"widen_p2b_nocoop: marker {i} overlap")
        chunk = inner[a:b].replace("        grid.sync();\n", "")
        chunk = chunk.rstrip() + "\n"
        bal = _balance(chunk)
        prefix = "    {\n" * max(0, -bal)
        suffix = "    }\n" * max(0, bal)
        chunks.append(f"    if (phase == {i}) {{\n{prefix}{chunk}{suffix}    }}\n")
    gated = "\n".join(chunks)
    if _balance(gated) != 0:
        raise SystemExit("widen_p2b_nocoop: gated body braces do not balance")
    if "grid.sync()" in gated:
        raise SystemExit("widen_p2b_nocoop: grid.sync leaked into gated body")
    return src[:start] + gated + src[body_end:]


def patch_cu(src: str) -> str:
    if _already(src):
        return src
    if "run_gemv_tile<BITS, 1, 1>" not in src:
        raise SystemExit("widen_p2b_nocoop: apply widen_p2b_cfg1 first")
    had_mrow = "m * experts * warps_per_exp" in src
    had_bounds = "__launch_bounds__(256, 4)" in src
    out = _replace_one(src, SIG_OLD, SIG_NEW, "kernel signature / this_grid")
    out = _gate_phases(out)
    out = _replace_one(out, LAUNCH_OLD, LAUNCH_NEW, "cooperative launch")
    if "grid.sync()" in out:
        raise SystemExit("widen_p2b_nocoop: grid.sync still present")
    if "cudaLaunchCooperativeKernel" in out:
        raise SystemExit("widen_p2b_nocoop: cooperative launch still present")
    if "cg::this_grid()" in out:
        raise SystemExit("widen_p2b_nocoop: this_grid still present")
    if "cudaLaunchKernel(kernel, dim3(grid), dim3(256)" not in out:
        raise SystemExit("widen_p2b_nocoop: non-coop 256-thread launch missing")
    if f"phase < {N_PHASES}" not in out:
        raise SystemExit("widen_p2b_nocoop: 9-phase launch loop missing")
    if "run_gemv_tile<BITS, 1, 1>" not in out:
        raise SystemExit("widen_p2b_nocoop: CFG=1 tile undone")
    if had_mrow and "m * experts * warps_per_exp" not in out:
        raise SystemExit("widen_p2b_nocoop: m-row work lists undone")
    if had_bounds and "__launch_bounds__(256, 4)" not in out:
        raise SystemExit("widen_p2b_nocoop: launch_bounds undone")
    if "resident * sms" not in out:
        raise SystemExit("widen_p2b_nocoop: occupancy grid undone")
    return out


def apply(root: Path) -> None:
    cu = root / "csrc" / "p2b_moe.cu"
    cu.write_text(patch_cu(cu.read_text()))
    print(f"patched {cu}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    apply(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
