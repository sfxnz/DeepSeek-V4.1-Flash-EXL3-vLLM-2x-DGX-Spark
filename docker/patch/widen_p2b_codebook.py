#!/usr/bin/env python3
"""Port vllm-exl3 p2b fused MoE from MCG-only (cb=1) to MCG or MUL1 (cb=2).

Stock p2b launches run_gemv_tile<BITS, 1, CFG> and TORCH_CHECKs mcg.
A MUL1 pack then fails native p2b and falls back to generic exl3_moe.
Apply after widen_p2b_cfg1.py. Keeps CFG=1 tiles. Host mcg=true still
selects cb=1 so the published 2.0bpw-mcg pack stays on the hillclimbed path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

TILE_OLD = "run_gemv_tile<BITS, 1, 1>"
TILE_CFG0_OLD = "run_gemv_tile<BITS, 1, 0>"
TILE_NEW = "run_gemv_tile<BITS, CB, 1>"
TILE_CFG0_NEW = "run_gemv_tile<BITS, CB, 0>"

KERNEL_TMPL_OLD = "template <int BITS>\n__global__"
KERNEL_TMPL_NEW = "template <int BITS, int CB>\n__global__"

LAUNCH_TMPL_OLD = "template <int BITS>\nstatic void launch_moe_batched"
LAUNCH_TMPL_NEW = "template <int BITS, int CB>\nstatic void launch_moe_batched"

KERNEL_PTR_OLD = "(void*) p2b_moe_batched_kernel<BITS>"
KERNEL_PTR_NEW = "(void*) p2b_moe_batched_kernel<BITS, CB>"

CHECK_OLD = (
    '    TORCH_CHECK(mcg && kg == ku && ku == kd && (kg == 2 || kg == 3 || kg == 4), '
    '"unsupported fused MoE K");'
)
CHECK_NEW = (
    '    TORCH_CHECK(kg == ku && ku == kd && (kg == 2 || kg == 3 || kg == 4), '
    '"unsupported fused MoE K");'
)

LAUNCH2_OLD = "    if (kg == 2) launch_moe_batched<2>("
LAUNCH3_OLD = "    else if (kg == 3) launch_moe_batched<3>("
LAUNCH4_OLD = "    else if (kg == 4) launch_moe_batched<4>("
LAUNCH2_NEW = "    if (mcg && kg == 2) launch_moe_batched<2, 1>("
LAUNCH3_NEW = "    else if (mcg && kg == 3) launch_moe_batched<3, 1>("
LAUNCH4_NEW = "    else if (mcg && kg == 4) launch_moe_batched<4, 1>("

PY_FLAGS_OLD = "    extra_args = (intermediate, clamp_limit) if extended_abi else ()\n"
PY_FLAGS_NEW = """    flags = getattr(layer, "_exl3_codebook_flags", (True, False, True, False, True, False))
    mcg = bool(flags[0])
    mul1 = bool(flags[1])
    if mcg == mul1:
        return None
    extra_args = (intermediate, clamp_limit) if extended_abi else ()
"""

# Post-mrow indent (8 spaces). Stock/pin indent is 12. Apply after mrow.
_MCG_TRUE_SHAPES = (
    """        k,
        k,
        k,
        True,
        *extra_args,""",
    """            k,
            k,
            k,
            True,
            *extra_args,""",
)
_MCG_VAR_SHAPES = (
    """        k,
        k,
        k,
        mcg,
        *extra_args,""",
    """            k,
            k,
            k,
            mcg,
            *extra_args,""",
)


def _already_cu(src: str) -> bool:
    return (
        "run_gemv_tile<BITS, CB," in src
        and "template <int BITS, int CB>" in src
        and "launch_moe_batched<2, 2>" in src
        and "mcg && kg == 2" in src
        and CHECK_OLD not in src
    )


def _replace_one(src: str, old: str, new: str, label: str) -> str:
    if new in src and old not in src:
        return src
    if old not in src:
        raise SystemExit(f"widen_p2b_codebook: {label} not found")
    return src.replace(old, new, 1)


def _replace_all(src: str, old: str, new: str, label: str) -> str:
    if old not in src:
        if new in src:
            return src
        raise SystemExit(f"widen_p2b_codebook: {label} not found")
    return src.replace(old, new)


def patch_cu(src: str) -> str:
    if _already_cu(src):
        return src
    out = src
    if TILE_OLD in out:
        out = _replace_all(out, TILE_OLD, TILE_NEW, "CFG=1 GEMV tile cb")
    elif TILE_CFG0_OLD in out:
        out = _replace_all(out, TILE_CFG0_OLD, TILE_CFG0_NEW, "CFG=0 GEMV tile cb")
    elif "run_gemv_tile<BITS, CB," not in out:
        raise SystemExit("widen_p2b_codebook: GEMV tile template not found")
    out = _replace_one(out, KERNEL_TMPL_OLD, KERNEL_TMPL_NEW, "kernel template")
    out = _replace_one(out, LAUNCH_TMPL_OLD, LAUNCH_TMPL_NEW, "launch template")
    out = _replace_all(out, KERNEL_PTR_OLD, KERNEL_PTR_NEW, "kernel pointer")
    out = _replace_one(out, CHECK_OLD, CHECK_NEW, "mcg-only TORCH_CHECK")
    out = _replace_one(out, LAUNCH2_OLD, LAUNCH2_NEW, "K=2 MCG launch")
    out = _replace_one(out, LAUNCH3_OLD, LAUNCH3_NEW, "K=3 MCG launch")
    out = _replace_one(out, LAUNCH4_OLD, LAUNCH4_NEW, "K=4 MCG launch")
    if "launch_moe_batched<2, 2>" not in out:
        start = out.find(LAUNCH2_NEW)
        end = out.find(";", out.find(LAUNCH4_NEW))
        if start < 0 or end < 0:
            raise SystemExit("widen_p2b_codebook: MCG launch block missing after rewrite")
        mcg_block = out[start : end + 1]
        mul1_block = (
            mcg_block.replace("mcg &&", "!mcg &&")
            .replace(", 1>", ", 2>")
            .replace("    if (", "    else if (", 1)
        )
        out = out[: end + 1] + "\n" + mul1_block + out[end + 1 :]
    if "run_gemv_tile<BITS, 1," in out:
        raise SystemExit("widen_p2b_codebook: hardcoded cb=1 tile still present")
    if "p2b_moe_batched_kernel<BITS>" in out:
        raise SystemExit("widen_p2b_codebook: kernel still untemplated on CB")
    if "launch_moe_batched<2, 2>" not in out:
        raise SystemExit("widen_p2b_codebook: MUL1 K=2 launch missing")
    if CHECK_OLD in out:
        raise SystemExit("widen_p2b_codebook: mcg-only TORCH_CHECK still present")
    return out


def _already_py(src: str) -> bool:
    return (
        'getattr(layer, "_exl3_codebook_flags"' in src
        and all(old not in src for old in _MCG_TRUE_SHAPES)
        and ("        mcg," in src or "            mcg," in src)
    )


def _swap_mcg_true(src: str) -> str:
    for old, new in zip(_MCG_TRUE_SHAPES, _MCG_VAR_SHAPES):
        if old in src:
            return src.replace(old, new)
    raise SystemExit("widen_p2b_codebook: hardcoded mcg=True not found")


def patch_py(src: str) -> str:
    if _already_py(src):
        return src
    if PY_FLAGS_OLD not in src:
        raise SystemExit("widen_p2b_codebook: extra_args assignment not found")
    out = src.replace(PY_FLAGS_OLD, PY_FLAGS_NEW, 1)
    out = _swap_mcg_true(out)
    if any(old in out for old in _MCG_TRUE_SHAPES):
        raise SystemExit("widen_p2b_codebook: hardcoded mcg=True still present")
    if 'getattr(layer, "_exl3_codebook_flags"' not in out:
        raise SystemExit("widen_p2b_codebook: codebook flags read missing")
    return out


def apply(root: Path) -> None:
    cu = root / "csrc" / "p2b_moe.cu"
    py = root / "src" / "vllm_exl3" / "exl3.py"
    if not py.is_file():
        py = root / "vllm_exl3" / "exl3.py"
    cu.write_text(patch_cu(cu.read_text()))
    print(f"patched {cu}")
    if py.is_file():
        py.write_text(patch_py(py.read_text()))
        print(f"patched {py}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    apply(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
