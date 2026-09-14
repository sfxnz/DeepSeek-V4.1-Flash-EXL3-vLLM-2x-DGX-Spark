#!/usr/bin/env python3
"""True m-row vllm-exl3 p2b fused MoE: scale each phase's work list by m.

Not a serial `for (row) { whole moe; grid.sync(); }`. That was slower on GB10
(13.2 tok/s vs 15.3). Each phase keeps the same barrier count as m=1. Work
items become m * experts * groups. Scratch is {e, m, dim}. GEMV tiles stay
one row (A2/C already point at that row).

Apply after widen_p2b_shapes.py. Idempotent. Row cap 8.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MAX_ROWS = 8

CU_HIDDEN_SHAPES_OLD = """    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1 && x.size(1) > 0 && x.size(1) % 128 == 0,
                "fused MoE requires one input row whose hidden width is a positive multiple of 128");"""
CU_HIDDEN_SHAPES_NEW = f"""    TORCH_CHECK(x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= {MAX_ROWS} && x.size(1) > 0 && x.size(1) % 128 == 0,
                "fused MoE requires 1-{MAX_ROWS} input rows whose hidden width is a positive multiple of 128");"""

CU_HIDDEN_STOCK_OLD = """    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1 && x.size(1) == 4096,
                "fused MoE requires one input row with hidden width 4096");"""
CU_HIDDEN_STOCK_NEW = f"""    TORCH_CHECK(x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= {MAX_ROWS} && x.size(1) == 4096,
                "fused MoE requires 1-{MAX_ROWS} input rows with hidden width 4096");"""

CU_IDS_OLD = """    TORCH_CHECK(ids.dim() == 1 && ids.scalar_type() == at::kInt && ids.numel() > 0,
                "fused MoE expert indices must be a nonempty int32 routing vector");"""
CU_IDS_NEW = """    TORCH_CHECK(ids.scalar_type() == at::kInt && ids.numel() > 0,
                "fused MoE expert indices must be a nonempty int32 routing tensor");
    TORCH_CHECK((ids.dim() == 1 && x.size(0) == 1)
                    || (ids.dim() == 2 && ids.size(0) == x.size(0) && ids.size(1) > 0),
                "fused MoE expert indices must be [K] for one row or [m, K] matching the input rows");"""

CU_E_OLD = """    const int e = static_cast<int>(ids.numel());"""
CU_E_NEW = """    const int e = static_cast<int>(ids.dim() == 1 ? ids.size(0) : ids.size(1));"""

CU_CONST_SHAPES_OLD = """    constexpr int m = 1;
    const int hidden = static_cast<int>(x.size(1));
    const int inter = static_cast<int>(intermediate_size);"""
CU_CONST_SHAPES_NEW = """    const int m = static_cast<int>(x.size(0));
    const int hidden = static_cast<int>(x.size(1));
    const int inter = static_cast<int>(intermediate_size);"""

CU_CONST_STOCK_OLD = """    constexpr int m = 1, hidden = 4096;
    const int inter = static_cast<int>(intermediate_size);"""
CU_CONST_STOCK_NEW = """    const int m = static_cast<int>(x.size(0));
    constexpr int hidden = 4096;
    const int inter = static_cast<int>(intermediate_size);"""

WARP_LOOP_OLD = """        for (; this_warp < total_warps; this_warp += grid_warps) {
            int e = this_warp / warps_per_exp;
            int w = this_warp % warps_per_exp;
            int src = ids[e];
"""
WARP_LOOP_NEW = """        for (; this_warp < total_warps; this_warp += grid_warps) {
            int row = this_warp / (experts * warps_per_exp);
            int rem = this_warp % (experts * warps_per_exp);
            int e = rem / warps_per_exp;
            int w = rem % warps_per_exp;
            int src = ids[row * experts + e];
            const half* x_row = x + (size_t) row * hidden;
"""

GEMV_GATE_OLD = """        int total_work = 2 * experts * num_groups_gate;
        for (int item = blockIdx.x; item < total_work; item += gridDim.x) {
            int is_up = item & 1;
            int rem = item >> 1;
            int e = rem / num_groups_gate;
            int group = rem % num_groups_gate;
            int src = ids[e];

            const uint32_t* B32 = reinterpret_cast<const uint32_t*>(is_up ? ut_ptrs[src] : gt_ptrs[src]);
            const half2* A2 = reinterpret_cast<const half2*>((is_up ? had_up : had_gate) + e * hidden);
            half* C = (is_up ? up : gate) + e * inter;
"""
GEMV_GATE_NEW = """        int total_work = 2 * m * experts * num_groups_gate;
        for (int item = blockIdx.x; item < total_work; item += gridDim.x) {
            int is_up = item & 1;
            int rem = item >> 1;
            int row = rem / (experts * num_groups_gate);
            int rem2 = rem % (experts * num_groups_gate);
            int e = rem2 / num_groups_gate;
            int group = rem2 % num_groups_gate;
            int src = ids[row * experts + e];
            size_t em = (size_t) e * m + row;

            const uint32_t* B32 = reinterpret_cast<const uint32_t*>(is_up ? ut_ptrs[src] : gt_ptrs[src]);
            const half2* A2 = reinterpret_cast<const half2*>((is_up ? had_up : had_gate) + em * hidden);
            half* C = (is_up ? up : gate) + em * inter;
"""

GEMV_DOWN_OLD = """        int total_work = experts * num_groups_down;
        for (int item = blockIdx.x; item < total_work; item += gridDim.x) {
            int e = item / num_groups_down;
            int group = item % num_groups_down;
            int src = ids[e];

            const uint32_t* B32 = reinterpret_cast<const uint32_t*>(dt_ptrs[src]);
            const half2* A2 = reinterpret_cast<const half2*>(had_down + e * inter);
            half* C = down + e * hidden;
"""
GEMV_DOWN_NEW = """        int total_work = m * experts * num_groups_down;
        for (int item = blockIdx.x; item < total_work; item += gridDim.x) {
            int row = item / (experts * num_groups_down);
            int rest = item % (experts * num_groups_down);
            int e = rest / num_groups_down;
            int group = rest % num_groups_down;
            int src = ids[row * experts + e];
            size_t em = (size_t) e * m + row;

            const uint32_t* B32 = reinterpret_cast<const uint32_t*>(dt_ptrs[src]);
            const half2* A2 = reinterpret_cast<const half2*>(had_down + em * inter);
            half* C = down + em * hidden;
"""

REDUCE_OLD = """        int total_elements = experts * hidden;
        for (int j = tid; j < total_elements; j += total_threads) {
            int e = j / hidden;
            int col = j % hidden;
            float w = __half2float(rw[e]);
            atomicAdd(accum + col, w * __half2float(down[j]));
        }"""
REDUCE_NEW = """        int total_elements = m * experts * hidden;
        for (int j = tid; j < total_elements; j += total_threads) {
            int e = j / (m * hidden);
            int rem = j % (m * hidden);
            int row = rem / hidden;
            int col = rem % hidden;
            float w = __half2float(rw[row * experts + e]);
            atomicAdd(accum + row * hidden + col, w * __half2float(down[j]));
        }"""

PY_LOOP_OLD = """    extra_args = (intermediate, clamp_limit) if extended_abi else ()
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
    return native_out.to(dtype=torch.float32)"""
PY_LOOP_NEW = """    extra_args = (intermediate, clamp_limit) if extended_abi else ()
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
        True,
        *extra_args,
    )
    # pybind returns the same output tensor, while lightweight test doubles
    # may return a fresh tensor.  Accommodate both without synchronizing.
    if isinstance(result, torch.Tensor) and result is not native_out:
        native_out.copy_(result.reshape_as(native_out))
    return native_out.to(dtype=torch.float32)"""

DONE_MARKERS = (
    "m * experts * warps_per_exp",
    "row * experts + e",
    "(size_t) e * m + row",
    f"x.size(0) >= 1 && x.size(0) <= {MAX_ROWS}",
)


def _already_patched_cu(src: str) -> bool:
    return all(marker in src for marker in DONE_MARKERS)


def _replace_one(src: str, old: str, new: str, label: str) -> str:
    if new in src and old not in src:
        return src
    if old not in src:
        raise SystemExit(f"widen_p2b_mrow: {label} not found")
    return src.replace(old, new, 1)


def _replace_all(src: str, old: str, new: str, label: str) -> str:
    if old not in src:
        raise SystemExit(f"widen_p2b_mrow: {label} not found")
    return src.replace(old, new)


def _replace_one_of(src: str, pairs: tuple[tuple[str, str], ...], label: str) -> str:
    for old, new in pairs:
        if new in src and old not in src:
            return src
        if old in src:
            return src.replace(old, new, 1)
    raise SystemExit(f"widen_p2b_mrow: {label} not found")


def patch_cu(src: str) -> str:
    if _already_patched_cu(src):
        return src
    out = _replace_one_of(
        src,
        (
            (CU_HIDDEN_SHAPES_OLD, CU_HIDDEN_SHAPES_NEW),
            (CU_HIDDEN_STOCK_OLD, CU_HIDDEN_STOCK_NEW),
        ),
        "p2b m-row TORCH_CHECK",
    )
    out = _replace_one(out, CU_IDS_OLD, CU_IDS_NEW, "p2b ids TORCH_CHECK")
    out = _replace_one(out, CU_E_OLD, CU_E_NEW, "p2b expert count")
    out = _replace_one_of(
        out,
        (
            (CU_CONST_SHAPES_OLD, CU_CONST_SHAPES_NEW),
            (CU_CONST_STOCK_OLD, CU_CONST_STOCK_NEW),
        ),
        "p2b m constexpr",
    )
    out = _replace_all(
        out,
        "int total_warps = experts * warps_per_exp;",
        "int total_warps = m * experts * warps_per_exp;",
        "hadamard warp count",
    )
    out = _replace_all(out, WARP_LOOP_OLD, WARP_LOOP_NEW, "hadamard warp decode")
    out = _replace_one(out, GEMV_GATE_OLD, GEMV_GATE_NEW, "gate/up GEMV work list")
    out = _replace_one(out, GEMV_DOWN_OLD, GEMV_DOWN_NEW, "down GEMV work list")
    out = _replace_all(
        out,
        "int total_elements = experts * inter;",
        "int total_elements = m * experts * inter;",
        "swiglu element count",
    )
    out = _replace_one(out, REDUCE_OLD, REDUCE_NEW, "weighted reduction")
    out = _replace_all(
        out,
        "had_gate + e * hidden",
        "had_gate + ((size_t) e * m + row) * hidden",
        "had_gate row stride",
    )
    out = _replace_all(
        out,
        "had_up + e * hidden",
        "had_up + ((size_t) e * m + row) * hidden",
        "had_up row stride",
    )
    out = _replace_all(
        out,
        "gate + e * inter",
        "gate + ((size_t) e * m + row) * inter",
        "gate row stride",
    )
    out = _replace_all(
        out,
        "up + e * inter",
        "up + ((size_t) e * m + row) * inter",
        "up row stride",
    )
    out = _replace_all(
        out,
        "had_down + e * inter",
        "had_down + ((size_t) e * m + row) * inter",
        "had_down row stride",
    )
    out = _replace_all(
        out,
        "down + e * hidden",
        "down + ((size_t) e * m + row) * hidden",
        "down row stride",
    )
    out = _replace_all(out, "x + w * 128", "x_row + w * 128", "input hadamard x row")
    if "constexpr int m = 1" in out:
        raise SystemExit("widen_p2b_mrow: constexpr int m = 1 still present")
    if "x.dim() == 2 && x.size(0) == 1" in out:
        raise SystemExit("widen_p2b_mrow: m=1 input TORCH_CHECK still present")
    if "for (int row = 0; row < m; ++row)" in out:
        raise SystemExit("widen_p2b_mrow: serial per-row moe loop is the reverted path")
    if "m * experts * warps_per_exp" not in out:
        raise SystemExit("widen_p2b_mrow: phase work lists did not scale by m")
    if out.count("grid.sync()") != src.count("grid.sync()"):
        raise SystemExit("widen_p2b_mrow: barrier count changed")
    return out


def patch_py(src: str) -> str:
    if PY_LOOP_NEW in src:
        return src
    if PY_LOOP_OLD not in src:
        raise SystemExit("widen_p2b_mrow: python row loop not found")
    out = src.replace(PY_LOOP_OLD, PY_LOOP_NEW, 1)
    if "for row in range(int(x2d.shape[0])):" in out:
        raise SystemExit("widen_p2b_mrow: python row loop still present")
    if "xh[row : row + 1]" in out:
        raise SystemExit("widen_p2b_mrow: python per-row slice still present")
    return out


def apply(root: Path) -> None:
    cu = root / "csrc" / "p2b_moe.cu"
    py = root / "src" / "vllm_exl3" / "exl3.py"
    if not py.is_file():
        py = root / "vllm_exl3" / "exl3.py"
    cu.write_text(patch_cu(cu.read_text()))
    py.write_text(patch_py(py.read_text()))
    print(f"patched {cu}")
    print(f"patched {py}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    apply(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
