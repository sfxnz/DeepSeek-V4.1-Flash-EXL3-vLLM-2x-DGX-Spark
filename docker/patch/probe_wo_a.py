#!/usr/bin/env python3
"""Boot-time probe: log wo_a/wo_b runtime dtypes in deep_gemm_fp8_o_proj.

The live decode profile shows one per-layer bf16 WMMA cutlass_80 kernel
(~176 us) inside the o_proj block although the pack stores wo_a as F8_E4M3
(block 32, ue8m0). This probe prints the runtime dtype/shape/use_fp8 decision
on the first call (and every call with DSV41_PROBE_WO_A_EVERY=1) so the
fallback can be attributed from docker logs. Diagnostic only, no behavior
change. Idempotent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

OLD = '''    use_fp8 = wo_a.weight.dtype == torch.float8_e4m3fn
    o_proj_input, o_scale = fused_inv_rope_fp8_quant('''

NEW = '''    use_fp8 = wo_a.weight.dtype == torch.float8_e4m3fn
    # --- probe_wo_a (diagnostic, no behavior change) ---
    import os as _probe_os
    _every = _probe_os.environ.get("DSV41_PROBE_WO_A_EVERY", "0") == "1"
    if _every or not globals().get("_PROBE_WO_A_LOGGED", False):
        globals()["_PROBE_WO_A_LOGGED"] = True
        print(
            f"[wo_a-probe] wo_a.dtype={wo_a.weight.dtype} "
            f"wo_a.shape={tuple(wo_a.weight.shape)} use_fp8={use_fp8} "
            f"wo_b.dtype={getattr(wo_b.weight, 'dtype', None)} "
            f"n_groups={n_groups}",
            flush=True,
        )
    # --- end probe_wo_a ---
    o_proj_input, o_scale = fused_inv_rope_fp8_quant('''


def patch(src: str) -> str:
    if "_PROBE_WO_A_LOGGED" in src:
        return src
    if src.count(OLD) != 1:
        raise SystemExit(f"probe_wo_a: anchor not found (count={src.count(OLD)})")
    return src.replace(OLD, NEW, 1)


def apply(path: Path) -> None:
    src = path.read_text()
    out = patch(src)
    if out != src:
        path.write_text(out)
    print(f"patched {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", type=Path)
    args = ap.parse_args()
    apply(args.target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
