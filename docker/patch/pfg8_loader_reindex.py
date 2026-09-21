#!/usr/bin/env python3
"""PF-G8 loader re-index — STAGED, DORMANT (do not add to a Dockerfile yet).

Applies to the vllm-exl3 plugin loader (kernel_study/cb2_vllm_exl3/exl3.py is
the reference copy; the image path is
/usr/local/lib/python3.12/dist-packages/vllm_exl3/exl3.py). Runtime behavior is
env-gated by DSV41_LOAD_PF_G8 (default "0" = exact stock):

  DSV41_LOAD_PF_G8=1 + a G8 pack (trellis [NT/8][KT][256], i.e. trailing dim
  8*16*K instead of 16*K) shards on the G8 dims and hands the G8 trellis to
  LinearEXL3 UNCHANGED — the serving GEMM/GEMV kernels must already be
  G8-aware (the exl3_gemm pf_g8 remap proven bit-exact by
  kernel_study/gemv_bench on 2026-09-21, plus the p2b decode G8 reader).

Why dims swap: stock trellis is [KT][NT][W] (dim0=k, dim1=n). G8 trellis is
[NT/8][KT][8*W] (dim0=n-groups, dim1=k). Gate/up is column-parallel (splits n
→ narrow dim0 on G8 instead of dim1); down is row-parallel (splits k → narrow
dim1 on G8 instead of dim0). Both splits stay whole: gate/up NT=144 tiles =
18 groups → 9 per rank at TP=2; down KT=144 → 72 per rank.

Idempotent; no-op if DSV41_LOAD_PF_G8 is unset in the SERVING environment
(the gate here only guards the patched branches, which default to stock).
"""
from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "# --- pfg8-loader-reindex ---"

COL_OLD = '''def shard_exl3_col(loaded: torch.Tensor, suffix: str, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Gate/up: trellis dim 1 and svh dim 0 are column-parallel."""
    if suffix == "trellis":
        return _narrow_tp(loaded, 1, tp_rank, tp_size)
'''

COL_NEW = '''def shard_exl3_col(loaded: torch.Tensor, suffix: str, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Gate/up: trellis dim 1 and svh dim 0 are column-parallel."""
    if suffix == "trellis":
        # ''' + MARKER + ''' G8 pack: [NT/8][KT][8W], n lives on dim 0.
        import os as _os
        if _os.environ.get("DSV41_LOAD_PF_G8", "0") == "1":
            return _narrow_tp(loaded, 0, tp_rank, tp_size)
        return _narrow_tp(loaded, 1, tp_rank, tp_size)
'''

ROW_OLD = '''def shard_exl3_row(loaded: torch.Tensor, suffix: str, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Down: trellis dim 0 and suh dim 0 are row-parallel."""
    if suffix == "trellis":
        return _narrow_tp(loaded, 0, tp_rank, tp_size)
'''

ROW_NEW = '''def shard_exl3_row(loaded: torch.Tensor, suffix: str, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Down: trellis dim 0 and suh dim 0 are row-parallel."""
    if suffix == "trellis":
        # ''' + MARKER + ''' G8 pack: [NT/8][KT][8W], k lives on dim 1.
        import os as _os
        if _os.environ.get("DSV41_LOAD_PF_G8", "0") == "1":
            return _narrow_tp(loaded, 1, tp_rank, tp_size)
        return _narrow_tp(loaded, 0, tp_rank, tp_size)
'''


def patch(text: str) -> str:
    if MARKER in text:
        return text  # idempotent
    if COL_OLD not in text:
        raise SystemExit("pfg8_loader_reindex: shard_exl3_col anchor missing")
    if ROW_OLD not in text:
        raise SystemExit("pfg8_loader_reindex: shard_exl3_row anchor missing")
    text = text.replace(COL_OLD, COL_NEW, 1)
    text = text.replace(ROW_OLD, ROW_NEW, 1)
    return text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("exl3_py", type=Path, help="path to vllm_exl3/exl3.py")
    args = p.parse_args()
    t = args.exl3_py.read_text()
    out = patch(t)
    if out != t:
        args.exl3_py.write_text(out)
        print("dsv41: pfg8 loader re-index installed (dormant; DSV41_LOAD_PF_G8=1 to engage)")
    else:
        print("dsv41: pfg8 loader re-index already present")


if __name__ == "__main__":
    main()
