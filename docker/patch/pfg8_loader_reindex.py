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


# ---- phase 2 (Round 29): create_weights dest allocation ----
# The image's exl3.py already carries phase 1 (narrow-swap) from the build.
# But create_weights allocates STOCK-shaped Parameter buffers
# (w13 [E,2,KT,NT_local,W], w2 [E,KT_local,NT,W]); a G8 shard
# ([NT/8][KT][8W]) then fails the dest-vs-loaded shape check:
#   RuntimeError: EXL3 load shape mismatch ... w13_trellis
#   dest (320, 72, 32) != loaded (9, 320, 256)
# Phase 2 allocates the G8 shapes under the same env flag:
#   w13: [E, 2, NT_local/8, KT, 8W]   (n-groups on dim 2 — matches col narrow dim 0 of fold)
#   w2:  [E, NT/8, KT_local, 8W]      (n-groups on dim 1 — matches row narrow dim 1 of fold)
MARKER2 = "# --- pfg8-loader-reindex-alloc ---"

ALLOC_W13_OLD = '''        w13_trellis = Parameter(
            torch.empty(
                num_experts, 2, in_tiles, out_tiles, k_words, dtype=torch.int16
            ),
            requires_grad=False,
        )'''

ALLOC_W13_NEW = '''        # ''' + MARKER2 + ''' G8 pack: trellis [n-groups][k-tiles][8W]
        import os as _os_alloc
        _g8_alloc = _os_alloc.environ.get("DSV41_LOAD_PF_G8", "0") == "1"
        _w13_shape = (
            (out_tiles // 8, in_tiles, k_words * 8) if _g8_alloc
            else (in_tiles, out_tiles, k_words)
        )
        w13_trellis = Parameter(
            torch.empty(num_experts, 2, *_w13_shape, dtype=torch.int16),
            requires_grad=False,
        )'''

ALLOC_W2_OLD = '''        w2_trellis = Parameter(
            torch.empty(
                num_experts, out_tiles, in_tiles, k_words, dtype=torch.int16
            ),
            requires_grad=False,
        )'''

ALLOC_W2_NEW = '''        _w2_shape = (
            (in_tiles // 8, out_tiles, k_words * 8) if _g8_alloc
            else (out_tiles, in_tiles, k_words)
        )
        w2_trellis = Parameter(
            torch.empty(num_experts, *_w2_shape, dtype=torch.int16),
            requires_grad=False,
        )'''


# tools/permute_pack_group_major.py --apply writes this into the G8 pack root.
MANIFEST_NAME = "permute_g8_manifest.json"


def serve_model_from_argv(argv: list[str]) -> str | None:
    """Model arg of a `vllm serve <model>` command line; None for other processes."""
    if "serve" not in argv:
        return None
    rest = argv[argv.index("serve") + 1:]
    if "--model" in rest[:-1]:
        return rest[rest.index("--model") + 1]
    if rest and not rest[0].startswith("-"):
        return rest[0]
    return None


def require_g8_manifest(argv: list[str]) -> None:
    """Refuse DSV41_LOAD_PF_G8=1 on a pack the permute tool did not write.

    A stock pack read with the G8 layout is garbage. Spawned engine/worker
    processes carry no `serve` argv; their `vllm serve` parent checked first.
    """
    model = serve_model_from_argv(argv)
    if model is None:
        return
    if not (Path(model) / MANIFEST_NAME).is_file():
        raise SystemExit(
            f"DSV41_LOAD_PF_G8=1 but {model} has no {MANIFEST_NAME}: not a G8 pack"
        )


def patch(text: str) -> str:
    if MARKER in text and MARKER2 in text:
        return text  # idempotent (both phases)
    if MARKER not in text:
        if COL_OLD not in text:
            raise SystemExit("pfg8_loader_reindex: shard_exl3_col anchor missing")
        if ROW_OLD not in text:
            raise SystemExit("pfg8_loader_reindex: shard_exl3_row anchor missing")
        text = text.replace(COL_OLD, COL_NEW, 1)
        text = text.replace(ROW_OLD, ROW_NEW, 1)
    # phase 2: dest allocation (idempotent on its own marker)
    if MARKER2 not in text:
        if ALLOC_W13_OLD not in text:
            raise SystemExit("pfg8_loader_reindex: w13_trellis alloc anchor missing")
        if ALLOC_W2_OLD not in text:
            raise SystemExit("pfg8_loader_reindex: w2_trellis alloc anchor missing")
        text = text.replace(ALLOC_W13_OLD, ALLOC_W13_NEW, 1)
        text = text.replace(ALLOC_W2_OLD, ALLOC_W2_NEW, 1)
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
