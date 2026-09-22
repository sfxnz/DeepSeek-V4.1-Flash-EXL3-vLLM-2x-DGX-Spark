#!/usr/bin/env python3
"""Targeted lm_head MXFP8 re-encode: rewrite ONE tensor inside an EXISTING
EXL3 pack snapshot copy. Staged, NOT run on the live 2.0bpw-mcg snapshot.

What it does (per shard file, CPU-only, ~2-4 min on the host):
  head.weight (bf16 [129280, 5120], ~1.32 GB) is removed and re-emitted as:
    lm_head.weight       F8_E4M3 [129280, 5120]      (~633 MB)
    lm_head.weight_scale F8_E8M0  [129280, 160]      (~20 MB)
  using the pack's dense-tensor recipe (per-32-block e8m0, ceil-log2 scale,
  same as layers.N.attn.* in the official checkpoint). All other tensors in
  the file are byte-copied unchanged. The safetensors index gains the two new
  keys. `head.weight` disappears, so a non-patched loader ignores the new keys
  (unexpected weights) and boots the stock bf16 DSpark markov/embed heads
  untouched — the only bf16 vocab head that exists in this pack is lm_head.

Notably it must be `lm_head.*`, not `head.weight_scale`: the DSV4.1
WeightsMapper renames `.scale$` -> `.{linear_scale_name}` and `head.weight`
-> `lm_head.weight` (model.py:924-942); `lm_head.*` passes through mapping
untouched and lands on the swapped fp8/e8m0 params directly.

Usage (on a COPY — never the live snapshot):
  python3 tools/quantize_lmhead_mxfp8.py --snapshot /path/to/snapshot-copy

Idempotency: refuses to run twice (key already present). Safe to abort; the
output file is written to *.tmp and renamed atomically.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docker" / "patch"))
from lmhead_mxfp8 import (  # noqa: E402
    CKPT_SCALE_KEY,
    CKPT_WEIGHT_KEY,
    quantize_weight_mxfp8,
)

OLD_KEY = "head.weight"


def rewrite_shard(path: Path, dry_run: bool = False) -> tuple[int, int]:
    """Replace OLD_KEY with the quantized pair inside one shard file."""
    tensors = dict(load_file(str(path), device="cpu"))
    if OLD_KEY not in tensors:
        return 0, 0
    w = tensors.pop(OLD_KEY)
    assert w.dtype == torch.bfloat16 and w.ndim == 2, (w.dtype, w.shape)
    values, scales = quantize_weight_mxfp8(w)
    tensors[CKPT_WEIGHT_KEY] = values
    tensors[CKPT_SCALE_KEY] = scales.view(torch.uint8)
    if dry_run:
        return values.numel(), scales.numel()
    tmp = path.with_suffix(path.suffix + ".tmp")
    save_file(tensors, str(tmp))
    tmp.replace(path)
    return values.numel(), scales.numel()


def update_index(snapshot: Path, shard_name: str) -> None:
    idx_path = snapshot / "model.safetensors.index.json"
    idx = json.loads(idx_path.read_text())
    wm = idx["weight_map"]
    if OLD_KEY not in wm:
        raise SystemExit(f"index has no {OLD_KEY}; already re-encoded?")
    shard = wm.pop(OLD_KEY)
    if shard != shard_name:
        raise SystemExit(f"index points at {shard}, file given is {shard_name}")
    wm[CKPT_WEIGHT_KEY] = shard_name
    wm[CKPT_SCALE_KEY] = shard_name
    tmp = idx_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(idx, indent=2))
    tmp.replace(idx_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    snap = args.snapshot
    idx_path = snap / "model.safetensors.index.json"
    if not idx_path.is_file():
        raise SystemExit(f"no index under {snap}")
    wm = json.loads(idx_path.read_text())["weight_map"]
    if CKPT_SCALE_KEY in wm:
        raise SystemExit(f"{CKPT_SCALE_KEY} already present; nothing to do")
    shard_name = wm.get(OLD_KEY)
    if shard_name is None:
        raise SystemExit(f"no {OLD_KEY} in index (not a bf16-head pack?)")
    shard = snap / shard_name
    nv, ns = rewrite_shard(shard, dry_run=args.dry_run)
    if args.dry_run:
        print(f"dry-run: would write {CKPT_WEIGHT_KEY} ({nv} fp8) + "
              f"{CKPT_SCALE_KEY} ({ns} e8m0) into {shard_name}")
        return 0
    update_index(snap, shard_name)
    print(f"rewrote {shard_name}: {OLD_KEY} -> mxfp8 pair "
          f"({nv:,} e4m3 values, {ns:,} e8m0 scales)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
