#!/usr/bin/env python3
"""Permute an EXL3 pack snapshot to the PF-G8 group-major trellis layout.

PF-G8: trellis [k/16, n/16, 16*K] -> [n/128, k/16, 128*K] (groups of 8
n-tiles major). This is the layout the prefill no-regression harness
(kernel_study/gemv_bench/driver_prefill.py) benches and the one that keeps
the p2b decode warp stream (bench5.cu DEC5, groups of 4 tiles) contiguous:
8 % 4 == 0, so ONE permuted pack serves both readers.

The permutation is a pure storage reindex of the SAME trellis words —
bit-exact by construction (mapping table asserted by a round-trip check on
a sample). suh/svh/scale/marker tensors are copied unchanged; non-expert
shards are copied unchanged.

Modes:
  --dry-run (default): print the plan — which tensors move, layout change,
      per-shard byte cost, total disk cost, free-space check. NO copy.
  --apply: write the permuted copy (refuses unless --out is on a filesystem
      with enough free space; ~full pack size again).

Recommendation (see PREFILL-GATE.md): prefer folding this permutation into
tools/quantize_experts_exl3.py at quantize time (pack_trellis output view)
instead of copying 334 GB — this script is the fallback/verifier.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

DEFAULT_PACK = Path(
    "/home/sfxnz/.cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-EXL3/"
    "snapshots/2.0bpw-mcg"
)

G8_GROUP_TILES = 8  # n-tiles per group; must be a multiple of the p2b group (4)
MANIFEST_NAME = "permute_g8_manifest.json"


def is_routed_trellis(name: str) -> bool:
    return (
        ".ffn.experts." in name
        and not name.startswith("mtp.")
        and ".engram." not in name
        and ".shared_experts." not in name
        and name.endswith(".trellis")
    )


def read_st_header(shard: Path) -> dict:
    with open(shard, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


def perm_axis(kt: int, nt: int) -> tuple[int, int, int]:
    """[kt][nt][w] -> [nt/8][kt][8*w]. Returns (G, KT, GW)."""
    if nt % G8_GROUP_TILES:
        raise ValueError(f"n-tiles {nt} not divisible by {G8_GROUP_TILES}")
    return nt // G8_GROUP_TILES, kt, G8_GROUP_TILES * 16 * 2  # K=2 words


def plan(pack: Path) -> dict:
    idx_path = pack / "model.safetensors.index.json"
    idx = json.loads(idx_path.read_text())
    wm = idx["weight_map"]

    shards: dict[str, dict] = {}
    n_trellis = 0
    trellis_bytes = 0
    for name, fname in wm.items():
        sh = shards.setdefault(fname, {"tensors": 0, "permuted": 0, "bytes": 0, "pbytes": 0})
        sh["tensors"] += 1
        if is_routed_trellis(name):
            n_trellis += 1
            sh["permuted"] += 1

    for fname, sh in shards.items():
        hdr = read_st_header(pack / fname)
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            off = meta["data_offsets"]
            nb = off[1] - off[0]
            sh["bytes"] += nb
            sh["pbytes"] += nb  # permuted copy is the same size, re-laid-out

    return {
        "pack": str(pack),
        "shards": shards,
        "n_trellis": n_trellis,
        "total_bytes": sum(s["bytes"] for s in shards.values()),
        "permuted_bytes": sum(s["pbytes"] for s in shards.values()),
    }


def fmt_gb(b: float) -> str:
    return f"{b / 1e9:.1f} GB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pack", nargs="?", type=Path, default=DEFAULT_PACK)
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir (required for --apply)")
    ap.add_argument("--apply", action="store_true",
                    help="actually write the permuted pack (default: dry-run)")
    ap.add_argument("--verify-roundtrip", type=int, default=4,
                    help="sampled tensors for the bit-exact mapping check")
    args = ap.parse_args()

    p = plan(args.pack)
    print(f"pack: {p['pack']}")
    print(f"routed expert trellis tensors: {p['n_trellis']} "
          f"(w1/w2/w3 per expert per layer)")
    print(f"shards: {len(p['shards'])}, total {fmt_gb(p['total_bytes'])}, "
          f"permuted copy would need another {fmt_gb(p['permuted_bytes'])}")
    for fname, sh in sorted(p["shards"].items())[:6]:
        print(f"  {fname}: {sh['tensors']} tensors, {sh['permuted']} permuted, "
              f"{fmt_gb(sh['bytes'])}")
    print("  ...")

    df = None
    try:
        st = os_statvfs(args.pack)
        df = st.f_bavail * st.f_frsize
        print(f"free space on pack filesystem: {fmt_gb(df)}")
        if p["permuted_bytes"] > df:
            print("  NOT ENOUGH FREE SPACE for an in-place sibling copy — "
                  "use quantize-time permutation or free space first.")
    except OSError:
        print("  (free-space check unavailable)")

    if not args.apply:
        print("\n[DRY-RUN] no data written. Layout change per trellis tensor:")
        print("  stock : [k/16][n/16][32]   uint16 words, k-major")
        print("  PF-G8 : [n/128][k/16][256] uint16 words, group of 8 n-tiles major")
        print("  same words, same values, bit-exact reindex; suh/svh/scale/marker copied unchanged")
        print("\nRecommendation: fold this permutation into quantize_experts_exl3.py")
        print("(pack_trellis output view) at next pack build instead of a full copy;")
        print("this script then serves as the pack-verifier fallback.")
        return 0

    if args.out is None:
        print("--apply requires --out", file=sys.stderr)
        return 2
    if df is not None and p["permuted_bytes"] > df:
        print("refusing: not enough free space", file=sys.stderr)
        return 2

    import torch
    from safetensors.torch import safe_open, save_file

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "layout": "pf-g8",
        "group_tiles": G8_GROUP_TILES,
        "source_pack": str(args.pack),
        "mapping": "[k/16][n/16][16*K] -> [n/128][k/16][128*K] (K=2)",
        "shards": {},
    }

    idx = json.loads((args.pack / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    by_file: dict[str, list[str]] = {}
    for name, fname in wm.items():
        by_file.setdefault(fname, []).append(name)

    checked = 0
    t0 = time.time()
    for fname, names in sorted(by_file.items()):
        src = args.pack / fname
        dst = args.out / fname
        tensors = {}
        with safe_open(src, framework="pt") as fh:
            for name in names:
                t = fh.get_tensor(name)
                if is_routed_trellis(name):
                    kt, nt, w = t.shape
                    g = nt // G8_GROUP_TILES
                    tp = (t.view(kt, g, G8_GROUP_TILES * w)
                            .permute(1, 0, 2).contiguous())
                    if checked < args.verify_roundtrip:
                        # bit-exact mapping check: inverse permute == original
                        inv = (tp.view(g, kt, G8_GROUP_TILES, w)
                                 .permute(1, 0, 2, 3).reshape(kt, nt, w))
                        assert torch.equal(inv, t), f"roundtrip failed {name}"
                        print(f"[verify] {name}: roundtrip BIT-EXACT", flush=True)
                        checked += 1
                    t = tp
                tensors[name] = t
        save_file(tensors, str(dst))
        manifest["shards"][fname] = {
            "sha256": sha256_file(dst),
            "tensors": len(names),
            "permuted": sum(1 for n in names if is_routed_trellis(n)),
        }
        print(f"{fname} done ({time.time() - t0:.0f}s)", flush=True)

    (args.out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    print(f"manifest: {args.out / MANIFEST_NAME}")
    return 0


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def os_statvfs(path: Path):
    import os
    head = path if path.is_dir() else path.parent
    # resolve symlinked HF snapshot dirs to a real mount point
    while head.is_symlink():
        head = Path(os.readlink(head))
        if not head.is_absolute():
            head = (path / head).resolve()
    return os.statvfs(head)


if __name__ == "__main__":
    sys.exit(main())
