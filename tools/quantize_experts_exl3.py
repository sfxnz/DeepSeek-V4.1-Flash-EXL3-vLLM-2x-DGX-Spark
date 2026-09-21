#!/usr/bin/env python3
"""Stream official V4.1 shards into an EXL3 mixed pack.

Copies / hardlinks non-routed tensors as stored. Replaces backbone routed-expert
w1/w2/w3 with EXL3 trellis (default 2.0 bpw, MUL1). Engram embed tables are
hardlinked so DSV41_ENGRAM_DISK=1 can pread them — they must not be loaded into
RAM (each is ~100 GiB). Stay at K=2 until a Spark UMA row exists. Do not raise
bits or pass --hq before an activation-Hessian rebuild.

Uncalibrated identity Hessian (no activation capture). Same-shape experts in a
shard share that Hessian and go through quantize_exl3_batch. Resume-safe per
output shard (atomic rename).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from pack_meta import (  # noqa: E402
    DEFAULT_CODEBOOK,
    apply_pack_config,
    build_quantization_config,
    get_codebook,
    is_routed_expert_tensor,
    is_routed_expert_weight,
    revision_for,
)

SIDECAR_SKIP = {"config.json", "model.safetensors.index.json"}


def quant_args_for(bits: int, device: str, codebook: str = DEFAULT_CODEBOOK) -> dict:
    cb = get_codebook(codebook)
    args = {
        "K": int(bits),
        "seed": 0,
        "sigma_reg": 0.025,
        "devices": [device],
        "apply_out_scales": None,
    }
    args[cb.quant_key] = True
    return args


def shard_needs_exl3(names: list[str]) -> bool:
    return any(is_routed_expert_weight(n) for n in names)


def link_or_copy(src: Path, dst: Path) -> str:
    """Hardlink the real blob, not a Hub snapshot symlink.

    Snapshot entries are relative links into ``../../blobs/<hash>``. Linking
    the symlink into another repo makes that relative path miss.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    real = src.resolve()
    try:
        os.link(real, dst)
        return "link"
    except OSError:
        shutil.copy2(real, dst)
        return "copy"


def _load_index(src: Path) -> dict:
    return json.loads((src / "model.safetensors.index.json").read_text())


def write_pack_config(
    src_config: Path, dest: Path, bits: int, codebook: str = DEFAULT_CODEBOOK
) -> None:
    cfg = json.loads(src_config.read_text())
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(
            apply_pack_config(
                cfg, build_quantization_config(bits=bits, codebook=codebook)
            ),
            indent=2,
        )
        + "\n"
    )


def copy_sidecars(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.name in SIDECAR_SKIP:
            continue
        if p.name.startswith("model-") and p.suffix == ".safetensors":
            continue
        dest = dst / p.name
        if p.is_file():
            shutil.copy2(p, dest)
        elif p.is_dir():
            if dest.exists():
                continue
            shutil.copytree(p, dest)


def map_existing_names(names: list[str], fname: str, new_map: dict[str, str]) -> None:
    for n in names:
        if is_routed_expert_weight(n):
            stem = n[: -len(".weight")]
            for suf in (".trellis", ".suh", ".svh", ".mcg", ".mul1"):
                new_map[stem + suf] = fname
        elif is_routed_expert_tensor(n):
            continue
        else:
            new_map[n] = fname


def _identity_h(in_features: int, device: str, cache: dict):
    import torch

    key = (in_features, device)
    hit = cache.get(key)
    if hit is not None:
        return hit
    dev = torch.device(device)
    h = {
        "H": torch.eye(in_features, dtype=torch.float32, device=dev),
        "device": dev,
        "finalized": False,
        "count": 1,
        "first_key": f"expert.in{in_features}",
        "num_total": 1,
        "inf_nan": torch.zeros(2, dtype=torch.long, device=dev),
        "L": None,
        "q_fallback": False,
    }
    cache[key] = h
    return h


def _dequant_t(weight, scale, device: str):
    from mxfp4 import dequant_mxfp4

    del device
    # Stay on CPU; quantize_exl3_batch stages to GPU. A shard of float32
    # experts would OOM UMA if dequantized onto the device together.
    return dequant_mxfp4(weight, scale).float().t().contiguous()


def _meta_h(in_features: int, device: str, cache: dict):
    """Uncalibrated Hessian on the meta device → q_fallback (no LDLQ walk)."""
    import torch

    key = ("meta", in_features, device)
    hit = cache.get(key)
    if hit is not None:
        hit["finalized"] = False
        return hit
    dev = torch.device(device)
    h = {
        "H": torch.empty(in_features, in_features, device="meta"),
        "device": dev,
        "finalized": False,
        "count": 0,
        "first_key": f"expert.in{in_features}",
        "num_total": 0,
        "inf_nan": torch.zeros(2, dtype=torch.long, device=dev),
        "L": None,
        "q_fallback": True,
    }
    cache[key] = h
    return h


def _pfg8_fold(trellis):
    # --- pfg8-pack-fold (dormant; DSV41_PACK_PF_G8=1 enables) ---
    # PF-G8 group-major layout: [KT][NT][W] -> [NT/8][KT][8*W], groups of 8
    # n-tiles major. Bit-exact pure re-index (gate PASS 2026-09-21,
    # kernel_study/gemv_bench/PREFILL-GATE-RESULT-2026-09-21.log). Serving
    # kernels must be G8-aware (or the loader must un-permute) before a G8
    # pack is booted — see results/2026-09-21-pfg8/REBUILD-PLAN.md.
    import os

    if os.environ.get("DSV41_PACK_PF_G8", "0") != "1":
        return trellis
    kt, nt, w = trellis.shape
    if nt % 8:
        raise SystemExit(f"pfg8 fold: n-tiles {nt} not divisible by 8")
    return trellis.view(kt, nt // 8, 8 * w).permute(1, 0, 2).contiguous()


def _pack_one(out: dict) -> dict:
    packed = {}
    for key in ("trellis", "suh", "svh", "mcg", "mul1"):
        if key in out:
            packed[key] = out[key].detach().cpu()
    if "trellis" in packed:
        packed["trellis"] = _pfg8_fold(packed["trellis"])
    return packed


def _quantize_greedy_tiles(
    tiles,
    bits: int,
    device: str,
    passes: int = 2,
    beam: int = 1,
    codebook: str = DEFAULT_CODEBOOK,
):
    """Greedy / beam K-bit sliding-window encode. tiles: (N, 256) float32.

    Each step tries 2**K next symbols per surviving path, keeps min-MSE via
    exllamav3_ext.decode (same codebook mapping as inference). beam=1 is pure
    greedy (4 vs 16384 Viterbi states at K=2). pack_trellis stores only the
    low K bits and unpack is tail-biting, so later passes start from the wrap
    state implied by the previous pass.
    """
    import torch
    from exllamav3.ext import exllamav3_ext as ext

    n, l = tiles.shape
    assert l == 256
    cb = get_codebook(codebook)
    mcg = cb.name == "mcg"
    mul1 = cb.name == "mul1"
    beam = max(1, int(beam))
    n_sym = 1 << bits
    wrap = (1 << (16 - bits)) - 1
    rng_n = torch.arange(n, device=device)
    start = torch.zeros(n, dtype=torch.int32, device=device)
    encoded = torch.empty(n, l, dtype=torch.int32, device=device)
    if beam == 1:
        ks = torch.arange(n_sym, device=device, dtype=torch.int32).unsqueeze(1)
        dec = torch.empty(n_sym, n, dtype=torch.float32, device=device)
        state = start
        for _ in range(max(1, int(passes))):
            for i in range(l):
                cands = ((state.unsqueeze(0) << bits) | ks) & 0xFFFF
                ext.decode(cands.to(torch.int16), dec, mcg, mul1)
                best = (dec - tiles[:, i]).square().argmin(dim=0)
                state = cands[best, rng_n]
                encoded[:, i] = state
            start = encoded[:, -1] & wrap
            state = start
        return encoded.to(torch.int16)

    ks = torch.arange(n_sym, device=device, dtype=torch.int32).view(n_sym, 1, 1)
    for _ in range(max(1, int(passes))):
        state = torch.zeros(beam, n, dtype=torch.int32, device=device)
        state[0] = start
        score = torch.full((beam, n), 1e30, dtype=torch.float32, device=device)
        score[0] = 0
        parent = torch.zeros(l, beam, n, dtype=torch.int32, device=device)
        chosen = torch.zeros(l, beam, n, dtype=torch.int32, device=device)
        for i in range(l):
            cands = ((state.unsqueeze(0) << bits) | ks) & 0xFFFF
            flat = cands.reshape(n_sym * beam, n)
            dec = torch.empty(n_sym * beam, n, dtype=torch.float32, device=device)
            ext.decode(flat.to(torch.int16), dec, mcg, mul1)
            total = score.unsqueeze(0) + (dec - tiles[:, i]).square().view(
                n_sym, beam, n
            )
            vals, pos = total.reshape(n_sym * beam, n).topk(beam, dim=0, largest=False)
            score = vals
            parent[i] = pos % beam
            gather_n = rng_n.unsqueeze(0).expand(beam, n)
            state = flat[pos, gather_n]
            chosen[i] = state
        cur = score.argmin(dim=0)
        for i in range(l - 1, -1, -1):
            encoded[:, i] = chosen[i, cur, rng_n]
            cur = parent[i, cur, rng_n]
        start = encoded[:, -1] & wrap
    return encoded.to(torch.int16)


def _quantize_fast(
    weights: list,
    bits: int,
    device: str,
    h_cache: dict,
    greedy: bool = False,
    beam: int = 1,
    codebook: str = DEFAULT_CODEBOOK,
) -> list[dict]:
    """All-tiles encode, no LDLQ strip walk and no global-scale search.

    Identity-Hessian LDLQ walks 320 K-tiles per matrix (~5.3 s/expert on GB10).
    Uncalibrated fallback plus skip_g_scale keeps the trellis the inference
    kernel expects, without the per-strip compensation GEMMs. greedy=True uses
    a 2**K sliding-window search instead of tail-biting Viterbi.
    """
    import torch
    from exllamav3.modules.quant.exl3_lib.quantize import (
        codebook_mcg_mult,
        codebook_mul1_mult,
        finalize_capture_H,
        pack_trellis,
        quantize_tiles,
        regularize,
        tensor_core_perm,
    )

    if not weights:
        return []
    dev = torch.device(device)
    qa = quant_args_for(bits, device, codebook)
    cb = get_codebook(codebook)
    marker = codebook_mul1_mult if cb.name == "mul1" else codebook_mcg_mult
    perm = tensor_core_perm(dev)
    packed_list = []
    chunk = 256
    for w in weights:
        wf = w.to(dev, dtype=torch.float32, non_blocking=True).contiguous()
        in_f = int(wf.shape[0])
        h_data = _meta_h(in_f, device, h_cache)
        q_fallback, _H, _L, su, H_diag = finalize_capture_H(h_data, qa, False)
        su = su.to(dev)
        sv = (torch.randn(wf.shape[1], device=dev).sign() + 1e-5).sign().float().unsqueeze(0)
        _aos, wr, _gs, su, sv = regularize(
            wf, su, sv, dict(qa), False, H_diag, None, skip_g_scale=True, q_fallback=q_fallback
        )
        k, n = wr.shape
        tiles_k, tiles_n = k // 16, n // 16
        tiles = (
            wr.reshape(tiles_k, 16, tiles_n, 16)
            .permute(0, 2, 1, 3)
            .reshape(tiles_k * tiles_n, 256)
            .contiguous()
        )
        tiles = tiles[:, perm]
        if greedy:
            encoded = _quantize_greedy_tiles(
                tiles, bits, device, beam=beam, codebook=codebook
            ).view(tiles_k, tiles_n, 256)
        else:
            idxs = []
            for i in range(0, tiles.shape[0], chunk):
                _qw, qi = quantize_tiles(tiles[i : i + chunk], qa)
                idxs.append(qi)
            encoded = torch.cat(idxs, 0).view(tiles_k, tiles_n, 256)
            del idxs
        trellis = pack_trellis(encoded, qa)
        trellis = _pfg8_fold(trellis)
        packed_list.append(
            {
                "trellis": trellis.detach().cpu(),
                "suh": su.flatten().contiguous().to(dtype=torch.half).cpu(),
                "svh": sv.flatten().contiguous().to(dtype=torch.half).cpu(),
                cb.suffix: torch.tensor(marker, dtype=torch.uint32).view(torch.int).cpu(),
            }
        )
        del wf, wr, tiles, encoded, trellis
    return packed_list


def _quantize_group(
    weights: list,
    bits: int,
    device: str,
    h_cache: dict,
    fast: bool = True,
    greedy: bool = False,
    beam: int = 1,
    codebook: str = DEFAULT_CODEBOOK,
) -> list[dict]:
    """weights: CPU/GPU float32 (in, out), same shape. Returns packed dicts."""
    from exllamav3.modules.quant.exl3_lib.quantize import (
        quantize_exl3,
        quantize_exl3_batch,
    )

    if not weights:
        return []
    if fast:
        return _quantize_fast(
            weights,
            bits,
            device,
            h_cache,
            greedy=greedy,
            beam=beam,
            codebook=codebook,
        )
    in_features = int(weights[0].shape[0])
    h_data = _identity_h(in_features, device, h_cache)
    qargs = [quant_args_for(bits, device, codebook) for _ in weights]
    h_list = [h_data] * len(weights)
    if len(weights) == 1:
        _, _, out = quantize_exl3(
            weights[0],
            h_data,
            qargs[0],
            return_weight_q=False,
            verbose=False,
            swap_to_device=__import__("torch").device(device),
        )
        return [_pack_one(out)]
    results = quantize_exl3_batch(weights, h_list, qargs, verbose=False)
    packed_list = []
    for item in results:
        if item is None:
            raise RuntimeError("quantize_exl3_batch returned None")
        _err, out = item
        packed_list.append(_pack_one(out))
    return packed_list


def convert_shards(
    src: Path,
    dst: Path,
    bits: int,
    device: str,
    batch: int,
    allow_partial: bool,
    only_files: set[str] | None = None,
    fast: bool = True,
    greedy: bool = False,
    beam: int = 1,
    codebook: str = DEFAULT_CODEBOOK,
) -> int:
    from safetensors.torch import safe_open, save_file

    idx = _load_index(src)
    weight_map = idx["weight_map"]
    by_file: dict[str, list[str]] = defaultdict(list)
    for name, fname in weight_map.items():
        by_file[fname].append(name)

    if only_files is not None:
        by_file = {k: v for k, v in by_file.items() if k in only_files}
        extra = only_files - set(by_file)
        if extra:
            print(f"unknown --only-files {sorted(extra)[:8]}", flush=True)

    new_map: dict[str, str] = {}
    dst.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    h_cache: dict = {}
    n_ex = 0
    t0 = time.time()

    for fname, names in sorted(by_file.items()):
        out_path = dst / fname
        src_file = src / fname
        if not src_file.is_file():
            print(f"missing {fname}", flush=True)
            missing.append(fname)
            continue
        if out_path.is_file():
            print(f"skip existing {fname}", flush=True)
            map_existing_names(names, fname, new_map)
            continue
        if not shard_needs_exl3(names):
            kind = link_or_copy(src_file, out_path)
            print(f"{kind} {fname}", flush=True)
            for n in names:
                new_map[n] = fname
            continue

        tensors: dict = {}
        expert_w = [n for n in names if is_routed_expert_weight(n)]
        with safe_open(src_file, framework="pt") as fh:
            for name in names:
                if is_routed_expert_tensor(name):
                    continue
                tensors[name] = fh.get_tensor(name)
                new_map[name] = fname

            by_kind: dict[str, list[str]] = defaultdict(list)
            for wname in expert_w:
                if wname.endswith(".w1.weight"):
                    by_kind["w1"].append(wname)
                elif wname.endswith(".w2.weight"):
                    by_kind["w2"].append(wname)
                elif wname.endswith(".w3.weight"):
                    by_kind["w3"].append(wname)
                else:
                    by_kind["other"].append(wname)

            for kind, group in by_kind.items():
                for i in range(0, len(group), batch):
                    chunk = group[i : i + batch]
                    ws = []
                    stems = []
                    bt = time.time()
                    for wname in chunk:
                        sname = wname[: -len(".weight")] + ".scale"
                        w = fh.get_tensor(wname)
                        scale = fh.get_tensor(sname)
                        ws.append(_dequant_t(w, scale, device))
                        stems.append(wname[: -len(".weight")])
                    packed_list = _quantize_group(
                        ws,
                        bits,
                        device,
                        h_cache,
                        fast=fast,
                        greedy=greedy,
                        beam=beam,
                        codebook=codebook,
                    )
                    del ws
                    for stem, packed in zip(stems, packed_list):
                        for suf, val in packed.items():
                            key = f"{stem}.{suf}"
                            tensors[key] = val
                            new_map[key] = fname
                    n_ex += len(chunk)
                    dt = time.time() - bt
                    rate = n_ex / max(time.time() - t0, 1e-6)
                    print(
                        f"exl3 {fname} {kind} {i}-{i+len(chunk)} "
                        f"n={len(chunk)} {dt:.1f}s tot={n_ex} {rate:.2f}/s",
                        flush=True,
                    )

        tmp = out_path.with_name(out_path.name + ".tmp")
        if tmp.exists():
            tmp.unlink()
        save_file(tensors, str(tmp))
        tmp.rename(out_path)
        os.chmod(out_path, 0o644)
        print(f"wrote {out_path} tensors={len(tensors)}", flush=True)
        del tensors

    idx_path = dst / "model.safetensors.index.json"
    if idx_path.is_file():
        prev = json.loads(idx_path.read_text()).get("weight_map") or {}
        prev.update(new_map)
        new_map = prev
    idx_path.write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": new_map}, indent=2)
        + "\n"
    )
    if missing:
        print(f"partial: missing {missing}", flush=True)
        if not allow_partial:
            return 3
    print(f"done experts={n_ex} elapsed={time.time()-t0:.0f}s", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--src",
        type=Path,
        default=Path.home()
        / ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash"
        / "snapshots/dba1be0a40aa45a94ad051997016db3960a90277",
    )
    ap.add_argument(
        "--dst",
        type=Path,
        default=Path.home()
        / ".cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-EXL3"
        / f"snapshots/{revision_for()}",
    )
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument(
        "--codebook",
        default=DEFAULT_CODEBOOK,
        choices=sorted(("mcg", "mul1")),
        help="EXL3 trellis codebook. Default mul1. Serve still pins 2.0bpw-mcg.",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument(
        "--ldlq",
        action="store_true",
        help="Use identity-Hessian LDLQ instead of the default all-tiles fallback.",
    )
    ap.add_argument(
        "--greedy",
        action="store_true",
        help="Greedy/beam K-bit sliding-window encode instead of tail-biting Viterbi.",
    )
    ap.add_argument(
        "--beam",
        type=int,
        default=16,
        help="Beam width for --greedy (1 = pure greedy). Ignored without --greedy.",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--link-only",
        action="store_true",
        help="Hardlink non-expert shards and write metadata; skip EXL3.",
    )
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="Exit 0 even if some source shards are not downloaded yet.",
    )
    ap.add_argument(
        "--only-files",
        nargs="*",
        default=None,
        help="If set, only process these shard filenames (split across Sparks).",
    )
    args = ap.parse_args()
    if not (args.src / "config.json").is_file():
        print(f"missing official snapshot: {args.src}", file=sys.stderr)
        return 1
    get_codebook(args.codebook)
    if args.bits != 2:
        print(
            "bits!=2 needs a new Spark UMA row after a calibrated K=2 MUL1 pack. "
            "Refusing to raise bits here.",
            file=sys.stderr,
        )
        return 1
    write_pack_config(
        args.src / "config.json",
        args.dst / "config.json",
        args.bits,
        codebook=args.codebook,
    )
    copy_sidecars(args.src, args.dst)
    idx = _load_index(args.src)
    expert = [n for n in idx["weight_map"] if is_routed_expert_tensor(n)]
    weights = [n for n in expert if is_routed_expert_weight(n)]
    print(
        f"routed expert tensors={len(expert)} weights={len(weights)} "
        f"bits={args.bits} codebook={args.codebook} "
        f"greedy={args.greedy} beam={args.beam if args.greedy else 0}"
    )
    print(f"wrote {args.dst / 'config.json'}")
    if args.dry_run:
        return 0
    if args.link_only:
        by_file: dict[str, list[str]] = defaultdict(list)
        for name, fname in idx["weight_map"].items():
            by_file[fname].append(name)
        for fname, names in sorted(by_file.items()):
            src_file = args.src / fname
            out_path = args.dst / fname
            if not src_file.is_file() or out_path.is_file():
                continue
            if shard_needs_exl3(names):
                continue
            kind = link_or_copy(src_file, out_path)
            print(f"{kind} {fname}", flush=True)
        return 0
    try:
        import exllamav3  # noqa: F401
    except ImportError:
        print("exllamav3 is not importable in this interpreter.", file=sys.stderr)
        return 2
    only = set(args.only_files) if args.only_files else None
    return convert_shards(
        args.src,
        args.dst,
        args.bits,
        args.device,
        args.batch,
        args.allow_partial,
        only,
        fast=not args.ldlq,
        greedy=args.greedy,
        beam=args.beam if args.greedy else 1,
        codebook=args.codebook,
    )


if __name__ == "__main__":
    raise SystemExit(main())
