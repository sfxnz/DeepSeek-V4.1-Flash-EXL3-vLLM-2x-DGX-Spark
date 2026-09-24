#!/usr/bin/env python3
"""requant-exl3-1.5 step 1: Viterbi re-encode vs the served greedy beam-16 pack.

The served 2.0bpw-mcg experts were encoded with `--greedy --beam 16`
(results/2026-09-21-pfg8/shard3-verify.md: relerr 0.3771-0.3775 vs source).
Commit 353eb28 measured that encoder at ~2.1x the tile MSE of tail-biting
Viterbi. This probe re-encodes a few routed experts from the MXFP4 source at
the same 2.0bpw MCG K=2 through the recipe's own _quantize_fast path and
reports relerr/MSE per arm against the same source weights:

  stock          the pack's trellis + suh/svh as served
  stock+refit    stock trellis, suh/svh refit (H = I); format unchanged
  greedy16       fresh greedy beam-16 encode (control: the pack's encoder)
  viterbi        exllamav3 quantize_tiles (Viterbi kernel of the image)
  viterbi+refit  viterbi, then the same refit

Refit is exllamav3 v1.5.1 refit_scales (7e2e6b065) specialised to H = I:
(QQ^T) o I is diagonal, so both alternating steps are closed-form per
row/column and minimise plain Frobenius error, which is what relerr measures.

GPU only; run with the serve down (exclusive GPUs). See
results/2026-09-24-review/research/requant-probe.md for the docker command.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

HUB = Path(os.environ.get("HF_HUB", "/cache/huggingface/hub"))
SRC = HUB / "models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277"
PACK = HUB / "models--sfxnz--DeepSeek-V4.1-Flash-EXL3/snapshots/2.0bpw-mcg"
SHARD = "model-00003-of-00048.safetensors"
K = 2
TENSORS_TOTAL = 46080  # routed expert w1/w2/w3 in the source index (40 shards x 1152)


def relerr(a, b) -> float:
    d = (a.double() - b.double()).square().sum() ** 0.5
    return float(d / (b.double().square().sum() ** 0.5))


def refit_identity(w, q, rounds: int = 2):
    """Row/column scale refit of q toward w in the Frobenius metric.

    w, q: (k, n) arrays in the original basis (numpy or torch). Returns
    (q_refit, r, c) with q_refit = diag(r) q diag(c) accumulated over rounds.
    Same guards as upstream: non-positive or degenerate factors become 1.
    """
    r_tot = 1.0
    c_tot = 1.0
    for _ in range(rounds):
        den = (q * q).sum(0)
        bad = den <= 1e-30
        c = (q * w).sum(0) / (den + bad) * ~bad + bad
        q = q * c[None, :]
        c_tot = c_tot * c
        den = (q * q).sum(1)
        bad = den <= 1e-30
        r = (q * w).sum(1) / (den + bad) * ~bad + bad
        r = r * (r > 0) + (r <= 0)
        q = q * r[:, None]
        r_tot = r_tot * r
    return q, r_tot, c_tot


def expert_stems(names: list[str], n: int) -> list[str]:
    """First n experts in the shard, as 'layers.L.ffn.experts.E' stems."""
    stems = sorted({x.rsplit(".", 2)[0] for x in names if x.endswith((".w1.weight", ".w2.weight", ".w3.weight"))},
                   key=lambda s: tuple(int(p) if p.isdigit() else p for p in s.split(".")))
    return stems[:n]


def full_pack_hours(sec_per_tensor: float, nodes: int) -> float:
    return sec_per_tensor * TENSORS_TOTAL / nodes / 3600


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--pack", type=Path, default=PACK)
    ap.add_argument("--shard", default=SHARD)
    ap.add_argument("--experts", type=int, default=2, help="experts to sample (3 tensors each)")
    ap.add_argument("--arms", default="stock,stock+refit,greedy16,viterbi,viterbi+refit")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    os.environ.pop("DSV41_PACK_PF_G8", None)  # stock layout for reconstruct

    import torch
    from exllamav3.ext import exllamav3_ext as ext
    import exllamav3
    from safetensors.torch import safe_open

    from quantize_experts_exl3 import _dequant_t, _load_index, _quantize_fast

    dev = torch.device(args.device)
    arms = args.arms.split(",")
    names = sorted(n for n, f in _load_index(args.src)["weight_map"].items() if f == args.shard)
    stems = expert_stems(names, args.experts)

    def recon(trellis, suh, svh, shape):
        out = torch.empty(shape, dtype=torch.half, device=dev)
        ext.reconstruct_had_slice(out, trellis.to(dev), suh.to(dev).half(), svh.to(dev).half(), K, True, False, 0)
        return out

    try:  # v1.5.1+: cross-check the H = I closed form against upstream
        from exllamav3.modules.quant.exl3_lib.quantize import refit_scales as upstream_refit
    except ImportError:
        upstream_refit = None

    def refit_arm(w, trellis, suh, svh, row, arm):
        q = recon(trellis, suh, svh, w.shape).float()
        _, r, c = refit_identity(w, q)
        if upstream_refit is not None:
            eye = torch.eye(w.shape[0], device=dev)
            _, su_u, sv_u, _, _ = upstream_refit(w, q, eye, suh.to(dev).float(), svh.to(dev).float())
            row[f"{arm}_upstream"] = relerr(recon(trellis, su_u.flatten().half(), sv_u.flatten().half(), w.shape), w)
        return recon(trellis, (suh.to(dev).float() * r).half(), (svh.to(dev).float() * c).half(), w.shape)

    rows = []
    with safe_open(args.src / args.shard, framework="pt") as src, \
            safe_open(args.pack / args.shard, framework="pt") as pack:
        for stem in stems:
            for kind in ("w1", "w2", "w3"):
                t = f"{stem}.{kind}"
                w = _dequant_t(src.get_tensor(t + ".weight"), src.get_tensor(t + ".scale"), args.device).to(dev)
                st = {x: pack.get_tensor(f"{t}.{x}") for x in ("trellis", "suh", "svh")}
                row = {"tensor": t, "shape": list(w.shape)}
                for arm in arms:
                    t0 = time.time()
                    if arm == "stock":
                        wq = recon(st["trellis"], st["suh"], st["svh"], w.shape)
                    elif arm == "stock+refit":
                        wq = refit_arm(w, st["trellis"], st["suh"], st["svh"], row, arm)
                    else:
                        greedy = arm.startswith("greedy")
                        enc = _quantize_fast([w.cpu()], K, args.device, {}, greedy=greedy,
                                             beam=16 if greedy else 1, codebook="mcg")[0]
                        torch.cuda.synchronize()
                        enc_s = time.time() - t0
                        if arm.endswith("+refit"):
                            wq = refit_arm(w, enc["trellis"], enc["suh"], enc["svh"], row, arm)
                        else:
                            wq = recon(enc["trellis"], enc["suh"], enc["svh"], w.shape)
                        row[f"{arm}_encode_s"] = enc_s
                    e = relerr(wq, w)
                    row[arm] = e
                    print(f"{t:36} {arm:14} relerr {e:.5f}  mse/stock {(e / row.get('stock', e)) ** 2:.3f}",
                          flush=True)
                rows.append(row)

    summary = {"exllamav3": getattr(exllamav3, "__version__", "?"), "experts": stems, "arms": {}}
    for arm in arms:
        errs = [r[arm] for r in rows]
        mean = sum(errs) / len(errs)
        s = {"relerr_mean": mean, "mse_ratio_vs_stock": (mean / (sum(r["stock"] for r in rows) / len(rows))) ** 2
             if "stock" in arms else None}
        enc = [r[f"{arm}_encode_s"] for r in rows if f"{arm}_encode_s" in r]
        if enc:
            spt = sum(enc) / len(enc)
            s.update(sec_per_tensor=spt, full_pack_h_1node=full_pack_hours(spt, 1),
                     full_pack_h_2nodes=full_pack_hours(spt, 2))
        summary["arms"][arm] = s
        print(f"== {arm:14} " + " ".join(f"{k}={v:.4g}" for k, v in s.items() if isinstance(v, float)))
    if args.json:
        args.json.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
