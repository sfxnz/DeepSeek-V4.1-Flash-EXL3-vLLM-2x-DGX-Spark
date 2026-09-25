"""Real per-rank (TP=2, rank 0) dense decode weights from the local pack, read-only.

Shapes follow the serve's TP=2 sharding (tools/decode_timeline.py WEIGHTS):
column-parallel layers keep rank 0's row half, row-parallel layers rank 0's
column half, replicated layers the whole tensor. MXFP8 scales are expanded
the way vLLM's KMxfp8Static loader does it (32x32 checkpoint blocks ->
repeat_interleave(32) over rows -> per-row [N, K/32] uint8); the lm_head
mxfp8 pack already stores per-row [N, K/32] scales.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

import torch

PACK = os.environ.get("DGEMV_PACK", "/pack/2.0bpw-mcg-lmhead-mxfp8")

# name -> (kind, N, K, layers that have it)
TARGET_LAYERS = list(range(40))
SHAPES = {
    "qkv_a": ("fp8", 1792, 5120, TARGET_LAYERS),
    "wq_b": ("fp8", 16384, 1280, TARGET_LAYERS),
    "wo_b": ("fp8", 5120, 4096, TARGET_LAYERS),
    "shared_gate_up": ("fp8", 2304, 5120, TARGET_LAYERS),
    "shared_down": ("fp8", 5120, 1152, TARGET_LAYERS),
    "indexer_wq_b": ("fp8", 4096, 1280, [2, 8, 14, 20, 24, 28, 32, 36]),
    "main_proj": ("fp8", 5120, 15360, [0]),  # mtp.0 only
    "engram_wkv": ("fp8", 25600, 6144, [1, 14]),
    "lm_head": ("fp8", 64640, 5120, [0]),
    "wo_a": ("fp8g", 4096, 4096, TARGET_LAYERS),  # 4 groups x [1024, 4096]
    "router_gate": ("bf16", 384, 5120, TARGET_LAYERS),
    "draft_router_gate": ("bf16", 128, 5120, [0, 1, 2]),  # mtp.N
    "compressor_wkv": ("bf16", 512, 5120, [2, 8, 14, 20]),
    "compressor_wgate": ("bf16", 512, 5120, [2, 8, 14, 20]),
}


@lru_cache(maxsize=None)
def _index() -> dict:
    return json.load(open(os.path.join(PACK, "model.safetensors.index.json")))["weight_map"]


def _get(name: str, rows=None, cols=None) -> torch.Tensor:
    from safetensors import safe_open

    f = os.path.join(PACK, _index()[name])
    with safe_open(f, framework="pt") as fh:
        sl = fh.get_slice(name)
        r = slice(None) if rows is None else slice(*rows)
        c = slice(None) if cols is None else slice(*cols)
        t = sl[r, c] if len(sl.get_shape()) == 2 else sl[r]
    return t


def _fp8(name: str, rows=None, cols=None, block_rows: int = 32):
    """(weight e4m3 [n, k], per-row scale uint8 [n, k/32]) for a 32x32-block checkpoint tensor."""
    w = _get(name + ".weight", rows, cols)
    srows = None if rows is None else (rows[0] // block_rows, -(-rows[1] // block_rows))
    scols = None if cols is None else (cols[0] // 32, cols[1] // 32)
    s = _get(name + ".scale", srows, scols).view(torch.uint8)
    s = s.repeat_interleave(block_rows, dim=0)
    return w.contiguous(), s.contiguous()


def load(shape: str, layer: int, device: str = "cuda"):
    """Rank-0 weight for `shape` at `layer`: fp8 -> (w, s2d), fp8g -> (w [G*n,k], s2d), bf16 -> (w,)."""
    L = f"layers.{layer}"
    if shape == "qkv_a":
        wa, sa = _fp8(f"{L}.attn.wq_a")
        wk, sk = _fp8(f"{L}.attn.wkv")
        out = (torch.cat([wa, wk]), torch.cat([sa, sk]))
    elif shape == "wq_b":
        out = _fp8(f"{L}.attn.wq_b", rows=(0, 16384))
    elif shape == "wo_b":
        out = _fp8(f"{L}.attn.wo_b", cols=(0, 4096))
    elif shape == "shared_gate_up":
        g, gs = _fp8(f"{L}.ffn.shared_experts.w1", rows=(0, 1152))
        u, us = _fp8(f"{L}.ffn.shared_experts.w3", rows=(0, 1152))
        out = (torch.cat([g, u]), torch.cat([gs, us]))
    elif shape == "shared_down":
        out = _fp8(f"{L}.ffn.shared_experts.w2", cols=(0, 1152))
    elif shape == "indexer_wq_b":
        out = _fp8(f"{L}.attn.indexer.wq_b")
    elif shape == "main_proj":
        out = _fp8(f"mtp.{layer}.main_proj")
    elif shape == "engram_wkv":
        out = _fp8(f"{L}.engram.wkv")
    elif shape == "lm_head":
        w = _get("lm_head.weight", (0, 64640))
        s = _get("lm_head.weight_scale", (0, 64640)).view(torch.uint8)
        out = (w.contiguous(), s.contiguous())
    elif shape == "wo_a":
        out = _fp8(f"{L}.attn.wo_a", rows=(0, 4096))
    elif shape == "router_gate":
        out = (_get(f"{L}.ffn.gate.weight").contiguous(),)
    elif shape == "draft_router_gate":
        out = (_get(f"mtp.{layer}.ffn.gate.weight").contiguous(),)
    elif shape in ("compressor_wkv", "compressor_wgate"):
        which = shape.split("_")[1]
        out = (_get(f"{L}.attn.compressor.{which}.weight").contiguous(),)
    else:
        raise KeyError(shape)
    return tuple(t.to(device) for t in out)


def compact_scale(s2d: torch.Tensor, block_rows: int = 32):
    """[N, K/32] per-row scale -> [N/32, K/32] if every 32-row group repeats one row, else None."""
    n = s2d.shape[0]
    if n % block_rows:
        return None
    g = s2d.view(n // block_rows, block_rows, -1)
    if not torch.equal(g, g[:, :1, :].expand_as(g)):
        return None
    return g[:, 0, :].contiguous()


def v3_scales_from(copy, N: int, K: int, kspan: int):
    """v3 scales from a timing copy (w, swizzled scale, compact-or-None)."""
    _, wsw, sc = copy
    nspan, kbs = K // kspan, kspan // 32
    if sc is not None:
        out = torch.zeros(N // 32, nspan, 16, dtype=torch.uint8, device=sc.device)
        out[:, :, :kbs] = sc.view(N // 32, nspan, kbs)
        return 0, out
    KB = K // 32
    mt, kt = (N + 127) // 128, (KB + 3) // 4
    s2d = wsw.view(mt, kt, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(mt * 128, kt * 4)[:N, :KB]
    return 1, s2d.reshape(N // 16, 16, nspan, kbs).permute(0, 2, 1, 3).contiguous()


def v3_scales(s2d: torch.Tensor, N: int, K: int, kspan: int):
    """(smode, tensor) for the v3 kernel: COMPACT32 [N/32][nspan][16] when every
    32-row group shares its scales, else TILE [N/16][nspan][16][KBS]."""
    nspan, kbs = K // kspan, kspan // 32
    sc = compact_scale(s2d)
    if sc is not None:
        out = torch.zeros(N // 32, nspan, 16, dtype=torch.uint8, device=s2d.device)
        out[:, :, :kbs] = sc.view(N // 32, nspan, kbs)
        return 0, out
    return 1, s2d.view(N // 16, 16, nspan, kbs).permute(0, 2, 1, 3).contiguous()
