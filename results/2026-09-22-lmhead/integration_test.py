#!/usr/bin/env python3
"""Real-image integration test for the lm_head MXFP8 checkpoint-key routing.

Runs INSIDE the canonical-e12 image (no GPUs needed): imports the REAL
vllm WeightsMapper / AutoWeightsLoader and the REAL DeepSeek-V4.1 mapper
makers, installs the lmhead patch (same install() sitecustomize calls),
and proves:

  1. Every key of the real re-encoded pack index (2.0bpw-mcg-lmhead-mxfp8)
     maps into the VL wrapper namespace; in particular
     lm_head.weight        -> language_model.lm_head.weight
     lm_head.weight_scale  -> language_model.lm_head.weight_scale
  2. The stock pack (2.0bpw-mcg) maps EXACTLY as without the patch
     (head.weight -> language_model.lm_head.weight), and no key of either
     pack maps to a bare wrapper-root name (the ValueError path).
  3. A real AutoWeightsLoader over a wrapper-shaped module tree loads the
     mapped head group without ValueError for BOTH packs (fp8+scale for the
     lm pack, bf16 for stock).

Exit 0 = PASS. Run from the repo root:
  docker run --rm --entrypoint python3 \
    -v $PWD:/repo:ro \
    -v $HOME/.cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-EXL3:/packs:ro \
    dsv41-flash-exl3-sm121:canonical-e12 /repo/results/2026-09-22-lmhead/integration_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/repo/docker/patch")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.nn.parameter import Parameter  # noqa: E402

STOCK = Path("/packs/snapshots/2.0bpw-mcg")
LM = Path("/packs/snapshots/2.0bpw-mcg-lmhead-mxfp8")

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    tag = "ok " if cond else "FAIL"
    print(f"[{tag}] {msg}", flush=True)
    if not cond:
        failures.append(msg)


# --- install the patch exactly as sitecustomize does -----------------------
from lmhead_mxfp8 import install, snapshot_has_lmhead_mxfp8  # noqa: E402

check(install() is True, "install() returned True")
check(install() is True, "install() idempotent on second call")
check(snapshot_has_lmhead_mxfp8(LM) is True, "snapshot probe sees lm pack marker")
check(snapshot_has_lmhead_mxfp8(STOCK) is False, "snapshot probe: stock not marked")

# --- real mapper, built the way the model __init__ builds it ---------------
from vllm.models.deepseek_v4_1.nvidia.vl_model import (  # noqa: E402
    _make_deepseek_v4_vl_weights_mapper,
)

# expert_dtype defaults to "fp4" for this pack (config.json has no
# expert_dtype -> getattr default in model __init__).
mapper = _make_deepseek_v4_vl_weights_mapper("fp4", "weight_scale")


def map_keys(path: Path) -> dict[str, str | None]:
    wm = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    out: dict[str, str | None] = {}
    for k in wm:
        res = mapper._map_name_with_shard(k)
        out[k] = res[0] if res is not None else None
    return out


lm_map = map_keys(LM)
stock_map = map_keys(STOCK)

# 1. lm pack routing
check(
    lm_map.get("lm_head.weight") == "language_model.lm_head.weight",
    f"lm pack: lm_head.weight -> {lm_map.get('lm_head.weight')!r}",
)
check(
    lm_map.get("lm_head.weight_scale") == "language_model.lm_head.weight_scale",
    f"lm pack: lm_head.weight_scale -> {lm_map.get('lm_head.weight_scale')!r}",
)

# 2. stock pack unchanged + no bare-root lm_head keys in either map
check(
    stock_map.get("head.weight") == "language_model.lm_head.weight",
    f"stock pack: head.weight -> {stock_map.get('head.weight')!r}",
)
check("lm_head.weight_scale" not in stock_map, "stock pack has no lm_head_scale key")
for tag, m in (("lm", lm_map), ("stock", stock_map)):
    bare = [
        (k, v)
        for k, v in m.items()
        if isinstance(v, str) and v.split(".")[0] in ("lm_head", "head")
    ]
    check(
        not bare,
        f"{tag} pack: no key maps to a wrapper-root lm_head/head name ({bare[:3]})",
    )

# every mapped key must land in a real wrapper namespace (wrapper-level
# params like image_start/image_end/image_newline are legitimate roots)
VALID_ROOTS = ("language_model", "vision", "aligner", "image_")
for tag, m in (("lm", lm_map), ("stock", stock_map)):
    bad = [
        (k, v)
        for k, v in m.items()
        if isinstance(v, str) and not v.startswith(VALID_ROOTS)
    ]
    check(not bad, f"{tag} pack: all mapped keys in valid namespaces (bad: {bad[:3]})")


# --- 3. real AutoWeightsLoader over the mapped head group ------------------
from vllm.model_executor.models.utils import AutoWeightsLoader  # noqa: E402


def _copy_loader(param, loaded):
    param.data.copy_(loaded)


class _Head(nn.Module):
    def __init__(self, fp8: bool, V: int, H: int):
        super().__init__()
        self.weight = Parameter(
            torch.empty(V, H, dtype=torch.float8_e4m3fn if fp8 else torch.bfloat16),
            requires_grad=False,
        )
        self.weight.weight_loader = _copy_loader
        if fp8:
            self.weight_scale = Parameter(
                torch.empty(V, H // 32, dtype=torch.uint8), requires_grad=False
            )
            self.weight_scale.weight_loader = _copy_loader


class _Child(nn.Module):
    def __init__(self, fp8: bool, V: int, H: int):
        super().__init__()
        self.lm_head = _Head(fp8, V, H)


class _Wrapper(nn.Module):
    def __init__(self, fp8: bool, V: int = 1024, H: int = 512):
        super().__init__()
        self.language_model = _Child(fp8, V, H)


def run_load(pack_map: dict, fp8: bool, label: str) -> None:
    V, H = 1024, 512
    keys = ["lm_head.weight", "lm_head.weight_scale"] if fp8 else ["head.weight"]
    weights = []
    for k in keys:
        tgt = pack_map[k]
        assert tgt is not None, f"{k} dropped by mapper"
        if k.endswith("weight_scale"):
            t = torch.full((V, H // 32), 129, dtype=torch.uint8)
        elif k == "head.weight":
            t = torch.zeros(V, H, dtype=torch.bfloat16)
        else:
            t = torch.zeros(V, H, dtype=torch.float8_e4m3fn)
        weights.append((tgt, t))
    tree = _Wrapper(fp8, V, H)
    loader = AutoWeightsLoader(tree)
    loaded = loader.load_weights(weights)
    head = tree.language_model.lm_head
    expect = (
        {"language_model.lm_head.weight", "language_model.lm_head.weight_scale"}
        if fp8
        else {"language_model.lm_head.weight"}
    )
    check(set(loaded) == expect, f"{label}: AutoWeightsLoader loaded {sorted(loaded)}")
    if fp8:
        check(
            bool((head.weight_scale.data == 129).all()),
            f"{label}: scale tensor actually copied into the param",
        )


run_load(lm_map, True, "lm pack")
run_load(stock_map, False, "stock pack")

print()
if failures:
    print(f"INTEGRATION TEST: {len(failures)} FAILURE(S)")
    sys.exit(1)
print("INTEGRATION TEST: ALL PASS")
