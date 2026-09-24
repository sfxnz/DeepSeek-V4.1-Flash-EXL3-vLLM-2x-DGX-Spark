#!/usr/bin/env python3
"""Small-M dense MXFP8 GEMMs on deep_gemm fp8_gemm_nt instead of b12x (opt-in).

Decode runs the dense MXFP8 projections at M = 1..8 rows through FlashInfer
b12x (`mm_mxfp8` backend=auto), at 193-210 GB/s (trace3). The deep_gemm
sm120 1d1d kernel family (the one the wo_a fp8 einsum uses with recipe
(1,1,32)) reads the same e4m3 + per-32 ue8m0 format. It has an in-kernel
split-K (kSplitKFactor + sm120_split_k_reduce); whether its host heuristic
picks it for under-filled grids (qkv_a N=1792, shared gate_up N=2304) is
unconfirmed until the bench runs with DG_PRINT_CONFIGS=1.

DSV41_DENSE_DG_SMALLM=1 routes the selected (K, N) shapes to
`fp8_gemm_nt` when the input has <= 8 rows. Larger M (prefill) and
pre-quantized activations keep the stock b12x path.

- Weight: the fp8 [N, K] tensor is shared with b12x (no second copy).
- Scales: deep_gemm packed int32 ue8m0 scales are built ONCE at load from the
  row-major e8m0 scales (~3% of the weight bytes, selected layers only). No
  per-call repack, unlike wo_a's transpose_and_pack_fp32_into_ue8m0.
- Activation: per_token_group_quant_fp8_packed_for_deepgemm (group 32,
  ue8m0), one kernel, replacing the swizzled mxfp8 quant.
- Load-time self-test per layer: M = 1..8 against the stock b12x output.
  Once per shape, one CUDA-graph capture + replay must match eager
  bit-for-bit. Any failure leaves that layer on b12x and logs one line.

DSV41_DENSE_DG_SHAPES: comma list of KxN (per-rank in_features x
out_features). Default: the five wireable decode shapes (qkv_a, wo_b, shared
gate_up, shared down, draft main_proj). Not bit-exact vs b12x (activation
scale rule and accumulation order differ).

wq_b (1280x16384) is not wireable here: DeepseekV4 attention fuses its
activation quant into the q/kv RMSNorm (fused_q_kv_rmsnorm_quant) and hands
wq_b and indexer.wq_b one shared QuantizedActivation, so maybe_apply never
sees a bf16 input for it.

Source rewrite of model_executor/kernels/linear/mxfp8/flashinfer.py in the
prefer_b12x_mxfp8 style, applied from sitecustomize only when the flag is 1.
The rewrite adds two hooks to FlashInferCutlassMxfp8LinearKernel. The hooks
import this module from /opt/dsv41-patch and do nothing when the flag is off.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ENV_FLAG = "DSV41_DENSE_DG_SMALLM"
ENV_SHAPES = "DSV41_DENSE_DG_SHAPES"
DEFAULT_SHAPES = "5120x1792,4096x5120,5120x2304,1152x5120,15360x5120"
MAX_M = 8
SELFTEST_TOL = 0.1  # normwise rel diff vs b12x; a layout bug gives ~1.0+
MARK = "_dsv41_dg_"

REL = Path("model_executor/kernels/linear/mxfp8/flashinfer.py")

IMPORT_OLD = "from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig\n"
IMPORT_NEW = IMPORT_OLD + (
    "\ntry:  # dsv41 dense_mxfp8_deepgemm (DSV41_DENSE_DG_SMALLM)\n"
    "    from dense_mxfp8_deepgemm import maybe_apply as _dsv41_dg_apply\n"
    "    from dense_mxfp8_deepgemm import prepare as _dsv41_dg_prepare\n"
    "except ImportError:\n"
    "    _dsv41_dg_apply = _dsv41_dg_prepare = lambda *a, **k: None\n"
)

# FlashInferCutlassMxfp8LinearKernel only (the CuTe-DSL twin stores weight.t()).
PWAL_OLD = """        layer.weight = Parameter(weight.contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(
            weight_scale_swizzled.contiguous(), requires_grad=False
        )
"""
PWAL_NEW = PWAL_OLD + "        _dsv41_dg_prepare(self, layer, weight_scale_2d)\n"

APPLY_OLD = """        weight = layer.weight
        weight_scale = layer.weight_scale
        N, K = weight.shape
"""
APPLY_NEW = """        _dsv41_dg_out = _dsv41_dg_apply(layer, x, bias)
        if _dsv41_dg_out is not None:
            return _dsv41_dg_out
""" + APPLY_OLD


# ---------------------------------------------------------------------------
# Env parsing (stdlib only; unit-tested on the host)
# ---------------------------------------------------------------------------

def enabled_from_env(env=None) -> bool:
    env = os.environ if env is None else env
    return env.get(ENV_FLAG, "0") == "1"


def shapes_from_env(env=None) -> frozenset[tuple[int, int]]:
    """Parse DSV41_DENSE_DG_SHAPES ('KxN,KxN') into {(K, N)}."""
    env = os.environ if env is None else env
    raw = env.get(ENV_SHAPES, "").strip() or DEFAULT_SHAPES
    out = set()
    for tok in raw.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        k, sep, n = tok.partition("x")
        if not sep or not k.isdigit() or not n.isdigit():
            raise ValueError(f"{ENV_SHAPES}: bad entry {tok!r} (want KxN)")
        out.add((int(k), int(n)))
    return frozenset(out)


# ---------------------------------------------------------------------------
# Source rewrite (stdlib only)
# ---------------------------------------------------------------------------

def patch_py(src: str) -> str:
    if MARK in src:
        return src
    for name, old in (("import", IMPORT_OLD), ("pwal", PWAL_OLD), ("apply", APPLY_OLD)):
        if src.count(old) != 1:
            raise SystemExit(
                f"dense_mxfp8_deepgemm: {name} anchor count={src.count(old)} (want 1)"
            )
    out = src.replace(IMPORT_OLD, IMPORT_NEW, 1)
    out = out.replace(PWAL_OLD, PWAL_NEW, 1)
    out = out.replace(APPLY_OLD, APPLY_NEW, 1)
    compile(out, str(REL), "exec")
    return out


def apply(tree: Path) -> bool:
    path = tree / REL
    if not path.is_file():
        raise SystemExit(f"dense_mxfp8_deepgemm: {path} missing")
    src = path.read_text()
    out = patch_py(src)
    if out == src:
        return False
    path.write_text(out)
    print(f"dsv41: dense deep_gemm small-M hooks patched into {path}", flush=True)
    return True


# ---------------------------------------------------------------------------
# Runtime (torch / vllm; runs inside the serve)
# ---------------------------------------------------------------------------

_shape_state: dict[tuple[int, int], str] = {}  # (K, N) -> "ok" | reason


def _rank() -> str:
    try:
        import torch.distributed as dist

        return str(dist.get_rank()) if dist.is_initialized() else "?"
    except Exception:  # noqa: BLE001 - logging only
        return "?"


# Boot-log markers for tools/engagement_audit.py: an armed shape that did not engage.
LOG_DISARMED = ("; b12x stays", "armed but got")


def _log(msg: str) -> None:
    print(f"[dense-dg] rank{_rank()} {msg}", flush=True)


def dg_mm(x2d, weight, sf):
    """bf16 [M, K] x fp8 [N, K] (packed ue8m0 sf) -> bf16 [M, N]."""
    import torch
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8_packed_for_deepgemm,
    )
    from vllm.utils.deep_gemm import fp8_gemm_nt

    xq, xs = per_token_group_quant_fp8_packed_for_deepgemm(
        x2d, group_size=32, use_ue8m0=True
    )
    out = torch.empty((x2d.shape[0], weight.shape[0]), dtype=x2d.dtype, device=x2d.device)
    fp8_gemm_nt((xq, xs), (weight, sf), out, recipe=(1, 1, 32), is_deep_gemm_e8m0_used=True)
    return out


def pack_weight_scale(weight, scale_2d):
    """Row-major e8m0 [N, K//32] -> deep_gemm packed int32 ue8m0 layout."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
    )

    _, sf = deepgemm_post_process_fp8_weight_block(
        wq=weight, ws=scale_2d, quant_block_shape=(1, 32), use_e8m0=False
    )
    return sf


def _selftest(kernel, layer, sf, key) -> None:
    import torch

    weight = layer.weight
    n, k = int(weight.shape[0]), int(weight.shape[1])
    gen = torch.Generator(device=weight.device).manual_seed(0)
    x = torch.randn(MAX_M, k, generator=gen, device=weight.device, dtype=torch.bfloat16)
    for m in range(1, MAX_M + 1):
        ref = kernel.apply_weights(layer, x[:m]).float()
        got = dg_mm(x[:m], weight, sf).float()
        rel = float((got - ref).norm() / ref.norm().clamp_min(1e-6))
        if not rel < SELFTEST_TOL:
            raise RuntimeError(f"M={m} rel diff vs b12x {rel:.3g} >= {SELFTEST_TOL}")
    if key in _shape_state:
        return
    xs = x[:4].clone()
    eager = dg_mm(xs, weight, sf)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        captured = dg_mm(xs, weight, sf)
    graph.replay()
    torch.cuda.synchronize()
    if not torch.equal(captured, eager):
        raise RuntimeError("cuda-graph replay != eager")
    del graph
    _log(f"K{k}xN{n} armed (M<=8 via deep_gemm fp8_gemm_nt; graph capture ok at M=4; last rel {rel:.3g})")


def prepare(kernel, layer, scale_2d) -> None:
    """Hook at the end of FlashInferCutlassMxfp8LinearKernel.process_weights_after_loading."""
    layer._dsv41_dg_sf = None
    if not enabled_from_env():
        return
    n, k = (int(d) for d in layer.weight.shape)
    key = (k, n)
    if _shape_state.get(key, "ok") != "ok":
        return  # this shape already failed on an earlier layer
    try:
        if key not in shapes_from_env():  # bad list -> except: b12x stays
            return
        from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

        if not is_deep_gemm_e8m0_used():
            raise RuntimeError("DeepGEMM E8M0 not in use")
        sf = pack_weight_scale(layer.weight.data, scale_2d)
        _selftest(kernel, layer, sf, key)
        _shape_state[key] = "ok"
        layer._dsv41_dg_sf = sf
    except Exception as exc:  # noqa: BLE001 - stock b12x stays on any failure
        _shape_state[key] = repr(exc)
        _log(f"K{k}xN{n} rejected ({exc!r}); b12x stays")


def maybe_apply(layer, x, bias):
    """Hook at the top of FlashInferCutlassMxfp8LinearKernel.apply_weights.

    Returns the output, or None to fall through to the stock b12x path.
    """
    sf = getattr(layer, "_dsv41_dg_sf", None)
    if sf is None:
        return None
    import torch

    if not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16:
        # QuantizedActivation (fused producer) or other dtype: an armed layer
        # that never engages. Say so once instead of staying silent.
        if not getattr(layer, "_dsv41_dg_warned", False):
            layer._dsv41_dg_warned = True
            n, k = (int(d) for d in layer.weight.shape)
            what = x.dtype if isinstance(x, torch.Tensor) else type(x).__name__
            _log(f"K{k}xN{n} armed but got {what} input; b12x runs")
        return None
    x2d = x.reshape(-1, x.shape[-1])
    if not 0 < x2d.shape[0] <= MAX_M:
        return None
    out = dg_mm(x2d, layer.weight, sf)
    if bias is not None:
        out = out + bias
    return out.view(*x.shape[:-1], out.shape[-1])


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: dense_mxfp8_deepgemm.py VLLM_TREE", file=sys.stderr)
        return 2
    apply(Path(argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
