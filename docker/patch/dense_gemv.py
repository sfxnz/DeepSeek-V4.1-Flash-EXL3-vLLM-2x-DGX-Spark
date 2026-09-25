#!/usr/bin/env python3
"""Dense MXFP8 decode GEMMs at M <= 8 on a purpose-built GEMV (opt-in, bit-exact).

DSV41_DENSE_GEMV=1 routes the dense MXFP8 projections whose (K, N) is in
DSV41_DENSE_GEMV_SHAPES (default: every shape in CONFIGS) to the kernel in
dense_gemv_kernel.cu whenever the input has <= 8 rows (decode: target verify
M=4/8, draft M=3/6). Larger M (prefill) keeps the stock b12x path.

Numerics: bit-exact with the stock path. The kernel quantizes a bf16 input
with FlashInfer's mxfp8_quantize rule (or takes the producer's
QuantizedActivation as is), issues the same block-scaled MMA as b12x per
32-wide k block and chains them in the same order, so the bf16 output equals
b12x's output bit for bit. Every armed layer is checked against the stock
path at load (see _selftest); the first eager pre-quantized call of each shape
is also checked against b12x on the real producer's activation. Any mismatch
leaves the shape on b12x with one log line.

Why it is faster (kernel_study/dense_gemv, results/2026-09-25-kernels/dense-gemv):
one warp per 16-row tile streams the stock [N, K] e4m3 weight through a
cp.async ring of >= 384-byte per-row bursts (b12x's 128-B tile rows lose DRAM
page locality), reads one ue8m0 per 32 rows instead of per row (-3% bytes),
fuses the activation quant (one launch fewer per GEMM; two staging warps per
CTA quantize one k-span ahead of the MMA warps). Cold isolated (real weights,
M=4, b12x incl. its quant kernel): qkv_a 44.1 vs 48.0 us, wo_b 94.2 vs 102.5,
wq_b 93.9 vs 99.4 (pre-quantized), shared gate_up 55.3 vs 60.4, shared down
29.8 vs 32.6, engram wkv 670 vs 726, main_proj 340 vs 375, lm_head 1450 vs
1515. Serve-like chain (8 layers of qkv_a/wq_b/wo_b/gate_up/down in one CUDA
graph): 345.7 -> 319.6 us per layer, outputs bit-identical.

- Weights: the stock fp8 [N, K] tensor, shared with b12x (no second copy).
- Scales: built ONCE at load from the row-major e8m0 scale: [N/32][spans][16]
  when every 32-row group shares its scales (the checkpoint's 32x32 blocks),
  else per-row [N/16][spans][16][KC/32] (lm_head mxfp8 pack, +10 MB/rank).
- Kernel: compiled at first use with torch.utils.cpp_extension (nvcc, ~1
  min per rank, cached for the process in /tmp). A failed build leaves every
  shape on b12x.

Source rewrite of model_executor/kernels/linear/mxfp8/flashinfer.py in the
dense_mxfp8_deepgemm style (two hooks in FlashInferCutlassMxfp8LinearKernel),
applied from sitecustomize only when DSV41_DENSE_GEMV=1; lmhead_mxfp8.py calls
prepare_lmhead / maybe_apply itself. Not combinable with
DSV41_DENSE_DG_SMALLM=1 (both replace the same call; this one refuses).
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading
from pathlib import Path

ENV_FLAG = "DSV41_DENSE_GEMV"
ENV_SHAPES = "DSV41_DENSE_GEMV_SHAPES"
ENV_DG = "DSV41_DENSE_DG_SMALLM"
MAX_M = 8
# Launch as a PDL dependent? Measured (seqbench.py, 8-layer dense chain in one graph):
# no gain (M=4: 324.7 us/layer with PDL vs 323.9 without), so off.
PDL = False
MARK = "_dsv41_gemv_"

# Boot-log markers for tools/engagement_audit.py.
LOG_ENGAGED = "dense GEMV armed"
LOG_DISARMED = ("; b12x stays", "; fp8_einsum stays")

# (K, N) per rank at TP=2 -> (name, KC, scale mode, ((max_m, W, S, MR), ...)).
# Scale mode 0 = one scale per 32 rows, 1 = per-row (lm_head pack). Each tuple
# is a kernel config compiled into dense_gemv_kernel.cu (kTable); picked from
# the cold sweeps in results/2026-09-25-kernels/dense-gemv (best per M bucket).
CONFIGS = {
    (5120, 1792): ("qkv_a", 512, 0, ((4, 4, 2, 4), (8, 3, 2, 8))),
    (1280, 16384): ("wq_b", 640, 0, ((8, 2, 2, 8),)),
    (4096, 5120): ("wo_b", 512, 0, ((4, 4, 2, 4), (8, 4, 2, 8))),
    (5120, 2304): ("shared_gate_up", 512, 0, ((4, 2, 2, 4), (8, 2, 2, 8))),
    (1152, 5120): ("shared_down", 384, 0, ((8, 4, 2, 8),)),
    (6144, 25600): ("engram_wkv", 512, 0, ((4, 2, 2, 4), (8, 2, 2, 8))),
    (15360, 5120): ("main_proj", 512, 0, ((4, 2, 2, 4),)),
    (5120, 64640): ("lm_head", 512, 1, ((8, 3, 2, 8),)),
}
DEFAULT_SHAPES = ",".join(f"{k}x{n}" for k, n in CONFIGS)
# o_proj wo_a: 4 groups x [1024, 4096] per rank, grouped GEMV on the DeepGEMM-layout
# activation from fused_inv_rope_fp8_quant (same KC/buckets as wo_b: K = 4096).
WOA_GROUPS, WOA_D, WOA_K = 4, 1024, 4096
WOA_CONFIG = ("wo_a", 512, 0, ((4, 4, 2, 4), (8, 4, 2, 8)))

REL = Path("model_executor/kernels/linear/mxfp8/flashinfer.py")

IMPORT_OLD = "from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig\n"
IMPORT_NEW = IMPORT_OLD + (
    "\ntry:  # dsv41 dense_gemv (DSV41_DENSE_GEMV)\n"
    "    from dense_gemv import maybe_apply as _dsv41_gemv_apply\n"
    "    from dense_gemv import prepare as _dsv41_gemv_prepare\n"
    "except ImportError:\n"
    "    _dsv41_gemv_apply = _dsv41_gemv_prepare = lambda *a, **k: None\n"
)

# FlashInferCutlassMxfp8LinearKernel only (the CuTe-DSL twin stores weight.t()).
PWAL_OLD = """        layer.weight = Parameter(weight.contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(
            weight_scale_swizzled.contiguous(), requires_grad=False
        )
"""
PWAL_NEW = PWAL_OLD + "        _dsv41_gemv_prepare(self, layer, weight_scale_2d)\n"

APPLY_OLD = """        weight = layer.weight
        weight_scale = layer.weight_scale
        N, K = weight.shape
"""
APPLY_NEW = """        _dsv41_gemv_out = _dsv41_gemv_apply(layer, x, bias)
        if _dsv41_gemv_out is not None:
            return _dsv41_gemv_out
""" + APPLY_OLD


# ---------------------------------------------------------------------------
# Env parsing (stdlib only; unit-tested on the host)
# ---------------------------------------------------------------------------

def enabled_from_env(env=None) -> bool:
    env = os.environ if env is None else env
    return env.get(ENV_FLAG, "0") == "1"


def shapes_from_env(env=None) -> frozenset:
    """Parse DSV41_DENSE_GEMV_SHAPES ('KxN,...,wo_a') into {(K, N), 'wo_a'}; unknown entries raise.

    Default: every CONFIGS shape plus the grouped o_proj wo_a."""
    env = os.environ if env is None else env
    raw = env.get(ENV_SHAPES, "").strip() or DEFAULT_SHAPES + "," + WOA_CONFIG[0]
    out = set()
    for tok in raw.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok == WOA_CONFIG[0]:
            out.add(tok)
            continue
        k, sep, n = tok.partition("x")
        if not sep or not k.isdigit() or not n.isdigit():
            raise ValueError(f"{ENV_SHAPES}: bad entry {tok!r} (want KxN or wo_a)")
        key = (int(k), int(n))
        if key not in CONFIGS:
            raise ValueError(f"{ENV_SHAPES}: {tok} has no tuned config (known: {DEFAULT_SHAPES},wo_a)")
        out.add(key)
    return frozenset(out)


def pick(buckets, m: int):
    """(W, S, MR) of the first bucket with max_m >= m, else None."""
    for max_m, w, s, mr in buckets:
        if m <= max_m:
            return w, s, mr
    return None


# ---------------------------------------------------------------------------
# Source rewrite (stdlib only)
# ---------------------------------------------------------------------------

def patch_py(src: str) -> str:
    if MARK in src:
        return src
    for name, old in (("import", IMPORT_OLD), ("pwal", PWAL_OLD), ("apply", APPLY_OLD)):
        if src.count(old) != 1:
            raise SystemExit(f"dense_gemv: {name} anchor count={src.count(old)} (want 1)")
    out = src.replace(IMPORT_OLD, IMPORT_NEW, 1)
    out = out.replace(PWAL_OLD, PWAL_NEW, 1)
    out = out.replace(APPLY_OLD, APPLY_NEW, 1)
    compile(out, str(REL), "exec")
    return out


def _rewrite(path: Path, fn) -> bool:
    if not path.is_file():
        raise SystemExit(f"dense_gemv: {path} missing")
    src = path.read_text()
    out = fn(src)
    if out == src:
        return False
    path.write_text(out)
    print(f"dsv41: dense GEMV hooks patched into {path}", flush=True)
    return True


def apply(tree: Path) -> bool:
    """flashinfer.py hooks (required: SystemExit on drift), then the o_proj wo_a hook
    (independent: a drifted o_proj.py only leaves wo_a on fp8_einsum)."""
    changed = _rewrite(tree / REL, patch_py)
    try:
        changed |= _rewrite(tree / OPROJ_REL, patch_oproj_py)
    except SystemExit as exc:
        print(f"dsv41: dense GEMV wo_a hook skipped ({exc}); fp8_einsum stays", flush=True)
    return changed


# o_proj wo_a: the image's o_proj.py (fix_o_proj_woa_fp8 requant + prepack stages
# already baked in) calls DeepGEMM's fp8_einsum; the hook runs first and returns
# True when it filled z (bit-identical), else the stock einsum runs.
OPROJ_REL = Path("models/deepseek_v4/nvidia/ops/o_proj.py")
OPROJ_IMPORT_OLD = "from vllm.utils.deep_gemm import fp8_einsum\n"
OPROJ_IMPORT_NEW = OPROJ_IMPORT_OLD + (
    "\ntry:  # dsv41 dense_gemv (DSV41_DENSE_GEMV): grouped wo_a GEMV\n"
    "    from dense_gemv import woa_apply as _dsv41_gemv_woa\n"
    "except ImportError:\n"
    "    _dsv41_gemv_woa = lambda *a, **k: False\n"
)
OPROJ_CALL_OLD = """        fp8_einsum(
            "bhr,hdr->bhd",
            (o_proj_input, o_scale),
            (wo_a.weight, weight_scale),
            z,
            recipe=einsum_recipe,
        )
"""
OPROJ_CALL_NEW = """        if not _dsv41_gemv_woa(wo_a, o_proj_input, o_scale, weight_scale, z, einsum_recipe):
            fp8_einsum(
                "bhr,hdr->bhd",
                (o_proj_input, o_scale),
                (wo_a.weight, weight_scale),
                z,
                recipe=einsum_recipe,
            )
"""


def patch_oproj_py(src: str) -> str:
    if MARK in src:
        return src
    for name, old in (("oproj import", OPROJ_IMPORT_OLD), ("oproj einsum", OPROJ_CALL_OLD)):
        if src.count(old) != 1:
            raise SystemExit(f"dense_gemv: {name} anchor count={src.count(old)} (want 1)")
    out = src.replace(OPROJ_IMPORT_OLD, OPROJ_IMPORT_NEW, 1)
    out = out.replace(OPROJ_CALL_OLD, OPROJ_CALL_NEW, 1)
    compile(out, str(OPROJ_REL), "exec")
    return out


# ---------------------------------------------------------------------------
# Runtime (torch / vllm; runs inside the serve)
# ---------------------------------------------------------------------------

_KERNEL_SRC = Path(__file__).resolve().with_name("dense_gemv_kernel.cu")
_ext_lock = threading.Lock()
_ext_mod = None
_ext_err: str | None = None
_shape_state: dict[tuple[int, int], str] = {}  # (K, N) -> "ok" | reason
_preq_checked: dict[tuple[int, int], bool] = {}  # in-situ QuantizedActivation check done


def _rank() -> str:
    try:
        import torch.distributed as dist

        return str(dist.get_rank()) if dist.is_initialized() else "?"
    except Exception:  # noqa: BLE001 - logging only
        return "?"


def _log(msg: str) -> None:
    print(f"[dense-gemv] rank{_rank()} {msg}", flush=True)


def _ext():
    """Build (once per process) and return the kernel extension; raises on failure."""
    global _ext_mod, _ext_err
    with _ext_lock:
        if _ext_mod is not None:
            return _ext_mod
        if _ext_err is not None:
            raise RuntimeError(_ext_err)
        try:
            from torch.utils.cpp_extension import load

            src = _KERNEL_SRC.read_bytes()
            tag = hashlib.sha1(src).hexdigest()[:12]
            build = Path(f"/tmp/dsv41-dense-gemv-{tag}")
            build.mkdir(parents=True, exist_ok=True)
            _ext_mod = load(
                name=f"dsv41_dense_gemv_{tag}",
                sources=[str(_KERNEL_SRC)],
                build_directory=str(build),
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_121a,code=sm_121a"],
                verbose=False,
            )
            return _ext_mod
        except Exception as exc:  # noqa: BLE001
            _ext_err = f"kernel build failed: {exc!r}"[:400]
            raise RuntimeError(_ext_err) from exc


class _Armed:
    __slots__ = ("key", "name", "kc", "smode", "scales", "plans")

    def __init__(self, key, name, kc, smode, scales, plans):
        self.key = key
        self.name = name
        self.kc = kc
        self.smode = smode
        self.scales = scales
        self.plans = plans  # [(max_m, W, S, MR, grid)]

    def plan(self, m: int):
        for max_m, w, s, mr, g in self.plans:
            if m <= max_m:
                return w, s, mr, g
        return None


def build_scales(scale_2d, n: int, k: int, kc: int, smode: int):
    """Kernel scale layout from the row-major [N, K/32] e8m0 scale (see module doc)."""
    import torch

    nspan, kbs = k // kc, kc // 32
    s = scale_2d.view(torch.uint8)
    if smode == 0:
        g = s.view(n // 32, 32, k // 32)
        if not torch.equal(g, g[:, :1, :].expand_as(g)):
            raise RuntimeError("scales are not shared by 32-row groups")
        out = torch.zeros(n // 32, nspan, (kbs + 15) // 16 * 16, dtype=torch.uint8, device=s.device)
        out[:, :, :kbs] = g[:, 0, :].reshape(n // 32, nspan, kbs)
        return out
    return s.reshape(n // 16, 16, nspan, kbs).permute(0, 2, 1, 3).contiguous()


def _run(armed: _Armed, layer, x2d=None, q=None, s=None):
    import torch

    m = (x2d if x2d is not None else q).shape[0]
    pl = armed.plan(m)
    if pl is None:
        return None
    w, st, mr, grid = pl
    n = layer.weight.shape[0]
    y = torch.empty((m, n), dtype=torch.bfloat16, device=layer.weight.device)
    _ext().gemv(x2d, q, s, layer.weight.view(torch.uint8), armed.scales, armed.smode, y, w, st, armed.kc, mr, grid,
                PDL)
    return y


def _selftest(ref_bf16, ref_quant, layer, armed: _Armed, full: bool) -> str:
    """Bitwise vs the stock b12x path; raises on any difference. full: M 1..8 x 3
    distributions + CUDA-graph replay; else M = 1, 4, 8 on one distribution."""
    import torch
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize

    n, k = (int(d) for d in layer.weight.shape)
    dev = layer.weight.device
    gen = torch.Generator(device=dev).manual_seed(20260925 + n + k)
    ms = [m for m in range(1, MAX_M + 1) if armed.plan(m) is not None]
    if not full:
        ms = sorted({ms[0], min(4, ms[-1]), ms[-1]})
    dists = ("normal", "lognormal", "outlier") if full else ("normal",)
    for dist in dists:
        for m in ms:
            x = torch.randn(m, k, generator=gen, device=dev)
            if dist == "lognormal":
                x = x * torch.exp(1.5 * torch.randn(m, k, generator=gen, device=dev))
            elif dist == "outlier":
                x[:, torch.randint(0, k, (8,), generator=gen, device=dev)] *= 300.0
            x = x.to(torch.bfloat16)
            ref = ref_bf16(x)
            got = _run(armed, layer, x2d=x)
            if not torch.equal(got.view(torch.int16), ref.reshape(m, n).view(torch.int16)):
                raise RuntimeError(f"bf16 input M={m} {dist}: not bitwise equal to b12x")
            if ref_quant is not None:
                q, s = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
                ref_q = ref_quant(q, s)
                got_q = _run(armed, layer, q=q.view(torch.uint8), s=s.view(torch.uint8))
                if not torch.equal(got_q.view(torch.int16), ref_q.reshape(m, n).view(torch.int16)):
                    raise RuntimeError(f"pre-quantized input M={m} {dist}: not bitwise equal to b12x")
    if full:
        m = ms[-1]
        x = torch.randn(m, k, generator=gen, device=dev).to(torch.bfloat16)
        eager = _run(armed, layer, x2d=x)
        torch.cuda.synchronize()
        pl = armed.plan(m)
        y = torch.empty_like(eager)
        graph = torch.cuda.CUDAGraph()
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                _ext().gemv(x, None, None, layer.weight.view(torch.uint8), armed.scales, armed.smode, y, pl[0], pl[1],
                            armed.kc, pl[2], pl[3], PDL)
        graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(y.view(torch.int16), eager.view(torch.int16)):
            raise RuntimeError("cuda-graph replay != eager")
        del graph
    return f"M={ms[0]}..{ms[-1]} x {len(dists)} dist bitwise" + (", graph replay ok" if full else "")


def _arm(ref_bf16, ref_quant, layer, scale_2d) -> None:
    layer._dsv41_gemv = None
    if not enabled_from_env():
        return
    n, k = (int(d) for d in layer.weight.shape)
    key = (k, n)
    if _shape_state.get(key, "ok") != "ok":
        return  # this shape already failed on an earlier layer
    try:
        if key not in shapes_from_env():  # bad list -> except: b12x stays
            return
        if os.environ.get(ENV_DG, "0") == "1":
            raise RuntimeError(f"{ENV_DG}=1 replaces the same call; turn one off")
        name, kc, smode, buckets = CONFIGS[key]
        ext = _ext()
        plans = []
        for max_m, w, s, mr in buckets:
            g = int(ext.plan_grid(w, s, kc, mr, smode, 0, n, k))
            if g < 1:
                raise RuntimeError(f"config W{w} S{s} KC{kc} MR{mr} cannot run here")
            plans.append((max_m, w, s, mr, g))
        armed = _Armed(key, name, kc, smode, build_scales(scale_2d, n, k, kc, smode), plans)
        first = key not in _shape_state
        detail = _selftest(ref_bf16, ref_quant, layer, armed, full=first)
        _shape_state[key] = "ok"
        layer._dsv41_gemv = armed
        if first:
            cfg = ", ".join(f"M<={p[0]} W{p[1]} S{p[2]} MR{p[3]} grid{p[4]}" for p in plans)
            _log(f"K{k}xN{n} ({name}) dense GEMV armed: KC{kc} {cfg}; self-test {detail}")
    except Exception as exc:  # noqa: BLE001 - stock b12x stays on any failure
        _shape_state[key] = repr(exc)
        layer._dsv41_gemv = None
        _log(f"K{k}xN{n} rejected ({exc!r}); b12x stays")


def prepare(kernel, layer, scale_2d) -> None:
    """Hook at the end of FlashInferCutlassMxfp8LinearKernel.process_weights_after_loading."""
    layer._dsv41_gemv = None
    if not enabled_from_env():
        return

    def ref_bf16(x):
        return kernel.apply_weights(layer, x)

    def ref_quant(q, s):
        import torch
        from vllm.utils import flashinfer as vllm_flashinfer

        return vllm_flashinfer.mm_mxfp8(q, layer.weight.t(), s, layer.weight_scale, out_dtype=torch.bfloat16,
                                        backend="auto")

    _arm(ref_bf16, ref_quant, layer, scale_2d)


def prepare_lmhead(method, layer, scale_2d) -> None:
    """Hook at the end of lmhead_mxfp8.Mxfp8LMHeadMethod.process_weights_after_loading."""
    layer._dsv41_gemv = None
    if not enabled_from_env():
        return
    _arm(lambda x: method.apply(layer, x), None, layer, scale_2d)


def _check_preq(armed: _Armed, layer, qa) -> bool:
    """First eager QuantizedActivation call per shape: compare with b12x once."""
    import torch

    if armed.key in _preq_checked:
        return _preq_checked[armed.key]
    if torch.cuda.is_current_stream_capturing():
        return False  # never checked yet: stay on b12x inside this graph
    from vllm.utils import flashinfer as vllm_flashinfer

    ok = False
    try:
        got = _run(armed, layer, q=qa.data.view(torch.uint8), s=qa.scale.view(torch.uint8))
        ref = vllm_flashinfer.mm_mxfp8(qa.data, layer.weight.t(), qa.scale, layer.weight_scale,
                                       out_dtype=torch.bfloat16, backend="auto")
        ok = got is not None and torch.equal(got.view(torch.int16), ref.view(torch.int16))
        why = "" if ok else "not bitwise equal to b12x on the producer's activation"
    except Exception as exc:  # noqa: BLE001
        why = repr(exc)
    _preq_checked[armed.key] = ok
    k, n = armed.key
    if ok:
        _log(f"K{k}xN{n} ({armed.name}) pre-quantized input checked bitwise vs b12x (M={qa.data.shape[0]})")
    else:
        _log(f"K{k}xN{n} ({armed.name}) pre-quantized input rejected ({why}); b12x stays")
    return ok


def maybe_apply(layer, x, bias):
    """Hook at the top of apply_weights / the lm_head apply: output, or None for b12x."""
    armed = getattr(layer, "_dsv41_gemv", None)
    if armed is None:
        return None
    import torch

    n, k = armed.key[1], armed.key[0]
    if isinstance(x, torch.Tensor):
        if x.dtype != torch.bfloat16 or x.shape[-1] != k:
            return None
        x2d = x.reshape(-1, k)
        m = x2d.shape[0]
        if not 0 < m <= MAX_M or x2d.stride(1) != 1 or x2d.stride(0) % 8 or x2d.data_ptr() % 16:
            return None
        y = _run(armed, layer, x2d=x2d)
        shape = x.shape
    else:
        data = getattr(x, "data", None)
        scale = getattr(x, "scale", None)
        if data is None or scale is None or getattr(x, "orig_dtype", None) != torch.bfloat16:
            return None
        if data.dim() != 2 or data.shape[1] != k or not 0 < data.shape[0] <= MAX_M or data.stride(1) != 1:
            return None
        if not _check_preq(armed, layer, x):
            return None
        y = _run(armed, layer, q=data.view(torch.uint8), s=scale.view(torch.uint8))
        shape = x.orig_shape
    if y is None:
        return None
    if bias is not None:
        y = y + bias
    return y.view(*shape[:-1], n)


# ---------------------------------------------------------------- o_proj wo_a

_woa_full_test_done = False


def _woa_run(armed: _Armed, wo_a, x8, xs, z) -> bool:
    import torch

    m = x8.shape[0]
    pl = armed.plan(m)
    if pl is None:
        return False
    w, st, mr, grid = pl
    g, d, k = WOA_GROUPS, WOA_D, WOA_K
    _ext().gemv_grouped(x8.view(torch.uint8), xs, wo_a.weight.view(g * d, k).view(torch.uint8), armed.scales,
                        armed.smode, z.view(m, g * d), w, st, armed.kc, mr, grid, PDL, g)
    return True


def _woa_selftest(armed: _Armed, wo_a, weight_scale, recipe) -> str:
    """M 1..8 x 3 distributions of the serve's own producer vs fp8_einsum, bitwise."""
    import torch
    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import fused_inv_rope_fp8_quant
    from vllm.utils.deep_gemm import fp8_einsum

    g, d, k = WOA_GROUPS, WOA_D, WOA_K
    dev = wo_a.weight.device
    gen = torch.Generator(device=dev).manual_seed(20260925)
    cos_sin = torch.randn(4096, 64, generator=gen, device=dev, dtype=torch.float32)
    heads = k // 512  # heads per group (head_dim 512)
    for dist in ("normal", "lognormal", "outlier"):
        for m in range(1, MAX_M + 1):
            o = torch.randn(m, g * heads, 512, generator=gen, device=dev)
            if dist == "lognormal":
                o = o * torch.exp(1.5 * torch.randn(m, g * heads, 512, generator=gen, device=dev))
            elif dist == "outlier":
                o[:, :, torch.randint(0, 448, (4,), generator=gen, device=dev)] *= 200.0
            pos = torch.randint(0, 4096, (m,), generator=gen, device=dev)
            x8, xs = fused_inv_rope_fp8_quant(o.to(torch.bfloat16), pos, cos_sin, n_groups=g, heads_per_group=heads,
                                              nope_dim=448, rope_dim=64, quant_group_size=recipe[2],
                                              tma_aligned_scales=True)
            ref = torch.empty(m, g, d, device=dev, dtype=torch.bfloat16)
            fp8_einsum("bhr,hdr->bhd", (x8, xs), (wo_a.weight, weight_scale), ref, recipe=recipe)
            got = torch.empty_like(ref)
            if not _woa_run(armed, wo_a, x8, xs, got):
                raise RuntimeError(f"no bucket for M={m}")
            if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
                raise RuntimeError(f"M={m} {dist}: not bitwise equal to fp8_einsum")
    return "M=1..8 x 3 dist bitwise vs fp8_einsum"


def _woa_arm(wo_a, x8, xs, weight_scale, z, recipe):
    """First eager call per wo_a layer: build scales, (once) the full self-test, then the
    in-situ check on this call's real inputs. Returns the armed state; raises on any
    mismatch (the caller then keeps fp8_einsum for this layer)."""
    import torch
    from vllm.utils.deep_gemm import fp8_einsum

    global _woa_full_test_done
    g, d, k = WOA_GROUPS, WOA_D, WOA_K
    w = wo_a.weight
    if tuple(recipe) != (1, 1, 32):
        raise RuntimeError(f"einsum recipe {tuple(recipe)} (want (1, 1, 32))")
    if w.dtype != torch.float8_e4m3fn or tuple(w.shape) != (g, d, k) or not w.is_contiguous():
        raise RuntimeError(f"wo_a weight {w.dtype} {tuple(w.shape)} (want e4m3 {(g, d, k)} contiguous)")
    s = wo_a.weight_scale if hasattr(wo_a, "weight_scale") else wo_a.weight_scale_inv
    if s.dtype != torch.float32 or tuple(s.shape) != (g, d, k // 32):
        raise RuntimeError(f"wo_a scale {s.dtype} {tuple(s.shape)} (want fp32 {(g, d, k // 32)})")
    bits = s.contiguous().view(torch.int32)
    if bool(((bits.to(torch.int64) & 0x807FFFFF) != 0).any()):
        raise RuntimeError("wo_a scale is not a positive power of two")
    ue = ((bits >> 23) & 0xFF).to(torch.uint8).view(g * d, k // 32)
    name, kc, smode, buckets = WOA_CONFIG
    ext = _ext()
    plans = []
    for max_m, wr, st, mr in buckets:
        grid = int(ext.plan_grid(wr, st, kc, mr, smode, 1, g * d, k, g))
        if grid < g:
            raise RuntimeError(f"config W{wr} S{st} KC{kc} MR{mr} cannot run here")
        plans.append((max_m, wr, st, mr, grid))
    armed = _Armed(("wo_a",), name, kc, smode, build_scales(ue, g * d, k, kc, smode), plans)
    detail = ""
    if not _woa_full_test_done:
        detail = _woa_selftest(armed, wo_a, weight_scale, recipe) + "; "
        _woa_full_test_done = True
        cfg = ", ".join(f"M<={p[0]} W{p[1]} S{p[2]} MR{p[3]} grid{p[4]}" for p in plans)
        first = True
    else:
        first = False
    ref = torch.empty_like(z)
    fp8_einsum("bhr,hdr->bhd", (x8, xs), (w, weight_scale), ref, recipe=recipe)
    if not _woa_run(armed, wo_a, x8, xs, z):
        raise RuntimeError(f"no bucket for M={x8.shape[0]}")
    if not torch.equal(z.view(torch.int16), ref.view(torch.int16)):
        z.copy_(ref)  # the caller gets the stock result either way
        raise RuntimeError(f"in-situ M={x8.shape[0]}: not bitwise equal to fp8_einsum")
    if first:
        _log(f"wo_a ({g}x{d}x{k}) dense GEMV armed: KC{kc} {cfg}; self-test {detail}in-situ bitwise "
             f"(M={x8.shape[0]})")
    return armed


def woa_apply(wo_a, x8, xs, weight_scale, z, recipe) -> bool:
    """Hook before o_proj's fp8_einsum: True when z was filled (bit-identical), else False."""
    st = getattr(wo_a, "_dsv41_gemv", None)
    if st is False:
        return False
    m = x8.shape[0]
    if not 0 < m <= MAX_M:
        return False
    if st is None:
        try:
            if not enabled_from_env() or WOA_CONFIG[0] not in shapes_from_env():
                wo_a._dsv41_gemv = False
                return False
        except ValueError as exc:
            wo_a._dsv41_gemv = False
            _log(f"wo_a rejected ({exc!r}); fp8_einsum stays")
            return False
        import torch

        if torch.cuda.is_current_stream_capturing():
            # Arm in an eager pass only. vLLM sets cudagraph_num_of_warmups = 1 when graphs
            # are on, so each capture size gets one eager run before it is captured.
            return False
        try:
            wo_a._dsv41_gemv = _woa_arm(wo_a, x8, xs, weight_scale, z, recipe)
            return True
        except Exception as exc:  # noqa: BLE001 - stock einsum stays on any failure
            wo_a._dsv41_gemv = False
            _log(f"wo_a rejected ({exc!r}); fp8_einsum stays")
            return False
    return _woa_run(st, wo_a, x8, xs, z)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: dense_gemv.py VLLM_TREE", file=sys.stderr)
        return 2
    apply(Path(argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
