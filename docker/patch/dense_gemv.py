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
fuses the activation quant (one launch fewer per GEMM) and starts streaming
weights before griddepcontrol.wait. Cold isolated (real weights, M=4):
qkv_a 45.1 vs 47.3 us, wo_b 94.2 vs 102.2, wq_b 95.2 vs 99.4, shared gate_up
56.4 vs 60.4, shared down 30.7 vs 32.7, engram wkv 670 vs 726, main_proj 341
vs 374, lm_head 1466 vs 1530.

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
PDL = True  # stream weights before the producer kernel finishes (griddepcontrol)
MARK = "_dsv41_gemv_"

# Boot-log markers for tools/engagement_audit.py.
LOG_ENGAGED = "dense GEMV armed"
LOG_DISARMED = ("; b12x stays",)

# (K, N) per rank at TP=2 -> (name, KC, scale mode, ((max_m, W, S, MR), ...)).
# Scale mode 0 = one scale per 32 rows, 1 = per-row (lm_head pack). Each tuple
# is a kernel config compiled into dense_gemv_kernel.cu (kTable); picked from
# the cold sweeps in results/2026-09-25-kernels/dense-gemv (best per M bucket).
CONFIGS = {
    (5120, 1792): ("qkv_a", 512, 0, ((4, 4, 2, 4), (8, 3, 2, 8))),
    (1280, 16384): ("wq_b", 640, 0, ((8, 2, 2, 8),)),
    (4096, 5120): ("wo_b", 512, 0, ((4, 4, 2, 4), (8, 4, 2, 8))),
    (5120, 2304): ("shared_gate_up", 512, 0, ((4, 4, 2, 4), (8, 3, 2, 8))),
    (1152, 5120): ("shared_down", 384, 0, ((8, 4, 2, 8),)),
    (6144, 25600): ("engram_wkv", 512, 0, ((4, 2, 2, 4), (8, 2, 2, 8))),
    (15360, 5120): ("main_proj", 512, 0, ((4, 2, 2, 4),)),
    (5120, 64640): ("lm_head", 512, 1, ((8, 3, 2, 8),)),
}
DEFAULT_SHAPES = ",".join(f"{k}x{n}" for k, n in CONFIGS)

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


def shapes_from_env(env=None) -> frozenset[tuple[int, int]]:
    """Parse DSV41_DENSE_GEMV_SHAPES ('KxN,KxN') into {(K, N)}; unknown shapes raise."""
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
        key = (int(k), int(n))
        if key not in CONFIGS:
            raise ValueError(f"{ENV_SHAPES}: {tok} has no tuned config (known: {DEFAULT_SHAPES})")
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


def apply(tree: Path) -> bool:
    path = tree / REL
    if not path.is_file():
        raise SystemExit(f"dense_gemv: {path} missing")
    src = path.read_text()
    out = patch_py(src)
    if out == src:
        return False
    path.write_text(out)
    print(f"dsv41: dense GEMV hooks patched into {path}", flush=True)
    return True


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


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: dense_gemv.py VLLM_TREE", file=sys.stderr)
        return 2
    apply(Path(argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
