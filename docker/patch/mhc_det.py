"""DSV41_MHC_DET_SPLITS: bitwise-stock decode mHC on faster kernels (kernels in mhc_det.cu).

The stock decode mHC path per sublayer is mhc_post (TileLang) -> prenorm GEMM (DeepGEMM TF32,
16-split split-K, one CTA per split) -> fused norm (TileLang, sums the 16 partials in split
order). It is deterministic; its cost is the GEMM's layout (16 CTAs, 8 warps each multiplying a
128-row tile for T <= 16 tokens). With DSV41_MHC_DET_SPLITS=16, for T <= 16 tokens:
- mhc_post -> det post (same fp32 FMA order and bf16 rounding, 20*T CTAs, launches its
  dependent GEMM on entry so the weights stream during the post),
- the prenorm GEMM -> det GEMM: the same 16 splits, the same mma.sync m16n8k8 TF32 chain per
  (split, 8-column n-tile), fn pre-packed once as its exact RNE-tf32 bits (19 of 32 bits), one
  warp per chain on 48 CTAs,
- the TileLang fused norm is unchanged.
Every output is bitwise the stock one (kernel_study/mhc_det, results/2026-09-25-kernels/mhc-det).
16 is the only accepted value: it names the stock decode split-K that the kernels reproduce.
After weight load every packed fn and the post kernel are checked bitwise against the stock
kernels on the GPU; any mismatch leaves the stock path in place (LOG_DISARMED).

Imported lazily; top level is stdlib only so importing this module cannot fail.
"""

from __future__ import annotations

import ctypes
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = 16  # the stock decode split-K the det kernels reproduce bitwise
STAGE_KB = 2
FS_PAD, XS_PAD = 4, 8
PK_KB_BYTES = 1280
MAX_T = 16


def _align16(n: int) -> int:
    return (n + 15) & ~15


def kb_per_split(k: int) -> int:
    if k % (64 * SPLITS):
        raise ValueError(f"K={k} is not a multiple of {64 * SPLITS}")
    return k // (64 * SPLITS)


def smem_bytes(t: int, k: int, packed: bool) -> int:
    kbps = kb_per_split(k)
    kspan = kbps * 64
    nstage = -(-kbps // STAGE_KB)
    fn_bytes = kbps * PK_KB_BYTES if packed else 8 * (kspan + FS_PAD) * 4
    return fn_bytes + _align16(t * (kspan + XS_PAD) * 2) + nstage * 8


def pack_fn(fn):
    """fp32 fn [24, K] -> uint8, stage-major [stage][split*3 + n-tile][nkb(stage) * 1280]:
    per k-block the RNE tf32 bits of the 32 lanes' B fragments (layout in mhc_det.cu)."""
    import torch

    n_out, k = fn.shape
    if n_out != 24:
        raise ValueError(f"fn rows {n_out} != 24")
    kbps = kb_per_split(k)
    dev = fn.device
    bits = fn.detach().contiguous().view(torch.int32)
    s = torch.arange(SPLITS, device=dev).view(SPLITS, 1, 1, 1, 1)
    nt = torch.arange(3, device=dev).view(1, 3, 1, 1, 1)
    kb = torch.arange(kbps, device=dev).view(1, 1, kbps, 1, 1)
    lane = torch.arange(32, device=dev).view(1, 1, 1, 32, 1)
    i = torch.arange(16, device=dev).view(1, 1, 1, 1, 16)
    row = nt * 8 + (lane >> 2)
    col = (s * kbps + kb) * 64 + (i >> 1) * 8 + (lane & 3) + 4 * (i & 1)
    vals = bits[row, col]  # [16, 3, kbps, 32, 16]
    vals = (vals + 0xFFF + ((vals >> 13) & 1)) & ~0x1FFF  # RNE to tf32, as the stock B tensor map
    hi = ((vals >> 16) & 0xFFFF).to(torch.int32)
    hi = hi.view(SPLITS, 3, kbps, 32, 2, 8).permute(0, 1, 2, 4, 3, 5)  # [.., half, lane, 8]
    hi16 = torch.where(hi >= 32768, hi - 65536, hi).to(torch.int16)  # same 16 bits, int16 view
    hi_bytes = hi16.contiguous().view(torch.uint8).view(SPLITS, 3, kbps, 1024)
    nib = ((vals >> 13) & 0x7).to(torch.int32)
    lo = (nib[..., 0::2] | (nib[..., 1::2] << 4)).to(torch.uint8)  # [16, 3, kbps, 32, 8]
    lo_bytes = lo.contiguous().view(SPLITS, 3, kbps, 256)
    blocks = torch.cat([hi_bytes, lo_bytes], dim=3).view(SPLITS * 3, kbps, PK_KB_BYTES)
    stages = [blocks[:, lo_kb : lo_kb + STAGE_KB].reshape(-1) for lo_kb in range(0, kbps, STAGE_KB)]
    return torch.cat(stages).contiguous()


class DetKernels:
    """Compiled mhc_det.cu for this GPU; launch wrappers on torch's current stream."""

    def __init__(self, src_path: str | None = None, opts=()) -> None:
        from mhc_det_rt import Module

        path = src_path or os.path.join(HERE, "mhc_det.cu")
        with open(path) as fh:
            self.src = fh.read()
        self.mod = Module(self.src, "mhc_det.cu", opts=opts)
        self._post = self.mod.function("mhc_det_post")
        self._gemm = {
            (packed, big): self.mod.function(f"mhc_det_gemm_{'pk' if packed else 'f32'}_t{16 if big else 8}")
            for packed in (False, True) for big in (False, True)
        }

    @staticmethod
    def _check(x, mixes, sqr) -> tuple[int, int]:
        t, k = x.shape
        if not 1 <= t <= MAX_T:
            raise ValueError(f"T={t} outside 1..{MAX_T}")
        if tuple(mixes.shape) != (SPLITS, t, 24) or tuple(sqr.shape) != (SPLITS, t):
            raise ValueError(f"bad output shapes {tuple(mixes.shape)} {tuple(sqr.shape)}")
        return t, k

    def gemm(self, x, fn, mixes, sqr, packed: bool, pdl: bool = True) -> None:
        t, k = self._check(x, mixes, sqr)
        self._gemm[(packed, t > 8)].launch(
            (3, SPLITS), (32,), smem_bytes(t, k, packed),
            [(x.data_ptr(), ctypes.c_void_p), (fn.data_ptr(), ctypes.c_void_p),
             (mixes.data_ptr(), ctypes.c_void_p), (sqr.data_ptr(), ctypes.c_void_p),
             (t, ctypes.c_int), (k, ctypes.c_int)],
            pdl=pdl,
        )

    def gemm_f32(self, x, fn, mixes, sqr, pdl: bool = True) -> None:
        self.gemm(x, fn, mixes, sqr, packed=False, pdl=pdl)

    def gemm_pk(self, x, fnp, mixes, sqr, pdl: bool = True) -> None:
        self.gemm(x, fnp, mixes, sqr, packed=True, pdl=pdl)

    def post(self, x, residual, post_mix, comb_mix, out, pdl: bool = True) -> None:
        """Bitwise TileLang mhc_post: x [T,H], residual/out [T,4,H] bf16, post_mix [T,4](,1),
        comb_mix [T,4,4] fp32, all contiguous."""
        t, hc, h = residual.shape
        if hc != 4 or h % 256 or not 1 <= t <= 65535:
            raise ValueError(f"bad residual shape {tuple(residual.shape)}")
        self._post.launch(
            (h // 256, t), (64,), 0,
            [(comb_mix.data_ptr(), ctypes.c_void_p), (residual.data_ptr(), ctypes.c_void_p),
             (post_mix.data_ptr(), ctypes.c_void_p), (x.data_ptr(), ctypes.c_void_p),
             (out.data_ptr(), ctypes.c_void_p), (h, ctypes.c_int)],
            pdl=pdl,
        )


def det_post(dk, x, residual, post_layer_mix, comb_res_mix):
    """Drop-in for vLLM mhc_post_tilelang (bitwise); inputs contiguous, T <= MAX_T."""
    import torch

    out = torch.empty_like(residual)
    dk.post(x, residual, post_layer_mix, comb_res_mix, out)
    return out


def det_pre_delayed(dk, fnp, residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                    hc_post_mult_value, sinkhorn_repeat, pre_mix=None, x=None, norm_weight=None,
                    norm_eps=1e-6):
    """vLLM mhc_pre_delayed_tilelang with the prenorm GEMM swapped for the det kernel.

    Same buffers, same 16-split partials (bitwise), same TileLang fused-norm kernel, so every
    output is bitwise the stock one. Callers check eligibility (T <= MAX_T, norm_weight set,
    stock split count 16)."""
    import torch
    from vllm.model_executor.kernels.mhc.warmup import MHC_PRE_NORM_KERNEL

    num_tokens, hc_mult, hidden_size = residual.shape
    if x is None:
        x = residual.view(num_tokens, hc_mult * hidden_size)
    input_size = x.shape[1]
    mix_size = hc_mult * (hc_mult + 2)
    dev = residual.device
    next_pre_mix = torch.empty(num_tokens, hc_mult, dtype=torch.float32, device=dev)
    post = torch.empty_like(next_pre_mix)
    comb = torch.empty(num_tokens, hc_mult * hc_mult, dtype=torch.float32, device=dev)
    layer_input = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=dev)
    outputs = (post.unsqueeze(-1), comb.view(num_tokens, hc_mult, hc_mult), layer_input, next_pre_mix)
    mixes = torch.empty(SPLITS, num_tokens, mix_size, dtype=torch.float32, device=dev)
    sqrsum = torch.empty(SPLITS, num_tokens, dtype=torch.float32, device=dev)
    dk.gemm_pk(x, fnp, mixes, sqrsum)
    MHC_PRE_NORM_KERNEL(
        mixes, sqrsum, hc_scale, hc_base, residual, post, comb, layer_input, norm_weight,
        pre_mix if pre_mix is not None else post, next_pre_mix,
        hidden_size=hidden_size, rms_eps=rms_eps, hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps, hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat, norm_eps=norm_eps, hc_mult=hc_mult,
        use_pre_mix_in=pre_mix is not None, save_pre_mix=True, rms_numel=input_size,
    )
    return outputs


# ---------------------------------------------------------------------------------------------
# Lever: env, weight packing + self-test after load, dispatch
# ---------------------------------------------------------------------------------------------
ENV = "DSV41_MHC_DET_SPLITS"
# Boot-log markers for tools/engagement_audit.py.
LOG_ENGAGED = "dsv41: mhc det engaged"
LOG_DISARMED = "dsv41: mhc det lever is OFF"
MODEL_MOD = "vllm.models.deepseek_v4_1.nvidia.model"
DSPARK_MOD = "vllm.models.deepseek_v4_1.nvidia.dspark"
STOCK_MOD = "vllm.model_executor.kernels.mhc.tilelang"


def splits_from_env(env) -> int | None:
    """None = off. 16 = on. Anything else raises ValueError (the lever stays off)."""
    raw = (env.get("DSV41_MHC_DET_SPLITS", "0") or "0").strip()
    n = int(raw)
    if n == 0:
        return None
    if n != SPLITS:
        raise ValueError(
            f"{ENV}={raw}: only {SPLITS} (the stock decode split-K, reproduced bitwise) is supported"
        )
    if (env.get("DSV41_MHC_DECODE_SPLITS", "0") or "0").strip() != "0":
        raise ValueError(f"{ENV} cannot combine with DSV41_MHC_DECODE_SPLITS (changes the stock numerics)")
    if (env.get("DSV41_MHC_NO_DEEPGEMM", "0") or "0").strip() == "1":
        raise ValueError(f"{ENV} cannot combine with DSV41_MHC_NO_DEEPGEMM=1 (changes the stock numerics)")
    return n


class _State:
    def __init__(self) -> None:
        self.on = False
        self.failed = False
        self.dk = None
        self.packed: dict[int, tuple] = {}  # fn.data_ptr() -> (shape, version, packed)
        self.stock_post = None
        self.stock_pre = None
        self.misses = 0


_S = _State()


def _disarm(msg: str) -> None:
    _S.on = False
    _S.failed = True
    print(f"dsv41: mhc det lever is OFF: {msg}", flush=True)


def _fn_attrs(module):
    """(owner, attribute) of every mHC projection in a model: DeepseekV4DecoderLayer instances."""
    out = []
    for mod in module.modules():
        if type(mod).__name__ != "DeepseekV4DecoderLayer":
            continue
        for name in ("hc_attn_fn", "hc_ffn_fn", "hc_attn_fn_broadcast"):
            fn = getattr(mod, name, None)
            if fn is not None:
                out.append((mod, name))
    return out


def _pack_one(fn):
    packed = pack_fn(fn)
    _S.packed[fn.data_ptr()] = (tuple(fn.shape), fn._version, packed)
    return packed


def _selftest_gemm(fn, packed) -> str | None:
    import torch
    from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm

    k = fn.shape[1]
    for t in (1, 4, 8, 16):
        g = torch.Generator(device=fn.device).manual_seed(1000 * t + k)
        x = (torch.randn(t, k, device=fn.device, generator=g) * 3).to(torch.bfloat16)
        ref = (torch.empty(SPLITS, t, 24, device=fn.device), torch.empty(SPLITS, t, device=fn.device))
        got = (torch.full_like(ref[0], float("nan")), torch.full_like(ref[1], float("nan")))
        tf32_hc_prenorm_gemm(x, fn, ref[0], ref[1], SPLITS)
        _S.dk.gemm_pk(x, packed, got[0], got[1])
        for a, b in zip(ref, got):
            if not torch.equal(a.view(torch.int32), b.view(torch.int32)):
                return f"GEMM K={k} T={t}: {int((a.view(torch.int32) != b.view(torch.int32)).sum())} elements differ"
    return None


def _selftest_post(device) -> str | None:
    import torch

    for t in (1, 4, 16):
        g = torch.Generator(device=device).manual_seed(77 + t)
        residual = (torch.randn(t, 4, 5120, device=device, generator=g) * 4).to(torch.bfloat16)
        x = (torch.randn(t, 5120, device=device, generator=g) * 2).to(torch.bfloat16)
        post_mix = (2 * torch.sigmoid(torch.randn(t, 4, 1, device=device, generator=g))).contiguous()
        comb = torch.softmax(torch.randn(t, 4, 4, device=device, generator=g) * 3, -1).contiguous()
        ref = _S.stock_post(x, residual, post_mix, comb)
        got = det_post(_S.dk, x, residual, post_mix, comb)
        if not torch.equal(ref.view(torch.int16), got.view(torch.int16)):
            return f"post T={t}: {int((ref.view(torch.int16) != got.view(torch.int16)).sum())} elements differ"
    return None


def prepare(model, label: str) -> None:
    """After weight load: compile the kernels, pack every mHC fn, self-test bitwise, engage."""
    if _S.failed:
        return
    try:
        import torch

        attrs = _fn_attrs(model)
        if not attrs:
            print(f"dsv41: mhc det: {label} has no mHC layers", flush=True)
            return
        if _S.dk is None:
            _S.dk = DetKernels()
        fns = [getattr(m, n) for m, n in attrs]
        fresh = [fn for fn in fns if _S.packed.get(fn.data_ptr(), (None, None))[:2] != (tuple(fn.shape), fn._version)]
        if not fresh:
            return  # load hooks can fire twice (VL wrapper + language model); already done
        fns = fresh
        packed = [_pack_one(fn) for fn in fns]
        for fn, pk in zip(fns, packed):
            err = _selftest_gemm(fn, pk)
            if err:
                _disarm(f"{label}: bitwise self-test failed ({err}); stock kernels stay")
                return
        err = _selftest_post(fns[0].device)
        if err:
            _disarm(f"{label}: bitwise self-test failed ({err}); stock kernels stay")
            return
        torch.cuda.synchronize()
        mb = sum(p.numel() for p in packed) / 2**20
        _S.on = True
        print(
            f"dsv41: mhc det engaged: {label}: {len(fns)} fn packed ({mb:.1f} MiB), det GEMM and post "
            f"bitwise equal to stock (DeepGEMM {SPLITS}-split, TileLang mhc_post); T <= {MAX_T}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 - the lever never breaks the load
        _disarm(f"{label}: prepare failed: {exc!r}; stock kernels stay")


def _lookup(fn):
    ent = _S.packed.get(fn.data_ptr())
    if ent is not None and ent[0] == tuple(fn.shape) and ent[1] == fn._version:
        return ent[2]
    import torch

    if torch.cuda.is_current_stream_capturing():
        _S.misses += 1
        if _S.misses == 1:
            print(f"dsv41: mhc det lever is OFF for an unpacked fn {tuple(fn.shape)} during graph "
                  "capture (stock path used)", flush=True)
        return None
    return _pack_one(fn)  # eager call on a new/changed weight: pack now


def _post(x, residual, post_layer_mix, comb_res_mix):
    import torch

    if (_S.on and residual.dim() == 3 and 1 <= residual.shape[0] <= MAX_T and residual.shape[1] == 4
            and residual.shape[2] % 256 == 0 and residual.dtype == torch.bfloat16
            and x.dtype == torch.bfloat16 and tuple(x.shape) == (residual.shape[0], residual.shape[2])
            and post_layer_mix.dtype == torch.float32 and comb_res_mix.dtype == torch.float32
            and post_layer_mix.numel() == residual.shape[0] * 4
            and comb_res_mix.numel() == residual.shape[0] * 16
            and all(t.is_contiguous() and t.is_cuda for t in (x, residual, post_layer_mix, comb_res_mix))):
        return det_post(_S.dk, x, residual, post_layer_mix, comb_res_mix)
    return _S.stock_post(x, residual, post_layer_mix, comb_res_mix)


def _pre_eligible(residual, fn, pre_mix, x, norm_weight) -> bool:
    import torch
    from vllm.model_executor.kernels.mhc.warmup import compute_mhc_pre_num_splits
    from vllm.utils.deep_gemm import is_deep_gemm_supported

    if not (_S.on and norm_weight is not None and residual.dim() == 3 and residual.dtype == torch.bfloat16
            and residual.is_contiguous() and residual.is_cuda and 1 <= residual.shape[0] <= MAX_T
            and residual.shape[1] == 4):
        return False
    t = residual.shape[0]
    k = x.shape[1] if x is not None else 4 * residual.shape[2]
    if x is not None and not (x.dtype == torch.bfloat16 and x.is_contiguous() and tuple(x.shape) == (t, k)):
        return False
    if pre_mix is not None and not (pre_mix.dtype == torch.float32 and pre_mix.is_contiguous()):
        return False
    if fn.dtype != torch.float32 or tuple(fn.shape) != (24, k) or k % (64 * SPLITS):
        return False
    return is_deep_gemm_supported() and compute_mhc_pre_num_splits(k, t) == SPLITS


def _pre(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
         sinkhorn_repeat, pre_mix=None, x=None, norm_weight=None, norm_eps=1e-6):
    if _pre_eligible(residual, fn, pre_mix, x, norm_weight):
        packed = _lookup(fn)
        if packed is not None:
            return det_pre_delayed(_S.dk, packed, residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                                   hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, pre_mix=pre_mix,
                                   x=x, norm_weight=norm_weight, norm_eps=norm_eps)
    return _S.stock_pre(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                        hc_post_mult_value, sinkhorn_repeat, pre_mix=pre_mix, x=x,
                        norm_weight=norm_weight, norm_eps=norm_eps)


def _after(cls, name: str, label: str, walk) -> None:
    orig = getattr(cls, name)

    def wrapped(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        prepare(walk(self), label)
        return out

    wrapped.__wrapped__ = orig
    setattr(cls, name, wrapped)


def install(env=None) -> None:
    env = os.environ if env is None else env
    try:
        n = splits_from_env(env)
    except ValueError as exc:
        _disarm(str(exc))
        return
    if n is None:
        return
    import importlib

    m = importlib.import_module(MODEL_MOD)
    d = importlib.import_module(DSPARK_MOD)
    stock = importlib.import_module(STOCK_MOD)
    anchors = (
        getattr(m, "mhc_post_tilelang", None) is stock.mhc_post_tilelang,
        getattr(m, "mhc_pre_delayed_tilelang", None) is stock.mhc_pre_delayed_tilelang,
        getattr(d, "mhc_post_tilelang", None) is stock.mhc_post_tilelang,
        hasattr(m.DeepseekV4Model, "finalize_mhc_broadcast_weights"),
        hasattr(d.DSparkDeepseekV4ForCausalLM, "load_weights"),
    )
    if not all(anchors):
        _disarm(f"image drift: mHC call sites or load hooks not as pinned {anchors}")
        return
    _S.stock_post = stock.mhc_post_tilelang
    _S.stock_pre = stock.mhc_pre_delayed_tilelang
    m.mhc_post_tilelang = _post
    m.mhc_pre_delayed_tilelang = _pre
    d.mhc_post_tilelang = _post
    _after(m.DeepseekV4Model, "finalize_mhc_broadcast_weights", "target", lambda self: self)
    _after(d.DSparkDeepseekV4ForCausalLM, "load_weights", "draft", lambda self: self)
    print(f"dsv41: mhc det armed ({ENV}={n}): packs and self-tests mHC weights after load", flush=True)
