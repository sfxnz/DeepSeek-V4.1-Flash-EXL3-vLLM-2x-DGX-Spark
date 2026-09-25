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
- the TileLang fused norm -> det norm: the same operations in the same order (split sums,
  sigmoids, 20-step sinkhorn with the TileLang butterflies, RMSNorm with the TileLang sumsq
  order); its layer_input half runs during the GEMM, its coefficient warp after it.
Every output is bitwise the stock one (kernel_study/mhc_det, results/2026-09-25-kernels/mhc-det).
16 is the only accepted value: it names the stock decode split-K that the kernels reproduce.
After weight load every packed fn (GEMM), every sublayer's whole pre (GEMM + norm, the layer's
real parameters) and the post are checked bitwise against the stock kernels on the GPU; any
mismatch leaves the stock path in place for the process (LOG_DISARMED).

Imported lazily; top level is stdlib only so importing this module cannot fail.
"""

from __future__ import annotations

import ctypes
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = 16  # the stock decode split-K the det kernels reproduce bitwise
STAGE_KB = 2
XS_PAD = 8
PK_KB_BYTES = 1280
MAX_T = 16
NORM_H = 5120  # hidden size compiled into mhc_det_norm


def _align16(n: int) -> int:
    return (n + 15) & ~15


def kb_per_split(k: int) -> int:
    if k % (64 * SPLITS):
        raise ValueError(f"K={k} is not a multiple of {64 * SPLITS}")
    return k // (64 * SPLITS)


def smem_bytes(t: int, k: int) -> int:
    kbps = kb_per_split(k)
    nstage = -(-kbps // STAGE_KB)
    return kbps * PK_KB_BYTES + _align16(t * (kbps * 64 + XS_PAD) * 2) + nstage * 8


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
        self._norm = self.mod.function("mhc_det_norm")
        self._gemm = {big: self.mod.function(f"mhc_det_gemm_t{16 if big else 8}") for big in (False, True)}

    @staticmethod
    def _check(x, mixes, sqr) -> tuple[int, int]:
        t, k = x.shape
        if not 1 <= t <= MAX_T:
            raise ValueError(f"T={t} outside 1..{MAX_T}")
        if tuple(mixes.shape) != (SPLITS, t, 24) or tuple(sqr.shape) != (SPLITS, t):
            raise ValueError(f"bad output shapes {tuple(mixes.shape)} {tuple(sqr.shape)}")
        return t, k

    def gemm_pk(self, x, fnp, mixes, sqr, pdl: bool = True) -> None:
        """Det prenorm GEMM: x [T, K] bf16, fnp = pack_fn(fn) -> mixes [16, T, 24], sqr [16, T]."""
        t, k = self._check(x, mixes, sqr)
        self._gemm[t > 8].launch(
            (3, SPLITS), (32,), smem_bytes(t, k),
            [(x.data_ptr(), ctypes.c_void_p), (fnp.data_ptr(), ctypes.c_void_p),
             (mixes.data_ptr(), ctypes.c_void_p), (sqr.data_ptr(), ctypes.c_void_p),
             (t, ctypes.c_int), (k, ctypes.c_int)],
            pdl=pdl,
        )

    def norm(self, mixes, sqrsum, hc_scale, hc_base, residual, pre_mix, norm_weight, post, comb,
             layer_input, pre_mix_out, rms_numel, rms_eps, hc_pre_eps, sinkhorn_eps, post_mult,
             sinkhorn_repeat, norm_eps, pdl: bool = True) -> None:
        """Bitwise TileLang mhc_pre_big_fuse_with_norm (hidden 5120, 16 splits, save_pre_mix)."""
        t = residual.shape[0]
        self._norm.launch(
            (t,), (288,), 0,
            [(mixes.data_ptr(), ctypes.c_void_p), (sqrsum.data_ptr(), ctypes.c_void_p),
             (hc_scale.data_ptr(), ctypes.c_void_p), (hc_base.data_ptr(), ctypes.c_void_p),
             (residual.data_ptr(), ctypes.c_void_p),
             (pre_mix.data_ptr() if pre_mix is not None else 0, ctypes.c_void_p),
             (norm_weight.data_ptr(), ctypes.c_void_p), (post.data_ptr(), ctypes.c_void_p),
             (comb.data_ptr(), ctypes.c_void_p), (layer_input.data_ptr(), ctypes.c_void_p),
             (pre_mix_out.data_ptr(), ctypes.c_void_p), (t, ctypes.c_int),
             (float(rms_numel), ctypes.c_float), (float(rms_eps), ctypes.c_float),
             (float(hc_pre_eps), ctypes.c_float), (float(sinkhorn_eps), ctypes.c_float),
             (float(post_mult), ctypes.c_float), (int(sinkhorn_repeat), ctypes.c_int),
             (float(norm_eps), ctypes.c_float)],
            pdl=pdl,
        )

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
                    norm_eps=1e-6, det_norm=True):
    """vLLM mhc_pre_delayed_tilelang on the det kernels: same buffers, the same 16-split
    partials and the same fused-norm arithmetic, so every output is bitwise the stock one.
    det_norm=False keeps the TileLang fused norm (kernel_study A/B only). Callers check
    eligibility (_pre_eligible: T <= MAX_T, hidden 5120, norm_weight set, stock split count 16)."""
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
    if det_norm:
        dk.norm(mixes, sqrsum, hc_scale, hc_base, residual, pre_mix, norm_weight, post, comb, layer_input,
                next_pre_mix, input_size, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
                sinkhorn_repeat, norm_eps)
        return outputs
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


def _layers(module):
    return [m for m in module.modules() if type(m).__name__ == "DeepseekV4DecoderLayer"]


def _pack_one(fn):
    packed = pack_fn(fn)
    _S.packed[fn.data_ptr()] = (tuple(fn.shape), fn._version, packed)
    return packed


def _is_packed(fn) -> bool:
    return _S.packed.get(fn.data_ptr(), (None, None))[:2] == (tuple(fn.shape), fn._version)


def _bits(t):
    import torch

    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


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
            if not torch.equal(_bits(a), _bits(b)):
                return f"GEMM K={k} T={t}: {int((_bits(a) != _bits(b)).sum())} elements differ"
    return None


def _selftest_pre(layer, sub, fn, packed, broadcast: bool, carried: bool, tokens) -> str | None:
    """Whole det pre (GEMM + fused norm) vs the stock one on this layer's real parameters."""
    import torch

    norm = getattr(layer, f"{sub}_norm")
    args = (getattr(layer, f"hc_{sub}_scale"), getattr(layer, f"hc_{sub}_base"), layer.rms_norm_eps,
            layer.hc_eps, layer.hc_eps, layer.hc_post_alpha, layer.hc_sinkhorn_iters)
    dev = fn.device
    for t in tokens:
        g = torch.Generator(device=dev).manual_seed(31 * t + fn.shape[1])
        x = None
        if broadcast:
            x = (torch.randn(t, 5120, device=dev, generator=g)).to(torch.bfloat16)
            residual = x.unsqueeze(1).expand(-1, 4, -1).contiguous()
        else:
            residual = (torch.randn(t, 4, 5120, device=dev, generator=g) * 4).to(torch.bfloat16)
        pre_mix = torch.softmax(torch.randn(t, 4, device=dev, generator=g), -1).contiguous() if carried else None
        kw = dict(pre_mix=pre_mix, x=x, norm_weight=norm.weight, norm_eps=norm.variance_epsilon)
        ref = _S.stock_pre(residual, fn, *args, **kw)
        got = det_pre_delayed(_S.dk, packed, residual, fn, *args, **kw)
        for name, a, b in zip(("post_mix", "comb_mix", "layer_input", "pre_mix"), ref, got):
            if not torch.equal(_bits(a), _bits(b)):
                return f"pre {sub} T={t} {name}: {int((_bits(a) != _bits(b)).sum())} elements differ"
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
        if not torch.equal(_bits(ref), _bits(got)):
            return f"post T={t}: {int((_bits(ref) != _bits(got)).sum())} elements differ"
    return None


def prepare(model, label: str) -> None:
    """After weight load: compile the kernels, pack every mHC fn, self-test bitwise, engage.

    Self-tests (all bitwise, on this GPU, with the layers' real weights): the det GEMM vs
    DeepGEMM at T 1/4/8/16 for every fn; the whole det pre vs the stock pre (T=4 every
    sublayer, T 1/16 on the first; the layer-0 broadcast input; no carried pre-mix); the det
    post vs TileLang mhc_post. Any mismatch leaves the stock path for the whole process."""
    if _S.failed:
        return
    try:
        import torch

        layers = _layers(model)
        if not layers:
            print(f"dsv41: mhc det: {label} has no mHC layers", flush=True)
            return
        if _S.dk is None:
            _S.dk = DetKernels()
        n_fn = n_pre = 0
        mib = 0.0
        for li, layer in enumerate(layers):
            cases = [("attn", layer.hc_attn_fn, False), ("ffn", layer.hc_ffn_fn, False)]
            if getattr(layer, "hc_attn_fn_broadcast", None) is not None:
                cases.append(("attn", layer.hc_attn_fn_broadcast, True))
            for sub, fn, broadcast in cases:
                if _is_packed(fn):
                    continue  # load hooks can fire twice (VL wrapper + language model)
                packed = _pack_one(fn)
                mib += packed.numel() / 2**20
                n_fn += 1
                err = _selftest_gemm(fn, packed)
                first = n_pre == 0
                for carried in ((False, True) if first and not broadcast else (not broadcast,)):
                    err = err or _selftest_pre(layer, sub, fn, packed, broadcast, carried,
                                               (1, 4, 16) if first else (4,))
                    n_pre += 1
                if err:
                    _disarm(f"{label}: bitwise self-test failed (layer {li}: {err}); stock kernels stay")
                    return
        if n_fn == 0:
            return
        err = _selftest_post(layers[0].hc_attn_fn.device)
        if err:
            _disarm(f"{label}: bitwise self-test failed ({err}); stock kernels stay")
            return
        torch.cuda.synchronize()
        _S.on = True
        print(
            f"dsv41: mhc det engaged: {label}: {n_fn} fn packed ({mib:.1f} MiB); det GEMM, fused norm and "
            f"post bitwise equal to stock on {n_pre} pre self-tests (DeepGEMM {SPLITS}-split, TileLang); "
            f"T <= {MAX_T}",
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
            and tuple(residual.shape[1:]) == (4, NORM_H) and norm_weight.dtype == torch.bfloat16
            and norm_weight.is_contiguous() and norm_weight.numel() == NORM_H):
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
