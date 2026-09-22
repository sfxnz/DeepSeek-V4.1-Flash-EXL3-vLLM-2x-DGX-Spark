#!/usr/bin/env python3
"""lm_head MXFP8: route the vocab head through the in-image b12x dense path.

The bf16 lm_head GEMM pair (DSpark draft propose m=1 + target verify m=4-6,
`cutlass_80_wmma` bf16, 2692us each, 5.35 ms/step, 100% in decode graphs) is
weight-streaming bound: ~735 MB bf16 streamed per call at TP2. Storing the head
as MXFP8 (e4m3 values + per-32 e8m0 scales, the same recipe as the pack's dense
attn tensors) halves the stream and rides the b12x `mm_mxfp8` kernel the dense
Q/O projections already use inside the decode graphs.

Dispatch map (why this is a runtime swap and not a quant-config entry):
- lm_head is a ParallelLMHead created WITHOUT quant_config (model.py:1028-1033,
  dspark.py:323-328), so VocabParallelEmbedding.__init__ hands it
  UnquantizedEmbeddingMethod (vocab_parallel_embedding.py:291-294). The dense
  b12x path is only reached for LinearBase layers via
  DeepseekV4FP8Config.get_quant_method (quant_config.py:186-208) ->
  ModelOptLinearMethod(QuantSpec(kMxfp8Static, kMxfp8Dynamic)).
- LogitsProcessor._apply_head (logits_processor.py:136-143) calls
  lm_head.quant_method.apply(lm_head, hidden_states, bias) whenever head_dtype
  is None (it is, on this serve), so replacing quant_method + params on the
  lm_head object re-routes both the target verify call (model.py:1068) and
  the DSpark draft call (dspark.py:364; the speculator aliases the target
  lm_head onto the draft model, spec_decode/dspark/utils.py:104-117).
- Checkpoint-key routing (Round-30 root cause): the serving model root is the
  VL wrapper DeepseekV41ForCausalLM (vl_model.py). Its WeightsMapper maps HF
  names fully into the wrapper namespace before AutoWeightsLoader strips the
  "language_model." prefix and delegates to the child
  (DeepseekV41LLMForCausalLM), whose own mapper is a NO-OP (vl_model.py:177
  — the suffix rules are not idempotent). The wrapper's suffix rule
  "head.weight" -> "language_model.lm_head.weight" (key.rsplit — prefix
  preserved) corrupted our staged "lm_head.weight" key to
  "lm_language_model.lm_head.weight" (garbage), while
  "lm_head.weight_scale" matched nothing at all (the "\.scale$" regex wants
  a literal dot; no suffix rule matches "_scale") and arrived at the wrapper
  root as a bare name -> ValueError "There is no module or parameter named
  'lm_head' in DeepseekV41ForCausalLM" (the crash text reported the scale
  key's arrival point, which is why the weight key's corruption went
  unnoticed until the Round-30 integration test).
  Fix (this module, _install_scale_suffix_rule): prepend REGEX rules to
  both mapper makers — "^lm_head\.weight$" -> "head.weight" (the stock
  suffix rule then performs the canonical rename) and
  "^lm_head\.weight_scale$" -> the fully-qualified param name. Regexes are
  required, not suffix rules: regexes apply BEFORE the suffix pass, and any
  suffix-rule result ending in "head.weight" would itself be re-fired by
  the stock suffix rule (the same corruption). The rules are inert for
  stock packs: a bf16-head index has no "lm_head.*" keys at all, and no
  other checkpoint key starts with "lm_head.".

Gating: DSV41_LMHEAD_MXFP8=1 (sitecustomize) AND the snapshot index contains
`lm_head.weight_scale`. Absent tensor -> stock bf16 path, one log line, never
a crash. Checkpoint keys are `lm_head.weight`/`lm_head.weight_scale`,
rewritten onto the model params by the mapper regex rules installed above
(see the checkpoint-key routing note).

Graph safety: apply() calls only mxfp8_e4m3_quantize (the same
torch.ops.vllm.mxfp8_quantize the dense b12x path uses in-graph, DRAFT-AUX
"quant 1.33 ms/step, 99% graph") and mm_mxfp8(backend="auto") (b12x, the
17.58 ms/step dense family, 100% graph). Branching is static: bias is always
None (ParallelLMHead bias=False), shapes are capture-time constants.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torch.nn.parameter import Parameter

MXFP8_BLOCK_SIZE = 32
ENV_FLAG = "DSV41_LMHEAD_MXFP8"
CKPT_SCALE_KEY = "lm_head.weight_scale"
CKPT_WEIGHT_KEY = "lm_head.weight"


# ---------------------------------------------------------------------------
# Pure quantization helpers (no vllm import; mirrors
# vllm/model_executor/layers/quantization/utils/mxfp8_utils.py
# _mxfp8_e4m3_quantize_torch, weight side, row-major scales).
# ---------------------------------------------------------------------------

def quantize_weight_mxfp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [N, K] -> (e4m3 [N, K], e8m0-as-uint8 [N, K//32]) row-major.

    Same math as the pack's dense tensors: per-32-element blocks along K,
    scale = ceil(log2(amax/448)) + 127 clamped to [0, 254].
    """
    N, K = w.shape
    assert K % MXFP8_BLOCK_SIZE == 0, f"K={K} not divisible by {MXFP8_BLOCK_SIZE}"
    x = w.to(torch.float32).view(N, K // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE)
    amax = x.abs().amax(dim=-1).clamp(min=torch.finfo(torch.float32).tiny)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    biased = torch.ceil(torch.log2(amax / fp8_max)) + 127.0
    biased = biased.clamp(0, 254)
    scales = biased.to(torch.uint8)
    descale = torch.exp2(biased - 127.0)
    values = (x / descale.unsqueeze(-1)).view(N, K).to(torch.float8_e4m3fn)
    return values, scales


def enabled_from_env(env=None) -> bool:
    env = dict(os.environ if env is None else env)
    return env.get(ENV_FLAG, "0") == "1"


def snapshot_has_lmhead_mxfp8(snapshot_dir: str | Path) -> bool:
    """True iff the pack index declares lm_head.weight_scale."""
    try:
        idx = Path(snapshot_dir) / "model.safetensors.index.json"
        if not idx.is_file():
            return False
        weight_map = json.loads(idx.read_text()).get("weight_map", {})
        return CKPT_SCALE_KEY in weight_map and CKPT_WEIGHT_KEY in weight_map
    except Exception:
        return False


def _model_snapshot_dir(model_config) -> str | None:
    """Best-effort local snapshot dir from a vLLM ModelConfig."""
    try:
        model = getattr(model_config, "model", None)
        path = getattr(model, "path", None) or model
        if isinstance(path, str) and Path(path).is_dir():
            return path
    except Exception:
        pass
    return None


# Checkpoint-key routing (see module docstring). Regex rules are REQUIRED,
# not suffix rules: regexes apply BEFORE the suffix pass, and any mapped
# name ending in "head.weight" is re-fired by the stock suffix rule
# ("head.weight" -> "language_model.lm_head.weight", rsplit keeps the
# prefix: "lm_head.weight" -> "lm_language_model.lm_head.weight", the
# corruption the Round-30 integration test caught). Routing
# "^lm_head\.weight$" back to "head.weight" lets the STOCK suffix rule do
# the final rename; the scale key maps directly to the fully-qualified
# param name (no stock rule matches "_scale", so nothing re-fires).
LMHEAD_REGEX_RULES_VL = {
    r"^lm_head\.weight$": "head.weight",
    r"^lm_head\.weight_scale$": "language_model.lm_head.weight_scale",
}
LMHEAD_REGEX_RULES_TEXT = {
    r"^lm_head\.weight$": "head.weight",
    r"^lm_head\.weight_scale$": "lm_head.weight_scale",
}


def _install_scale_suffix_rule() -> bool:
    """Teach the DeepSeek-V4.1 WeightsMappers the lm_head.* quantized keys.

    The re-encoded pack stores the head as ``lm_head.weight`` (e4m3) +
    ``lm_head.weight_scale`` (e8m0); stock mappers route the first key to
    garbage (see module docstring) and the second nowhere, so the load
    aborts with ValueError at the VL wrapper root. Appending the regex
    rules above to BOTH mapper makers fixes routing and is inert for stock
    packs (neither key exists there; no other key starts with "lm_head.").

    Returns True if at least one maker gained the rules.
    """
    import re as _re

    from vllm.models.deepseek_v4_1.nvidia import model as _model_mod
    from vllm.models.deepseek_v4_1.nvidia import vl_model as _vl_mod

    targets = (
        (_model_mod, "_make_deepseek_v4_weights_mapper", LMHEAD_REGEX_RULES_TEXT),
        (_vl_mod, "_make_deepseek_v4_vl_weights_mapper", LMHEAD_REGEX_RULES_VL),
    )
    ok = False
    seen: dict[int, bool] = {}
    for mod, name, rules in targets:
        fn = getattr(mod, name, None)
        if fn is None:
            continue
        if id(fn) in seen:
            continue  # module aliasing (vl re-imports the text maker)
        seen[id(fn)] = True
        if getattr(fn, "_lmhead_regex_rules", False):
            ok = True  # already installed (double install() call)
            continue
        orig = fn

        def _patched(*args, _orig=orig, _rules=rules, **kwargs):
            mapper = _orig(*args, **kwargs)
            regex = dict(getattr(mapper, "orig_to_new_regex", None) or {})
            for pat, repl in _rules.items():
                regex[_re.compile(pat)] = repl
            mapper.orig_to_new_regex = regex
            return mapper

        _patched._lmhead_regex_rules = True
        setattr(mod, name, _patched)
        ok = True
    # Rebuild the class-attr default (built at import time from the stock
    # maker) so even a consumer reading it before __init__ sees the rules.
    cls = getattr(_model_mod, "DeepseekV41LLMForCausalLM", None)
    if cls is not None:
        try:
            cls.hf_to_vllm_mapper = _model_mod._make_deepseek_v4_weights_mapper(
                "fp4"
            )
        except Exception:
            pass
    return ok


# ---------------------------------------------------------------------------
# install(): wrap DeepseekV41LLMForCausalLM.__init__. Presence of the
# quantized tensor is checked lazily at model-init time (sitecustomize runs
# before the vllm config exists). Self-disarms to stock on any miss.
# ---------------------------------------------------------------------------

def install() -> bool:
    # Round-30 fix: checkpoint-key routing FIRST (see module docstring) —
    # without the suffix rule the load dies at the first
    # ``lm_head.weight_scale`` key regardless of the swap below.
    _install_scale_suffix_rule()
    # Round-29 fix: this vllm build keeps the model classes under
    # vllm.models.deepseek_v4_1 (NOT vllm.model_executor.models.deepseek_v4_1
    # as the c32be56 draft assumed) — the old import raised ModuleNotFoundError
    # in sitecustomize and the hook silently never installed.
    from vllm.models.deepseek_v4_1.nvidia.model import (
        DeepseekV41LLMForCausalLM,
    )
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizeMethodBase,
    )
    from vllm.model_executor.utils import set_weight_attrs
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        swizzle_mxfp8_scale,
    )
    # Round-29 fix: this build has no vllm.utils.flashinfer.vllm_flashinfer
    # wrapper — flashinfer.mm_mxfp8 is imported directly (present, verified).
    import flashinfer as _flashinfer
    vllm_flashinfer = _flashinfer

    class Mxfp8LMHeadMethod(QuantizeMethodBase):
        """ParallelLMHead quant method riding the b12x mm_mxfp8 kernel.

        apply() mirrors FlashInferCutlassMxfp8LinearKernel.apply_weights
        (kernels/linear/mxfp8/flashinfer.py:81-113) with backend="auto" (the
        image's prefer_b12x_mxfp8 patch puts b12x first on SM120/121).
        process_weights_after_loading mirrors flashinfer.py:50-57.
        """

        def create_weights(self, *a, **k) -> None:  # swapped in post-init
            raise RuntimeError("mxfp8 lm_head weights are created by the swap")

        def process_weights_after_loading(self, layer) -> None:
            weight = layer.weight.data
            N, K = weight.shape
            scale_k = K // MXFP8_BLOCK_SIZE
            scale_2d = layer.weight_scale.data[:N, :scale_k].contiguous()
            layer.weight = Parameter(weight.contiguous(), requires_grad=False)
            layer.weight_scale = Parameter(
                swizzle_mxfp8_scale(scale_2d, M=N, K=K).contiguous(),
                requires_grad=False,
            )

        def apply(self, layer, x, bias=None):
            weight = layer.weight
            N, K = weight.shape
            assert K >= 128 and K % MXFP8_BLOCK_SIZE == 0 and N >= 128
            from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
                mxfp8_e4m3_quantize,
            )

            flat = x.reshape(-1, K)
            input_mxfp8, input_scale = mxfp8_e4m3_quantize(
                flat, is_sf_swizzled_layout=True
            )
            output = vllm_flashinfer.mm_mxfp8(
                input_mxfp8,
                weight.t(),
                input_scale,
                layer.weight_scale,
                out_dtype=x.dtype,
                backend="auto",
            )
            output = output.view(*x.shape[:-1], N)
            if bias is not None:
                output = output + bias
            return output

    def _swap_lm_head(lm_head) -> None:
        N, K = lm_head.weight.shape
        assert K % MXFP8_BLOCK_SIZE == 0
        device = lm_head.weight.device
        weight = Parameter(
            torch.empty(N, K, dtype=torch.float8_e4m3fn, device=device),
            requires_grad=False,
        )
        scale = Parameter(
            torch.empty(N, K // MXFP8_BLOCK_SIZE, dtype=torch.uint8, device=device),
            requires_grad=False,
        )
        attrs = {"input_dim": 1, "output_dim": 0}
        set_weight_attrs(weight, dict(attrs, weight_loader=lm_head.weight_loader))
        set_weight_attrs(scale, dict(attrs, weight_loader=lm_head.weight_loader))
        lm_head.weight = weight
        lm_head.weight_scale = scale
        lm_head.quant_method = Mxfp8LMHeadMethod()

    orig_init = DeepseekV41LLMForCausalLM.__init__

    def _init_lmhead_mxfp8(self, *, vllm_config, prefix: str = "") -> None:
        orig_init(self, vllm_config=vllm_config, prefix=prefix)
        try:
            lm_head = getattr(self, "lm_head", None)
            if lm_head is None or not hasattr(lm_head, "weight_loader"):
                return  # PP non-last rank: PPMissingLayer, stock path
            model_config = vllm_config.model_config
            # Round-30 fix: head_dtype is a PROPERTY in this build
            # (config/model.py:1970) that always returns a dtype — the model
            # dtype for generation models (bfloat16 here), never None. The
            # real branch is in LogitsProcessor._apply_head
            # (logits_processor.py:143): quant_method.apply() runs whenever
            # head_dtype == hidden_states.dtype. Only a genuine OVERRIDE
            # (head_dtype != model dtype, e.g. --hf-overrides float32 for
            # RL parity, which would .to(float32) a plain lm_head.weight)
            # must disarm.
            head_dtype = getattr(model_config, "head_dtype", None)
            model_dtype = getattr(model_config, "dtype", None)
            if head_dtype is not None and head_dtype != model_dtype:
                print(
                    f"dsv41: lm_head mxfp8 self-disarmed "
                    f"(head_dtype override {head_dtype} != {model_dtype})",
                    flush=True,
                )
                return
            snap = _model_snapshot_dir(model_config)
            if snap is None or not snapshot_has_lmhead_mxfp8(snap):
                print(
                    "dsv41: lm_head mxfp8 self-disarmed "
                    f"(no {CKPT_SCALE_KEY} under {snap})",
                    flush=True,
                )
                return
            _swap_lm_head(lm_head)
            print(
                f"dsv41: lm_head mxfp8 enabled (b12x, {tuple(lm_head.weight.shape)})",
                flush=True,
            )
        except Exception as exc:  # never crash the boot
            print(f"dsv41: lm_head mxfp8 self-disarmed ({exc!r})", flush=True)

    DeepseekV41LLMForCausalLM.__init__ = _init_lmhead_mxfp8
    return True
