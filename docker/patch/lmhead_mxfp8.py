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
  lm_head object re-routes both the target verify call (model.py:1068) and the
  DSpark draft call (dspark.py:364; the speculator aliases the target lm_head
  onto the draft model, spec_decode/dspark/utils.py:104-117).

Gating: DSV41_LMHEAD_MXFP8=1 (sitecustomize) AND the snapshot index contains
`lm_head.weight_scale`. Absent tensor -> stock bf16 path, one log line, never
a crash. The checkpoint key is `lm_head.weight`/`lm_head.weight_scale` (NOT
`head.weight_scale`: the DSV4.1 WeightsMapper regex `\.scale$` matches "." not
"_", so a `head.weight_scale` key would never be renamed onto the param and
would be dropped as an unexpected weight; `lm_head.*` passes through mapping
untouched and lands on the module directly).

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


# ---------------------------------------------------------------------------
# install(): wrap DeepseekV41LLMForCausalLM.__init__. Presence of the
# quantized tensor is checked lazily at model-init time (sitecustomize runs
# before the vllm config exists). Self-disarms to stock on any miss.
# ---------------------------------------------------------------------------

def install() -> bool:
    from vllm.model_executor.models.deepseek_v4_1.nvidia.model import (
        DeepseekV41LLMForCausalLM,
    )
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizeMethodBase,
    )
    from vllm.model_executor.utils import set_weight_attrs
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        swizzle_mxfp8_scale,
    )
    from vllm.utils.flashinfer import vllm_flashinfer

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
            if getattr(model_config, "head_dtype", None) is not None:
                print(
                    "dsv41: lm_head mxfp8 self-disarmed (head_dtype override)",
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
