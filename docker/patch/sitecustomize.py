# install the apport exception handler if available
try:
    import apport_python_hook
except ImportError:
    pass
else:
    apport_python_hook.install()

import sys

if "/opt/dsv41-patch" not in sys.path:
    sys.path.insert(0, "/opt/dsv41-patch")

# Load vLLM general plugins in every process (API, EngineCore, workers).
# VLLM_PLUGINS=vllm_exl3 is not enough on this image: EngineCore can resolve
# --quantization exl3 before load_general_plugins() runs.
try:
    from vllm.plugins import load_general_plugins

    load_general_plugins()
except Exception:
    pass

# DSV4.1 mapper picks weight_scale vs weight_scale_inv from
# quant_config.weight_block_size == [32, 32]. Exl3Config keeps that field
# inside non_routed_quantization, so copy it onto the config object.
try:
    from vllm_exl3.exl3 import Exl3Config

    _exl3_from_config = Exl3Config.from_config.__func__

    @classmethod
    def _exl3_from_config_with_block_size(cls, config):
        inst = _exl3_from_config(cls, config)
        nr = getattr(inst, "non_routed_quantization", None) or {}
        wbs = nr.get("weight_block_size") or config.get("weight_block_size")
        if wbs is not None:
            inst.weight_block_size = list(wbs)
        return inst

    Exl3Config.from_config = _exl3_from_config_with_block_size
except Exception:
    pass

# DSv4 sparse-MLA mixed warmup still dummy-forwards through DeepGEMM paged-MQA
# (block_kv must be 32 or 64) after autotune is disabled. Skip that warmup.
try:
    import vllm.model_executor.warmup.kernel_warmup as _kw

    _kw.deepseek_v4_sparse_mla_attention_warmup = lambda worker: None
    _kw.kernel_warmup = lambda worker: None
except Exception:
    pass

try:
    from vllm.v1.worker.gpu_worker import Worker
    from vllm.v1.worker.worker_base import CompilationTimes

    Worker.compile_or_warm_up_model = lambda self: CompilationTimes(0.0, 0.0)
except Exception:
    pass

# FlashInfer SM120 DSV4 decode is compiled only for page_block_size=64.
# Upstream V4.1 hardcodes SWA pages to 32 (DeepGEMM paged-MQA).
try:
    from sm120_page import coerce_swa_block_size
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache

    _swa_init = DeepseekV4SWACache.__init__

    def _swa_init_sm120_page(self, *args, **kwargs):
        if "block_size" in kwargs:
            kwargs["block_size"] = coerce_swa_block_size(kwargs["block_size"])
        elif len(args) >= 7:
            args = list(args)
            args[6] = coerce_swa_block_size(args[6])
            args = tuple(args)
        return _swa_init(self, *args, **kwargs)

    DeepseekV4SWACache.__init__ = _swa_init_sm120_page
except Exception:
    pass

# Indexer decode metadata feeds DeepGEMM paged-MQA, which asserts
# block_kv in {32, 64}. Upstream V4 indexer reports 128 on SM12.
try:
    from sm120_page import indexer_kernel_block_sizes
    from vllm.v1.attention.backends.mla.indexer import DeepseekV4IndexerBackend

    DeepseekV4IndexerBackend.get_supported_kernel_block_sizes = staticmethod(
        lambda: list(indexer_kernel_block_sizes())
    )
except Exception:
    pass

# --language-model-only still flattens vision_max_n_token onto hf_config, so
# SWA prefill index rows widen to window+1024=1152. SM120 DSV4 decode topk is
# {128,192,256,512,1024}. Do not zero vision_n_layers: VL checkpoints ship
# gate.bias_vl and load_weights KeyErrors without that param.
try:
    from sm120_page import text_only_max_image_tokens
    from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

    _v41_cfg_init = DeepseekV41Config.__init__

    def _v41_cfg_init_text_only(self, *args, **kwargs):
        _v41_cfg_init(self, *args, **kwargs)
        self.vision_max_n_token = text_only_max_image_tokens(
            getattr(self, "vision_max_n_token", 0), True
        )

    DeepseekV41Config.__init__ = _v41_cfg_init_text_only
except Exception:
    pass
