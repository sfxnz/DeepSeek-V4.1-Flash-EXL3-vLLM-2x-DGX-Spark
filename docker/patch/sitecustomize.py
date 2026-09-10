# install the apport exception handler if available
try:
    import apport_python_hook
except ImportError:
    pass
else:
    apport_python_hook.install()

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
except Exception:
    pass
