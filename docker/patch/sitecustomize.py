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

