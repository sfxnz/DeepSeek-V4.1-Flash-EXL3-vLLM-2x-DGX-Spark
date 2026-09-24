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

# --- pfg8 load tracer (Round 31): when DSV41_LOAD_PF_G8=1, sample the worker
# process every 10s during weight load: RSS, gc type census (top objects by
# retained count), torch tensor count, and main-thread stack. Writes
# /tmp/g8trace.log inside the container. Zero-risk: pure reads, env-gated.
try:
    import os as _os_tr

    if _os_tr.environ.get("DSV41_LOAD_PF_G8", "0") == "1":
        import threading as _th_tr
        import time as _t_tr

        def _g8_tracer_loop():
            path = "/tmp/g8trace.log"
            with open(path, "a") as _fh:
                _fh.write(f"tracer start {_t_tr.time()}\n")
            while True:
                try:
                    _t_tr.sleep(10)
                    lines = []
                    with open("/proc/self/status") as _st:
                        for _ln in _st:
                            if _ln.startswith(("VmRSS:", "VmSwap:")):
                                lines.append(_ln.strip())
                    import gc

                    _stats = {}
                    for _o in gc.get_objects():
                        _t = type(_o).__name__
                        if _t in ("Tensor", "Parameter", "Storage", "dict", "list"):
                            _stats[_t] = _stats.get(_t, 0) + 1
                    lines.append("gc: " + repr(_stats))
                    try:
                        import torch as _torch_tr

                        lines.append(
                            f"torch cuda alloc={_torch_tr.cuda.memory_allocated()//2**20}MiB"
                        )
                    except Exception:
                        pass
                    # main thread stack
                    try:
                        for _th in _th_tr.enumerate():
                            if _th is not _th_tr.current_thread() and getattr(_th, "__dict__", None):
                                _f = sys._current_frames().get(_th.ident)
                                if _f is not None and _th.name == "MainThread":
                                    lines.append(
                                        "stack: "
                                        + "".join(
                                            f"{_fr.f_code.co_filename}:{_fr.f_lineno}:{_fr.f_code.co_name} "
                                            for _fr in [
                                                _f,
                                                *(_f.f_back for _ in range(0))
                                            ][:1]
                                        )
                                    )
                    except Exception:
                        pass
                    with open(path, "a") as _fh:
                        _fh.write(f"--- {_t_tr.time()}\n" + "\n".join(lines) + "\n")
                except Exception as _e:
                    try:
                        with open(path, "a") as _fh:
                            _fh.write(f"tracer err {_e!r}\n")
                    except Exception:
                        pass

        _th_tr.Thread(target=_g8_tracer_loop, daemon=True).start()
except Exception:
    pass


# c1_graph_safe_adaptive is unwired. Extra-graphs and pin-budget=2 both
# failed L.A.I.L (22.5 and 21.1). 6-token-verify family is closed.

# SM120 MXFP8 Q/O: vLLM hardcodes mm_mxfp8 backend=cutlass. auto prefers
# b12x small-M tiles on SM120/SM121 and falls back to cutlass if needed.
try:
    from pathlib import Path as _P2

    from prefer_b12x_mxfp8 import apply as _apply_b12x_mxfp8

    _apply_b12x_mxfp8(
        _P2("/usr/local/lib/python3.12/dist-packages/vllm")
    )
except Exception:
    pass

# Indexer prefill gather workspace: stock max_model_len*40 entries (~5.3 GiB
# per rank at 1M ctx) locked for process life. DSV41_INDEXER_PREFILL_FACTOR=1
# right-sizes it to max_model_len*1 (~130 MB). Unset = stock no-op.
try:
    from pathlib import Path as _Pw

    from indexer_workspace import apply as _apply_idx_ws

    _apply_idx_ws(_Pw("/usr/local/lib/python3.12/dist-packages/vllm"))
except Exception as _idx_ws_err:
    print(f"dsv41: indexer workspace patch skipped: {_idx_ws_err!r}", flush=True)

# sm120_wo_a unwired. b12x wo_a_dense_gemm_mxfp8 waves 21.23/23.00; not
# faster than Emulation torch.bmm. wo_a is not the remaining 22ms.

# Two IO warps on DSV4 decode. Default off. Naive IO_WARPS=4 races mbarriers.
try:
    import os
    from pathlib import Path as _Pio

    if os.environ.get("DSV41_MLA_IO_WARPS", "0") == "2":
        from widen_mla_io2 import apply as _apply_mla_io2

        _apply_mla_io2(_Pio("/usr/local/lib/python3.12/dist-packages/flashinfer"))
        print("dsv41: MLA decode DSV4_IO_WARPS=2 linear io_tid", flush=True)
except Exception as _mla_io2_err:
    print(f"dsv41: MLA IO=2 patch skipped: {_mla_io2_err!r}", flush=True)

# Tile32 MLA: workspace mid_out splits must match CAND_WINDOW. The JIT image
# already has WINDOW=32; stock _core.py still sizes scratch with split_tile=64.
try:
    from pathlib import Path as _P

    from widen_mla_tile32 import apply as _apply_mla_tile32

    _fi = _P("/usr/local/lib/python3.12/dist-packages/flashinfer")
    _cuh = (
        _fi
        / "data/include/flashinfer/attention/sparse_mla_sm120/decode_dsv4_kernel.cuh"
    )
    if _cuh.is_file() and "DSV4_CAND_WINDOW = 32" in _cuh.read_text():
        _apply_mla_tile32(_fi)
except Exception:
    pass

# GB10 (SM120) persistent_topk oversubscribes at 2 decode rows (TopK=512).
# Patch installed vLLM before it imports. Qwen already excludes family 120
# for cooperative topk; V4.1 indexer still calls persistent_topk.
try:
    from pathlib import Path

    from sm120_page import (
        patch_kpool_persistent_topk_source,
        patch_persistent_topk_source,
    )

    _idx = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers"
        "/sparse_attn_indexer.py"
    )
    if _idx.is_file():
        _idx.write_text(patch_persistent_topk_source(_idx.read_text()))
    _kpool = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers"
        "/sparse_attn_indexer_kpool.py"
    )
    if _kpool.is_file():
        _kpool.write_text(patch_kpool_persistent_topk_source(_kpool.read_text()))
except Exception:
    pass

# PF-G8 loader re-index (results/2026-09-21-pfg8/BOOT-CHAIN-AUDIT.md):
# DSV41_LOAD_PF_G8=1 serves a G8 pack (trellis [NT/8][KT][8W]) — the loader
# must narrow the G8 dims (gate/up dim0, down dim1) or TP sharding reads the
# pack wrong. Install is idempotent; failure here means the image's
# vllm_exl3/exl3.py no longer matches the patch anchors — serving on would
# produce garbage, so abort the boot instead of falling back silently.
try:
    import os as _os_g8

    if _os_g8.environ.get("DSV41_LOAD_PF_G8", "0") == "1":
        from pathlib import Path as _Pg8

        from pfg8_loader_reindex import patch as _pfg8_patch
        from pfg8_loader_reindex import require_g8_manifest as _pfg8_require

        _pfg8_require(sys.argv)

        _exl3_py = _Pg8(
            "/usr/local/lib/python3.12/dist-packages/vllm_exl3/exl3.py"
        )
        _t = _exl3_py.read_text()
        _out = _pfg8_patch(_t)
        if _out != _t:
            _exl3_py.write_text(_out)
            print("dsv41: pfg8 loader re-index installed", flush=True)
        else:
            print("dsv41: pfg8 loader re-index already present", flush=True)
except SystemExit as _g8_exit:
    print(f"dsv41: FATAL pfg8 loader re-index failed: {_g8_exit}", flush=True)
    import os as _os_g8x

    _os_g8x._exit(1)
except Exception as _g8_err:
    print(f"dsv41: FATAL pfg8 loader wiring error: {_g8_err!r}", flush=True)
    import os as _os_g8x

    _os_g8x._exit(1)

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

# Graph capture still dummy-forwards DeepGEMM paged-MQA unless the warmup
# stubs above stay. Default serve sets DSV41_ALLOW_CUDA_GRAPHS=1 and
# ENFORCE_EAGER=0. Disk Engram rows are staged in prepare_inputs.
try:
    import os

    from vllm.v1.worker.gpu_worker import Worker
    from vllm.v1.worker.worker_base import CompilationTimes

    if os.environ.get("DSV41_ALLOW_CUDA_GRAPHS") != "1":
        Worker.compile_or_warm_up_model = lambda self: CompilationTimes(0.0, 0.0)
except Exception:
    pass

# mem-hygiene bundle (env-guarded no-ops when their envs are unset):
# 1) fadvise DONTNEED on shard files after weight load — the GB10 driver
#    allocates from MemFree and does not reclaim clean page cache
#    (DSV41_DROP_PAGE_CACHE=1).
# 2) torch.cuda.empty_cache() after long prefill chunks while MemAvailable is
#    under the floor (DSV41_PREFILL_EMPTY_CACHE_TOKENS / _MEMAVAIL_GIB).
try:
    from drop_page_cache import install as _install_dpc

    _install_dpc()
except Exception as _dpc_err:
    print(f"dsv41: drop-page-cache install skipped: {_dpc_err!r}", flush=True)
try:
    from prefill_empty_cache import install as _install_pec

    _install_pec()
except Exception as _pec_err:
    print(f"dsv41: prefill-empty-cache install skipped: {_pec_err!r}", flush=True)

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

# Indexer + compressed MLA share a packed KV group, so they must agree.
# DeepGEMM paged-MQA asserts block_kv in {32, 64}; FlashInfer DSV4 decode
# wants page 64. Upstream reports 128 on SM12, which then has no common size
# if only the indexer is pinned to 64.
try:
    from sm120_page import indexer_kernel_block_sizes
    from vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse import (
        DeepseekV4FlashInferMLASparseBackend,
    )
    from vllm.v1.attention.backends.mla.indexer import DeepseekV4IndexerBackend

    _kbs = staticmethod(lambda: list(indexer_kernel_block_sizes()))
    DeepseekV4IndexerBackend.get_supported_kernel_block_sizes = _kbs
    DeepseekV4FlashInferMLASparseBackend.get_supported_kernel_block_sizes = _kbs
except Exception:
    pass

# compress_ratio=2 at manager 64 yields extra_page_block_size=32. SM120
# prefill (num_tokens>64) rejects that. Bump those specs to 128 so extra
# pages stay 64 and DeepGEMM still sees 128/2=64 states.
try:
    from dataclasses import replace

    from sm120_page import manager_block_for_flashinfer_extra
    from vllm.models.deepseek_v4_1.attention import (
        DeepseekV4Attention,
        DeepseekV4IndexerCache,
    )

    def _pin_extra_page(spec, compress_ratio: int):
        if spec is None:
            return spec
        new_bs = manager_block_for_flashinfer_extra(spec.block_size, compress_ratio)
        if new_bs == spec.block_size:
            return spec
        return replace(spec, block_size=new_bs)

    _attn_spec = DeepseekV4Attention.get_kv_cache_spec

    def _attn_spec_extra_page(self, vllm_config):
        return _pin_extra_page(
            _attn_spec(self, vllm_config), int(getattr(self, "compress_ratio", 1) or 1)
        )

    DeepseekV4Attention.get_kv_cache_spec = _attn_spec_extra_page

    _idx_spec = DeepseekV4IndexerCache.get_kv_cache_spec

    def _idx_spec_extra_page(self, vllm_config):
        return _pin_extra_page(
            _idx_spec(self, vllm_config), int(getattr(self, "compress_ratio", 1) or 1)
        )

    DeepseekV4IndexerCache.get_kv_cache_spec = _idx_spec_extra_page
except Exception:
    pass

# Vision-on: keep hf_config.vision_max_n_token and vision_n_layers so VL
# weights load. SM120 dual-cache prefill only instantiates SWA topk=128.
# SWA index width stays window=128. Decode still uses window=128.
try:
    import os
    from pathlib import Path

    from sm120_page import (
        clamp_index_topk,
        language_model_only_from_env,
        patch_attention_image_width_source,
        patch_indexer_adaptive_source,
        patch_indexer_short_context_source,
        patch_native_indexer_decode_source,
        patch_swa_prefill_image_width_source,
        text_only_max_image_tokens,
    )
    from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

    _v41_cfg_init = DeepseekV41Config.__init__

    def _v41_cfg_init_vision(self, *args, **kwargs):
        _v41_cfg_init(self, *args, **kwargs)
        lm_only = language_model_only_from_env(sys.argv, os.environ)
        self.vision_max_n_token = text_only_max_image_tokens(
            getattr(self, "vision_max_n_token", 0), lm_only
        )
        idx_clamp = int(os.environ.get("DSV41_INDEX_TOPK", "0") or "0")
        text_cfg = getattr(self, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "index_topk"):
            text_cfg.index_topk = clamp_index_topk(
                getattr(text_cfg, "index_topk", 512), idx_clamp
            )
        if hasattr(self, "index_topk"):
            self.index_topk = clamp_index_topk(
                getattr(self, "index_topk", 512), idx_clamp
            )
    DeepseekV41Config.__init__ = _v41_cfg_init_vision

    _swa = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends"
        "/mla/sparse_swa.py"
    )
    if _swa.is_file():
        _swa.write_text(patch_swa_prefill_image_width_source(_swa.read_text()))
    _attn = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1"
        "/attention.py"
    )
    if _attn.is_file():
        _attn.write_text(patch_attention_image_width_source(_attn.read_text()))
    _mla_idx = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends"
        "/mla/indexer.py"
    )
    if _mla_idx.is_file():
        _mla_idx.write_text(patch_native_indexer_decode_source(_mla_idx.read_text()))
        _mla_idx.write_text(patch_indexer_adaptive_source(_mla_idx.read_text()))
    _v41_attn = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1"
        "/attention.py"
    )
    if _v41_attn.is_file():
        _v41_attn.write_text(patch_indexer_short_context_source(_v41_attn.read_text()))
except Exception:
    pass

# lm_head MXFP8 (b12x): DSV41_LMHEAD_MXFP8=1 + quantized tensor in the
# snapshot. Default-off; self-disarms (one line, never a crash) when the
# snapshot keeps the bf16 head. Halves the ~735 MB/call vocab-head weight
# stream (5.35 ms/step bf16 pair, DRAFT-AUX-13MS lever #1).
try:
    import os as _os_lmh

    from lmhead_mxfp8 import enabled_from_env as _lmh_enabled, install as _lmh_install

    if _lmh_enabled(_os_lmh.environ):
        _lmh_install()
except Exception as _lmh_err:
    print(f"dsv41: lm_head mxfp8 hook skipped: {_lmh_err!r}", flush=True)

# DSpark Markov scale. 1 = stock sequential bias. 0 = parallel backbone drafts.
try:
    import os

    from sm120_page import scale_markov_bias
    from vllm.models.deepseek_v4_1.nvidia.dspark import DSparkDeepseekV4ForCausalLM

    _markov_scale = float(os.environ.get("DSV41_DSPARK_MARKOV_SCALE", "1"))
    if _markov_scale != 1.0:
        _markov_bias = DSparkDeepseekV4ForCausalLM.markov_bias

        def _markov_bias_scaled(self, markov_embed):
            return scale_markov_bias(_markov_bias(self, markov_embed), _markov_scale)

        DSparkDeepseekV4ForCausalLM.markov_bias = _markov_bias_scaled
except Exception:
    pass

# Optional decode-step census (draft graph, target forward, Engram staging).
try:
    import os

    if os.environ.get("DSV41_STEP_CENSUS", "0") == "1":
        from sm120_page import install_step_census

        install_step_census()
except Exception:
    pass

# Decode MHC prenorm split-K: DeepGEMM heuristic returns 16 at m=6.
try:
    import os

    if os.environ.get("DSV41_MHC_DECODE_SPLITS", "0") == "1":
        from sm120_page import decode_mhc_pre_num_splits
        from vllm.model_executor.kernels.mhc import tilelang as _mhc_tl
        from vllm.model_executor.kernels.mhc import warmup as _mhc_wu

        _mhc_splits = _mhc_wu.compute_mhc_pre_num_splits

        def _mhc_splits_decode(input_size: int, num_tokens: int) -> int:
            return decode_mhc_pre_num_splits(
                num_tokens, _mhc_splits(input_size, num_tokens)
            )

        _mhc_wu.compute_mhc_pre_num_splits = _mhc_splits_decode
        _mhc_tl.compute_mhc_pre_num_splits = _mhc_splits_decode
        print("dsv41: MHC decode prenorm splits collapsed to 1", flush=True)
except Exception:
    pass

# Host LRU for disk Engram rows. Staging still hashes on GPU; this skips
# NVMe pread+dequant on repeated file rows (overlapping n-grams).
try:
    import os

    if os.environ.get("DSV41_ENGRAM_CACHE", "0") == "1":
        from sm120_page import (
            engram_row_cache_split,
            engram_row_cache_store,
            fill_engram_cache_hits,
        )
        from vllm.models.deepseek_v4_1.common.engram_disk import DiskEngramTable

        _engram_cache: dict = {}
        _engram_order: list = []
        _engram_cap = int(os.environ.get("DSV41_ENGRAM_CACHE_ROWS", "32768"))
        _engram_gather = DiskEngramTable.gather_dequant

        def _cached_gather_dequant(self, rel, owned):
            keys = [int(x) for x in rel.reshape(-1).tolist()]
            hits, missing = engram_row_cache_split(_engram_cache, keys)
            fetched: dict = {}
            if missing:
                import torch

                miss_rel = rel.new_tensor(missing)
                own_map = {
                    int(k): bool(o)
                    for k, o in zip(keys, owned.reshape(-1).tolist())
                }
                miss_owned = owned.new_tensor([own_map[k] for k in missing])
                fetched_rows = _engram_gather(self, miss_rel, miss_owned)
                for i, k in enumerate(missing):
                    row = fetched_rows[i].detach().clone()
                    fetched[k] = row
                    engram_row_cache_store(
                        _engram_cache, _engram_order, k, row, _engram_cap
                    )
            rows = fill_engram_cache_hits(keys, hits, fetched)
            import torch

            return torch.stack(rows, 0)

        DiskEngramTable.gather_dequant = _cached_gather_dequant
        print("dsv41: Engram disk row cache enabled", flush=True)
except Exception:
    pass

# MHC prenorm: stock DeepGEMM uses 16 split-K at m=6. Forcing the TileLang
# GEMM (n_splits=1) is a separate path from DSV41_MHC_DECODE_SPLITS.
try:
    import os

    if os.environ.get("DSV41_MHC_NO_DEEPGEMM", "0") == "1":
        from vllm.model_executor.kernels.mhc import tilelang as _mhc_tl

        def _mhc_no_deep_gemm() -> bool:
            return False

        _mhc_tl.is_deep_gemm_supported = _mhc_no_deep_gemm
        print("dsv41: MHC prenorm uses TileLang GEMM not DeepGEMM", flush=True)
except Exception:
    pass

# Greedy propose, softmax verify. Allocate draft_logits on greedy DSpark
# and cache pre-temperature U+Markov logits, then argmax. Rejection then
# uses q=softmax(draft) instead of one-hot. Do not Gumbel-sample drafts.
try:
    import os

    from sm120_page import dspark_softmax_verify_from_env

    if dspark_softmax_verify_from_env(
        int(os.environ.get("DSV41_DSPARK_SOFTMAX_VERIFY", "0") or "0")
    ):
        import torch
        from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (
            DSparkSpeculator,
        )

        _dspark_init = DSparkSpeculator.__init__

        def _dspark_init_logits_cache(self, *args, **kwargs):
            _dspark_init(self, *args, **kwargs)
            if self.draft_logits is None:
                dtype, fill = self.draft_logits_spec(self.vllm_config)
                self.draft_logits = torch.full(
                    (
                        self.max_num_reqs,
                        self.num_speculative_steps,
                        self.vocab_size,
                    ),
                    fill,
                    dtype=dtype,
                    device=self.device,
                )
            self._zero_temperature = torch.zeros_like(self.temperature)

        DSparkSpeculator.__init__ = _dspark_init_logits_cache

        from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

        def _sample_logits_greedy_cache(
            self, logits, idx_map, sample_pos, step
        ):
            if self._d2t_scatter_index is not None:
                assert self._draft_scatter_buf is not None
                buf = self._draft_scatter_buf[: logits.shape[0]]
                buf.index_copy_(
                    1, self._d2t_scatter_index, logits.to(buf.dtype)
                )
                logits = buf
            # T=0 is plain argmax. gumbel_sample still caches pre-temperature
            # logits and skips idx_map < 0, which index_copy_ does not.
            return gumbel_sample(
                logits,
                idx_map,
                self._zero_temperature,
                self.seeds,
                sample_pos - 1,
                apply_temperature=True,
                is_drafting=True,
                logits_cache=self.draft_logits,
                logits_cache_col=self._step_cols[step],
                use_fp64=self.use_fp64_gumbel,
            )

        DSparkSpeculator._sample_logits = _sample_logits_greedy_cache
        print("dsv41: DSpark greedy propose, softmax verify", flush=True)
except Exception as _softmax_verify_err:
    print(
        f"dsv41: DSpark softmax-verify wrap skipped: {_softmax_verify_err!r}",
        flush=True,
    )

# Gate Markov bias by the unused confidence head: logits = base + conf * bias.
try:
    import os

    from dspark_conf_gate import (
        apply_confidence_gate,
        dspark_conf_gate_from_env,
    )

    if dspark_conf_gate_from_env(
        int(os.environ.get("DSV41_DSPARK_CONF_GATE", "0") or "0")
    ):
        from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (
            DSparkSpeculator,
        )

        _dspark_sample_seq = DSparkSpeculator._sample_sequential

        def _sample_sequential_conf_gate(self, num_reqs, head_hidden):
            if self._draft_topk is not None:
                return _dspark_sample_seq(self, num_reqs, head_hidden)
            n_spec = self.num_speculative_steps
            num_sample = num_reqs * n_spec
            sample_hidden = head_hidden[self.sample_indices[:num_sample]]
            base_logits = self.model.compute_draft_logits(sample_hidden)
            vocab_size = base_logits.shape[-1]
            base_logits = base_logits.view(num_reqs, n_spec, vocab_size)
            idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
            sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
            hidden_by_step = sample_hidden.view(num_reqs, n_spec, -1)
            prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
            for i in range(n_spec):
                markov_embed = self.model.markov_embed(prev)
                bias = self.model.markov_bias(markov_embed)
                conf = self.model.compute_confidence(
                    hidden_by_step[:, i], markov_embed
                )
                logits_i = apply_confidence_gate(
                    base_logits[:, i], bias, conf
                )
                draft_sampled_i = self._sample_logits(
                    logits_i, idx_map[:, i], sample_pos[:, i], i
                )
                self.draft_tokens[:num_reqs, i] = draft_sampled_i
                prev = draft_sampled_i

        DSparkSpeculator._sample_sequential = _sample_sequential_conf_gate
        print("dsv41: DSpark confidence-gated Markov", flush=True)
except Exception as _conf_gate_err:
    print(
        f"dsv41: DSpark conf-gate wrap skipped: {_conf_gate_err!r}",
        flush=True,
    )

# Second DSpark pass: write pass-1 samples into noise query slots and
# replay _generate_draft so later hiddens attend to tokens, not the mask.
try:
    import os

    from dspark_refine_pass import (
        apply_refine_fill,
        dspark_refine_pass_from_env,
        refine_query_index,
    )

    if dspark_refine_pass_from_env(
        int(os.environ.get("DSV41_DSPARK_REFINE_PASS", "0") or "0")
    ):
        from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (
            DSparkSpeculator,
        )

        _dspark_refine_init = DSparkSpeculator.__init__

        def _dspark_init_refine_idx(self, *args, **kwargs):
            _dspark_refine_init(self, *args, **kwargs)
            self._refine_idx = refine_query_index(
                int(self.max_num_reqs),
                int(self.num_query_per_req),
                int(self.num_speculative_steps),
                self.device,
            )

        DSparkSpeculator.__init__ = _dspark_init_refine_idx

        _dspark_generate_draft = DSparkSpeculator._generate_draft

        def _dspark_generate_draft_refine(
            self,
            num_reqs,
            *args,
            **kwargs,
        ):
            _dspark_generate_draft(self, num_reqs, *args, **kwargs)
            apply_refine_fill(
                self.input_buffers.input_ids,
                self.draft_tokens,
                self._refine_idx,
                int(num_reqs),
            )
            _dspark_generate_draft(self, num_reqs, *args, **kwargs)

        DSparkSpeculator._generate_draft = _dspark_generate_draft_refine
        print("dsv41: DSpark refine pass 2 on pass-1 query fill", flush=True)
except Exception as _refine_pass_err:
    print(
        f"dsv41: DSpark refine-pass wrap skipped: {_refine_pass_err!r}",
        flush=True,
    )

# DSpark tail n-gram: keep draft pos 0..start_pos-1, fill the dead Markov
# tail from prompt-lookup of prefix+head. Graph-safe: runs after propose()
# returns, which is after FULL draft-graph replay.
try:
    import os

    from sm120_page import (
        dspark_tail_ngram_start_pos,
        overlay_ngram_on_draft_tail,
    )

    _tail_pos = dspark_tail_ngram_start_pos(
        int(os.environ.get("DSV41_DSPARK_TAIL_NGRAM", "0") or "0"),
        int(os.environ.get("DSV41_DSPARK_TAIL_NGRAM_POS", "3") or "3"),
    )
    if _tail_pos is not None:
        import torch
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner

        _sample_tokens = GPUModelRunner.sample_tokens

        def _sample_tokens_tail_ngram(self, *args, **kwargs):
            spec = getattr(self, "speculator", None)
            orig_propose = spec.propose if spec is not None else None
            if orig_propose is None:
                return _sample_tokens(self, *args, **kwargs)

            def _propose_overlay(input_batch, *a, **k):
                drafts = orig_propose(input_batch, *a, **k)
                try:
                    n_req = int(drafts.shape[0])
                    idx_np = input_batch.idx_mapping_np
                    computed = self.req_states.num_computed_tokens_np
                    ids = self.req_states.all_token_ids.gpu
                    prefixes = []
                    for i in range(n_req):
                        req_idx = int(idx_np[i])
                        slen = int(computed[req_idx])
                        if slen <= 0:
                            prefixes.append([])
                        else:
                            prefixes.append(
                                [int(x) for x in ids[req_idx, :slen].tolist()]
                            )
                    cpu = drafts.detach().to("cpu").tolist()
                    merged = overlay_ngram_on_draft_tail(
                        cpu, prefixes, start_pos=_tail_pos
                    )
                    drafts.copy_(
                        torch.tensor(
                            merged, dtype=drafts.dtype, device=drafts.device
                        )
                    )
                except Exception as _tail_err:
                    print(
                        f"dsv41: tail ngram overlay skipped: {_tail_err!r}",
                        flush=True,
                    )
                return drafts

            spec.propose = _propose_overlay
            try:
                return _sample_tokens(self, *args, **kwargs)
            finally:
                spec.propose = orig_propose

        GPUModelRunner.sample_tokens = _sample_tokens_tail_ngram
        print(
            f"dsv41: DSpark tail ngram overlay start_pos={_tail_pos}",
            flush=True,
        )
except Exception as _tail_ngram_err:
    print(
        f"dsv41: DSpark tail ngram wrap skipped: {_tail_ngram_err!r}",
        flush=True,
    )

# Restrict DSpark backbone logits to top-k before sequential Markov.
# Do not set hf_config.dspark_draft_topk: SpeculativeConfig only allows
# that field on Qwen3DSpark, and DeepSeek lacks apply_markov_bias_gathered.
# Mask in-place (fill_ + scatter_) so CUDA graph capture does not allocate
# a second full-vocab tensor. Dense _sample_sequential still runs.
try:
    import os

    from sm120_page import dspark_draft_topk_from_env

    _draft_k = dspark_draft_topk_from_env(
        int(os.environ.get("DSV41_DSPARK_DRAFT_TOPK", "0") or "0")
    )
    if _draft_k is not None:
        from vllm.models.deepseek_v4_1.nvidia.dspark import (
            DSparkDeepseekV4ForCausalLM,
        )

        _cdl = DSparkDeepseekV4ForCausalLM.compute_draft_logits

        def _cdl_topk(self, hidden_states):
            logits = _cdl(self, hidden_states)
            vals, idx = logits.topk(_draft_k, dim=-1)
            logits.fill_(float("-inf"))
            return logits.scatter_(-1, idx, vals)

        DSparkDeepseekV4ForCausalLM.compute_draft_logits = _cdl_topk
        print(
            f"dsv41: DSpark compute_draft_logits topk={_draft_k}",
            flush=True,
        )
except Exception as _draft_topk_err:
    print(
        f"dsv41: DSpark compute_draft_logits wrap skipped: {_draft_topk_err!r}",
        flush=True,
    )

# Bake SM120 sparse-MLA chunks_per_block into the captured decode graph.
# Default 0 leaves FlashInfer AutoTuner/heuristic. A positive k is the
# cubin tactic (1..num_splits) and skips autotune.
try:
    import os

    from sm120_page import mla_chunks_per_block_from_env

    _mla_cpb = mla_chunks_per_block_from_env(
        int(os.environ.get("DSV41_MLA_CHUNKS_PER_BLOCK", "0") or "0")
    )
    if _mla_cpb is not None:
        import flashinfer.mla._sparse_mla_sm120 as _sm120_mla

        _sm120_decode = _sm120_mla.sparse_mla_sm120_decode_dsv4

        def _sm120_decode_cpb(*args, chunks_per_block=None, **kwargs):
            kwargs["chunks_per_block"] = _mla_cpb
            return _sm120_decode(*args, **kwargs)

        _sm120_mla.sparse_mla_sm120_decode_dsv4 = _sm120_decode_cpb
        print(
            f"dsv41: SM120 MLA chunks_per_block={_mla_cpb}",
            flush=True,
        )
except Exception as _mla_cpb_err:
    print(
        f"dsv41: SM120 MLA chunks_per_block wrap skipped: {_mla_cpb_err!r}",
        flush=True,
    )

# Engram disk stager census: time the per-step NVMe gather phases. The
# inter-step gap is dominated by EngramDiskStager.stage (15-28 ms/step at
# L.A.I.L shapes); this census names the phase so the fast-stager patch can
# target it. Diagnostic only; enable with DSV41_ENGRAM_CENSUS=1.
try:
    from pathlib import Path as _Pc

    from engram_stage_census import apply as _apply_engram_census

    _apply_engram_census(
        _Pc("/usr/local/lib/python3.12/dist-packages/vllm")
    )
except Exception as _engram_census_err:
    print(f"dsv41: engram census skipped: {_engram_census_err!r}", flush=True)

# Engram fast stage: the 13 per-layer disk gathers run concurrently instead
# of serially. Trace forensics: ~25.6 ms/step of GPU idle is
# _read_rows Future.result lock-wait (cold NVMe rows in the serial loop).
# DSV41_ENGRAM_FAST_STAGE=1 default; self-check reverts on any mismatch.
try:
    from pathlib import Path as _Pf

    from engram_stage_fast import apply as _apply_engram_fast

    _apply_engram_fast(
        _Pf("/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1")
    )
except Exception as _engram_fast_err:
    print(f"dsv41: engram fast stage skipped: {_engram_fast_err!r}", flush=True)

# Engram next-step prefetch: fadvise WILLNEED for the next decode chunk's
# rows, hashed on CPU at postprocess time so kernel readahead overlaps the
# draft graph. Advisory only (correctness never depends on it). Requires
# engram_stage_fast. DSV41_ENGRAM_PREFETCH=1 enables.
try:
    from pathlib import Path as _Pp

    from engram_prefetch_v3 import apply as _apply_engram_pf

    _apply_engram_pf(
        _Pp("/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1"),
        _Pp("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py"),
        _Pp("/usr/local/lib/python3.12/dist-packages/vllm"),
    )
except Exception as _engram_pf_err:
    print(f"dsv41: engram prefetch skipped: {_engram_pf_err!r}", flush=True)

# Engram CPU-side hash (Round 18 attribution fix): compute the next step's
# n-gram hashes on the host from a post-propose snapshot and delete the
# prepare_inputs D2H + hashes_ready.synchronize() (99.9% of the ~14.2
# ms/step GPU-idle pool). Bit-exact warmup verify + async canary; any
# mismatch self-disables to the stock path. DSV41_ENGRAM_CPU_HASH=1
# enables (default off). Requires the engram_stage_fast chain.
# cpu-hash (R20) and defer (R24) are reverted: their text installs only when
# one of them is enabled. Defer anchors on the cpu-hash stage text, so either
# flag installs both.
import os as _os_chd

_ENGRAM_CH_DEFER = (
    _os_chd.environ.get("DSV41_ENGRAM_CPU_HASH", "0") == "1"
    or _os_chd.environ.get("DSV41_ENGRAM_DEFER", "0") == "1"
)
try:
    if _ENGRAM_CH_DEFER:
        from pathlib import Path as _Pch

        from engram_cpu_hash import apply as _apply_cpu_hash

        _apply_cpu_hash(
            _Pch("/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1"),
            _Pch("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py"),
        )
except Exception as _cpu_hash_err:
    print(f"dsv41: engram cpu-hash skipped: {_cpu_hash_err!r}", flush=True)

# Engram gather v2 (Round 22 attribution fix): replace the stock chunk-pool
# _read_rows inside DiskEngramTable.gather_dequant with inline preadv
# per contiguous row run + the stock dequant math verbatim. The Round-22
# trace attributes ~7.5 ms/step of GPU idle to host execution of the
# gather loop (pool dispatch + per-row syscalls), NOT IO waits (read_w
# 0.1 ms, page-cached). DSV41_ENGRAM_GATHER_V2=1 enables (default off);
# one-shot bit-exact self-check, any error disarms to stock with one line.
# Requires the engram_stage_census chain (applies after it).
try:
    from pathlib import Path as _Pg

    from engram_gather_v2 import apply as _apply_gather_v2

    _apply_gather_v2(_Pg("/usr/local/lib/python3.12/dist-packages/vllm"))
except Exception as _gather_v2_err:
    print(f"dsv41: engram gather v2 skipped: {_gather_v2_err!r}", flush=True)

# Engram stage defer (Round 23 chain): ONE persistent worker per rank
# gathers the NEXT step's rows off the critical path (v3-rule prediction,
# cpu-hash numpy mirror, v2 preadv gather into double-buffered pinned
# slots + side-stream H2D with proper wait-before-with ordering); stage()
# consumes the signaled buffer with a per-table DtoD at replay time and
# falls back to the sync v2 path on ANY anomaly (miss / late / mismatch ->
# ONE warning line, never a crash, never slower than sync).
# DSV41_ENGRAM_DEFER=1 enables (default off). Requires the full chain
# (prestage -> census -> fast -> v3 -> cpu-hash -> gather v2).
try:
    if _ENGRAM_CH_DEFER:
        from pathlib import Path as _Pd

        from engram_defer import apply as _apply_defer

        _apply_defer(
            _Pd("/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1"),
            _Pd("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py"),
            _Pd("/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/nvidia/model_state.py"),
        )
except Exception as _defer_err:
    print(f"dsv41: engram defer skipped: {_defer_err!r}", flush=True)

# PF-G8 fat-expert routing gate (BOOT-CHAIN-AUDIT.md R3, 2026-09-21): under
# DSV41_LOAD_PF_G8=1 the fat-GEMM/reconstruct fallback readers are
# stock-layout and must not run on a G8 pack. The kernel-side guard
# (exl3_fat_gemm TORCH_CHECK) is belt-and-braces; the primary gate is here:
# raise VLLM_EXL3_FAT_THRESHOLD to 2**30 unless the operator explicitly set
# it, so no expert is ever routed fat and all rows go through the G8-aware
# exl3_moe kernel. The module-level read of the env in vllm_exl3.exl3
# happens at import time — sitecustomize runs before any vllm_exl3 import,
# so setting os.environ here wins. Default (env unset) = untouched stock
# behavior with the stock threshold of 256.
try:
    import os as _os_fat

    if (
        _os_fat.environ.get("DSV41_LOAD_PF_G8", "0") == "1"
        and "VLLM_EXL3_FAT_THRESHOLD" not in _os_fat.environ
    ):
        _os_fat.environ["VLLM_EXL3_FAT_THRESHOLD"] = str(2**30)
        print("dsv41: pfg8 fat-expert routing OFF (VLLM_EXL3_FAT_THRESHOLD=2^30)", flush=True)
except Exception as _fat_err:
    print(f"dsv41: FATAL pfg8 fat gate wiring error: {_fat_err!r}", flush=True)
    import os as _os_fat_x

    _os_fat_x._exit(1)

# G8 stream feed (g8final r31): env-gated DSV41_LOAD_PF_G8=1 — drain the VL
# wrapper's sorted mapped list in place during load_weights so per-tensor H2D
# page pins are released as consumed (G8 boot OOM root cause; see
# docker/patch/g8_stream_feed.py). Idempotent; no-op for stock boots.
try:
    import os as _os_sf

    if _os_sf.environ.get("DSV41_LOAD_PF_G8", "0") == "1":
        from g8_stream_feed import install as _g8sf_install

        _g8sf_install()
except SystemExit as _g8sf_exit:
    print(f"dsv41: FATAL g8 stream feed failed: {_g8sf_exit}", flush=True)
    import os as _os_sfx

    _os_sfx._exit(1)
except Exception as _g8sf_err:
    print(f"dsv41: g8 stream feed skipped: {_g8sf_err!r}", flush=True)
