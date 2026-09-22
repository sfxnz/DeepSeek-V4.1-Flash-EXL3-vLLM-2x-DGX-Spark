#!/usr/bin/env python3
"""G8 stream feed — env-gated (DSV41_LOAD_PF_G8=1 only), stock path untouched.

Root cause (diag v4/v5, 2026-09-22 g8final): the VL wrapper's load_weights
(vllm/models/deseek_v4_1/nvidia/vl_model.py, DeepseekV41ForCausalLM) does

    mapped = sorted(self.hf_to_vllm_mapper.apply(weights), key=...)
    loader.load_weights(mapped)

The retained sorted list holds EVERY expert tensor alive for the whole load.
Under GB10 unified memory every dest.copy_(sharded) is an H2D from a pageable
mmap view; the H2D pins the source pages and the kernel only releases them at
tensor free. With the full list retained, pins accumulate linearly: measured
(diag_v5, full 40-layer topology on spark2 GPU) stock peaks ≈27.6 GiB anon+swap
(survives), G8 ≈53.3 GiB (dies at 59.4 GiB in the real boot, ×3 deterministic).

Fix: drain the sorted list IN PLACE while AutoWeightsLoader consumes it — each
item slot is set to None right after it is yielded, so each source tensor (and
its pinned pages) is dropped as soon as the child loader moves past it. diag_v6
A/B: G8 consume-phase anon peak 39.3 GiB → 1.9 GiB (flat), swap 14.0 → 0.6.

The sort order is preserved exactly (same key, same order); only lifetime
changes. The contiguous-group contract of AutoWeightsLoader is unaffected: it
delegates per contiguous group as before, it just observes a generator that
yields the identical sequence.
"""
from __future__ import annotations

MARKER = "# --- g8-stream-feed ---"

INSTALL_OLD = '''    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Map HF names into this wrapper's namespace up front and sort, so
        # the "language_model." group reaches the child loader as one
        # contiguous block (AutoWeightsLoader delegates per contiguous group,
        # and the child's load_weights finalizes fused expert weights, which
        # must not run on a partially loaded model).
        mapped = sorted(self.hf_to_vllm_mapper.apply(weights), key=lambda x: x[0])
        loader = AutoWeightsLoader(self)
        loaded_params = loader.load_weights(mapped)'''

INSTALL_NEW = '''    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Map HF names into this wrapper's namespace up front and sort, so
        # the "language_model." group reaches the child loader as one
        # contiguous block (AutoWeightsLoader delegates per contiguous group,
        # and the child's load_weights finalizes fused expert weights, which
        # must not run on a partially loaded model).
        mapped = sorted(self.hf_to_vllm_mapper.apply(weights), key=lambda x: x[0])
        loader = AutoWeightsLoader(self)
        # ''' + MARKER + ''' drain-in-place: H2D copies pin each source
        # tensor's host pages until the tensor is freed; retaining the whole
        # sorted list (184k expert tensors) balloons host anon ~27 GiB stock /
        # ~53 GiB G8 (OOM). Yield the identical order but drop each item as
        # consumed (diag_v6: anon peak 39.3 -> 1.9 GiB). G8-gated only.
        import os as _os

        def _drained(seq):
            if _os.environ.get("DSV41_LOAD_PF_G8", "0") != "1":
                yield from seq
                return
            lst = list(seq)
            for _i in range(len(lst)):
                _item = lst[_i]
                lst[_i] = None
                yield _item

        loaded_params = loader.load_weights(_drained(mapped))'''


def install() -> bool:
    """Rewrite vl_model.py's wrapper load_weights on disk (idempotent)."""
    import os

    path = os.environ.get(
        "DSV41_VL_MODEL_PATH",
        "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/nvidia/vl_model.py",
    )
    with open(path) as fh:
        text = fh.read()
    if MARKER in text:
        print("dsv41: g8 stream feed already present", flush=True)
        return True
    if INSTALL_OLD not in text:
        raise SystemExit(f"anchor not found in {path} (vl_model.py drifted)")
    with open(path, "w") as fh:
        fh.write(text.replace(INSTALL_OLD, INSTALL_NEW))
    print("dsv41: g8 stream feed installed", flush=True)
    return True
