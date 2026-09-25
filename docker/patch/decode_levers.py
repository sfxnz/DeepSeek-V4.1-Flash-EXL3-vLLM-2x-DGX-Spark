"""Env-gated decode levers, all default off. sitecustomize calls install().

- DSV41_MHC_DECODE_SPLITS=N, N >= 2: mHC prenorm split-K is N for
  num_tokens <= 64, capped at the k-block limit cdiv(K, 64) // 4. The stock
  heuristic wants 48 on GB10 but warmup.py clamps it to {1, 4, 16} to bound
  startup JIT work. 0 keeps stock. 1 is the older collapse-to-1 block in
  sitecustomize. Only warmup.compute_mhc_pre_num_splits is patched: the
  tilelang mhc_pre imports it at call time and MHCPreNormKernel's
  get_warmup_keys calls it at num_tokens=1, so dispatch and warmup keys both
  see N.
- DSV41_DSPARK_SPARSE_MARKOV=1: the DSpark drafter takes the speculator's own
  top-k path (_sample_sequential_topk) with k = DSV41_DSPARK_SPARSE_MARKOV_TOPK
  (default 256), so the Markov bias is a k x 256 gather-dot instead of the
  129280 x 256 bf16 GEMM. The full-vocab argmax stays. DeepSeek cannot set
  hf_config.dspark_draft_topk (SpeculativeConfig allows it on Qwen only), so
  _draft_topk is set after DSparkSpeculator.__init__ and DeepSeek gets the
  apply_markov_bias_gathered hook Qwen already has. Greedy verify keeps the
  output exact; only acceptance can move (a winner outside the base top-k).
- DSV41_WOA_PREPACK=1 lives in the image's o_proj.py (fix_o_proj_woa_fp8.py
  stage 2). Here: a loud warning when the image lacks that stage.
- DSV41_P2B_SRC_SORT=1 is read by the compiled vllm_exl3_c (widen_p2b_srcsort
  in docker/Dockerfile). Here: a loud warning when that .so has no srcsort
  code (built before the patch), where the env would silently do nothing.
- DSV41_P2B_COOP=1 is read by the compiled vllm_exl3_c (widen_p2b_coop in
  docker/Dockerfile.e14). Same warning when that .so has no coop code.

Top-level imports are stdlib only, so importing this module cannot fail.
"""

from __future__ import annotations

import os
from pathlib import Path

# Boot-log marker for tools/engagement_audit.py: an armed lever that did not engage.
LOG_DISARMED = "lever is OFF"

MHC_DECODE_MAX_TOKENS = 64
SPARSE_MARKOV_TOPK_DEFAULT = 256
O_PROJ_PY = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/ops/o_proj.py"
)


def mhc_forced_splits(env_value: str | None) -> int | None:
    """None keeps stock (0) or the legacy collapse block (1). N >= 2 forces N."""
    n = int(env_value or "0")
    return n if n >= 2 else None


def mhc_pre_num_splits(input_size: int, num_tokens: int, forced: int, stock) -> int:
    """Forced split-K at decode widths; stock heuristic above 64 tokens."""
    if int(num_tokens) > MHC_DECODE_MAX_TOKENS:
        return stock(input_size, num_tokens)
    kblock_cap = max(1, -(-int(input_size) // 64) // 4)
    return max(1, min(int(forced), kblock_cap))


def sparse_markov_topk(env) -> int | None:
    """None keeps the dense Markov GEMM. Otherwise the candidate count k."""
    if (env.get("DSV41_DSPARK_SPARSE_MARKOV", "0") or "0") != "1":
        return None
    k = int(env.get("DSV41_DSPARK_SPARSE_MARKOV_TOPK", "") or SPARSE_MARKOV_TOPK_DEFAULT)
    if not 1 <= k <= 4096:
        raise ValueError(f"DSV41_DSPARK_SPARSE_MARKOV_TOPK={k} outside 1..4096")
    return k


def sparse_markov_conflicts(env) -> list[str]:
    """Levers the gathered path would silently bypass."""
    out = []
    if float(env.get("DSV41_DSPARK_MARKOV_SCALE", "1") or "1") != 1.0:
        out.append("DSV41_DSPARK_MARKOV_SCALE")  # wraps markov_bias only
    if int(env.get("DSV41_DSPARK_CONF_GATE", "0") or "0") == 1:
        out.append("DSV41_DSPARK_CONF_GATE")  # falls through when _draft_topk is set
    if int(env.get("DSV41_DSPARK_DRAFT_TOPK", "0") or "0") > 0:
        out.append("DSV41_DSPARK_DRAFT_TOPK")  # dense mask wrap, rejected lever
    return out


def _install_mhc_splits(env) -> None:
    forced = mhc_forced_splits(env.get("DSV41_MHC_DECODE_SPLITS"))
    if forced is None:
        return
    from vllm.model_executor.kernels.mhc import warmup as _mhc_wu

    stock = _mhc_wu.compute_mhc_pre_num_splits

    def _splits(input_size: int, num_tokens: int) -> int:
        return mhc_pre_num_splits(input_size, num_tokens, forced, stock)

    _mhc_wu.compute_mhc_pre_num_splits = _splits
    print(
        f"dsv41: MHC prenorm split-K forced to {forced} for num_tokens <= "
        f"{MHC_DECODE_MAX_TOKENS}",
        flush=True,
    )


def _install_sparse_markov(env) -> None:
    k = sparse_markov_topk(env)
    if k is None:
        return
    conflicts = sparse_markov_conflicts(env)
    if conflicts:
        raise ValueError(f"DSV41_DSPARK_SPARSE_MARKOV=1 cannot combine with {conflicts}")
    from vllm.models.deepseek_v4_1.nvidia.dspark import DSparkDeepseekV4ForCausalLM
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    if not hasattr(DSparkDeepseekV4ForCausalLM, "apply_markov_bias_gathered"):

        def apply_markov_bias_gathered(self, markov_embed, logits, values, index):
            return self.model.markov_head.apply_bias_gathered(
                markov_embed, logits, values, index, self.logits_processor.scale
            )

        DSparkDeepseekV4ForCausalLM.apply_markov_bias_gathered = apply_markov_bias_gathered

    init = DSparkSpeculator.__init__

    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        if self._draft_topk is None:
            self._draft_topk = k

    DSparkSpeculator.__init__ = __init__
    print(f"dsv41: DSpark sparse Markov (gathered top-k) k={k}", flush=True)


def _check_woa_prepack(env) -> None:
    if (env.get("DSV41_WOA_PREPACK", "0") or "0") != "1" or not O_PROJ_PY.exists():
        return
    if "_woa_prepacked_scale" not in O_PROJ_PY.read_text():
        print(
            "dsv41: WARNING DSV41_WOA_PREPACK=1 but this image's o_proj.py has no "
            "prepack stage; the lever is OFF. Build docker/Dockerfile.woa-prepack.",
            flush=True,
        )


def _check_p2b_env(env, name: str, lever: str, code: str, dockerfile: str) -> None:
    if (env.get(name, "0") or "0") != "1":
        return
    import importlib.util
    import mmap

    spec = importlib.util.find_spec("vllm_exl3_c")
    if spec is None or not spec.origin:
        raise RuntimeError("vllm_exl3_c extension not found")
    # The patched host code reads the env by name; the literal is in .rodata.
    with open(spec.origin, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as so:
        if so.find(name.encode()) < 0:
            print(
                f"dsv41: decode lever {lever}: {spec.origin} has no {code} code; "
                f"the lever is OFF. Rebuild {dockerfile}.",
                flush=True,
            )


def _check_p2b_src_sort(env) -> None:
    _check_p2b_env(env, "DSV41_P2B_SRC_SORT", "p2b_src_sort", "srcsort", "docker/Dockerfile")


def _check_p2b_coop(env) -> None:
    _check_p2b_env(env, "DSV41_P2B_COOP", "p2b_coop", "coop", "docker/Dockerfile.e14")


def install(env=None) -> None:
    env = os.environ if env is None else env
    for name, step in (
        ("mhc-prenorm-splits", _install_mhc_splits),
        ("sparse-markov", _install_sparse_markov),
        ("woa-prepack", _check_woa_prepack),
        ("p2b_src_sort", _check_p2b_src_sort),
        ("p2b_coop", _check_p2b_coop),
    ):
        try:
            step(env)
        except Exception as exc:  # noqa: BLE001 - one lever never blocks the others
            print(f"dsv41: decode lever {name} FAILED, lever is OFF: {exc!r}", flush=True)
