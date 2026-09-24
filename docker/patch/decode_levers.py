"""Env-gated decode levers, all default off. sitecustomize calls install().

- DSV41_MHC_DECODE_SPLITS=N, N >= 2: mHC prenorm split-K is N for
  num_tokens <= 64, capped at the k-block limit cdiv(K, 64) // 4. The stock
  heuristic wants 48 on GB10 but warmup.py clamps it to {1, 4, 16} to bound
  startup JIT work. 0 keeps stock. 1 is the older collapse-to-1 block in
  sitecustomize. Only warmup.compute_mhc_pre_num_splits is patched: the
  tilelang mhc_pre imports it at call time and MHCPreNormKernel's
  get_warmup_keys calls it at num_tokens=1, so dispatch and warmup keys both
  see N.
- DSV41_WOA_PREPACK=1 lives in the image's o_proj.py (fix_o_proj_woa_fp8.py
  stage 2). Here: a loud warning when the image lacks that stage.

Top-level imports are stdlib only, so importing this module cannot fail.
"""

from __future__ import annotations

import os
from pathlib import Path

MHC_DECODE_MAX_TOKENS = 64
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


def _check_woa_prepack(env) -> None:
    if (env.get("DSV41_WOA_PREPACK", "0") or "0") != "1" or not O_PROJ_PY.exists():
        return
    if "_woa_prepacked_scale" not in O_PROJ_PY.read_text():
        print(
            "dsv41: WARNING DSV41_WOA_PREPACK=1 but this image's o_proj.py has no "
            "prepack stage; the lever is OFF. Build docker/Dockerfile.woa-prepack.",
            flush=True,
        )


def install(env=None) -> None:
    env = os.environ if env is None else env
    for name, step in (
        ("mhc-prenorm-splits", _install_mhc_splits),
        ("woa-prepack", _check_woa_prepack),
    ):
        try:
            step(env)
        except Exception as exc:  # noqa: BLE001 - one lever never blocks the others
            print(f"dsv41: decode lever {name} FAILED, lever is OFF: {exc!r}", flush=True)
