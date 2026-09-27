"""Engram early hash: launch the stage's hash + DtoH before the attention metadata.

The stock stage (model_state.prepare_inputs -> EngramDiskStager.stage) hashes
the step's tokens as the LAST eager launch before the target graph, so the
host's gather (native stage: ~0.15-0.35 ms) runs while the GPU has nothing
queued. Every input of that hash is final once GPUModelRunner.prepare_inputs
returns (input_ids, positions, query_start_loc; the lookback window reads
num_computed_tokens / all_token_ids, which only the previous step's
post_update writes). DSV41_ENGRAM_EARLY_HASH=1 (needs
DSV41_ENGRAM_NATIVE_STAGE=1) launches the lookback kernel, the hash and the
DtoH at the start of GPUModelRunner.prepare_attn; stage() then only waits for
that DtoH and gathers while the GPU still runs the ~0.5-1 ms of block-table,
slot-mapping and attention-metadata kernels queued behind it.

Safety: stage() uses the early hashes only for the batch they were made for
(same InputBatch object and token count, consumed once). The first
DSV41_ENGRAM_EARLY_VERIFY steps (default 8, minimum 1) recompute the stock late
hash and compare int32 for int32; a mismatch prints one LOG_DISARMED line, that
step keeps the late hashes, and every later step takes the stock order.
Prefill (more tokens than the native stage takes) is never hashed early.

Top-level imports are stdlib only.
"""

from __future__ import annotations

import os
import weakref

# Boot-log markers for tools/engagement_audit.py.
LOG_ENGAGED = "dsv41: engram early hash matches the stock hash"
LOG_DISARMED = "dsv41: engram early hash DISABLED ->"

DEFAULT_VERIFY = 8
_STATE = {"armed": True, "verify_left": DEFAULT_VERIFY, "engaged": False}


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return env.get("DSV41_ENGRAM_EARLY_HASH", "0") == "1"


def verify_steps(env=None) -> int:
    env = os.environ if env is None else env
    return max(1, int(env.get("DSV41_ENGRAM_EARLY_VERIFY", "") or DEFAULT_VERIFY))


def _disarm(exc: BaseException) -> None:
    if _STATE["armed"]:
        _STATE["armed"] = False
        print("dsv41: engram early hash DISABLED -> stock order: %r" % (exc,), flush=True)


class EarlyHash:
    """Launch (prepare_attn) and consume (stage) the early hashes."""

    def __init__(self, torch, image_sentinel_mask, lookback_kernel, triton):
        self.torch = torch
        self.mask = image_sentinel_mask
        self.lookback_kernel = lookback_kernel
        self.triton = triton

    def launch(self, stager, model_state, input_batch, req_states) -> None:
        """model_state.prepare_inputs' lookback + stage()'s hash + DtoH, now."""
        nat = getattr(stager, "_eng_native", None)
        window = getattr(model_state, "lookback_token_ids", None)
        if (
            not _STATE["armed"]
            or nat is None
            or window is None
            or input_batch.input_ids is None
            or getattr(model_state, "rope_state", None) is not None
        ):
            return
        n = min(int(input_batch.num_tokens), stager.max_tokens)
        if n <= 0 or n > nat.max_tokens or not stager.hash_state.ensure_cache():
            return
        # DeepseekV41ModelState.prepare_inputs, verbatim (it runs again there).
        all_token_ids = req_states.all_token_ids.gpu
        depth = window.shape[1]
        self.lookback_kernel[(window.shape[0],)](
            window,
            input_batch.idx_mapping,
            req_states.num_computed_tokens.gpu,
            all_token_ids,
            all_token_ids.stride(0),
            input_batch.idx_mapping.shape[0],
            DEPTH=depth,
            BLOCK_DEPTH=self.triton.next_power_of_2(depth),
        )
        # EngramDiskStager.stage's hash, with the arguments prepare_inputs passes.
        ids = input_batch.input_ids[:n]
        hashes = stager.hash_state(
            ids,
            input_batch.positions[:n],
            input_batch.query_start_loc[: input_batch.num_reqs + 1],
            self.mask(ids),
            window,
            self.mask(window),
            None,
            None,
        )
        host = stager.hash_host[:n]
        host.copy_(hashes[:, :, stager.head_start : stager.head_end], non_blocking=True)
        stager.hashes_ready.record()
        stager._early_rec = (weakref.ref(input_batch), n)

    def consume(self, stager, input_ids, positions, query_start_loc, lookback, n) -> bool:
        """True: hash_host[:n] holds this step's hashes and the DtoH is done."""
        rec, cur = getattr(stager, "_early_rec", None), getattr(stager, "_early_cur", None)
        stager._early_rec = None
        if not _STATE["armed"] or rec is None or cur is None:
            return False
        if rec[0]() is not cur or rec[1] != n or input_ids is not cur.input_ids:
            return False
        stager.hashes_ready.synchronize()
        if _STATE["verify_left"] > 0:
            torch = self.torch
            early = stager.hash_host[:n].clone()
            ids = input_ids[:n]
            hashes = stager.hash_state(
                ids, positions[:n], query_start_loc, self.mask(ids), lookback, self.mask(lookback), None, None
            )
            host = stager.hash_host[:n]
            host.copy_(hashes[:, :, stager.head_start : stager.head_end], non_blocking=True)
            stager.hashes_ready.record()
            stager.hashes_ready.synchronize()
            if not torch.equal(early, host):
                _disarm(RuntimeError(f"early hash != stock hash (n={n})"))
                return True  # hash_host holds the stock (late) hashes
            _STATE["verify_left"] -= 1
            if not _STATE["engaged"]:
                _STATE["engaged"] = True
                print("dsv41: engram early hash matches the stock hash (n=%d)" % n, flush=True)
        return True


def install(torch, image_sentinel_mask):
    """Wrap GPUModelRunner.prepare_attn (launch) and the model state's
    prepare_inputs (marks the batch stage() may consume). Returns the hook."""
    from vllm.models.deepseek_v4_1.nvidia import model_state as ms_mod
    from vllm.triton_utils import triton
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    _STATE["verify_left"] = verify_steps()
    hook = EarlyHash(torch, image_sentinel_mask, ms_mod._gather_lookback_kernel, triton)
    orig_attn = GPUModelRunner.prepare_attn
    cls = ms_mod.DeepseekV41ModelState
    orig_inputs = cls.prepare_inputs

    def prepare_attn(self, input_batch):
        ms = getattr(self, "model_state", None)
        stager = getattr(ms, "engram_stager", None)
        if stager is not None and _STATE["armed"]:
            try:
                hook.launch(stager, ms, input_batch, self.req_states)
            except Exception as exc:  # noqa: BLE001 - stage() then hashes late
                stager._early_rec = None
                _disarm(exc)
        return orig_attn(self, input_batch)

    def prepare_inputs(self, input_batch, req_states):
        stager = self.engram_stager
        if stager is None:
            return orig_inputs(self, input_batch, req_states)
        stager._early_cur = input_batch
        try:
            return orig_inputs(self, input_batch, req_states)
        finally:
            stager._early_cur = None
            stager._early_rec = None

    GPUModelRunner.prepare_attn = prepare_attn
    cls.prepare_inputs = prepare_inputs
    return hook
