#!/usr/bin/env python3
"""All fusion-host levers on at once: the baked sitecustomize ran decode_levers.install()
at interpreter start (envs set by the caller); report what each patched and the dry-run
of the install lines. No GPU work. Prints one JSON line.

The sitecustomize text patches (engram_cpu_hash / engram_defer, gated by
DSV41_ENGRAM_CPU_HASH / DSV41_ENGRAM_DEFER) rewrite vLLM's files on disk in the first
interpreter of a container, where engram.py may already be imported; serve workers are
fresh interpreters that import the rewritten files. So this runs once more in a child
interpreter (the worker's view, "worker" in the JSON) after the first one ("first").
With either env set, the native stage must refuse there (one LOG_DISARMED line) and
EngramDiskStager.stage must take the input_batch keyword the rewired model_state passes."""
import inspect
import json
import os
import subprocess
import sys


def view():
    from vllm.models.deepseek_v4_1.attention import DeepseekV4Indexer
    from vllm.models.deepseek_v4_1.common.engram import Engram, EngramDiskStager
    from vllm.v1.attention.backend import CommonAttentionMetadata as CAM
    from vllm.v1.attention.backends.mla import sparse_swa as swa

    out = {
        "swa_meta_fused": bool(getattr(swa.DeepseekSparseSWAMetadataBuilder.build, "_dsv41_swa_fused", False)),
        "t2r_dedup": bool(getattr(CAM.token_to_req_indices, "_dsv41_t2r_dedup", False)),
        "wp_gemv": bool(getattr(DeepseekV4Indexer.__init__, "_dsv41_wp_gemv", False)),
        "native_stage": bool(getattr(EngramDiskStager.stage, "_dsv41_native", False)),
        "wkv_tp": bool(getattr(Engram.__init__, "_dsv41_wkv_tp", False)),
        "cpu_hash_defer_text": [n for n in ("_ch_try_stage", "_defer_try_stage") if hasattr(EngramDiskStager, n)],
    }
    sig = inspect.signature(EngramDiskStager.stage)
    out["stage_params"] = list(sig.parameters)
    try:  # the call the defer-rewired model_state.prepare_inputs makes every step
        sig.bind(None, "ids", "pos", "qsl", "lookback", 4, input_batch=None)
        out["stage_accepts_input_batch"] = True
    except TypeError as exc:
        out["stage_accepts_input_batch"] = repr(exc)
    out["modules"] = sorted(m for m in sys.modules if m in (
        "engram_native_stage", "engram_early_hash", "moe_prep_fused", "candidate_mask_bounded",
        "indexer_wp_gemv", "swa_meta_fused", "attn_t2r_dedup", "engram_wkv_tp"))
    return out


if os.environ.get("FH_DRYRUN_CHILD") == "1":
    print(json.dumps(view()))
else:
    first = view()
    sys.stdout.flush()
    child = subprocess.run([sys.executable, __file__], env=dict(os.environ, FH_DRYRUN_CHILD="1"),
                           capture_output=True, text=True)
    print("---- worker-view interpreter output:\n" + child.stdout + child.stderr, flush=True)
    lines = [ln for ln in child.stdout.splitlines() if ln.startswith("{")]
    worker = json.loads(lines[-1]) if lines else {"error": child.returncode}
    print(json.dumps({"first": first, "worker": worker}))
