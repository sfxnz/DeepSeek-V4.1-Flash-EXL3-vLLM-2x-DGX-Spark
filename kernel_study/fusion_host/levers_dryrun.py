#!/usr/bin/env python3
"""All fusion-host levers on at once: the baked sitecustomize ran decode_levers.install()
at interpreter start (envs set by the caller); report what each patched and the dry-run
of the install lines. No GPU work. Prints one JSON line."""
import json

out = {}
from vllm.v1.attention.backends.mla import sparse_swa as swa  # noqa: E402
from vllm.v1.attention.backend import CommonAttentionMetadata as CAM  # noqa: E402
from vllm.models.deepseek_v4_1.attention import DeepseekV4Indexer  # noqa: E402

out["swa_meta_fused"] = bool(getattr(swa.DeepseekSparseSWAMetadataBuilder.build, "_dsv41_swa_fused", False))
out["t2r_dedup"] = bool(getattr(CAM.token_to_req_indices, "_dsv41_t2r_dedup", False))
out["wp_gemv"] = bool(getattr(DeepseekV4Indexer.__init__, "_dsv41_wp_gemv", False))
import sys  # noqa: E402

out["modules"] = sorted(m for m in sys.modules if m in (
    "engram_native_stage", "engram_early_hash", "moe_prep_fused", "candidate_mask_bounded",
    "indexer_wp_gemv", "swa_meta_fused"))
print(json.dumps(out))
