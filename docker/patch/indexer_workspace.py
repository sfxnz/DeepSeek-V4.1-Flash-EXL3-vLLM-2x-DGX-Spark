"""Right-size the sparse-indexer prefill gather workspace (GB10 UMA hygiene).

Stock vLLM sizes the indexer prefill buffer as max_model_len * 40 entries of
132 B, locked for the life of the process: 0.8 GiB at 128k context, 2.6 GiB
at 500k, ~5.3 GiB at 1M (per rank). The chunk planner only needs the summed
(compressed) prefix of the prefill requests it packs into one chunk.

Source (READ ONLY reference): mia-exl3-ref overlay/patch_sm120_block64.py:230-250
(patch tuple on v1/attention/backends/mla/indexer.py), used with
DSV41_INDEXER_PREFILL_FACTOR=1 in her .env.example:242-244.

Stock line verified in OUR base image (dsv41-flash-exl3-sm121:canonical-e12):
vllm/v1/attention/backends/mla/indexer.py:647 `return max_model_len * 40`.

Adaptation vs mia: her patched code falls back to
min(40, max_num_seqs) when the env is empty; ours keeps the stock factor 40
when the env is unset so the default is a true no-op (our chain's opt-in
convention). DSV41_INDEXER_PREFILL_FACTOR=1 gives her measured setting;
40 restores stock behaviour explicitly.

Source-rewrite patch (idempotent via marker, self-check read-back), applied
at boot from sitecustomize the same way prefer_b12x_mxfp8 is.
"""

from __future__ import annotations

from pathlib import Path

MARK = "dsv41-indexer-workspace"

OLD = """    # For DeepSeek-V3.2, the max_model_len is 163840.
    #   40 * 163840 * 132 = 865075200 bytes = 825 MB
    return max_model_len * 40
"""

NEW = """    # For DeepSeek-V3.2, the max_model_len is 163840.
    #   40 * 163840 * 132 = 865075200 bytes = 825 MB
    _factor = __import__("os").environ.get("DSV41_INDEXER_PREFILL_FACTOR", "")  # dsv41-indexer-workspace
    if _factor.strip():
        return max_model_len * max(1, int(_factor))
    return max_model_len * 40
"""


def apply(vllm_root: Path) -> bool:
    """Patch <vllm_root>/v1/attention/backends/mla/indexer.py. Idempotent."""
    path = vllm_root / "v1/attention/backends/mla/indexer.py"
    if not path.is_file():
        print(f"{MARK}: {path} missing; skipped", flush=True)
        return False
    src = path.read_text()
    if MARK in src:
        print(f"{MARK}: already applied", flush=True)
        return True
    if OLD not in src:
        print(f"{MARK}: stock marker not found; skipped", flush=True)
        return False
    path.write_text(src.replace(OLD, NEW, 1))
    # Self-check: marker landed and the file still parses.
    import ast

    out = path.read_text()
    assert MARK in out and "max_model_len * 40" in out
    ast.parse(out)
    print(f"{MARK}: applied ({path})", flush=True)
    return True
