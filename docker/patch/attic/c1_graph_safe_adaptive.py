#!/usr/bin/env python3
"""c=1 DSpark adaptive verify pinned to draft_budget=2 (query_len=3).

Stock adaptive wants varlen decode + query-lens mismatch. SM120 sparse MLA
does not support that (CUDA IMA). At c=1 the batch is uniform, so we:
  * skip the ALWAYS/mismatch gates
  * keep varlen_decode=False
  * force draft_budget=2 when any drafts are scheduled

Essay accept is ~2.4 of 5, so bonus+2 covers the typical accept. Verifying
6 tokens pays for a rejected tail. Snapping to {0..5} still picked 5.
Prefill with zero scheduled drafts stays budget 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

CAPTURED_DRAFTS = (2,)
CAPTURE_SIZES = (1, 2, 3, 4, 5, 6, 10, 12)
PINNED_DRAFT_BUDGET = 2

GATE_OLD = """    if not enable_adaptive_verification:
        return None

    # The selector rejects unsupported backends"""
GATE_NEW = """    if not enable_adaptive_verification:
        return None

    if getattr(vllm_config.scheduler_config, "max_num_seqs", 99) <= 2:
        return AdaptiveVerificationManager(
            req_states,
            query_start_loc,
            num_bonus_tokens,
            max_total_logits=max_total_logits,
        )

    # The selector rejects unsupported backends"""

BUDGET_OLD = """        draft_budget = int(np.argmax(num_tokens_to_estimated_accepted_tokens / costs))
        self._batch_budget = ("""
BUDGET_NEW = """        draft_budget = int(np.argmax(num_tokens_to_estimated_accepted_tokens / costs))
        if int(scheduled_drafts.sum()) > 0:
            draft_budget = 2
        self._batch_budget = ("""

VARLEN_OLD = "            varlen_decode=self.adaptive_verification is not None,"
VARLEN_NEW = "            varlen_decode=False,"


def snap_draft_budget(
    raw: int, captured_drafts: tuple[int, ...] = CAPTURED_DRAFTS
) -> int:
    """Pin to captured drafts. Never AR when the captured set is nonempty.

    Prefill with no scheduled drafts is handled by the caller (keep 0).
    """
    if not captured_drafts:
        return 0
    floor = min(captured_drafts)
    if raw < floor:
        return floor
    return max((c for c in captured_drafts if c <= raw), default=floor)


def patch_adaptive(src: str) -> str:
    out = src
    if GATE_OLD in out:
        out = out.replace(GATE_OLD, GATE_NEW, 1)
    if BUDGET_OLD in out:
        out = out.replace(BUDGET_OLD, BUDGET_NEW, 1)
    if GATE_NEW not in out:
        raise SystemExit("c1_graph_safe_adaptive: c=1 gate skip not present")
    if "draft_budget = 2" not in out:
        raise SystemExit("c1_graph_safe_adaptive: pinned draft_budget=2 not present")
    return out


def patch_runner(src: str) -> str:
    out = src
    if VARLEN_OLD in out:
        out = out.replace(VARLEN_OLD, VARLEN_NEW, 1)
    if VARLEN_NEW not in out:
        raise SystemExit("c1_graph_safe_adaptive: varlen_decode=False not present")
    return out


def apply(tree: Path) -> bool:
    adaptive = tree / "v1/worker/gpu/spec_decode/adaptive_verification.py"
    runner = tree / "v1/worker/gpu/model_runner.py"
    if not adaptive.is_file():
        hits = list(tree.rglob("adaptive_verification.py"))
        if not hits:
            raise SystemExit(f"c1_graph_safe_adaptive: no adaptive_verification.py under {tree}")
        adaptive = hits[0]
    if not runner.is_file():
        hits = list(tree.rglob("v1/worker/gpu/model_runner.py"))
        if not hits:
            raise SystemExit(f"c1_graph_safe_adaptive: no model_runner.py under {tree}")
        runner = hits[0]
    changed = False
    for path, patch in ((adaptive, patch_adaptive), (runner, patch_runner)):
        src = path.read_text()
        out = patch(src)
        if out != src:
            path.write_text(out)
            print(f"patched {path}")
            changed = True
    return changed


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: c1_graph_safe_adaptive.py TREE", file=sys.stderr)
        return 2
    apply(Path(argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
