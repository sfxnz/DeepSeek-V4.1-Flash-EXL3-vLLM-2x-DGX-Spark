#!/usr/bin/env python3
"""Parallel Engram disk staging across per-layer tables.

Census (DSV41_ENGRAM_CENSUS=1) + trace forensics on the L.A.I.L decode
step: the GPU idles 15-28 ms between decode steps, and the innermost frame
spanning those gaps is `_thread.lock.acquire` under
`engram_disk._read_rows -> Future.result`. The EngramDiskStager stages 13
per-layer tables SERIALLY; most gathers are page-cache warm (~0.5 ms) but
one or two per step hit cold rows and block 3-15 ms on NVMe while the GPU
sits idle. 717 ms of lock-wait over a 28-step window = ~25.6 ms/step.

This patch runs the per-table CPU work (pread + dequant into the pinned
host row buffer) concurrently on a small pool. H2D copies stay on the
calling thread so stream ordering before the graph replay is unchanged;
only WHO performs the CPU-side gather changes, and each table touches
disjoint buffers. A first-call self-test compares the parallel results
against the stock serial loop and permanently reverts on any mismatch.

Enable with DSV41_ENGRAM_FAST_STAGE=1 (default on). Idempotent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "# --- engram-fast-stage ---"

STAGE_ANCHOR = """    @torch.inference_mode()
    def stage(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        lookback_token_ids: torch.Tensor,
        num_tokens: int,
    ) -> int:
        n = min(int(num_tokens), self.max_tokens)
        if n <= 0 or not self.hash_state.ensure_cache():
            return 0
        from .mm_preprocess import image_sentinel_mask

        ids = input_ids[:n]
        hashes = self.hash_state(
            ids,
            positions[:n],
            query_start_loc,
            image_sentinel_mask(ids),
            lookback_token_ids,
            image_sentinel_mask(lookback_token_ids),
            None,
            None,
        )
        host = self.hash_host[:n]
        host.copy_(hashes[:, :, self.head_start : self.head_end], non_blocking=True)
        self.hashes_ready.record()
        self.hashes_ready.synchronize()
        for engram, buf in zip(self.engrams, self.rows_host):
            local = host[:, engram.layer_hash_index, :].to(torch.int64)
            file_rows, owned = engram.embed_tokens.disk_file_rows_owned(local)
            rows = engram.embed_tokens.disk.gather_dequant(file_rows, owned)
            staged = buf[:n]
            staged.copy_(rows.view(n, self.local_heads, self.dim))
            dest = engram._staged_rows_for_ubatch()[:n]
            dest.copy_(staged, non_blocking=True)
        self.num_staged = n
        return n
"""

STAGE_FAST = """    @torch.inference_mode()
    def stage(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        lookback_token_ids: torch.Tensor,
        num_tokens: int,
    ) -> int:
        n = min(int(num_tokens), self.max_tokens)
        if n <= 0 or not self.hash_state.ensure_cache():
            return 0
        from .mm_preprocess import image_sentinel_mask

        ids = input_ids[:n]
        import os as _pf_os

        if _pf_os.environ.get("DSV41_ENGRAM_PF_DUMP", "0") == "1":
            _win = (
                lookback_token_ids[0].tolist()
                if lookback_token_ids is not None
                and lookback_token_ids.numel()
                else []
            )
            print(
                "[pf-dump-stage] n=%d ids=%s pos=%s win=%s"
                % (n, ids.tolist()[:8], positions[:n].tolist()[:8], _win),
                flush=True,
            )
        hashes = self.hash_state(
            ids,
            positions[:n],
            query_start_loc,
            image_sentinel_mask(ids),
            lookback_token_ids,
            image_sentinel_mask(lookback_token_ids),
            None,
            None,
        )
        host = self.hash_host[:n]
        host.copy_(hashes[:, :, self.head_start : self.head_end], non_blocking=True)
        self.hashes_ready.record()
        self.hashes_ready.synchronize()

        def _fast_stage_one(engram, buf):
            # CPU-side gather + dequant only; the H2D copy_ stays on the
            # calling thread so it keeps the graph replay's stream order.
            local = host[:, engram.layer_hash_index, :].to(torch.int64)
            file_rows, owned = engram.embed_tokens.disk_file_rows_owned(local)
            rows = engram.embed_tokens.disk.gather_dequant(file_rows, owned)
            staged = buf[:n]
            staged.copy_(rows.view(n, self.local_heads, self.dim))

        pairs = list(zip(self.engrams, self.rows_host))
        if (
            not _ENG_FAST_STAGE_OK[0]
            or len(pairs) == 1
        ):
            for engram, buf in pairs:
                _fast_stage_one(engram, buf)
        else:
            futs = [
                _ENG_STAGE_POOL.submit(_fast_stage_one, engram, buf)
                for engram, buf in pairs
            ]
            first_exc = None
            for fut in futs:
                exc = fut.exception()
                if exc is not None and first_exc is None:
                    first_exc = exc
            if first_exc is not None:
                _ENG_FAST_STAGE_OK[0] = False
                print(
                    "dsv41: engram fast stage disabled after worker error: "
                    f"{first_exc!r}",
                    flush=True,
                )
                raise first_exc

        if _ENG_FAST_STAGE_SELFCHECK[0]:
            _ENG_FAST_STAGE_SELFCHECK[0] = False
            try:
                ref = [
                    (
                        engram._staged_rows_for_ubatch()[:n].clone(),
                        buf[:n].clone(),
                    )
                    for engram, buf in pairs
                ]
                for (engram, buf), (dest_ref, staged_ref) in zip(pairs, ref):
                    _fast_stage_one(engram, buf)
                    dest = engram._staged_rows_for_ubatch()[:n]
                    if not torch.equal(dest, dest_ref) or not torch.equal(
                        buf[:n], staged_ref
                    ):
                        raise RuntimeError(
                            "engram fast stage self-check mismatch"
                        )
                print(
                    "dsv41: engram fast stage self-check bit-exact "
                    f"({len(pairs)} tables, n={n})",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                _ENG_FAST_STAGE_OK[0] = False
                print(
                    "dsv41: engram fast stage REVERTED to serial: "
                    f"{exc!r}",
                    flush=True,
                )

        for engram, buf in pairs:
            staged = buf[:n]
            dest = engram._staged_rows_for_ubatch()[:n]
            dest.copy_(staged, non_blocking=True)
        self.num_staged = n
        return n
"""

INIT_ANCHOR = """        self.hashes_ready = torch.cuda.Event()
        self.num_staged = 0
"""

INIT_FAST = """        self.hashes_ready = torch.cuda.Event()
        self.num_staged = 0
""" + MARKER + """
        import os as _os
        from concurrent.futures import ThreadPoolExecutor as _TPE

        global _ENG_STAGE_POOL, _ENG_FAST_STAGE_OK, _ENG_FAST_STAGE_SELFCHECK
        if _ENG_STAGE_POOL is None:
            _ENG_STAGE_POOL = _TPE(
                max_workers=int(_os.environ.get("DSV41_ENGRAM_STAGE_THREADS", "16"))
            )
"""


def apply(model_root: Path) -> None:
    engram = model_root / "common" / "engram.py"
    text = engram.read_text()
    if MARKER in text:
        return
    if STAGE_ANCHOR not in text:
        raise SystemExit("engram_stage_fast: stage() anchor missing")
    if INIT_ANCHOR not in text:
        raise SystemExit("engram_stage_fast: __init__ anchor missing")

    # Module-level pool + switches, inserted above the stager class.
    cls = "class EngramDiskStager:"
    if cls not in text:
        raise SystemExit("engram_stage_fast: stager class anchor missing")
    mod_block = (
        MARKER
        + "\n"
        "import os as _eng_os\n"
        "_ENG_FAST_STAGE_OK = [\n"
        "    _eng_os.environ.get(\"DSV41_ENGRAM_FAST_STAGE\", \"1\") == \"1\"\n"
        "]\n"
        "_ENG_FAST_STAGE_SELFCHECK = [True]\n"
        "_ENG_STAGE_POOL = None\n"
        "\n"
        "\n"
    )
    text = text.replace(cls, mod_block + cls, 1)
    text = text.replace(INIT_ANCHOR, INIT_FAST, 1)
    text = text.replace(STAGE_ANCHOR, STAGE_FAST, 1)
    engram.write_text(text)
    print("dsv41: engram parallel stage installed")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "model_root",
        type=Path,
        help=".../vllm/models/deepseek_v4_1",
    )
    args = p.parse_args()
    apply(args.model_root)


if __name__ == "__main__":
    main()
