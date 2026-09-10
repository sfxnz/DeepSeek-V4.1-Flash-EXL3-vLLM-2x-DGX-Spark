#!/usr/bin/env python3
"""Patch official dsv41-flash Engram + weight iterator for disk tables.

Idempotent. Run against the files in the pinned vLLM image.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

MARKER = "DSV41_ENGRAM_DISK disk-backed Engram"

IMPORT_OLD = "import torch\nfrom torch import nn\n"
IMPORT_NEW = (
    "import torch\nfrom torch import nn\n\n"
    "# --- " + MARKER + " ---\n"
    "from vllm.models.deepseek_v4_1.common.engram_disk import (\n"
    "    DiskEngramTable,\n"
    "    engram_disk_enabled,\n"
    ")\n"
)

SIG_OLD = """        block_size: int = 32,
        cpu_offload: bool = False,
    ):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()"""

SIG_NEW = """        block_size: int = 32,
        cpu_offload: bool = False,
        model_dir: str | None = None,
        layer_id: int | None = None,
    ):
        super().__init__()
        self.disk: DiskEngramTable | None = None
        if engram_disk_enabled():
            cpu_offload = False
        tp_size = get_tensor_model_parallel_world_size()"""

ALLOC_OLD = """        kwargs = {"device": "cpu", "pin_memory": True} if cpu_offload else {}
        self.weight = nn.Parameter("""

ALLOC_NEW = """        kwargs = {"device": "cpu", "pin_memory": True} if cpu_offload else {}
        if engram_disk_enabled():
            if model_dir is None or layer_id is None:
                raise RuntimeError("DSV41_ENGRAM_DISK=1 needs model_dir and layer_id")
            self.disk = DiskEngramTable(model_dir, layer_id, dim, block_size)
            # 1-row placeholders as buffers so AutoWeightsLoader does not see
            # uninitialized Parameters. The iterator skips embed tensors.
            self.register_buffer(
                "weight",
                torch.zeros(1, dim, dtype=torch.float8_e4m3fn, device="cpu"),
                persistent=False,
            )
            self.register_buffer(
                "weight_scale_inv",
                torch.zeros(1, dim // block_size, dtype=torch.uint8, device="cpu"),
                persistent=False,
            )
            return
        self.weight = nn.Parameter("""

LOOKUP_OLD = """        rows = indices.shape[0] * self.part_n_hash_cols
        if not rows:
            return
        weight, scales = self._storage()"""

LOOKUP_NEW = """        rows = indices.shape[0] * self.part_n_hash_cols
        if not rows:
            return
        if self.disk is not None:
            self._disk_lookup(indices, out)
            return
        weight, scales = self._storage()"""

FWD_OLD = """    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        \"\"\"indices: [num_tokens, n_hash_cols] -> [num_tokens, n_hash_cols, dim]
        bf16, gathered from all shards for this replica's tokens.\"\"\""""

FWD_NEW = """    def _disk_lookup(self, indices: torch.Tensor, out: torch.Tensor) -> None:
        \"\"\"Host gather + dequant, then H2D into `out` ([T, local_heads, dim] bf16).\"\"\"
        t = indices.shape[0]
        local_heads = self.part_n_hash_cols
        ids = indices.detach().to("cpu", dtype=torch.int64)
        head_end = min(self.head_start + local_heads, self.n_hash_cols)
        local = ids[:, self.head_start:head_end]
        if local.shape[1] < local_heads:
            pad = torch.full((t, local_heads - local.shape[1]), -1, dtype=torch.int64)
            local = torch.cat([local, pad], dim=1)
        rows = local.reshape(-1)
        owned = (rows >= self.vocab_start_idx) & (rows < self.vocab_end_idx)
        # HF shards store the full unsharded table; file rows are global ids.
        file_rows = torch.where(owned, rows, torch.zeros_like(rows))
        assert self.disk is not None
        deq = self.disk.gather_dequant(file_rows, owned)
        out[:t].copy_(deq.view(t, local_heads, self.dim))

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        \"\"\"indices: [num_tokens, n_hash_cols] -> [num_tokens, n_hash_cols, dim]
        bf16, gathered from all shards for this replica's tokens.\"\"\""""

CTOR_OLD = """            cpu_offload=engram_config.cpu_offload if engram_config else True,
        )"""

CTOR_NEW = """            cpu_offload=False if engram_disk_enabled() else (
                engram_config.cpu_offload if engram_config else True
            ),
            model_dir=get_current_vllm_config().model_config.model,
            layer_id=layout.layer_ids[layer_hash_index],
        )"""

WU_OLD = """                    if should_skip_weight(name, local_expert_ids):
                        continue
                    param = f.get_tensor(name)
                    yield name, param"""

WU_NEW = """                    if should_skip_weight(name, local_expert_ids):
                        continue
                    # """ + MARKER + """
                    if os.environ.get("DSV41_ENGRAM_DISK", "0") == "1" and name.endswith(
                        (".engram.embed.weight", ".engram.embed.scale")
                    ):
                        continue
                    param = f.get_tensor(name)
                    yield name, param"""


def _once(src: str, old: str, new: str, label: str) -> str:
    if MARKER in src and old not in src:
        return src
    if old not in src:
        raise SystemExit(f"apply_engram_disk: missing marker {label}")
    return src.replace(old, new, 1)


def patch_engram(src: str) -> str:
    src = src.replace(
        "rel = torch.where(owned, rows - self.vocab_start_idx, torch.zeros_like(rows))\n"
        "        assert self.disk is not None\n"
        "        deq = self.disk.gather_dequant(rel, owned)",
        "file_rows = torch.where(owned, rows, torch.zeros_like(rows))\n"
        "        assert self.disk is not None\n"
        "        deq = self.disk.gather_dequant(file_rows, owned)",
    )
    if (
        MARKER in src
        and "DiskEngramTable" in src
        and "_disk_lookup" in src
        and "rows - self.vocab_start_idx" not in src
    ):
        return src
    src = _once(src, IMPORT_OLD, IMPORT_NEW, "import")
    src = _once(src, SIG_OLD, SIG_NEW, "sig")
    src = _once(src, ALLOC_OLD, ALLOC_NEW, "alloc")
    src = _once(src, LOOKUP_OLD, LOOKUP_NEW, "lookup")
    src = _once(src, FWD_OLD, FWD_NEW, "fwd")
    src = _once(src, CTOR_OLD, CTOR_NEW, "ctor")
    return src


def patch_weight_utils(src: str) -> str:
    if MARKER in src:
        return src
    if "import os" not in src:
        raise SystemExit("apply_engram_disk: weight_utils has no import os")
    return _once(src, WU_OLD, WU_NEW, "weight_iter")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engram", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument(
        "--disk-module",
        type=Path,
        default=Path(__file__).with_name("engram_disk.py"),
    )
    args = ap.parse_args()
    dest = args.engram.parent / "engram_disk.py"
    if args.disk_module.resolve() != dest.resolve():
        shutil.copy2(args.disk_module, dest)
    args.engram.write_text(patch_engram(args.engram.read_text()))
    args.weights.write_text(patch_weight_utils(args.weights.read_text()))
    print(f"apply_engram_disk: patched {args.engram} and {args.weights}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
