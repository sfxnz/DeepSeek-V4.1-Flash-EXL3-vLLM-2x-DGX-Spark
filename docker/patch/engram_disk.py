#!/usr/bin/env python3
"""Disk-backed Engram tables for DGX Spark unified memory.

Official vLLM stores Engram shards in pinned host RAM when cpu_offload=True.
On a Spark that pin is the same 121 GiB pool the GPU uses, so the two MXFP8
tables (~189 GiB) OOM before experts finish loading.

This reader keeps the safetensors shards on NVMe and preads only the hashed
rows. Dequant is CPU fp8 e4m3 x ue8m0 -> bf16, matching the Triton lookup
kernel in vllm.models.deepseek_v4_1.common.engram.

Enable with DSV41_ENGRAM_DISK=1. Lineage: tonyd2wild Engram-on-disk (2026-09-10)
and this lab's Qwen3.8 PLE-on-disk gather.
"""
from __future__ import annotations

import json
import os
import struct
from concurrent.futures import ThreadPoolExecutor

_DSV41_ENGRAM_DISK = os.environ.get("DSV41_ENGRAM_DISK", "0") == "1"


def _torch():
    import torch

    return torch


def engram_disk_enabled() -> bool:
    return os.environ.get("DSV41_ENGRAM_DISK", "0") == "1"


class DiskEngramTable:
    """Pread rows from safetensors shards. Never materializes the full table."""

    def __init__(self, model_dir: str, layer_id: int, dim: int, block_size: int):
        idx_path = os.path.join(model_dir, "model.safetensors.index.json")
        with open(idx_path, encoding="utf-8") as fh:
            weight_map = json.load(fh)["weight_map"]
        wname = f"layers.{layer_id}.engram.embed.weight"
        sname = f"layers.{layer_id}.engram.embed.scale"
        if wname not in weight_map or sname not in weight_map:
            raise FileNotFoundError(
                f"Engram tensors missing from index: {wname} / {sname}"
            )
        self.w_fd, self.w_off, self.w_shape = self._open(
            model_dir, weight_map[wname], wname
        )
        self.s_fd, self.s_off, self.s_shape = self._open(
            model_dir, weight_map[sname], sname
        )
        self.dim = dim
        self.sb = dim // block_size
        if self.w_shape[1] != dim:
            raise ValueError(f"engram weight width {self.w_shape} != dim {dim}")
        if self.s_shape[1] != self.sb:
            raise ValueError(f"engram scale width {self.s_shape} != {self.sb}")
        self.threads = int(os.environ.get("DSV41_ENGRAM_DISK_THREADS", "32"))
        self.chunk = int(os.environ.get("DSV41_ENGRAM_DISK_CHUNK", "16"))
        self.pool = ThreadPoolExecutor(max_workers=self.threads)

    @staticmethod
    def _open(model_dir: str, fname: str, tname: str):
        path = os.path.join(model_dir, fname)
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        except OSError:
            pass
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        meta = hdr[tname]
        start = meta["data_offsets"][0]
        return fd, 8 + n + start, tuple(meta["shape"])

    def _read_rows(self, fd: int, base: int, rel: list[int], row_bytes: int, buf) -> None:
        def work(lo: int, hi: int) -> None:
            for i in range(lo, hi):
                off = base + rel[i] * row_bytes
                view = buf[i * row_bytes : (i + 1) * row_bytes]
                got = 0
                while got < row_bytes:
                    n = os.preadv(fd, [view[got:]], off + got)
                    if n <= 0:
                        raise OSError("engram disk table: short read")
                    got += n

        n = len(rel)
        if n <= self.chunk:
            work(0, n)
            return
        futs = [
            self.pool.submit(work, lo, min(lo + self.chunk, n))
            for lo in range(0, n, self.chunk)
        ]
        for fut in futs:
            fut.result()

    def gather_dequant(self, rel, owned):
        """rel: [R] int64 CPU local row ids; owned: [R] bool. Returns [R, dim] bf16 CPU."""
        torch = _torch()
        r = int(rel.numel())
        w = torch.empty((r, self.dim), dtype=torch.uint8)
        s = torch.empty((r, self.sb), dtype=torch.uint8)
        rel_l = rel.tolist()
        self._read_rows(
            self.w_fd, self.w_off, rel_l, self.dim, memoryview(w.numpy()).cast("B")
        )
        self._read_rows(
            self.s_fd, self.s_off, rel_l, self.sb, memoryview(s.numpy()).cast("B")
        )
        vals = w.view(torch.float8_e4m3fn).to(torch.float32).view(r, self.sb, -1)
        scale = (s.to(torch.int32) << 23).view(torch.float32)
        out = (vals * scale[:, :, None]).reshape(r, self.dim)
        out[~owned] = 0
        return out.to(torch.bfloat16)


def is_engram_embed_tensor(name: str) -> bool:
    """Checkpoint names the disk reader (and the loader skip) must ignore."""
    return name.endswith((".engram.embed.weight", ".engram.embed.scale"))
