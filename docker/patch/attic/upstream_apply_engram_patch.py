import sys
src=open('dsv41_engram.py').read()
marker = "import torch\nfrom torch import nn\n"
assert marker in src
src = src.replace(marker, "import torch\nfrom torch import nn\n\n# --- Tech2Wild/Kai 2026-09-10: disk-backed Engram tables for DGX Spark (UMA) ---\nimport json as _kai_json\nimport os as _kai_os\nimport struct as _kai_struct\nfrom concurrent.futures import ThreadPoolExecutor as _KaiPool\n", 1)

helper = open('engram_helper.py').read()
assert src.count("class ParallelEngramEmbedding(nn.Module):") == 1
src = src.replace("class ParallelEngramEmbedding(nn.Module):", helper + "\n\nclass ParallelEngramEmbedding(nn.Module):", 1)

old_sig = '''        block_size: int = 32,
        cpu_offload: bool = False,
    ):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()'''
new_sig = '''        block_size: int = 32,
        cpu_offload: bool = False,
        model_dir: str | None = None,
        layer_id: int | None = None,
    ):
        super().__init__()
        self.disk: DiskEngramTable | None = None
        if _DSV41_ENGRAM_DISK:
            cpu_offload = False
        tp_size = get_tensor_model_parallel_world_size()'''
assert old_sig in src, "sig"; src = src.replace(old_sig, new_sig, 1)

old_alloc = '''        kwargs = {"device": "cpu", "pin_memory": True} if cpu_offload else {}
        self.weight = nn.Parameter('''
new_alloc = '''        kwargs = {"device": "cpu", "pin_memory": True} if cpu_offload else {}
        if _DSV41_ENGRAM_DISK:
            assert model_dir is not None and layer_id is not None
            self.disk = DiskEngramTable(model_dir, layer_id, dim, block_size)
            # 1-row placeholders as BUFFERS (not Parameters): the checkpoint
            # rows are skipped by the weights iterator and must not trip the
            # "weights not initialized" check.
            self.register_buffer("weight", torch.zeros(1, dim, dtype=torch.float8_e4m3fn, device="cpu"), persistent=False)
            self.register_buffer("weight_scale_inv", torch.zeros(1, dim // block_size, dtype=torch.uint8, device="cpu"), persistent=False)
            logger.info(
                "Engram table DISK-backed: %d rows x %d per rank stay on disk (%.2f GiB not allocated)",
                self.part_num_embeddings, dim,
                self.part_num_embeddings * (dim + dim // block_size) / 1024**3,
            )
            return
        self.weight = nn.Parameter('''
assert old_alloc in src, "alloc"; src = src.replace(old_alloc, new_alloc, 1)

old_lookup = '''        rows = indices.shape[0] * self.part_n_hash_cols
        if not rows:
            return
        weight, scales = self._storage()'''
new_lookup = '''        rows = indices.shape[0] * self.part_n_hash_cols
        if not rows:
            return
        if self.disk is not None:
            self._disk_lookup(indices, out)
            return
        weight, scales = self._storage()'''
assert old_lookup in src, "lookup"; src = src.replace(old_lookup, new_lookup, 1)

old_fwd = '''    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: [num_tokens, n_hash_cols] -> [num_tokens, n_hash_cols, dim]'''
new_fwd = '''    def _disk_lookup(self, indices: torch.Tensor, out: torch.Tensor) -> None:
        """Host-side gather + dequant for DISK mode, then H2D into `out`
        ([T, local_heads, dim] bf16). Padded heads (beyond TOTAL_HEADS) and
        rows outside this rank's range write zeros, matching the kernel."""
        T = indices.shape[0]
        L = self.part_n_hash_cols
        ids = indices.detach().to("cpu", dtype=torch.int64)
        head_end = min(self.head_start + L, self.n_hash_cols)
        local = ids[:, self.head_start:head_end]
        if local.shape[1] < L:
            pad = torch.full((T, L - local.shape[1]), -1, dtype=torch.int64)
            local = torch.cat([local, pad], dim=1)
        rows = local.reshape(-1)
        owned = (rows >= self.vocab_start_idx) & (rows < self.vocab_end_idx)
        rel = torch.where(owned, rows - self.vocab_start_idx, torch.zeros_like(rows))
        assert self.disk is not None
        deq = self.disk.gather_dequant(rel, owned)
        out[:T].copy_(deq.view(T, L, self.dim))

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: [num_tokens, n_hash_cols] -> [num_tokens, n_hash_cols, dim]'''
assert old_fwd in src, "fwd"; src = src.replace(old_fwd, new_fwd, 1)

old_ctor = '''            cpu_offload=engram_config.cpu_offload if engram_config else True,
        )'''
new_ctor = '''            cpu_offload=engram_config.cpu_offload if engram_config else True,
            model_dir=get_current_vllm_config().model_config.model,
            layer_id=layout.layer_ids[layer_hash_index],
        )'''
assert src.count(old_ctor)==1, "ctor"; src = src.replace(old_ctor, new_ctor, 1)
open('patched/engram.py','w').write(src)

wu=open('dsv41_weight_utils.py').read()
old_it = '''        else:
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    if should_skip_weight(name, local_expert_ids):
                        continue
                    param = f.get_tensor(name)
                    yield name, param'''
new_it = '''        else:
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    if should_skip_weight(name, local_expert_ids):
                        continue
                    # Tech2Wild/Kai 2026-09-10: DSV41_ENGRAM_DISK=1 leaves the
                    # Engram tables on disk (see deepseek_v4_1/common/engram.py)
                    if _DSV41_ENGRAM_DISK and name.endswith((".engram.embed.weight", ".engram.embed.scale")):
                        continue
                    param = f.get_tensor(name)
                    yield name, param'''
assert old_it in wu, "iter"; wu = wu.replace(old_it, new_it, 1)
assert "\nimport os\n" in wu, "os import"
wu = wu.replace("\nimport os\n", "\nimport os\n\n_DSV41_ENGRAM_DISK = os.environ.get(\"DSV41_ENGRAM_DISK\", \"0\") == \"1\"\n", 1)
open('patched/weight_utils.py','w').write(wu)
print("patched ok")
