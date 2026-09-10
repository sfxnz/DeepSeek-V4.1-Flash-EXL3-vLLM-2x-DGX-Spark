_DSV41_ENGRAM_DISK = _kai_os.environ.get("DSV41_ENGRAM_DISK", "0") == "1"


class DiskEngramTable:
    """Tech2Wild/Kai 2026-09-10: read Engram rows straight from the safetensors
    shards with positional preads on a thread pool, so the per-rank Engram
    shard (~47 GiB at TP4) never occupies memory. Built for DGX Spark, where
    "pinned host memory" is the same 128 GB pool the GPU uses. Rows are
    dequantized on the CPU (fp8 e4m3 x ue8m0 block scales -> bf16) and copied
    into the GPU staging buffer. Enable with DSV41_ENGRAM_DISK=1.
    Lineage: our own PLE-on-disk patch for Qwen3.8-Flash-Next (2026-09-05).
    """

    def __init__(self, model_dir: str, layer_id: int, dim: int, block_size: int):
        idx_path = _kai_os.path.join(model_dir, "model.safetensors.index.json")
        with open(idx_path) as f:
            weight_map = _kai_json.load(f)["weight_map"]
        wname = f"layers.{layer_id}.engram.embed.weight"
        sname = f"layers.{layer_id}.engram.embed.scale"
        self.w_fd, self.w_off, self.w_shape = self._open(model_dir, weight_map[wname], wname)
        self.s_fd, self.s_off, self.s_shape = self._open(model_dir, weight_map[sname], sname)
        self.dim = dim
        self.sb = dim // block_size
        assert self.w_shape[1] == dim, (self.w_shape, dim)
        assert self.s_shape[1] == self.sb, (self.s_shape, self.sb)
        self.threads = int(_kai_os.environ.get("DSV41_ENGRAM_DISK_THREADS", "32"))
        self.chunk = int(_kai_os.environ.get("DSV41_ENGRAM_DISK_CHUNK", "16"))
        self.pool = _KaiPool(max_workers=self.threads)
        logger.info(
            "Engram DISK mode: layer %d rows read from %s (off=%d) and %s (off=%d); "
            "%d threads, chunk %d",
            layer_id, weight_map[wname], self.w_off, weight_map[sname], self.s_off,
            self.threads, self.chunk,
        )

    @staticmethod
    def _open(model_dir: str, fname: str, tname: str):
        path = _kai_os.path.join(model_dir, fname)
        fd = _kai_os.open(path, _kai_os.O_RDONLY)
        try:
            _kai_os.posix_fadvise(fd, 0, 0, _kai_os.POSIX_FADV_RANDOM)
        except Exception:  # noqa: BLE001
            pass
        with open(path, "rb") as f:
            n = _kai_struct.unpack("<Q", f.read(8))[0]
            hdr = _kai_json.loads(f.read(n))
        meta = hdr[tname]
        start = meta["data_offsets"][0]
        return fd, 8 + n + start, tuple(meta["shape"])

    def _read_rows(self, fd: int, base: int, rel: list, row_bytes: int, buf) -> None:
        def work(lo: int, hi: int) -> None:
            for i in range(lo, hi):
                off = base + rel[i] * row_bytes
                view = buf[i * row_bytes : (i + 1) * row_bytes]
                got = 0
                while got < row_bytes:
                    n = _kai_os.preadv(fd, [view[got:]], off + got)
                    if n <= 0:
                        raise OSError("engram disk table: short read")
                    got += n

        n = len(rel)
        if n <= self.chunk:
            work(0, n)
            return
        futs = [self.pool.submit(work, lo, min(lo + self.chunk, n)) for lo in range(0, n, self.chunk)]
        for fut in futs:
            fut.result()

    def gather_dequant(self, rel: torch.Tensor, owned: torch.Tensor) -> torch.Tensor:
        """rel: [R] int64 CPU local row ids; owned: [R] bool. Returns [R, dim] bf16 CPU."""
        R = rel.numel()
        w = torch.empty((R, self.dim), dtype=torch.uint8)
        s = torch.empty((R, self.sb), dtype=torch.uint8)
        rel_l = rel.tolist()
        self._read_rows(self.w_fd, self.w_off, rel_l, self.dim, memoryview(w.numpy()).cast("B"))
        self._read_rows(self.s_fd, self.s_off, rel_l, self.sb, memoryview(s.numpy()).cast("B"))
        vals = w.view(torch.float8_e4m3fn).to(torch.float32).view(R, self.sb, -1)
        # ue8m0 byte is the fp32 exponent field: 2^(e-127)
        scale = (s.to(torch.int32) << 23).view(torch.float32)
        out = (vals * scale[:, :, None]).reshape(R, self.dim)
        out[~owned] = 0
        return out.to(torch.bfloat16)
