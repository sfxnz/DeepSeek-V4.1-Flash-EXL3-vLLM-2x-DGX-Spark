"""Engram native stage: the per-step disk gather in C, off the GIL.

The stock EngramDiskStager.stage() (engram_stage_fast + engram_gather_v2)
runs, after hashes_ready.synchronize(), one _fast_stage_one per engram table
on the 16-thread stage pool and blocks on the futures. The GPU is idle for
that whole span (r3 profile: 3.2 ms/step idle at c=1, the largest idle block;
the main thread sits in fut.exception() while two pool threads trade the GIL
~200 times per step around per-row preadv calls and a torch dequant chain).

DSV41_ENGRAM_NATIVE_STAGE=1 replaces that span, for calls of at most
DSV41_ENGRAM_NATIVE_MAX_TOKENS tokens (default 64; decode is 1-8), with one
ctypes call per table into engram_native.c (compiled with gcc at the first
EngramDiskStager, cached by source hash): owned rows are read with
preadv2(RWF_NOWAIT) from the page cache, misses get WILLNEED together and a
blocking pread, and the fp8 x ue8m0 dequant writes bf16 straight into the
pinned staging buffer. The hash kernel, the sync, the H2D copies and every
larger (prefill) call are the stock code.

Safety: the first DSV41_ENGRAM_NATIVE_VERIFY calls (default 8, minimum 1)
also run the stock per-table gather and compare raw bf16 bits. Any mismatch,
error or unexpected layout prints one LOG_DISARMED line and every later call
takes the stock stage; the step that failed is refilled by the stock gather.
DSV41_ENGRAM_CENSUS=1 adds a [native-census] line every
DSV41_ENGRAM_CENSUS_EVERY calls (rows, page-cache misses, syscalls, us).

Top-level imports are stdlib only. decode_levers.install() (called by the
baked sitecustomize) calls install() here when the env is on.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import time
from pathlib import Path

# Boot-log markers for tools/engagement_audit.py.
LOG_ENGAGED = "dsv41: engram native stage self-check bit-exact"
LOG_DISARMED = "dsv41: engram native stage DISABLED ->"

ENV_ON = "DSV41_ENGRAM_NATIVE_STAGE"
SRC = Path(__file__).with_name("engram_native.c")
CACHE_DIR = Path("/tmp/dsv41-native")  # container-local; rebuilt once per container
ABI_VERSION = 1
NSTATS = 5  # rows, owned, miss, syscalls, nowait (engram_native.c)
DEFAULT_MAX_TOKENS = 64
DEFAULT_VERIFY = 8

_LIB = None
_STATE = {"armed": True, "verify_left": DEFAULT_VERIFY, "engaged": False}


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return env.get("DSV41_ENGRAM_NATIVE_STAGE", "0") == "1"


def max_tokens(env=None) -> int:
    env = os.environ if env is None else env
    n = int(env.get("DSV41_ENGRAM_NATIVE_MAX_TOKENS", "") or DEFAULT_MAX_TOKENS)
    if n < 1:
        raise ValueError(f"DSV41_ENGRAM_NATIVE_MAX_TOKENS={n} must be >= 1")
    return n


def verify_calls(env=None) -> int:
    """Initial calls checked against the stock gather; at least one."""
    env = os.environ if env is None else env
    return max(1, int(env.get("DSV41_ENGRAM_NATIVE_VERIFY", "") or DEFAULT_VERIFY))


def census_every(env=None) -> int:
    """0 = no census line. Shares the engram census switches."""
    env = os.environ if env is None else env
    if env.get("DSV41_ENGRAM_CENSUS", "0") != "1":
        return 0
    return max(1, int(env.get("DSV41_ENGRAM_CENSUS_EVERY", "32") or 32))


def so_name(src: bytes) -> str:
    return "engram_native-%s.so" % hashlib.sha256(src).hexdigest()[:16]


def compile_cmd(src: Path, out: Path) -> list[str]:
    return ["gcc", "-O3", "-shared", "-fPIC", "-o", str(out), str(src)]


def build(src: Path = SRC, cache_dir: Path = CACHE_DIR) -> Path:
    """Compile src once per content hash; the .so is renamed into place."""
    cache_dir = Path(cache_dir)
    so = cache_dir / so_name(src.read_bytes())
    if so.is_file():
        return so
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = so.with_name(f".{so.name}.{os.getpid()}.tmp")
    try:
        res = subprocess.run(compile_cmd(src, tmp), capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"gcc failed: {res.stderr.strip()[-400:]}")
        os.replace(tmp, so)
    finally:
        tmp.unlink(missing_ok=True)
    return so


def load(so: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(so))
    lib.eng_abi_version.restype = ctypes.c_int
    if lib.eng_abi_version() != ABI_VERSION:
        raise RuntimeError(f"{so}: ABI {lib.eng_abi_version()} != {ABI_VERSION}")
    lib.eng_gather_bf16.restype = ctypes.c_int
    lib.eng_gather_bf16.argtypes = [
        ctypes.c_int, ctypes.c_int64, ctypes.c_int, ctypes.c_int64,  # w_fd w_off s_fd s_off
        ctypes.c_int, ctypes.c_int, ctypes.c_int64, ctypes.c_int64,  # dim sb vocab range
        ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # ids
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # lut out sw ss
        ctypes.c_int, ctypes.c_void_p,  # flags stats
    ]
    return lib


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = load(build())
    return _LIB


def _disarm(exc: BaseException) -> None:
    if _STATE["armed"]:
        _STATE["armed"] = False
        print("dsv41: engram native stage DISABLED -> stock stage: %r" % (exc,), flush=True)


class _NativeStager:
    """Per-stager table descriptors, scratch and counters."""

    def __init__(self, stager, torch):
        self.torch = torch
        self.lib = _lib()
        hh = stager.hash_host
        if hh.dtype != torch.int32 or hh.dim() != 3 or not hh.is_contiguous():
            raise ValueError(f"hash_host layout {tuple(hh.shape)} {hh.dtype}")
        self.nl = hh.shape[1]
        self.heads_valid = stager.head_end - stager.head_start
        self.local_heads = stager.local_heads
        self.dim = stager.dim
        if hh.shape[2] != self.heads_valid or not 0 < self.heads_valid <= self.local_heads:
            raise ValueError(f"heads {hh.shape[2]} / {self.heads_valid} / {self.local_heads}")
        self.max_tokens = min(stager.max_tokens, max_tokens())
        self.tables = []
        for engram, buf in zip(stager.engrams, stager.rows_host):
            emb, disk = engram.embed_tokens, engram.embed_tokens.disk
            if buf.dtype != torch.bfloat16 or not buf.is_contiguous() or tuple(buf.shape[1:]) != (
                self.local_heads,
                self.dim,
            ):
                raise ValueError(f"rows_host layout {tuple(buf.shape)} {buf.dtype}")
            if disk.dim != self.dim or disk.dim % disk.sb or not 0 <= engram.layer_hash_index < self.nl:
                raise ValueError(f"table geometry dim={disk.dim} sb={disk.sb}")
            self.tables.append(
                (disk.w_fd, disk.w_off, disk.s_fd, disk.s_off, disk.dim, disk.sb,
                 emb.vocab_start_idx, emb.vocab_end_idx, engram.layer_hash_index, buf)
            )
        sb = max(t[5] for t in self.tables)
        rows = self.max_tokens * self.local_heads
        self.sw = torch.empty((rows, self.dim), dtype=torch.uint8)
        self.ss = torch.empty((rows, sb), dtype=torch.uint8)
        self.lut = (
            torch.arange(256, dtype=torch.int32).to(torch.uint8)
            .view(torch.float8_e4m3fn).to(torch.float32).contiguous()
        )
        self.stats = (ctypes.c_int64 * NSTATS)()
        self.census_every = census_every()
        self.acc = [0, 0, 0, 0, 0.0]  # calls, rows, misses, syscalls, seconds

    def gather(self, stager, n: int) -> None:
        """Fill rows_host[i][:n] for every table from hash_host[:n]."""
        t0 = time.perf_counter()
        base = stager.hash_host.data_ptr()
        stride = self.nl * self.heads_valid
        rows = misses = calls = 0
        for w_fd, w_off, s_fd, s_off, dim, sb, v0, v1, layer, buf in self.tables:
            rc = self.lib.eng_gather_bf16(
                w_fd, w_off, s_fd, s_off, dim, sb, v0, v1,
                base + layer * self.heads_valid * 4, stride, n, self.heads_valid,
                self.local_heads, self.lut.data_ptr(), buf.data_ptr(),
                self.sw.data_ptr(), self.ss.data_ptr(), 1, self.stats,
            )
            if rc:
                raise OSError(-rc, f"eng_gather_bf16 layer {layer}: {os.strerror(-rc)}")
            rows += self.stats[0]
            misses += self.stats[2]
            calls += self.stats[3]
        if self.census_every:
            a = self.acc
            a[0] += 1
            a[1] += rows
            a[2] += misses
            a[3] += calls
            a[4] += time.perf_counter() - t0
            if a[0] % self.census_every == 0:
                print(
                    "[native-census] calls=%d rows/call=%.1f miss/call=%.2f "
                    "syscalls/call=%.1f us/call=%.1f"
                    % (a[0], a[1] / a[0], a[2] / a[0], a[3] / a[0], 1e6 * a[4] / a[0]),
                    flush=True,
                )
                self.acc = [0, 0, 0, 0, 0.0]

    def verify(self, stager, n: int) -> None:
        """Compare against the stock per-table gather; raise on any bit."""
        torch = self.torch
        host = stager.hash_host[:n]
        for engram, buf in zip(stager.engrams, stager.rows_host):
            ref = _stock_rows(engram, host, n, stager.local_heads, stager.dim)
            if not torch.equal(ref.view(torch.int16), buf[:n].view(torch.int16)):
                raise RuntimeError(
                    f"native != stock (layer {engram.layer_hash_index}, n={n})"
                )


def _stock_rows(engram, host, n, local_heads, dim):
    """The engram_stage_fast _fast_stage_one body, into a fresh tensor."""
    import torch

    local = host[:, engram.layer_hash_index, :].to(torch.int64)
    file_rows, owned = engram.embed_tokens.disk_file_rows_owned(local)
    rows = engram.embed_tokens.disk.gather_dequant(file_rows, owned)
    return rows.view(n, local_heads, dim)


def _make_stage(orig_stage, torch, image_sentinel_mask):
    pf_dump = os.environ.get("DSV41_ENGRAM_PF_DUMP", "0") == "1"

    @torch.inference_mode()
    def stage(self, input_ids, positions, query_start_loc, lookback_token_ids, num_tokens):
        nat = getattr(self, "_eng_native", None)
        n = min(int(num_tokens), self.max_tokens)
        if nat is None or not _STATE["armed"] or n > nat.max_tokens or pf_dump:
            return orig_stage(
                self, input_ids, positions, query_start_loc, lookback_token_ids, num_tokens
            )
        if n <= 0 or not self.hash_state.ensure_cache():
            return 0
        # Stock hash + sync (engram_stage_fast STAGE_FAST, verbatim).
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
        try:
            nat.gather(self, n)
            if _STATE["verify_left"] > 0:
                nat.verify(self, n)
                _STATE["verify_left"] -= 1
                if not _STATE["engaged"]:
                    _STATE["engaged"] = True
                    print(
                        "dsv41: engram native stage self-check bit-exact "
                        "(%d tables, n=%d, max_tokens=%d)" % (len(nat.tables), n, nat.max_tokens),
                        flush=True,
                    )
        except Exception as exc:  # noqa: BLE001 - this step falls back below
            _disarm(exc)
            for engram, buf in zip(self.engrams, self.rows_host):
                buf[:n].copy_(_stock_rows(engram, host, n, self.local_heads, self.dim))
        for engram, buf in zip(self.engrams, self.rows_host):
            dest = engram._staged_rows_for_ubatch()[:n]
            dest.copy_(buf[:n], non_blocking=True)
        self.num_staged = n
        return n

    stage._dsv41_native = True
    return stage


def install() -> str:
    """Wrap EngramDiskStager (__init__ builds the tables, stage() gathers)."""
    import torch
    from vllm.models.deepseek_v4_1.common import engram as eng
    from vllm.models.deepseek_v4_1.common.mm_preprocess import image_sentinel_mask

    cls = eng.EngramDiskStager
    if getattr(cls.stage, "_dsv41_native", False):
        return "already installed"
    _STATE["verify_left"] = verify_calls()
    max_tokens()  # validate the env before arming
    orig_init, orig_stage = cls.__init__, cls.stage

    def __init__(self, hash_state, engrams):
        orig_init(self, hash_state, engrams)
        self._eng_native = None
        if not _STATE["armed"]:
            return
        try:
            self._eng_native = _NativeStager(self, torch)
            print(
                "dsv41: engram native stage armed (%d tables, max_tokens=%d, verify=%d)"
                % (len(self._eng_native.tables), self._eng_native.max_tokens, _STATE["verify_left"]),
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            _disarm(exc)

    cls.__init__ = __init__
    cls.stage = _make_stage(orig_stage, torch, image_sentinel_mask)
    return f"EngramDiskStager wrapped (max_tokens={max_tokens()}, verify={_STATE['verify_left']})"
