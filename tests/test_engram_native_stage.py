"""engram_native.c + engram_native_stage.py on the host (CPU, numpy, gcc).

The C gather is checked against a numpy transcription of the stock torch
chain (fp8 e4m3fn -> f32, x 2^(e-127), bf16 round-to-nearest-even with
torch's NaN bits) on a synthetic table file. The torch-vs-C proof on the
real tables and all 65536 byte pairs is
kernel_study/fusion_host/engram_native_check.py (runs in the image).
"""

from __future__ import annotations

import ast
import ctypes
import errno
import os
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "docker" / "patch"
sys.path.insert(0, str(PATCH))
import engram_native_stage as ens  # noqa: E402
import engram_stage_fast  # noqa: E402

DIM, SB = 256, 8
W_OFF = 664  # the real shard's w offset: rows are not 256-aligned


def e4m3fn_lut() -> np.ndarray:
    """torch's float8_e4m3fn -> float32 (0x7F/0xFF -> 0x7FF00000/0xFFF00000)."""
    out = np.zeros(256, np.float32)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e, m = (b >> 3) & 0xF, b & 7
        out[b] = s * (m / 8.0) * 2.0**-6 if e == 0 else s * (1 + m / 8.0) * 2.0 ** (e - 7)
    bits = out.view(np.uint32)
    bits[0x7F], bits[0xFF] = 0x7FF00000, 0xFFF00000
    return out


LUT = e4m3fn_lut()


def bf16_bits(x: np.ndarray) -> np.ndarray:
    u = x.view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    nan = np.isnan(x)
    r[nan] = ((u[nan] >> 16) | 0x40).astype(np.uint16)
    return r


def ref_dequant(w: np.ndarray, s: np.ndarray) -> np.ndarray:
    """w [R, DIM] uint8, s [R, SB] uint8 -> [R, DIM] bf16 bits."""
    scale = (s.astype(np.uint32) << 23).view(np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        prod = LUT[w].reshape(len(w), SB, -1) * scale[:, :, None]
    return bf16_bits(prod.reshape(len(w), DIM).astype(np.float32))


class NativeGatherTests(unittest.TestCase):
    NROWS = 5000

    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("gcc") is None:
            raise unittest.SkipTest("gcc not on PATH")
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.so = ens.build(ens.SRC, tmp / "cache")
        cls.lib = ens.load(cls.so)
        cls.lib.eng_dequant_one.restype = ctypes.c_uint16
        cls.lib.eng_dequant_one.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint8]
        rng = np.random.default_rng(7)
        cls.w = rng.integers(0, 256, (cls.NROWS, DIM), dtype=np.uint8)
        cls.s = rng.integers(0, 256, (cls.NROWS, SB), dtype=np.uint8)
        cls.w[:3] = np.array([0x7F, 0xFF, 0x00], np.uint8)[:, None]  # NaN / -NaN / zero rows
        cls.s[:3] = np.array([0, 255, 255], np.uint8)[:, None]  # 0 * inf -> default NaN
        cls.s_off = W_OFF + cls.NROWS * DIM + 40
        path = tmp / "table.bin"
        with open(path, "wb") as fh:
            fh.write(b"\0" * W_OFF)
            fh.write(cls.w.tobytes())
            fh.write(b"\0" * 40)
            fh.write(cls.s.tobytes())
        cls.fd = os.open(path, os.O_RDONLY)
        cls.lut = np.ascontiguousarray(LUT)

    @classmethod
    def tearDownClass(cls) -> None:
        os.close(cls.fd)
        cls._tmp.cleanup()

    def gather(self, ids: np.ndarray, local_heads: int, v0: int, v1: int, flags: int = 1, fd=None, n_tok=None):
        ids = np.ascontiguousarray(ids, dtype=np.int32)
        n = ids.shape[0] if n_tok is None else n_tok
        rows = max(n * local_heads, 1)
        out = np.full((rows, DIM), 0xABCD, np.uint16)
        sw = np.zeros((rows, DIM), np.uint8)
        ss = np.zeros((rows, SB), np.uint8)
        st = (ctypes.c_int64 * ens.NSTATS)()
        fd = self.fd if fd is None else fd
        rc = self.lib.eng_gather_bf16(
            fd, W_OFF, fd, self.s_off, DIM, SB, v0, v1,
            ids.ctypes.data, ids.shape[1] if ids.ndim == 2 else 0, n, ids.shape[1], local_heads,
            self.lut.ctypes.data, out.ctypes.data, sw.ctypes.data, ss.ctypes.data, flags, st,
        )
        return rc, out[: n * local_heads], list(st)

    def reference(self, ids: np.ndarray, local_heads: int, v0: int, v1: int) -> np.ndarray:
        n, hv = ids.shape
        local = np.full((n, local_heads), -1, np.int64)
        local[:, :hv] = ids
        local = local.reshape(-1)
        owned = (local >= v0) & (local < v1)
        rows = np.where(owned, local, 0)
        out = ref_dequant(self.w[rows], self.s[rows])
        out[~owned] = 0
        return out

    def test_dequant_one_all_byte_pairs(self) -> None:
        w = np.repeat(np.arange(256, dtype=np.uint8)[None, :], 256, 0)
        s = np.repeat(np.arange(256, dtype=np.uint8)[:, None], SB, 1)
        ref = ref_dequant(w, s)  # row = scale byte, col = fp8 byte
        got = np.array(
            [[self.lib.eng_dequant_one(self.lut.ctypes.data, wb, sv) for wb in range(256)] for sv in range(256)],
            np.uint16,
        )
        self.assertEqual(int((got != ref).sum()), 0)

    def test_gather_matches_stock_semantics(self) -> None:
        rng = random.Random(3)
        for n in (1, 2, 3, 4, 5, 6, 7, 8, 64):
            for hv, lh in ((12, 12), (11, 12), (12, 16)):
                for flags in (0, 1):
                    v0, v1 = 3, self.NROWS - 7
                    ids = np.array(
                        [[rng.randrange(-2, self.NROWS) for _ in range(hv)] for _ in range(n)], np.int32
                    )
                    ids[0, 0] = min(2, ids[0, 0])  # unowned (below v0) or NaN rows
                    ids[-1, -1] = v0  # first owned row
                    rc, got, st = self.gather(ids, lh, v0, v1, flags)
                    self.assertEqual(rc, 0, (n, hv, lh, flags))
                    ref = self.reference(ids, lh, v0, v1)
                    self.assertTrue(np.array_equal(got, ref), (n, hv, lh, flags))
                    owned = int(((ids >= v0) & (ids < v1)).sum())
                    self.assertEqual(st[:2], [n * lh, owned])

    def test_nan_rows_keep_torch_bits(self) -> None:
        ids = np.array([[0, 1, 2]], np.int32)  # NaN, -NaN, 0 x inf
        rc, got, _ = self.gather(ids, 3, 0, 3)
        self.assertEqual(rc, 0)
        self.assertEqual({int(x) for x in got[0]}, {0x7FF0})
        self.assertEqual({int(x) for x in got[1]}, {0xFFF0})
        self.assertEqual({int(x) for x in got[2]}, {0x7FC0})

    def test_unowned_rows_are_zero_and_never_read(self) -> None:
        ids = np.array([[-1, 0, 1], [2, -5, 9]], np.int32)
        rc, got, st = self.gather(ids, 4, 100, 200)
        self.assertEqual(rc, 0)
        self.assertFalse(got.any())
        self.assertEqual(st[1:4], [0, 0, 0])  # owned, miss, syscalls

    def test_zero_tokens(self) -> None:
        rc, got, st = self.gather(np.zeros((0, 12), np.int32), 12, 0, 10)
        self.assertEqual((rc, got.shape[0], st[0]), (0, 0, 0))

    def test_errors_are_negative_errno(self) -> None:
        ids = np.array([[5]], np.int32)
        rc, _, _ = self.gather(ids, 1, 0, 10, fd=-1)
        self.assertEqual(rc, -errno.EBADF)
        # a row past EOF: the table geometry is wrong
        past = np.array([[self.NROWS + 10_000_000]], np.int32)
        rc, _, _ = self.gather(past, 1, 0, 1 << 30)
        self.assertEqual(rc, -errno.EIO)
        st = (ctypes.c_int64 * ens.NSTATS)()
        bad = self.lib.eng_gather_bf16(
            self.fd, W_OFF, self.fd, self.s_off, 250, SB, 0, 10, ids.ctypes.data, 1, 1, 1, 1,
            self.lut.ctypes.data, None, None, None, 1, st,
        )
        self.assertEqual(bad, -errno.EINVAL)  # dim % sb != 0

    def test_build_is_cached_by_source_hash(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            a = ens.build(ens.SRC, Path(d))
            mtime = a.stat().st_mtime_ns
            b = ens.build(ens.SRC, Path(d))
            self.assertEqual(a, b)
            self.assertEqual(b.stat().st_mtime_ns, mtime)
            self.assertEqual(a.name, ens.so_name(ens.SRC.read_bytes()))
            self.assertEqual([p.name for p in Path(d).iterdir()], [a.name])


class NativeStageModuleTests(unittest.TestCase):
    def test_env_parsing(self) -> None:
        self.assertFalse(ens.enabled({}))
        self.assertTrue(ens.enabled({"DSV41_ENGRAM_NATIVE_STAGE": "1"}))
        self.assertEqual(ens.max_tokens({}), 64)
        self.assertEqual(ens.max_tokens({"DSV41_ENGRAM_NATIVE_MAX_TOKENS": "8"}), 8)
        with self.assertRaises(ValueError):
            ens.max_tokens({"DSV41_ENGRAM_NATIVE_MAX_TOKENS": "0"})
        self.assertEqual(ens.verify_calls({}), 8)
        self.assertEqual(ens.verify_calls({"DSV41_ENGRAM_NATIVE_VERIFY": "0"}), 1)
        self.assertEqual(ens.census_every({}), 0)
        self.assertEqual(ens.census_every({"DSV41_ENGRAM_CENSUS": "1"}), 32)
        self.assertEqual(ens.census_every({"DSV41_ENGRAM_CENSUS": "1", "DSV41_ENGRAM_CENSUS_EVERY": "5"}), 5)

    def test_top_level_imports_are_stdlib_only(self) -> None:
        tree = ast.parse(Path(ens.__file__).read_text())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        mods = {a.name for n in top if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module for n in top if isinstance(n, ast.ImportFrom)}
        self.assertEqual(mods, {"__future__", "ctypes", "hashlib", "os", "subprocess", "time", "pathlib"})

    def test_hash_and_h2d_statements_match_the_stock_stage(self) -> None:
        # The native stage re-implements the stock stage around the gather;
        # these statements must stay verbatim (catches drift in STAGE_FAST).
        src = Path(ens.__file__).read_text()
        stock = engram_stage_fast.STAGE_FAST
        for stmt in (
            "n = min(int(num_tokens), self.max_tokens)",
            "if n <= 0 or not self.hash_state.ensure_cache():",
            "ids = input_ids[:n]",
            "host = self.hash_host[:n]",
            "host.copy_(hashes[:, :, self.head_start : self.head_end], non_blocking=True)",
            "self.hashes_ready.record()",
            "self.hashes_ready.synchronize()",
            "local = host[:, engram.layer_hash_index, :].to(torch.int64)",
            "file_rows, owned = engram.embed_tokens.disk_file_rows_owned(local)",
            "rows = engram.embed_tokens.disk.gather_dequant(file_rows, owned)",
            "dest = engram._staged_rows_for_ubatch()[:n]",
            "dest.copy_(",
            "self.num_staged = n",
        ):
            self.assertIn(stmt, stock, stmt)
            self.assertIn(stmt, src, stmt)
        for arg in ("positions[:n],", "image_sentinel_mask(ids),", "image_sentinel_mask(lookback_token_ids),"):
            self.assertIn(arg, stock)
            self.assertIn(arg, src)

    def test_stager_attributes_exist_in_the_image_fixture(self) -> None:
        fixture = (ROOT / "tests/fixtures/engram_e12.pin.py").read_text()
        cls = fixture[fixture.index("class EngramDiskStager:") :]
        for attr in ("self.hash_host =", "self.rows_host =", "self.local_heads =", "self.head_start =",
                     "self.head_end =", "self.dim =", "self.max_tokens =", "self.engrams ="):
            self.assertIn(attr, cls, attr)

    def test_decode_levers_wiring(self) -> None:
        # decode_levers.install() runs from the image's baked sitecustomize and
        # imports from the mounted patch dir, so no image rebuild is needed.
        import decode_levers

        src = (PATCH / "decode_levers.py").read_text()
        self.assertIn('("engram-native-stage", _install_engram_native_stage)', src)
        calls = []
        fake = type(sys)("engram_native_stage")
        fake.install = lambda: calls.append(1) or "wrapped"
        saved = sys.modules.get("engram_native_stage")
        sys.modules["engram_native_stage"] = fake
        try:
            decode_levers._install_engram_native_stage({})
            decode_levers._install_engram_native_stage({"DSV41_ENGRAM_NATIVE_STAGE": "0"})
            self.assertEqual(calls, [])
            decode_levers._install_engram_native_stage({"DSV41_ENGRAM_NATIVE_STAGE": "1"})
            self.assertEqual(calls, [1])
        finally:
            if saved is None:
                sys.modules.pop("engram_native_stage", None)
            else:
                sys.modules["engram_native_stage"] = saved
        site = (PATCH / "sitecustomize.py").read_text()
        self.assertNotIn("engram_native_stage", site)

if __name__ == "__main__":
    unittest.main()
