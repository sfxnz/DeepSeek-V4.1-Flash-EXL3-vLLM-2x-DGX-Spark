"""swa_meta_fused (DSV41_SWA_META_FUSED): source rewrite, routing, wiring, host only.

The GPU proof (48 builds of the rewritten build vs the image's on the same
batches incl. padded requests/tokens and short sequences, 20 CUDA-graph
replays with new inputs, the in-serve verify path and its sabotage fallback,
install() on the image class: all bit-exact; 15 builds 562 -> 431 us, 12 -> 9
GPU ops per build) is kernel_study/fusion_host/swa_meta_check.py. The stock
kernel and build are pinned from canonical-e13 in
tests/fixtures/sparse_swa_build_e13.pin.py.
"""

from __future__ import annotations

import ast
import sys
import textwrap
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "docker" / "patch"
FIX = ROOT / "tests" / "fixtures" / "sparse_swa_build_e13.pin.py"
sys.path.insert(0, str(PATCH))
import decode_levers as dl  # noqa: E402
import swa_meta_fused as smf  # noqa: E402


def pinned(name: str) -> str:
    src = FIX.read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            lines = src.splitlines(keepends=True)
            first = node.decorator_list[0].lineno if node.decorator_list else node.lineno
            return "".join(lines[first - 1 : node.end_lineno])
    raise KeyError(name)


class FT:
    """A named stand-in tensor that logs the ops the build issues on it."""

    dtype = "int32"

    def __init__(self, name, log, n=16):
        self.name, self.log, self.shape = name, log, (n, 1, 128)

    def __getitem__(self, key):
        return FT(f"{self.name}[{key.start}:{key.stop}]", self.log, self.shape[0])

    def __setitem__(self, key, value):
        self.log.append(("fill", f"{self.name}[{key.start}:{key.stop}]", value))

    def __ge__(self, other):
        self.log.append(("ge", self.name, other))
        return FT(f"({self.name} >= {other})", self.log)

    def copy_(self, src):
        self.log.append(("copy", self.name, src.name))
        return self


class Recorder:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def __call__(self, *args, **kwargs):
        self.log.append((self.name, [getattr(a, "name", a) for a in args], kwargs))


class FakeHelper:
    def __init__(self, ok, log):
        self.ok, self.log = ok, log

    def fused_ok(self, builder, cm, nd, np_):
        self.log.append(("fused_ok", nd, np_))
        return self.ok

    def decode(self, *args):
        self.log.append(("fused", [getattr(a, "name", a) for a in args]))


def rewritten_build(ok, log, split):
    ns = {
        "split_decodes_and_prefills": lambda cm, decode_threshold: split,
        "_COMPUTE_SWA_INDICES_AND_LENS_KERNEL": Recorder("swa_kernel", log),
        "_COMPUTE_DSPARK_NONCAUSAL_SWA_INDICES_KERNEL": Recorder("dspark_kernel", log),
        "DeepseekSparseSWAMetadata": lambda **kw: types.SimpleNamespace(**kw),
        "CommonAttentionMetadata": object,
        "torch": types.SimpleNamespace(zeros=lambda *a, **k: FT("noncausal_buf", log), int32="int32"),
        "_dsv41_swa": FakeHelper(ok, log),
    }
    for k in ("SWAONLY", "C4A", "C128A", "C1A", "C2A"):
        ns[f"_LAYER_TYPE_{k}"] = k
    src = textwrap.dedent(smf.patch_source(pinned("build")))
    exec(compile(src, "build", "exec", dont_inherit=True), ns)
    return ns["build"]


def builder(log):
    b = types.SimpleNamespace(
        decode_threshold=4, window_size=128, noncausal_index_width=256, is_dspark=True,
        decode_swa_indices_noncausal=None, _max_tokens=16, device="cuda", block_size=64,
        max_image_tokens=0, prefill_index_width=128,
    )
    for name in ("is_valid_token", "decode_swa_indices", "decode_swa_lens", "prefill_swa_indices",
                 "prefill_swa_lens", "token_to_req_indices"):
        setattr(b, name, FT(f"self.{name}", log))
    b._build_deepseek_v4_metadata = lambda *a: {}
    b.build_tile_scheduler = lambda n: {k: None for k in ("SWAONLY", "C4A", "C128A", "C1A", "C2A")}
    return b


def cm(log, causal=True, n=4):
    return types.SimpleNamespace(
        seq_lens=FT("seq_lens", log), seq_lens_cpu_upper_bound=None, query_start_loc=FT("qsl", log),
        query_start_loc_cpu=None, block_table_tensor=FT("block_table", log),
        slot_mapping=FT("slot_mapping", log, n), causal=causal, mm_req_doc_ranges=None, num_reqs=1,
        max_query_len=4, token_to_req_indices=lambda buf: FT("t2r", log),
    )


class RewriteTests(unittest.TestCase):
    def test_only_the_two_anchors_change(self) -> None:
        src = pinned("build")
        out = smf.patch_source(src)
        ast.parse(textwrap.dedent(out))
        self.assertEqual(out.replace(smf.DECODE_NEW, smf.DECODE_OLD).replace(smf.VALID_NEW, smf.VALID_OLD), src)
        self.assertEqual(out.count("_dsv41_swa.decode("), 1)
        self.assertEqual(out.count("_dsv41_swa.fused_ok("), 1)

    def test_class_scope_and_drift_are_refused(self) -> None:
        src = pinned("build")
        for bad in (src + "        super().build(0, None)\n", src + "        self.__cache = 1\n",
                    src.replace("is_valid_token.copy_(slot_mapping >= 0)", "is_valid_token.copy_(slot_mapping > 0)"),
                    src + smf.DECODE_OLD):
            with self.assertRaises(ValueError):
                smf.patch_source(bad)
        smf.patch_source(src + "        self.__dict__\n")  # dunders are fine

    def test_fused_decode_issues_one_call_instead_of_the_stock_ops(self) -> None:
        log = []
        b = builder(log)
        md = rewritten_build(True, log, (1, 0, 4, 0))(b, 0, cm(log))
        self.assertEqual(log[0], ("fused_ok", 4, 0))
        self.assertEqual([e[0] for e in log], ["fused_ok", "fused"])  # no compare/copy/fill/kernel
        args = log[1][1]
        self.assertIs(args[0], b)
        self.assertEqual(args[1:], [
            "self.decode_swa_indices", "qsl", "seq_lens", "t2r", "slot_mapping",
            "self.is_valid_token[None:4]", "block_table", 4])
        self.assertEqual(md.is_valid_token.name, "self.is_valid_token[None:4]")
        self.assertEqual(md.decode_swa_indices.name, "self.decode_swa_indices[None:4]")
        self.assertEqual(md.decode_swa_lens.name, "self.decode_swa_lens[None:4]")
        self.assertEqual(md.decode_swa_width, 128)

    def test_unfused_builds_are_the_stock_ops_in_order(self) -> None:
        cases = (
            # (causal, split, the ops after fused_ok)
            (True, (1, 0, 4, 0), ["ge", "copy", "fill", "swa_kernel"]),
            (False, (1, 0, 4, 0), ["ge", "copy", "fill", "dspark_kernel"]),
            (True, (0, 1, 0, 4), ["ge", "copy", "swa_kernel"]),
            (True, (1, 1, 1, 3), ["ge", "copy", "fill", "swa_kernel", "swa_kernel"]),
        )
        for causal, split, ops in cases:
            log = []
            rewritten_build(False, log, split)(builder(log), 0, cm(log, causal))
            self.assertEqual(log[0], ("fused_ok",) + split[2:], split)
            self.assertEqual([e[0] for e in log[1:]], ops, (causal, split))
            if split[2]:
                self.assertIn(("fill", f"self.decode_swa_lens[{split[2]}:None]", 0), log)

    def test_kernel_mirrors_the_stock_causal_arithmetic(self) -> None:
        stock = pinned("_compute_swa_indices_and_lens_kernel")
        mine = (PATCH / "swa_meta_fused.py").read_text()
        for s, m in (
            ("query_len = query_end - query_start", None),
            ("prefix_len = seq_len - query_len", None),
            ("pos = prefix_len + token_idx - query_start", "pos = prefix_len + pid - query_start"),
            ("start_pos = tl.maximum(pos - (window_size - 1) - left_add, 0)",
             "start_pos = tl.maximum(pos - (window_size - 1), 0)"),
            ("end_pos = pos + right + 1", "end_pos = pos + 1"),
            ("swa_len = end_pos - start_pos", None),
            ("block_indices = pos_offset // block_size", None),
            ("block_offsets = pos_offset % block_size", None),
            ("slot_ids = block_numbers * block_size + block_offsets", None),
            ("slot_ids = tl.where(offset < swa_len, slot_ids, -1)", None),
            ("mask=pos_offset < end_pos,", "mask=pos_offset < end_pos"),
            ("tl.store(swa_lens_ptr + pid, 0)", "tl.store(swa_lens + pid, 0)"),
        ):
            self.assertIn(s, stock, s)
            self.assertIn(m or s, mine, m or s)
        # no image: left = right = 0, so left_add = max(-(window - 1), 0) = 0
        self.assertIn("left = 0\n        right = 0", stock)
        self.assertIn("left_add = tl.maximum(left - (window_size - 1), 0)", stock)

    def test_build_call_sites_the_rewrite_relies_on(self) -> None:
        src = pinned("build")
        # the stock kernel call the fused path replaces: causal, width = window, offset 0
        self.assertIn("decode_swa_indices.shape[-1],", src)
        self.assertIn("num_tokens=num_decode_tokens,\n                    token_offset=0,", src)
        self.assertIn("self.decode_swa_lens,  # unused (HAS_IMAGE=False)", src)


class FusedOkTests(unittest.TestCase):
    def setUp(self) -> None:
        smf._STATE.update(armed=True, verify_left=0, engaged=True)
        self.h = smf.SwaFused(types.SimpleNamespace(int32="int32"), None, None, None)
        self.b = types.SimpleNamespace(decode_swa_indices=types.SimpleNamespace(dtype="int32"))

    def cm(self, causal=True, slots=4):
        return types.SimpleNamespace(causal=causal, slot_mapping=types.SimpleNamespace(shape=(slots,)))

    def test_only_causal_pure_decode(self) -> None:
        self.assertTrue(self.h.fused_ok(self.b, self.cm(), 4, 0))
        self.assertFalse(self.h.fused_ok(self.b, self.cm(), 0, 4))
        self.assertFalse(self.h.fused_ok(self.b, self.cm(), 1, 3))
        self.assertFalse(self.h.fused_ok(self.b, self.cm(causal=False), 4, 0))
        self.assertFalse(self.h.fused_ok(self.b, self.cm(causal=object()), 4, 0))  # tensor-valued flag
        self.assertFalse(self.h.fused_ok(self.b, self.cm(slots=3), 4, 0))
        self.b.decode_swa_indices.dtype = "int64"
        self.assertFalse(self.h.fused_ok(self.b, self.cm(), 4, 0))

    def test_disarmed_is_stock(self) -> None:
        smf._STATE["armed"] = False
        self.assertFalse(self.h.fused_ok(self.b, self.cm(), 4, 0))

    def test_launch_error_disarms_to_the_stock_ops(self) -> None:
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_current_stream_capturing=lambda: False))
        h = smf.SwaFused(torch, None, None, None)
        calls = []
        h._launch = lambda *a: (_ for _ in ()).throw(RuntimeError("boom"))
        h._stock = lambda *a: calls.append("stock")
        with redirect_stdout(StringIO()) as out:
            h.decode(*range(9))
        self.assertEqual(calls, ["stock"])
        self.assertFalse(smf._STATE["armed"])
        self.assertIn(smf.LOG_DISARMED, out.getvalue())

    def test_capture_never_runs_the_verify_ops(self) -> None:
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_current_stream_capturing=lambda: True))
        h = smf.SwaFused(torch, None, None, None)
        smf._STATE["verify_left"] = 8
        calls = []
        h._launch = lambda *a: calls.append("fused")
        h._stock = lambda *a: calls.append("stock")
        h.decode(*range(9))
        self.assertEqual(calls, ["fused"])
        self.assertEqual(smf._STATE["verify_left"], 8)


class WiringTests(unittest.TestCase):
    def test_env(self) -> None:
        self.assertFalse(smf.enabled({}))
        self.assertTrue(smf.enabled({"DSV41_SWA_META_FUSED": "1"}))
        self.assertEqual(smf.verify_calls({}), 8)
        self.assertEqual(smf.verify_calls({"DSV41_SWA_META_VERIFY": "0"}), 1)

    def test_decode_levers_step(self) -> None:
        self.assertIn('("swa-meta-fused", _install_swa_meta_fused)', (PATCH / "decode_levers.py").read_text())
        calls = []
        fake = types.ModuleType("swa_meta_fused")
        fake.install = lambda: calls.append(1) or "fused"
        with mock.patch.dict(sys.modules, {"swa_meta_fused": fake}), redirect_stdout(StringIO()) as out:
            dl._install_swa_meta_fused({})
            dl._install_swa_meta_fused({"DSV41_SWA_META_FUSED": "0"})
            self.assertEqual(calls, [])
            dl._install_swa_meta_fused({"DSV41_SWA_META_FUSED": "1"})
        self.assertEqual(calls, [1])
        self.assertIn("swa metadata fused: fused", out.getvalue())

    def test_forwarded_default_on(self) -> None:
        run = (ROOT / "run.sh").read_text()
        self.assertIn("  DSV41_SWA_META_FUSED=\n", run)  # default from the generated block
        self.assertIn('DSV41_SWA_META_FUSED="${DSV41_SWA_META_FUSED:-1}"\n', run)  # round 35
        self.assertIn("  DSV41_SWA_META_VERIFY=\n", run)

    def test_top_level_imports_are_stdlib_only(self) -> None:
        tree = ast.parse((PATCH / "swa_meta_fused.py").read_text())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        mods = {a.name for n in top if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module for n in top if isinstance(n, ast.ImportFrom)}
        self.assertEqual(mods, {"__future__", "os", "re"})


if __name__ == "__main__":
    unittest.main()
