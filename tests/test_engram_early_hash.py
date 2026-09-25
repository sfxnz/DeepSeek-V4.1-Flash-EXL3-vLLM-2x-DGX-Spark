"""engram_early_hash (DSV41_ENGRAM_EARLY_HASH): launch/consume rules on the host.

No torch here: fakes stand in for the stager, the batch and the kernels. The
GPU proof (the image's lookback kernel, early vs late hash equal, staged rows
bit-exact, GPU idle 496.5 -> 322.9 us at n=4) is
kernel_study/fusion_host/engram_stage_gpu_bench.py.
"""

from __future__ import annotations

import ast
import sys
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "docker" / "patch"
FIX = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(PATCH))
import engram_early_hash as eeh  # noqa: E402
import engram_stage_fast  # noqa: E402


class Arr:
    """The slicing/copy surface launch() touches."""

    def __init__(self, n=8):
        self.n = n
        self.copies = []

    def __getitem__(self, key):
        return self

    def copy_(self, other, non_blocking=False):
        self.copies.append((other, non_blocking))
        return self

    @property
    def shape(self):
        return (2, 3)

    def stride(self, i):
        return 4096


class Event:
    def __init__(self):
        self.calls = []

    def record(self):
        self.calls.append("record")

    def synchronize(self):
        self.calls.append("sync")


class Stager:
    def __init__(self, max_native=64):
        self.max_tokens = 8192
        self.head_start, self.head_end = 0, 12
        self.hash_host = Arr()
        self.hashes_ready = Event()
        self.hash_calls = []
        self._eng_native = types.SimpleNamespace(max_tokens=max_native)
        self.hash_state = types.SimpleNamespace(
            ensure_cache=lambda: True, __call__=None
        )
        stager = self

        class HS:
            def ensure_cache(self):
                return True

            def __call__(self, *args):
                stager.hash_calls.append(args)
                return Arr()

        self.hash_state = HS()


class Batch:
    def __init__(self, n=4):
        self.input_ids = Arr()
        self.positions = Arr()
        self.query_start_loc = Arr()
        self.num_reqs = 1
        self.num_tokens = n
        self.idx_mapping = Arr()


class Kernel:
    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        def run(*args, **kw):
            self.launches.append((grid, args, kw))

        return run


def hook():
    k = Kernel()
    triton = types.SimpleNamespace(next_power_of_2=lambda x: 4)
    return eeh.EarlyHash(torch=None, image_sentinel_mask=lambda t: ("mask", t), lookback_kernel=k, triton=triton), k


def model_state(rope=None):
    return types.SimpleNamespace(lookback_token_ids=Arr(), rope_state=rope)


def req_states():
    return types.SimpleNamespace(
        all_token_ids=types.SimpleNamespace(gpu=Arr()), num_computed_tokens=types.SimpleNamespace(gpu=Arr())
    )


class LaunchTests(unittest.TestCase):
    def setUp(self):
        eeh._STATE.update(armed=True, verify_left=0, engaged=False)

    def test_launch_records_hash_dtoh_and_batch(self):
        h, k = hook()
        st, ms, b = Stager(), model_state(), Batch(4)
        h.launch(st, ms, b, req_states())
        self.assertEqual(len(k.launches), 1)  # the lookback kernel
        self.assertEqual(len(st.hash_calls), 1)
        self.assertEqual(st.hashes_ready.calls, ["record"])
        self.assertEqual(st.hash_host.copies[-1][1], True)  # non_blocking DtoH
        self.assertIs(st._early_rec[0](), b)
        self.assertEqual(st._early_rec[1], 4)

    def test_launch_skips_what_stage_would_not_take(self):
        cases = [
            ("prefill-size", Stager(max_native=64), model_state(), Batch(65)),
            ("rope state", Stager(), model_state(rope=object()), Batch(4)),
            ("no window", Stager(), types.SimpleNamespace(lookback_token_ids=None, rope_state=None), Batch(4)),
            ("no native", Stager(), model_state(), Batch(4)),
            ("zero tokens", Stager(), model_state(), Batch(0)),
        ]
        cases[3][1]._eng_native = None
        for label, st, ms, b in cases:
            h, k = hook()
            h.launch(st, ms, b, req_states())
            self.assertEqual((k.launches, st.hash_calls), ([], []), label)
            self.assertIsNone(getattr(st, "_early_rec", None), label)
        b = Batch(4)
        b.input_ids = None
        h, k = hook()
        st = Stager()
        h.launch(st, model_state(), b, req_states())
        self.assertEqual(st.hash_calls, [])

    def test_disarmed_never_launches(self):
        eeh._STATE["armed"] = False
        h, k = hook()
        st = Stager()
        h.launch(st, model_state(), Batch(4), req_states())
        self.assertEqual(st.hash_calls, [])


class ConsumeTests(unittest.TestCase):
    def setUp(self):
        eeh._STATE.update(armed=True, verify_left=0, engaged=False)

    def launched(self, n=4):
        h, _ = hook()
        st, b = Stager(), Batch(n)
        h.launch(st, model_state(), b, req_states())
        st.hashes_ready.calls.clear()
        return h, st, b

    def test_consumes_once_for_the_marked_batch(self):
        h, st, b = self.launched()
        st._early_cur = b
        self.assertTrue(h.consume(st, b.input_ids, b.positions, b.query_start_loc, Arr(), 4))
        self.assertEqual(st.hashes_ready.calls, ["sync"])
        self.assertIsNone(st._early_rec)
        self.assertFalse(h.consume(st, b.input_ids, b.positions, b.query_start_loc, Arr(), 4))

    def test_mismatches_fall_back_to_the_late_hash(self):
        for label, change in (
            ("other batch", lambda st, b: setattr(st, "_early_cur", Batch(4))),
            ("no current batch", lambda st, b: setattr(st, "_early_cur", None)),
        ):
            h, st, b = self.launched()
            st._early_cur = b
            change(st, b)
            self.assertFalse(h.consume(st, b.input_ids, b.positions, b.query_start_loc, Arr(), 4), label)
            self.assertEqual(st.hashes_ready.calls, [], label)
        h, st, b = self.launched()
        st._early_cur = b
        self.assertFalse(h.consume(st, b.input_ids, b.positions, b.query_start_loc, Arr(), 3))  # n differs
        h, st, b = self.launched()
        st._early_cur = b
        self.assertFalse(h.consume(st, Arr(), b.positions, b.query_start_loc, Arr(), 4))  # other ids tensor

    def test_verify_mismatch_disarms_and_keeps_late_hashes(self):
        eeh._STATE["verify_left"] = 2
        h, st, b = self.launched()
        st._early_cur = b
        fake_torch = types.SimpleNamespace(equal=lambda a, c: False)
        h.torch = fake_torch
        st.hash_host.clone = lambda: "early"
        with redirect_stdout(StringIO()) as out:
            ok = h.consume(st, b.input_ids, b.positions, b.query_start_loc, Arr(), 4)
        self.assertTrue(ok)  # hash_host holds the late (stock) hashes
        self.assertEqual(len(st.hash_calls), 2)  # early + late
        self.assertIn("dsv41: engram early hash DISABLED ->", out.getvalue())
        self.assertFalse(eeh._STATE["armed"])

    def test_verify_match_prints_engaged_once(self):
        eeh._STATE["verify_left"] = 2
        with redirect_stdout(StringIO()) as out:
            for _ in range(3):
                h, st, b = self.launched()
                st._early_cur = b
                h.torch = types.SimpleNamespace(equal=lambda a, c: True)
                st.hash_host.clone = lambda: "early"
                self.assertTrue(h.consume(st, b.input_ids, b.positions, b.query_start_loc, Arr(), 4))
        self.assertEqual(out.getvalue().count("dsv41: engram early hash matches the stock hash"), 1)
        self.assertEqual(eeh._STATE["verify_left"], 0)


class ContractTests(unittest.TestCase):
    """The early launch repeats the image's code; keep it in step."""

    def test_env(self):
        self.assertFalse(eeh.enabled({}))
        self.assertTrue(eeh.enabled({"DSV41_ENGRAM_EARLY_HASH": "1"}))
        self.assertEqual(eeh.verify_steps({}), 8)
        self.assertEqual(eeh.verify_steps({"DSV41_ENGRAM_EARLY_VERIFY": "0"}), 1)

    def test_lookback_launch_matches_model_state(self):
        src = (PATCH / "engram_early_hash.py").read_text()
        fixture = (FIX / "model_state_e12.pin.py").read_text()
        for arg in (
            "input_batch.idx_mapping,",
            "req_states.num_computed_tokens.gpu,",
            "all_token_ids.stride(0),",
            "input_batch.idx_mapping.shape[0],",
            "DEPTH=depth,",
        ):
            self.assertIn(arg, fixture, arg)
            self.assertIn(arg, src, arg)
        self.assertIn("input_batch.query_start_loc[: input_batch.num_reqs + 1],", fixture)
        self.assertIn("input_batch.query_start_loc[: input_batch.num_reqs + 1],", src)
        self.assertIn("positions if positions is not None else input_batch.positions,", fixture)

    def test_hash_statements_match_the_stock_stage(self):
        src = (PATCH / "engram_early_hash.py").read_text()
        stock = engram_stage_fast.STAGE_FAST
        for stmt in (
            "host.copy_(hashes[:, :, self.head_start : self.head_end], non_blocking=True)",
            "self.hashes_ready.record()",
        ):
            self.assertIn(stmt, stock)
            self.assertIn(stmt.replace("self.", "stager."), src)

    def test_runner_calls_prepare_attn_between_inputs_and_model_state(self):
        runner = (FIX / "gpu_model_runner_e12.pin.py").read_text()
        body = runner[runner.index("    def execute_model(") :]
        a = body.index("input_batch = self.prepare_inputs(")
        b = body.index("block_tables, slot_mappings = self.prepare_attn(input_batch)")
        c = body.index("attn_metadata = self.model_state.prepare_attn(")
        d = body.index("**self.model_state.prepare_inputs(input_batch, self.req_states)")
        self.assertLess(a, b)
        self.assertLess(b, c)
        self.assertLess(c, d)

    def test_top_level_imports_are_stdlib_only(self):
        tree = ast.parse((PATCH / "engram_early_hash.py").read_text())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        mods = {a.name for n in top if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module for n in top if isinstance(n, ast.ImportFrom)}
        self.assertEqual(mods, {"__future__", "os", "weakref"})


class InstallTests(unittest.TestCase):
    def test_wrappers_mark_and_clear_the_batch(self):
        eeh._STATE.update(armed=True, verify_left=0, engaged=False)
        seen = {}

        class Runner:
            def prepare_attn(self, input_batch):
                seen["attn"] = input_batch
                return "tables"

        class MS:
            def prepare_inputs(self, input_batch, req_states):
                seen["cur"] = self.engram_stager._early_cur
                return {"x": 1}

        ms_mod = types.ModuleType("vllm.models.deepseek_v4_1.nvidia.model_state")
        ms_mod.DeepseekV41ModelState = MS
        ms_mod._gather_lookback_kernel = Kernel()
        runner_mod = types.ModuleType("vllm.v1.worker.gpu.model_runner")
        runner_mod.GPUModelRunner = Runner
        tr = types.ModuleType("vllm.triton_utils")
        tr.triton = types.SimpleNamespace(next_power_of_2=lambda x: 4)
        mods = {
            "vllm": types.ModuleType("vllm"),
            "vllm.models": types.ModuleType("vllm.models"),
            "vllm.models.deepseek_v4_1": types.ModuleType("vllm.models.deepseek_v4_1"),
            "vllm.models.deepseek_v4_1.nvidia": types.ModuleType("vllm.models.deepseek_v4_1.nvidia"),
            ms_mod.__name__: ms_mod,
            "vllm.v1": types.ModuleType("vllm.v1"),
            "vllm.v1.worker": types.ModuleType("vllm.v1.worker"),
            "vllm.v1.worker.gpu": types.ModuleType("vllm.v1.worker.gpu"),
            runner_mod.__name__: runner_mod,
            tr.__name__: tr,
        }
        with mock.patch.dict(sys.modules, mods):
            eeh.install(torch=None, image_sentinel_mask=lambda t: t)
        stager = Stager()
        ms = MS()
        ms.engram_stager = stager
        ms.lookback_token_ids = Arr()
        ms.rope_state = None
        r = Runner()
        r.model_state = ms
        r.req_states = req_states()
        b = Batch(4)
        self.assertEqual(r.prepare_attn(b), "tables")
        self.assertIs(seen["attn"], b)
        self.assertIs(stager._early_rec[0](), b)
        self.assertEqual(ms.prepare_inputs(b, None), {"x": 1})
        self.assertIs(seen["cur"], b)
        self.assertIsNone(stager._early_cur)
        self.assertIsNone(stager._early_rec)


if __name__ == "__main__":
    unittest.main()
