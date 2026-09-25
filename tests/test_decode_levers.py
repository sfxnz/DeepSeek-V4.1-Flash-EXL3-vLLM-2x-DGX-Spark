"""decode_levers: env parsing, split math, anchors in pinned canonical-e12 sources.

Fixtures are read-only copies from image dsv41-flash-exl3-sm121:canonical-e12
(sha256:984ea608...), vllm/ paths:
  mhc_warmup.pin.py        model_executor/kernels/mhc/warmup.py
  mhc_tilelang.pin.py      model_executor/kernels/mhc/tilelang.py
  dspark_speculator.pin.py v1/worker/gpu/spec_decode/dspark/speculator.py
  dsv41_dspark.pin.py      models/deepseek_v4_1/nvidia/dspark.py
  qwen3_dspark.pin.py      model_executor/models/qwen3_dspark.py
  o_proj_e12.pin.py        models/deepseek_v4/nvidia/ops/o_proj.py (probe + requant applied)
"""

import ast
import sys
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(ROOT / "docker" / "patch"))
sys.path.insert(0, str(ROOT / "tests"))

import decode_levers as dl  # noqa: E402


def _func_src(path: Path, name: str, cls: str | None = None) -> str:
    src = path.read_text()
    tree = ast.parse(src)
    scope = tree.body
    if cls is not None:
        scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    node = next(n for n in scope if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(src, node)


class MhcSplitsTests(unittest.TestCase):
    def test_env_values(self):
        self.assertIsNone(dl.mhc_forced_splits(None))
        self.assertIsNone(dl.mhc_forced_splits(""))
        self.assertIsNone(dl.mhc_forced_splits("0"))
        self.assertIsNone(dl.mhc_forced_splits("1"))  # legacy sitecustomize block
        self.assertEqual(dl.mhc_forced_splits("32"), 32)
        with self.assertRaises(ValueError):
            dl.mhc_forced_splits("many")

    def test_forced_at_decode_stock_above_64(self):
        stock = lambda k, t: 16  # noqa: E731
        for t in (1, 4, 6, 8, 12, 64):
            for n in (16, 24, 32, 40, 48):
                self.assertEqual(dl.mhc_pre_num_splits(20480, t, n, stock), n)
        self.assertEqual(dl.mhc_pre_num_splits(20480, 65, 48, stock), 16)
        self.assertEqual(dl.mhc_pre_num_splits(20480, 8192, 48, stock), 16)

    def test_kblock_cap(self):
        stock = lambda k, t: 16  # noqa: E731
        # K=5120 (layer-0 broadcast input): 80 k-blocks -> cap 20.
        self.assertEqual(dl.mhc_pre_num_splits(5120, 4, 48, stock), 20)
        self.assertEqual(dl.mhc_pre_num_splits(5120, 4, 16, stock), 16)
        # K=20480: 320 k-blocks -> cap 80, 48 stays.
        self.assertEqual(dl.mhc_pre_num_splits(20480, 4, 48, stock), 48)
        self.assertEqual(dl.mhc_pre_num_splits(64, 1, 48, stock), 1)

    def test_warmup_keys_see_forced_value(self):
        # Pinned warmup.py: keys = {compute_mhc_pre_num_splits(K, t) for t in 1, 65, ...}
        # through the module global, which is what install() replaces.
        body = _func_src(FIX / "mhc_warmup.pin.py", "get_warmup_keys", "MHCPreNormKernel")
        self.assertIn('compute_mhc_pre_num_splits(fields["rms_numel"], num_tokens)', body)
        self.assertIn("range(1, max_tokens + 1, 64)", body)
        stock = lambda k, t: 16 if t <= 64 else 4  # noqa: E731
        keys = sorted({dl.mhc_pre_num_splits(20480, t, 40, stock) for t in range(1, 8193, 64)})
        self.assertEqual(keys, [4, 40])

    def test_dispatch_imports_splits_at_call_time(self):
        body = _func_src(FIX / "mhc_tilelang.pin.py", "mhc_pre_delayed_tilelang")
        self.assertIn("from vllm.model_executor.kernels.mhc.warmup import (", body)
        self.assertIn("compute_mhc_pre_num_splits(input_size, num_tokens) if use_deep_gemm else 1", body)
        self.assertIn("tf32_hc_prenorm_gemm(x, fn, mixes, sqrsum, n_splits)", body)
        stock = _func_src(FIX / "mhc_warmup.pin.py", "compute_mhc_pre_num_splits")
        self.assertIn("return 1 if splits == 1 else 4 if splits <= 4 else 16", stock)

    def test_install_patches_warmup_module(self):
        wu = types.ModuleType("vllm.model_executor.kernels.mhc.warmup")
        wu.compute_mhc_pre_num_splits = lambda k, t: 16
        mhc = types.ModuleType("vllm.model_executor.kernels.mhc")
        mhc.warmup = wu
        mods = {
            "vllm": types.ModuleType("vllm"),
            "vllm.model_executor": types.ModuleType("vllm.model_executor"),
            "vllm.model_executor.kernels": types.ModuleType("vllm.model_executor.kernels"),
            "vllm.model_executor.kernels.mhc": mhc,
            "vllm.model_executor.kernels.mhc.warmup": wu,
        }
        with mock.patch.dict(sys.modules, mods), redirect_stdout(StringIO()) as out:
            dl.install({"DSV41_MHC_DECODE_SPLITS": "40"})
        self.assertEqual(wu.compute_mhc_pre_num_splits(20480, 4), 40)
        self.assertEqual(wu.compute_mhc_pre_num_splits(20480, 100), 16)
        self.assertIn("forced to 40", out.getvalue())
        wu.compute_mhc_pre_num_splits = lambda k, t: 16
        with mock.patch.dict(sys.modules, mods), redirect_stdout(StringIO()):
            dl.install({"DSV41_MHC_DECODE_SPLITS": "1"})
        self.assertEqual(wu.compute_mhc_pre_num_splits(20480, 4), 16)

    def test_legacy_block_only_takes_1(self):
        site = (ROOT / "docker/patch/sitecustomize.py").read_text()
        self.assertIn('os.environ.get("DSV41_MHC_DECODE_SPLITS", "0") == "1"', site)


class SparseMarkovTests(unittest.TestCase):
    def test_env_values(self):
        self.assertIsNone(dl.sparse_markov_topk({}))
        self.assertIsNone(dl.sparse_markov_topk({"DSV41_DSPARK_SPARSE_MARKOV": "0"}))
        self.assertEqual(dl.sparse_markov_topk({"DSV41_DSPARK_SPARSE_MARKOV": "1"}), 256)
        self.assertEqual(
            dl.sparse_markov_topk(
                {"DSV41_DSPARK_SPARSE_MARKOV": "1", "DSV41_DSPARK_SPARSE_MARKOV_TOPK": "128"}
            ),
            128,
        )
        for bad in ("0", "-5", "5000"):
            with self.assertRaises(ValueError):
                dl.sparse_markov_topk(
                    {"DSV41_DSPARK_SPARSE_MARKOV": "1", "DSV41_DSPARK_SPARSE_MARKOV_TOPK": bad}
                )

    def test_conflicts(self):
        self.assertEqual(dl.sparse_markov_conflicts({}), [])
        self.assertEqual(
            dl.sparse_markov_conflicts(
                {"DSV41_DSPARK_MARKOV_SCALE": "1", "DSV41_DSPARK_CONF_GATE": "0", "DSV41_DSPARK_DRAFT_TOPK": "0"}
            ),
            [],
        )
        self.assertEqual(
            dl.sparse_markov_conflicts(
                {"DSV41_DSPARK_MARKOV_SCALE": "0.5", "DSV41_DSPARK_CONF_GATE": "1", "DSV41_DSPARK_DRAFT_TOPK": "32"}
            ),
            ["DSV41_DSPARK_MARKOV_SCALE", "DSV41_DSPARK_CONF_GATE", "DSV41_DSPARK_DRAFT_TOPK"],
        )

    def test_conflict_refuses_loudly(self):
        env = {"DSV41_DSPARK_SPARSE_MARKOV": "1", "DSV41_DSPARK_CONF_GATE": "1"}
        with redirect_stdout(StringIO()) as out:
            dl.install(env)
        self.assertIn("sparse-markov FAILED, lever is OFF", out.getvalue())
        self.assertIn("DSV41_DSPARK_CONF_GATE", out.getvalue())

    def test_speculator_topk_path_anchors(self):
        spec = FIX / "dspark_speculator.pin.py"
        init = _func_src(spec, "__init__", "DSparkSpeculator")
        self.assertIn("self._draft_topk: int | None = getattr(", init)
        seq = _func_src(spec, "_sample_sequential", "DSparkSpeculator")
        self.assertIn("if self._draft_topk is not None:\n            self._sample_sequential_topk(", seq)
        topk = _func_src(spec, "_sample_sequential_topk", "DSparkSpeculator")
        self.assertIn("base_logits.topk(self._draft_topk, dim=-1)", topk)
        self.assertIn("self.model.apply_markov_bias_gathered(", topk)
        # Greedy argmax over the full-vocab buffer stays (corrections).
        self.assertIn("self._sample_logits(", topk)
        sample = _func_src(spec, "_sample_logits", "DSparkSpeculator")
        self.assertIn("logits.argmax(dim=-1)", sample)

    def test_deepseek_drafter_lacks_hook_and_shares_head(self):
        dsv = (FIX / "dsv41_dspark.pin.py").read_text()
        self.assertNotIn("apply_markov_bias_gathered", dsv)
        self.assertIn("self.model.markov_head.bias(markov_embed, self.logits_processor)", dsv)
        self.assertIn("self.logits_processor = LogitsProcessor(self.config.vocab_size)", dsv)
        head = _func_src(FIX / "qwen3_dspark.pin.py", "apply_bias_gathered", "DSparkMarkovHead")
        self.assertIn("scale: float = 1.0", head)
        self.assertIn("weight = self.markov_w2.weight[index]", head)
        self.assertIn("logits.scatter_(1, index, corrected.squeeze(-1))", head)
        # decode_levers mirrors Qwen's hook one to one.
        qwen_hook = _func_src(FIX / "qwen3_dspark.pin.py", "apply_markov_bias_gathered", "Qwen3DSparkForCausalLM")
        self.assertIn("return self.model.markov_head.apply_bias_gathered(", qwen_hook)
        self.assertIn("self.logits_processor.scale", qwen_hook)

    def _fake_vllm(self):
        class Head:
            def apply_bias_gathered(self, e, logits, values, index, scale=1.0):
                return ("gathered", e, logits, values, index, scale)

        class Drafter:
            def __init__(self):
                self.model = types.SimpleNamespace(markov_head=Head())
                self.logits_processor = types.SimpleNamespace(scale=1.0)

        class Spec:
            def __init__(self, topk=None):
                self._draft_topk = topk

        dspark = types.ModuleType("vllm.models.deepseek_v4_1.nvidia.dspark")
        dspark.DSparkDeepseekV4ForCausalLM = Drafter
        specmod = types.ModuleType("vllm.v1.worker.gpu.spec_decode.dspark.speculator")
        specmod.DSparkSpeculator = Spec
        names = [
            "vllm", "vllm.models", "vllm.models.deepseek_v4_1", "vllm.models.deepseek_v4_1.nvidia",
            "vllm.v1", "vllm.v1.worker", "vllm.v1.worker.gpu", "vllm.v1.worker.gpu.spec_decode",
            "vllm.v1.worker.gpu.spec_decode.dspark",
        ]
        mods = {n: types.ModuleType(n) for n in names}
        mods[dspark.__name__] = dspark
        mods[specmod.__name__] = specmod
        return mods, Drafter, Spec

    def test_install_sets_topk_and_hook(self):
        mods, Drafter, Spec = self._fake_vllm()
        with mock.patch.dict(sys.modules, mods), redirect_stdout(StringIO()) as out:
            dl.install({"DSV41_DSPARK_SPARSE_MARKOV": "1"})
        self.assertIn("k=256", out.getvalue())
        self.assertEqual(Spec()._draft_topk, 256)
        self.assertEqual(Spec(topk=64)._draft_topk, 64)  # an explicit config value wins
        r = Drafter().apply_markov_bias_gathered("e", "l", "v", "i")
        self.assertEqual(r, ("gathered", "e", "l", "v", "i", 1.0))

    def test_off_by_default_touches_nothing(self):
        mods, Drafter, Spec = self._fake_vllm()
        with mock.patch.dict(sys.modules, mods), redirect_stdout(StringIO()) as out:
            dl.install({})
        self.assertEqual(out.getvalue(), "")
        self.assertIsNone(Spec()._draft_topk)
        self.assertFalse(hasattr(Drafter, "apply_markov_bias_gathered"))


class WiringTests(unittest.TestCase):
    NAMES = ("DSV41_DSPARK_SPARSE_MARKOV", "DSV41_DSPARK_SPARSE_MARKOV_TOPK", "DSV41_WOA_PREPACK")

    def test_new_envs_forwarded(self):
        from test_recipe_ops import _forward_envs, _patch_env_reads

        fwd = _forward_envs()
        reads = _patch_env_reads()
        for name in self.NAMES + ("DSV41_MHC_DECODE_SPLITS",):
            self.assertIn(name, fwd)
            self.assertIn(name, reads)
        for name in self.NAMES:
            self.assertEqual(fwd[name], "", name)  # default comes from the generated block, if any

    def test_dry_run_forwards_levers_to_both_ranks(self):
        from run_sh_harness import container_env, dry_run

        res = dry_run(
            DSV41_WOA_PREPACK="1",
            DSV41_MHC_DECODE_SPLITS="40",
            DSV41_DSPARK_SPARSE_MARKOV="1",
            DSV41_DSPARK_SPARSE_MARKOV_TOPK="128",
        )
        self.assertEqual(res["returncode"], 0, res["stdout"] + res["stderr"])
        for role in ("head", "worker"):
            env = container_env(res[role])
            self.assertEqual(env["DSV41_WOA_PREPACK"], "1", role)
            self.assertEqual(env["DSV41_MHC_DECODE_SPLITS"], "40", role)
            self.assertEqual(env["DSV41_DSPARK_SPARSE_MARKOV"], "1", role)
            self.assertEqual(env["DSV41_DSPARK_SPARSE_MARKOV_TOPK"], "128", role)
        plain = container_env(dry_run()["head"])
        # Round 34 (s7/s8) promoted WOA_PREPACK and SPARSE_MARKOV; TOPK keeps the in-code 256.
        self.assertEqual(plain["DSV41_WOA_PREPACK"], "1")
        self.assertEqual(plain["DSV41_DSPARK_SPARSE_MARKOV"], "1")
        self.assertNotIn("DSV41_DSPARK_SPARSE_MARKOV_TOPK", plain)
        self.assertEqual(plain["DSV41_MHC_DECODE_SPLITS"], "0")

    def test_sitecustomize_calls_install_once_at_the_end(self):
        site = (ROOT / "docker/patch/sitecustomize.py").read_text()
        self.assertEqual(site.count('__import__("decode_levers").install()'), 1)
        self.assertTrue(
            site.rstrip().endswith('_patch("decode_levers", lambda: __import__("decode_levers").install())')
        )
        # Importing the module must not pull anything beyond the stdlib.
        tree = ast.parse((ROOT / "docker/patch/decode_levers.py").read_text())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        mods = {a.name for n in top if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module for n in top if isinstance(n, ast.ImportFrom)}
        self.assertEqual(mods, {"__future__", "os", "pathlib"})

    def test_woa_prepack_warns_on_image_without_stage(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "o_proj.py"
            p.write_text((FIX / "o_proj_e12.pin.py").read_text())
            with mock.patch.object(dl, "O_PROJ_PY", p), redirect_stdout(StringIO()) as out:
                dl.install({"DSV41_WOA_PREPACK": "1"})
            self.assertIn("WARNING DSV41_WOA_PREPACK=1", out.getvalue())
            import fix_o_proj_woa_fp8

            with redirect_stdout(StringIO()):
                fix_o_proj_woa_fp8.apply(p)
            with mock.patch.object(dl, "O_PROJ_PY", p), redirect_stdout(StringIO()) as out:
                dl.install({"DSV41_WOA_PREPACK": "1"})
            self.assertEqual(out.getvalue(), "")

    def test_p2b_src_sort_warns_on_extension_without_srcsort(self):
        import importlib.util
        import tempfile

        sys.path.insert(0, str(ROOT / "tools"))
        from engagement_audit import audit

        with tempfile.TemporaryDirectory() as td:
            so = Path(td) / "vllm_exl3_c.so"
            spec = types.SimpleNamespace(origin=str(so))
            with mock.patch.object(importlib.util, "find_spec", lambda name: spec):
                so.write_bytes(b"\x7fELF" + b"\0" * 4096 + b"p2b_fused_moe\0")  # pre-srcsort build
                with redirect_stdout(StringIO()) as out:
                    dl.install({"DSV41_P2B_SRC_SORT": "1"})
                line = out.getvalue()
                self.assertIn("decode lever p2b_src_sort", line)
                self.assertTrue(any("p2b_src_sort" in x for x in audit({"head": line}, {})), "the audit reports it")
                so.write_bytes(b"\x7fELF" + b"\0" * 4096 + b"DSV41_P2B_SRC_SORT\0")
                with redirect_stdout(StringIO()) as out:
                    dl.install({"DSV41_P2B_SRC_SORT": "1"})
                    dl.install({})  # off: the .so is never opened
                self.assertEqual(out.getvalue(), "")
            with mock.patch.object(importlib.util, "find_spec", lambda name: None), redirect_stdout(StringIO()) as out:
                dl.install({"DSV41_P2B_SRC_SORT": "1"})
            self.assertIn("decode lever p2b_src_sort FAILED, lever is OFF", out.getvalue())

    def test_p2b_coop_warns_on_extension_without_coop(self):
        import importlib.util
        import tempfile

        sys.path.insert(0, str(ROOT / "tools"))
        from engagement_audit import audit

        with tempfile.TemporaryDirectory() as td:
            so = Path(td) / "vllm_exl3_c.so"
            spec = types.SimpleNamespace(origin=str(so))
            with mock.patch.object(importlib.util, "find_spec", lambda name: spec):
                so.write_bytes(b"\x7fELF" + b"\0" * 4096 + b"DSV41_P2B_SRC_SORT\0")  # canonical-e13 build
                with redirect_stdout(StringIO()) as out:
                    dl.install({"DSV41_P2B_COOP": "1"})
                line = out.getvalue()
                self.assertIn("decode lever p2b_coop:", line)
                self.assertIn("Rebuild docker/Dockerfile.e14", line)
                self.assertTrue(any("p2b_coop" in x for x in audit({"head": line}, {})), "the audit reports it")
                so.write_bytes(b"\x7fELF" + b"\0" * 4096 + b"DSV41_P2B_SRC_SORT\0DSV41_P2B_COOP\0")
                with redirect_stdout(StringIO()) as out:
                    dl.install({"DSV41_P2B_COOP": "1", "DSV41_P2B_SRC_SORT": "1"})
                    dl.install({})  # off: the .so is never opened
                self.assertEqual(out.getvalue(), "")


class TraceKernelsTests(unittest.TestCase):
    def test_streamed_per_step_sums(self):
        import gzip
        import json
        import tempfile

        sys.path.insert(0, str(ROOT / "kernel_study" / "decode_levers"))
        import trace_kernels as tk

        def k(name, dur, grid):
            return {"ph": "X", "cat": "kernel", "name": name, "dur": dur, "args": {"grid": grid}}

        ev = [{"ph": "X", "cat": "cpu_op", "name": "aten::mm", "dur": 999}]
        for _ in range(2):  # two verify steps: 40 p2b calls each
            ev += [k("void p2b_moe_batched_kernel<2, 1>(x)", 500, [1, 1, 1])] * 40
            ev += [k("void deep_gemm::transpose_and_pack_fp32_into_ue8m0<512u, 48u, 128u>", 19, [22, 4, 1])] * 43
            ev += [k("void deep_gemm::transpose_and_pack_fp32_into_ue8m0<512u, 48u, 40u>", 2, [12, 1, 1])] * 3
            ev += [k("void deep_gemm::sm120_tf32_hc_prenorm_gemm_impl<24u, 20480u>", 20, [16, 1, 1])] * 86
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "t.json.gz")
            with gzip.open(path, "wt") as fh:
                json.dump({"schemaVersion": 1, "traceEvents": ev, "deviceProperties": []}, fh, indent=1)
            events = tk.load_events(path)
            self.assertEqual(len(events), 2 * (40 + 43 + 3 + 86))
            small = list(tk.iter_events(path, chunk=97))  # objects straddle chunk edges
            self.assertEqual(len(small), len(ev))
        res = tk.summarize(events, "p2b_moe_batched_kernel", 40)
        self.assertEqual(res["steps"], 2.0)
        rows = {(r["pattern"], tuple(r["grid"])): r for r in res["rows"]}
        pack = rows[("woa_pack", (22, 4, 1))]
        self.assertEqual(pack["calls_per_step"], 43.0)
        self.assertEqual(pack["ms_per_step"], 0.817)
        self.assertEqual(rows[("woa_pack", (12, 1, 1))]["calls_per_step"], 3.0)
        self.assertEqual(rows[("mhc_prenorm_gemm", (16, 1, 1))]["ms_per_step"], 1.72)


if __name__ == "__main__":
    unittest.main()
