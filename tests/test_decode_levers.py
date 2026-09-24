"""decode_levers: env parsing, split math, anchors in pinned canonical-e12 sources.

Fixtures are read-only copies from image dsv41-flash-exl3-sm121:canonical-e12
(sha256:984ea608...), vllm/ paths:
  mhc_warmup.pin.py        model_executor/kernels/mhc/warmup.py
  mhc_tilelang.pin.py      model_executor/kernels/mhc/tilelang.py
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


class WiringTests(unittest.TestCase):
    NAMES = ("DSV41_WOA_PREPACK",)

    def test_new_envs_forwarded_off_by_default(self):
        from test_recipe_ops import _forward_envs, _patch_env_reads

        fwd = _forward_envs()
        reads = _patch_env_reads()
        for name in self.NAMES + ("DSV41_MHC_DECODE_SPLITS",):
            self.assertIn(name, fwd)
            self.assertIn(name, reads)
        for name in self.NAMES:
            self.assertEqual(fwd[name], "", name)  # unset in the container = off

    def test_dry_run_forwards_levers_to_both_ranks(self):
        from run_sh_harness import container_env, dry_run

        res = dry_run(
            DSV41_WOA_PREPACK="1",
            DSV41_MHC_DECODE_SPLITS="40",
        )
        self.assertEqual(res["returncode"], 0, res["stdout"] + res["stderr"])
        for role in ("head", "worker"):
            env = container_env(res[role])
            self.assertEqual(env["DSV41_WOA_PREPACK"], "1", role)
            self.assertEqual(env["DSV41_MHC_DECODE_SPLITS"], "40", role)
        plain = dry_run()
        for name in self.NAMES:
            self.assertNotIn(name, container_env(plain["head"]))

    def test_sitecustomize_calls_install_once_at_the_end(self):
        site = (ROOT / "docker/patch/sitecustomize.py").read_text()
        self.assertEqual(site.count('__import__("decode_levers").install()'), 1)
        self.assertTrue(site.rstrip().endswith('__import__("decode_levers").install()'))
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


if __name__ == "__main__":
    unittest.main()
