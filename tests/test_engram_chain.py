"""Offline tests for the boot-time Engram patch chain (CPU only, no torch).

Fixtures are pinned read-only from dsv41-flash-exl3-sm121:canonical-e12,
whose engram.py / model_state.py already carry the build-time
apply_engram_disk + apply_engram_prestage edits. engram_disk.py is the
recipe's own docker/patch/engram_disk.py (the image copy is identical).
Refresh the pins on any image or base-digest bump.
"""

from __future__ import annotations

import contextlib
import io
import os
import py_compile
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "docker/patch"
FIX = ROOT / "tests/fixtures"
sys.path.insert(0, str(PATCH))

import apply_engram_disk  # noqa: E402
import apply_engram_prestage  # noqa: E402
import engram_cpu_hash  # noqa: E402
import engram_defer  # noqa: E402
import engram_gather_v2  # noqa: E402
import engram_prefetch_v3  # noqa: E402
import engram_stage_census  # noqa: E402
import engram_stage_fast  # noqa: E402
import indexer_workspace  # noqa: E402

TREE = {
    "models/deepseek_v4_1/common/engram.py": FIX / "engram_e12.pin.py",
    "models/deepseek_v4_1/common/engram_disk.py": PATCH / "engram_disk.py",
    "models/deepseek_v4_1/nvidia/model_state.py": FIX / "model_state_e12.pin.py",
    "v1/worker/gpu/model_runner.py": FIX / "gpu_model_runner_e12.pin.py",
    "v1/attention/backends/mla/indexer.py": FIX / "mla_indexer_e12.pin.py",
}
MARKERS = {
    "census": engram_stage_census.CENSUS_MARKER,
    "fast": engram_stage_fast.MARKER,
    "pfv3": engram_prefetch_v3.MARKER,
    "pfv3_hook": engram_prefetch_v3.V3_MARKER,
    "cpu_hash": engram_cpu_hash.MARKER,
    "cpu_hash_commit": engram_cpu_hash.COMMIT_MARKER,
    "gv2": engram_gather_v2.MARKER,
    "defer": engram_defer.MARKER,
    "indexer": indexer_workspace.MARK,
}
# Designed marker counts after one pass. A doubled apply shows up here.
EXPECTED = {
    "models/deepseek_v4_1/common/engram.py": {"fast": 2, "pfv3": 2, "cpu_hash": 1, "defer": 4},
    "models/deepseek_v4_1/common/engram_disk.py": {"census": 1, "pfv3_hook": 1, "gv2": 3},
    "models/deepseek_v4_1/nvidia/model_state.py": {"defer": 1},
    "v1/worker/gpu/model_runner.py": {"pfv3_hook": 1, "cpu_hash": 1, "cpu_hash_commit": 1, "defer": 1},
    "v1/attention/backends/mla/indexer.py": {"indexer": 1},
}


def _build_tree(dst: Path) -> Path:
    vllm = dst / "vllm"
    for rel, src in TREE.items():
        (vllm / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, vllm / rel)
    return vllm


def _apply_chain(vllm: Path) -> None:
    """sitecustomize.py order."""
    model = vllm / "models/deepseek_v4_1"
    runner = vllm / "v1/worker/gpu/model_runner.py"
    with contextlib.redirect_stdout(io.StringIO()):
        indexer_workspace.apply(vllm)
        engram_stage_census.apply(vllm)
        engram_stage_fast.apply(model)
        engram_prefetch_v3.apply(model, runner, vllm)
        engram_cpu_hash.apply(model, runner)
        engram_gather_v2.apply(vllm)
        engram_defer.apply(model, runner, model / "nvidia/model_state.py")


class EngramChainTests(unittest.TestCase):
    def test_fixtures_carry_build_time_patches(self) -> None:
        eng = (FIX / "engram_e12.pin.py").read_text()
        ms = (FIX / "model_state_e12.pin.py").read_text()
        self.assertEqual(eng.count(apply_engram_disk.MARKER), 1)
        self.assertIn(apply_engram_prestage.MARKER, eng)
        self.assertIn(apply_engram_prestage.MARKER, ms)
        self.assertEqual(apply_engram_disk.patch_engram(eng), eng)
        self.assertEqual(apply_engram_prestage.patch_engram(eng), eng)
        self.assertEqual(apply_engram_prestage.patch_model_state(ms), ms)

    def test_chain_applies_once_idempotent_and_compiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vllm = _build_tree(Path(tmp))
            _apply_chain(vllm)
            first = {rel: (vllm / rel).read_text() for rel in TREE}
            for rel, text in first.items():
                counts = {n: text.count(m) for n, m in MARKERS.items() if m in text}
                self.assertEqual(counts, EXPECTED[rel], rel)
            _apply_chain(vllm)
            for rel in TREE:
                self.assertEqual((vllm / rel).read_text(), first[rel], f"{rel} changed on re-apply")
                py_compile.compile(str(vllm / rel), cfile=os.path.join(tmp, "c.pyc"), doraise=True)


class BakedOldTextUpgradeTests(unittest.TestCase):
    """canonical-g8 baked the pre-2026-09-24 prefetch v3 / gather v2 text.

    apply() used to return on the old MARKER, so a G8 boot ran the old code
    while the audit saw the same engaged lines. The stale text is rebuilt
    here from the fresh chain: old marker, plus a line the old code had.
    """

    def _fresh(self, tmp: str):
        vllm = _build_tree(Path(tmp))
        _apply_chain(vllm)
        return vllm, {rel: (vllm / rel).read_text() for rel in TREE}

    def test_baked_v3_stager_block_is_swapped_and_neighbours_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vllm, fresh = self._fresh(tmp)
            rel = "models/deepseek_v4_1/common/engram.py"
            new, old = engram_prefetch_v3.MARKER, engram_prefetch_v3.V3_MARKER
            stale = fresh[rel].replace(new, old).replace(
                "        self._pf_tm_list = self._pf_tm.tolist()\n",
                "        self._pf_stale_v3 = True\n",
            )
            self.assertEqual(stale.count(old), 2)
            # cpu-hash and defer methods sit between the v3 block and stage().
            self.assertLess(stale.index(old + "\n    def "), stale.index(engram_cpu_hash.MARKER + "\n"))
            (vllm / rel).write_text(stale)
            model = vllm / "models/deepseek_v4_1"
            with contextlib.redirect_stdout(io.StringIO()) as out:
                engram_prefetch_v3.apply(model, vllm / "v1/worker/gpu/model_runner.py", vllm)
            self.assertIn("(replaced v3 methods)", out.getvalue())
            self.assertEqual((vllm / rel).read_text(), fresh[rel])
            for r in TREE:
                self.assertEqual((vllm / r).read_text(), fresh[r], r)

    def test_baked_gv2_blocks_are_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vllm, fresh = self._fresh(tmp)
            rel = "models/deepseek_v4_1/common/engram_disk.py"
            stale = fresh[rel].replace(engram_gather_v2.MARKER, engram_gather_v2.OLD_MARKER)
            stale = stale.replace("_ENG_GV2_LOCK = _gv2_threading.Lock()\n", "")
            stale = stale.replace(
                "        if _ENG_GATHER_V2[0] and int(rel.numel()) <= _ENG_GV2_MAX_ROWS:",
                "        if _ENG_GATHER_V2[0]:",
            )
            self.assertEqual(stale.count(engram_gather_v2.OLD_MARKER), 3)
            (vllm / rel).write_text(stale)
            with contextlib.redirect_stdout(io.StringIO()) as out:
                engram_gather_v2.apply(vllm)
            self.assertIn("(replaced baked pre-v2.1 text)", out.getvalue())
            self.assertEqual((vllm / rel).read_text(), fresh[rel])

    def test_unrecognised_old_gv2_layout_fails_loud(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vllm, fresh = self._fresh(tmp)
            rel = "models/deepseek_v4_1/common/engram_disk.py"
            stale = fresh[rel].replace(engram_gather_v2.MARKER, engram_gather_v2.OLD_MARKER)
            (vllm / rel).write_text(stale.replace("\nimport os as _gv2_os\n", "\nimport os\n_gv2_os = os\n", 1))
            with self.assertRaises(SystemExit):
                engram_gather_v2.apply(vllm)


class MemHygieneInstallTests(unittest.TestCase):
    """drop_page_cache / prefill_empty_cache are runtime installs, not rewrites."""

    def _stub_vllm(self):
        worker = type("Worker", (), {"load_model": lambda self: "l", "compile_or_warm_up_model": lambda self: "c", "execute_model": lambda self, so: "e"})
        mods = {n: types.ModuleType(n) for n in ("vllm", "vllm.v1", "vllm.v1.worker", "vllm.v1.worker.gpu_worker")}
        mods["vllm.v1.worker.gpu_worker"].Worker = worker
        return worker, mods

    def test_env_unset_installs_nothing(self) -> None:
        import drop_page_cache
        import prefill_empty_cache

        env = {k: v for k, v in os.environ.items() if k not in ("DSV41_DROP_PAGE_CACHE", "DSV41_PREFILL_EMPTY_CACHE_TOKENS")}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertIs(drop_page_cache.install(), False)
            self.assertIs(prefill_empty_cache.install(), False)

    def test_env_set_wraps_worker_once(self) -> None:
        import drop_page_cache
        import prefill_empty_cache

        worker, mods = self._stub_vllm()
        env = {"DSV41_DROP_PAGE_CACHE": "1", "DSV41_PREFILL_EMPTY_CACHE_TOKENS": "8192"}
        with mock.patch.dict(sys.modules, mods), mock.patch.dict(os.environ, env), contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(drop_page_cache.install())
            self.assertTrue(prefill_empty_cache.install())
            wrapped = (worker.load_model, worker.compile_or_warm_up_model, worker.execute_model)
            self.assertTrue(drop_page_cache.install())
            self.assertTrue(prefill_empty_cache.install())
        self.assertEqual((worker.load_model, worker.compile_or_warm_up_model, worker.execute_model), wrapped)
        self.assertTrue(worker.load_model._dsv41_dpc_wrapped)
        self.assertTrue(worker.compile_or_warm_up_model._dsv41_dpc_wrapped)
        self.assertTrue(worker.execute_model._dsv41_empty_cache)


if __name__ == "__main__":
    unittest.main()
