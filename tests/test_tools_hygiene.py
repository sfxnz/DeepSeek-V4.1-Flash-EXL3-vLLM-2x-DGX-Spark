#!/usr/bin/env python3
"""Pack-tool footguns and the trace kernel extractor (CPU only)."""
from __future__ import annotations

import gzip
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(rel: str):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PermuteToolTests(unittest.TestCase):
    def test_pack_argument_is_required(self) -> None:
        r = subprocess.run(
            [sys.executable, str(ROOT / "tools/permute_pack_group_major.py")],
            capture_output=True,
            text=True,
        )
        self.assertEqual(r.returncode, 2)
        self.assertIn("pack", r.stderr)

    def test_no_home_default_and_no_dead_perm_axis(self) -> None:
        mod = _load("tools/permute_pack_group_major.py")
        self.assertFalse(hasattr(mod, "DEFAULT_PACK"))
        self.assertFalse(hasattr(mod, "perm_axis"))
        self.assertNotIn("/home/", (ROOT / "tools/permute_pack_group_major.py").read_text())

    def test_manifest_name_matches_loader(self) -> None:
        tool = _load("tools/permute_pack_group_major.py")
        loader = _load("docker/patch/pfg8_loader_reindex.py")
        self.assertEqual(tool.MANIFEST_NAME, loader.MANIFEST_NAME)


class G8ManifestGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _load("docker/patch/pfg8_loader_reindex.py")

    def test_serve_model_from_argv(self) -> None:
        f = self.mod.serve_model_from_argv
        self.assertEqual(f(["/usr/local/bin/vllm", "serve", "/cache/snap", "--tp", "2"]), "/cache/snap")
        self.assertEqual(f(["vllm", "serve", "--model", "/cache/snap"]), "/cache/snap")
        self.assertIsNone(f(["-c", "--multiprocessing-fork"]))

    def test_refuses_pack_without_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit) as cm:
                self.mod.require_g8_manifest(["vllm", "serve", d])
            self.assertIn(self.mod.MANIFEST_NAME, str(cm.exception))

    def test_accepts_pack_with_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / self.mod.MANIFEST_NAME).write_text("{}")
            self.mod.require_g8_manifest(["vllm", "serve", d])

    def test_spawned_process_is_not_checked(self) -> None:
        self.mod.require_g8_manifest(["-c", "--multiprocessing-fork"])

    def test_sitecustomize_checks_manifest_before_patching(self) -> None:
        site = (ROOT / "docker/patch/sitecustomize.py").read_text()
        block = site[site.index("from pfg8_loader_reindex import patch"):]
        self.assertLess(block.index("_pfg8_require(sys.argv)"), block.index("_pfg8_patch(_t)"))


class ExtractKernelsTests(unittest.TestCase):
    def test_extracts_only_kernel_events(self) -> None:
        try:
            import ijson  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("ijson not installed")
        mod = _load("tools/extract_kernels.py")
        trace = {"traceEvents": [
            {"cat": "kernel", "name": "ncclDevKernel_AllReduce", "ts": 10.0, "dur": 24.0,
             "tid": 7, "args": {"grid": [5, 1, 1], "block": [512, 1, 1]}},
            {"cat": "cpu_op", "name": "aten::mm", "ts": 1.0, "dur": 2.0},
            {"cat": "kernel", "name": "p2b_moe", "ts": 40.0, "dur": 539.0, "tid": 7, "args": {}},
        ]}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.json.gz"
            with gzip.open(path, "wt") as fh:
                json.dump(trace, fh)
            names, rows = mod.extract(str(path))
        self.assertEqual(sorted(names.values()), ["ncclDevKernel_AllReduce", "p2b_moe"])
        self.assertEqual(len(rows), 2)
        ts, dur, nid, grid, block, tid, _, _ = rows[0]
        self.assertEqual((ts, dur, names[nid], grid, block, tid),
                         (10.0, 24.0, "ncclDevKernel_AllReduce", (5, 1, 1), (512, 1, 1), 7))


if __name__ == "__main__":
    unittest.main()
