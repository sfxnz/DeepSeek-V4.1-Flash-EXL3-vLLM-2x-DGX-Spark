#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "docker/patch/dspark_refine_pass.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DsparkRefinePassTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load()

    def test_env_flag(self) -> None:
        self.assertFalse(self.mod.dspark_refine_pass_from_env(0))
        self.assertTrue(self.mod.dspark_refine_pass_from_env(1))
        self.assertFalse(self.mod.dspark_refine_pass_from_env(2))

    def test_indices_skip_offset_zero(self) -> None:
        idx = self.mod.refine_query_indices(2, 5, 5)
        self.assertEqual(idx, [[1, 2, 3, 4], [6, 7, 8, 9]])
        self.assertNotIn(0, idx[0])
        self.assertNotIn(5, idx[1])

    def test_fill_writes_draft_into_noise_slots(self) -> None:
        ids = [100, 0, 0, 0, 0, 200, 0, 0, 0, 0]
        drafts = [
            [11, 12, 13, 14, 15],
            [21, 22, 23, 24, 25],
        ]
        out = self.mod.apply_refine_fill_list(ids, drafts, 5)
        self.assertEqual(out[0], 100)
        self.assertEqual(out[5], 200)
        self.assertEqual(out[1:5], [11, 12, 13, 14])
        self.assertEqual(out[6:10], [21, 22, 23, 24])

    def test_n_spec_one_is_noop(self) -> None:
        self.assertEqual(self.mod.refine_query_indices(1, 1, 1), [[]])
        ids = [7]
        self.assertEqual(self.mod.apply_refine_fill_list(ids, [[9]], 1), [7])
