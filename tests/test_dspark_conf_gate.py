#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "docker/patch/dspark_conf_gate.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DsparkConfGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load()

    def test_env_flag(self) -> None:
        self.assertFalse(self.mod.dspark_conf_gate_from_env(0))
        self.assertTrue(self.mod.dspark_conf_gate_from_env(1))
        self.assertFalse(self.mod.dspark_conf_gate_from_env(2))

    def test_conf_one_is_full_markov(self) -> None:
        base = [[1.0, 2.0], [3.0, 4.0]]
        bias = [[10.0, 20.0], [30.0, 40.0]]
        out = self.mod.gate_markov_logits_list(base, bias, [1.0, 1.0])
        self.assertEqual(out, [[11.0, 22.0], [33.0, 44.0]])

    def test_conf_zero_is_backbone_only(self) -> None:
        base = [[1.0, 2.0]]
        bias = [[99.0, 99.0]]
        out = self.mod.gate_markov_logits_list(base, bias, [0.0])
        self.assertEqual(out, [[1.0, 2.0]])

    def test_per_row_gate(self) -> None:
        base = [[0.0, 0.0], [0.0, 0.0]]
        bias = [[8.0, 4.0], [8.0, 4.0]]
        out = self.mod.gate_markov_logits_list(base, bias, [0.5, 0.25])
        self.assertEqual(out[0], [4.0, 2.0])
        self.assertEqual(out[1], [2.0, 1.0])
