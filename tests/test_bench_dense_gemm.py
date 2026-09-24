"""kernel_study/dense_gemm/bench_dense_gemm.py: shape table and promote gate (CPU)."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


BENCH = _load("kernel_study/dense_gemm/bench_dense_gemm.py", "bench_dense_gemm")
PATCH = _load("docker/patch/dense_mxfp8_deepgemm.py", "dense_mxfp8_deepgemm_b")


def _row(shape, backend, us, rel=0.02, M=None):
    m = M if M is not None else (3 if shape == "main_proj" else 4)
    return {"shape": shape, "M": m, "backend": backend, "us_e2e": us, "rel_fro": rel}


class BenchTests(unittest.TestCase):
    def test_bench_shapes_match_patch_defaults(self) -> None:
        bench = {(k, n) for _, k, n, _ in BENCH.SHAPES}
        self.assertEqual(bench, set(PATCH.shapes_from_env({})))

    def test_target_m(self) -> None:
        ms = {name: m for name, _, _, m in BENCH.SHAPES}
        self.assertEqual(ms["main_proj"], (3,))
        self.assertTrue(all(m == (4,) for n, m in ms.items() if n != "main_proj"))

    def test_split_k_only_on_underfilled_shapes(self) -> None:
        # Corrections: split-K helps qkv_a and shared gate_up only; wq_b K=1280 gets nothing.
        self.assertEqual(BENCH.SPLITK_SHAPES, {"qkv_a", "shared_gate_up"})

    def test_gate_promotes_only_ten_percent_dg_wins_with_sane_error(self) -> None:
        rows = [
            _row("qkv_a", "b12x", 46.5),
            _row("qkv_a", "dg", 40.0),  # +16% -> promote
            _row("wq_b", "b12x", 112.1),
            _row("wq_b", "dg", 105.0),  # +6.8% -> no
            _row("wo_b", "b12x", 105.2),
            _row("wo_b", "dg", 90.0, rel=0.2),  # fast but error 10x -> no
            _row("shared_gate_up", "b12x", 57.9),
            _row("shared_gate_up", "sk2", 40.0),  # not wireable -> no
            _row("shared_gate_up", "dg", 50.0),  # +15.8% -> promote (shared)
        ]
        s = BENCH.summarize(rows)
        self.assertEqual(s["DSV41_DENSE_DG_SHAPES"], "5120x1792,5120x2304")
        self.assertAlmostEqual(s["est_ms_per_step_serial"], -(6.5 * 45.8) / 1000)
        self.assertAlmostEqual(s["est_ms_per_step_shared_maybe_hidden"], -(7.9 * 42.8) / 1000)
        self.assertIn("sk2", s["per_shape"]["shared_gate_up"])

    def test_gate_ignores_failed_rows(self) -> None:
        rows = [_row("qkv_a", "b12x", 46.5), {"shape": "qkv_a", "M": 4, "backend": "dg", "error": "x"}]
        self.assertEqual(BENCH.summarize(rows)["DSV41_DENSE_DG_SHAPES"], "")


if __name__ == "__main__":
    unittest.main()
