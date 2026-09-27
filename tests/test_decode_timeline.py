#!/usr/bin/env python3
"""CPU tests for tools/decode_timeline.py (synthetic traces, no GPU)."""
from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import decode_timeline as dt  # noqa: E402

try:
    import ijson  # noqa: F401
    HAVE_IJSON = True
except ImportError:  # the extract step needs ijson; analyze does not
    HAVE_IJSON = False

B12X = "kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm120_b12xDenseGemmKernel"
AR = "ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)"
AG = "ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)"
WOA = "void deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0u, 4u, 4096u, 32u, 32u, 4u, 128u>"
PRE = "void deep_gemm::sm120_tf32_hc_prenorm_gemm_impl<24u, 20480u>"
P2B = "void p2b_moe_batched_kernel<2, 1, 0>(__half const*)"
MLA = "void flashinfer::sparse_mla_sm120::sparse_mla_decode_dsv4_kernel<1>"
ADD = "void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float> >"


def layer(engram: bool = False) -> list[tuple[str, tuple, float]]:
    """(name, grid, dur_us) for one target decoder layer, in order."""
    out = [(AR, (5, 1, 1), 20.0)]
    if engram:
        out += [(AG, (6, 1, 1), 15.0), (B12X, (1, 1, 48), 750.0)]
    out += [(PRE, (40, 1, 1), 16.0), (B12X, (1, 1, 28), 47.0), (B12X, (1, 1, 48), 113.0), (MLA, (4, 2, 2), 15.0),
            (WOA, (48, 1, 1), 81.0), (B12X, (1, 1, 48), 105.0), (AR, (5, 1, 1), 22.0), (PRE, (40, 1, 1), 16.0),
            (B12X, (1, 1, 36), 58.0), (B12X, (1, 1, 48), 30.0), (ADD, (1, 1, 1), 2.0), (P2B, (192, 1, 1), 540.0)]
    return out


class HelperTests(unittest.TestCase):
    def test_union_busy_merges_overlaps(self) -> None:
        self.assertEqual(dt.union_busy([(0, 10), (5, 12), (20, 25)]), 17)
        self.assertEqual(dt.union_busy([]), 0.0)

    def test_idle_gaps_report_neighbours(self) -> None:
        gaps = dt.idle_gaps([(0, 10, 0), (5, 12, 1), (20, 25, 2)], 0, 40, 3.0)
        self.assertEqual(gaps, [(12, 20, 1, 2), (25, 40, 2, None)])

    def test_classify(self) -> None:
        self.assertEqual(dt.classify(AR), "nccl_allreduce")
        self.assertEqual(dt.classify(P2B), "p2b_moe")
        self.assertEqual(dt.classify(B12X), "dense_b12x")
        self.assertEqual(dt.classify(WOA), "woa_einsum")
        self.assertEqual(dt.classify(WOA.replace("<0u, 4u,", "<0u, 8u,")), "woa_einsum")  # c=2 verify, m=8
        self.assertEqual(dt.classify("void deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0u, 2304u, 5120u>"),
                         "deepgemm_fp8fp4_other")
        self.assertEqual(dt.classify(ADD), "torch_eltwise")

    def test_label_roles_target_layer_with_engram(self) -> None:
        ks = layer(engram=True) + layer()
        seq = [(i, dt.classify(n), g, d, n) for i, (n, g, d) in enumerate(ks)]
        roles = dt.label_roles(seq, "target")
        b12x = [roles[i] for i, (n, _, _) in enumerate(ks) if n == B12X]
        self.assertEqual(b12x, ["engram_wkv", "qkv_a", "wq_b", "wo_b", "shared_gate_up", "shared_down",
                                "qkv_a", "wq_b", "wo_b", "shared_gate_up", "shared_down"])
        self.assertEqual([roles[i] for i, (n, _, _) in enumerate(ks) if n == WOA], ["wo_a", "wo_a"])

    def test_slow_target_gemm_is_not_a_head(self) -> None:
        ks = layer()
        ks[3] = (B12X, (1, 1, 48), 1500.0)  # a contended wq_b
        seq = [(i, dt.classify(n), g, d, n) for i, (n, g, d) in enumerate(ks)]
        self.assertEqual(dt.label_roles(seq, "target")[3], "wq_b")

    def test_label_roles_lm_head_and_eager_main_proj(self) -> None:
        seq = [(0, "dense_b12x", (1, 1, 48), 1700.0, B12X), (1, "dense_b12x", (1, 1, 48), 460.0, B12X),
               (2, "dense_b12x", (1, 1, 28), 46.0, B12X)]
        self.assertEqual(dt.label_roles(seq, "eager"), {0: "lm_head", 1: "main_proj", 2: "draft_ctx_kv"})

    def test_nccl_skew_sign_and_wait(self) -> None:
        def mk(durs):
            return {"names": {0: AR}, "rt": [],
                    "dev": [(i * 100.0, d, 0, (5, 1, 1), (), 7, i, 1, 0, 0, None, None, 0) for i, d in enumerate(durs)]}
        r = dt.nccl_skew(mk([30.0, 20.0, 40.0]), mk([20.0, 30.0, 20.0]))
        c = r["classes"][0]
        self.assertEqual((r["matched"], r["op_or_grid_mismatch"]), (3, 0))
        self.assertEqual(c["skew_median_us(+=rank1 late)"], 10.0)  # rank 0 waited longer: rank 1 late
        self.assertAlmostEqual(c["frac_rank1_late"], 0.667)
        self.assertAlmostEqual(c["r0_wait_over_floor_ms"], 0.03)


def _trace(steps: int = 3) -> dict:
    """Two graphs per step (target: 2 layers, 80-AR stand-in by count; draft: 1 AR),
    an eager lm_head, and a host gap before the next target graph."""
    ev, t, corr = [], 1000.0, 1
    for _ in range(steps):
        # target graph launch
        ev.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaGraphLaunch", "ts": t - 5, "dur": 300.0,
                   "pid": 10, "tid": 10, "args": {"correlation": corr}})
        for name, grid, dur in layer(engram=True) + layer():
            ev.append({"ph": "X", "cat": "kernel", "name": name, "ts": t, "dur": dur, "pid": 0, "tid": 7,
                       "args": {"grid": list(grid), "block": [128, 1, 1], "stream": 7, "correlation": corr,
                                "graph id": 5, "graph node id": 1}})
            t += dur + 2
        corr += 1
        # eager lm_head
        ev.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": t - 3, "dur": 4.0,
                   "pid": 10, "tid": 10, "args": {"correlation": corr}})
        ev.append({"ph": "X", "cat": "kernel", "name": B12X, "ts": t, "dur": 1700.0, "pid": 0, "tid": 7,
                   "args": {"grid": [1, 1, 48], "stream": 7, "correlation": corr, "graph id": 0}})
        t += 1702
        corr += 1
        # draft graph (1 AR)
        ev.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaGraphLaunch", "ts": t - 5, "dur": 40.0,
                   "pid": 10, "tid": 10, "args": {"correlation": corr}})
        ev.append({"ph": "X", "cat": "kernel", "name": AR, "ts": t, "dur": 20.0, "pid": 0, "tid": 7,
                   "args": {"grid": [3, 1, 1], "stream": 7, "correlation": corr, "graph id": 6}})
        t += 22
        corr += 1
        # host gap (engram stage) then next step
        ev.append({"ph": "X", "cat": "python_function", "name": "engram.py(1747): stage", "ts": t, "dur": 2500.0,
                   "pid": 10, "tid": 10, "args": {}})
        t += 2500
    ev.append({"ph": "M", "name": "process_name", "pid": 10, "tid": 0, "args": {"name": "VLLM::Worker_TP0"}})
    return {"traceEvents": ev}


class AnalyzeTests(unittest.TestCase):
    @unittest.skipUnless(HAVE_IJSON, "ijson not installed")
    def test_extract_and_analyze_synthetic_trace(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.pt.trace.json.gz"
            with gzip.open(p, "wt") as fh:
                json.dump(_trace(), fh)
            x = dt.extract(str(p))
        res = dt.analyze(x, "synthetic")
        self.assertEqual(res["steps_total"], 2)
        g = res["groups"]["5"]
        self.assertEqual(g["verify_rows_m"], 4)
        roles = {r["role"]: r for r in g["gemm_roles"]}
        self.assertEqual(roles["wq_b"]["calls_per_step"], 2.0)
        self.assertEqual(roles["lm_head"]["calls_per_step"], 1.0)
        self.assertEqual(roles["engram_wkv"]["calls_per_step"], 1.0)
        self.assertEqual(roles["routed_moe_p2b"]["pairs"], 24)
        # 24 pairs x EXPERT_BYTES at 540 us
        self.assertAlmostEqual(roles["routed_moe_p2b"]["GBps_at_median"],
                               round(24 * dt.EXPERT_BYTES / 540e-6 / 1e9, 1))
        big = g["idle_gaps"][0]
        self.assertGreater(big["us_mean"], 2400)
        self.assertTrue(any("engram.py(1747): stage" in f for f, _ in big["host_frames_top"]))
        self.assertIn("step wall ms", dt.render(res))


if __name__ == "__main__":
    unittest.main()
