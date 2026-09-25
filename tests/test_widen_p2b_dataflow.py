#!/usr/bin/env python3
"""widen_p2b_dataflow.py (DSV41_P2B_COOP=2): patch anchors, untouched kernels, host gates,
task-list coverage mirrored in Python, and the image / lever / audit wiring. CPU only."""
from __future__ import annotations

import importlib.util
import random
import re
import sys
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PIN_CU = ROOT / "tests/fixtures/p2b_moe.pin.cu"
CHAIN = ("shapes", "mrow", "cfg1", "codebook", "fshift", "srcsort")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "docker/patch"))
sys.path.insert(0, str(ROOT / "tools"))
from test_widen_p2b_coop import coop_units, em  # noqa: E402


def _load(name: str, path: Path | None = None):
    path = path or ROOT / "docker/patch" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _block(src: str, start: str, end: str) -> str:
    a = src.index(start)
    return src[a : src.index(end, a)]


class DataflowPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df = _load("widen_p2b_dataflow")
        src = PIN_CU.read_text()
        for name in CHAIN:
            src = _load(f"widen_p2b_{name}").patch_cu(src)
        cls.coop = _load("widen_p2b_coop").patch_cu(src)
        cls.patched = cls.df.patch_cu(cls.coop)

    def test_applies_after_coop_and_is_idempotent(self) -> None:
        self.assertNotEqual(self.patched, self.coop)
        self.assertEqual(self.df.patch_cu(self.patched), self.patched)
        self.assertEqual(self.patched.count("grid.sync()"), self.coop.count("grid.sync()") + 2)
        with self.assertRaises(SystemExit):
            src = PIN_CU.read_text()
            for name in CHAIN:
                src = _load(f"widen_p2b_{name}").patch_cu(src)
            self.df.patch_cu(src)  # no coop stage

    def test_undoing_the_edits_gives_the_coop_source_back(self) -> None:
        d = self.df
        out = self.patched.replace(d.KERNEL.lstrip("\n") + "\n", "", 1)
        for old, new in ((d.LAUNCH_OLD, d.LAUNCH_NEW), (d.ACCUM_OLD, d.ACCUM_NEW), (d.INCLUDE_OLD, d.INCLUDE_NEW)):
            self.assertEqual(out.count(new), 1)
            out = out.replace(new, old, 1)
        self.assertEqual(out, self.coop)
        self.assertTrue(d.LAUNCH_NEW.startswith(d.LAUNCH_OLD) and d.INCLUDE_NEW.startswith(d.INCLUDE_OLD))

    def test_existing_kernels_are_untouched(self) -> None:
        for start, end in (
            ("template <int BITS, int CB, int SORT = 0>\n__global__", "// --- widen_p2b_dataflow"),
            ("template <int bits, int cb, int CFG>\n__device__ __forceinline__ void run_gemv_tile(", "// --- widen_p2b_coop"),
            ("template <int bits, int cb, int CFG>\n__device__ __forceinline__ void run_gemv_tile_coop(",
             "template <int BITS, int CB, int SORT = 0>"),
        ):
            self.assertEqual(_block(self.patched, start, end), _block(self.coop, start, end.replace(
                "// --- widen_p2b_dataflow", "template <int BITS, int CB>\nstatic void launch_moe_batched(")))

    def test_launch_gate_is_mode_2_k2_mcg_and_cap(self) -> None:
        p = self.patched
        self.assertIn('std::getenv("DSV41_P2B_COOP")', p)
        self.assertIn("return v == nullptr ? 0 : std::atoi(v);", p)
        self.assertIn('return v != nullptr && std::strcmp(v, "1") == 0;', p)  # round-2 coop still "1"
        block = _block(p, "    if constexpr (BITS == 2 && CB == 1) {", "    cudaOccupancyMaxActiveBlocksPerMultiprocessor")
        self.assertIn("if (p2b_coop_mode() == 2 && m * e <= P2B_SORT_CAP && p2b_df_ctl_words(inter) <= m * hidden) {", block)
        self.assertIn("kernel = (void*) p2b_coop_df_kernel<BITS, CB>;", block)
        self.assertEqual(p.count("p2b_coop_df_kernel<BITS, CB>"), 1)
        # One cooperative launch, the grid from the occupancy of the chosen kernel (3 blocks/SM).
        self.assertEqual(p.count("cudaLaunchCooperativeKernel("), 1)
        self.assertIn("__global__ __launch_bounds__(256, 3)\nvoid p2b_coop_df_kernel(", p)

    def test_host_df_condition_matches_the_launch(self) -> None:
        d = self.df
        self.assertIn("const bool df = mcg && kg == 2 && p2b_coop_mode() == 2 && m * e <= P2B_SORT_CAP && "
                      "p2b_df_ctl_words(inter) <= m * hidden;", d.ACCUM_NEW)
        self.assertIn("auto accum = df ? at::empty({m, hidden}", d.ACCUM_NEW)
        self.assertIn(": at::zeros({m, hidden}, x.options().dtype(at::kFloat));", d.ACCUM_NEW)
        host = self.patched[self.patched.index("at::Tensor p2b_fused_moe_cuda(") :]
        self.assertIn("launch_moe_batched<2, 1>(", host)
        self.assertEqual(host.count("at::empty("), self.coop[self.coop.index("at::Tensor p2b_fused_moe_cuda(") :].count("at::empty(") + 1)

    def test_markers_are_printed_once_in_the_extension(self) -> None:
        d = self.df
        self.assertIn(f'std::fprintf(stderr, "%s\\n", "{d.LOG_ENGAGED}");', self.patched)
        self.assertIn(f'"{d.LOG_DISARMED}",', self.patched)
        self.assertEqual(self.patched.count("static bool logged = false;"), 1)
        self.assertEqual(self.patched.count("static bool warned = false;"), 1)
        self.assertIn("lever is OFF", d.LOG_DISARMED)

    def test_tile_math_is_run_gemv_tile_coops(self) -> None:
        coop = _block(self.patched, "__device__ __forceinline__ void run_gemv_tile_coop(", "template <int BITS, int CB, int SORT = 0>")
        tile = _block(self.patched, "__device__ __forceinline__ void p2b_df_tile(", "// L2 prefetch of the first n k-slices")
        for line in (
            "const int chunk = CEIL_DIVIDE(kslices, WK);",
            "const int ks0 = warp * chunk;",
            "const int myn = max(0, min(chunk, kslices - ks0));",
            "const int r0 = lane >> 2;",
            "const bool r0_ok = r0 < len;",
            "bench_fshift::dq8_regs_2bits_fs<cb>(awv, bwv, lane << 3, f0, f1);",
            "exl3_gemv_ns::mma_ab_h(a01, a23, f0, ch[t][0]);",
            "exl3_gemv_ns::mma_ab_h(a01, a23, f1, ch[t][1]);",
            "if ((d + 1) % FOLD == 0 || i + 1 == myn) {",
            "acc0[t][f].x += __low2float(ch[t][f][0]);",
            "acc0[t][f].y += __high2float(ch[t][f][0]);",
            "sh_red[warp][r0][col + 0] = acc0[t][f].x;",
            "for (int j = 0; j < WK; ++j)",
            "sum += sh_red[j][r][c];",
            "const uint32_t* bp = B32 + (size_t) ks0 * slice_stride + group * WNT * TWORDS + lane;",
        ):
            self.assertIn(line, coop, line)
            self.assertIn(line.replace("<cb>", "<1>"), tile, line)
        for name, value in (("WK", 8), ("WNT", 4), ("PF", 2), ("FOLD", 2)):
            self.assertIn(f"constexpr int {name} = {value};", tile)
            self.assertIn(f"constexpr int {name} = CFG == 0 ? ", coop)
        # coop loads the ring through ld_b(i, l) = __ldcs(bp + i * slice_stride + l * LSTRIDE) (bits 2)
        self.assertIn("return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE);", coop)
        self.assertIn("pf[d][l] = ld_b(i + PF, l);", coop)
        self.assertIn("pf[d][l] = __ldcs(bp + (size_t) d * slice_stride + l * LSTRIDE);", tile)
        self.assertIn("pf[d][l] = __ldcs(bp + (size_t) (i + PF) * slice_stride + l * LSTRIDE);", tile)
        self.assertNotIn("atomicAdd", tile)

    def test_epilogues_repeat_p2bs_elementwise_math(self) -> None:
        p = self.patched
        had = _block(p, "__device__ __forceinline__ half4 p2b_df_had(", "// L2 load of 4 halves")
        for line in ("float h0 = s0 + s1;", "float h1 = d0 + d1;", "float h2 = s0 - s1;", "float h3 = d0 - d1;",
                     "shuffle_had_f4x32(h0, h1, h2, h3, lane);", "const float r = 0.088388347648f;",
                     "__floats2half2_rn(h0 * r, h1 * r)", "__floats2half2_rn(h2 * r, h3 * r)"):
            self.assertIn(line, had)
        swiglu = _block(p, "__device__ __forceinline__ half p2b_df_swiglu(", "// Epilogue of one 128-block")
        phase3 = _block(p, "    // Phase 3: SwiGLU activation", "        grid.sync();")
        for line in ("g = fminf(g, limit);", "u = fminf(fmaxf(u, -limit), limit);", "float s = g / (1.0f + expf(-g));",
                     "return __float2half(s * u);"):
            self.assertIn(line, swiglu)
            self.assertIn(line.replace("limit", "swiglu_limit").replace("return __float2half(s * u);",
                                                                          "had_down[j] = __float2half(s * u);"), phase3)
        out = _block(p, "__device__ __forceinline__ void p2b_df_out6(", "template <int BITS, int CB>\n__global__")
        self.assertIn("s0 = __fadd_rn(s0, __fmul_rn(wt[e], __half2float(__low2half(d.x))));", out)
        self.assertIn("sum = __fadd_rn(sum, __fmul_rn(__half2float(rw[row * experts + e]),", p)  # round-2 order

    def test_ordering_primitives(self) -> None:
        k = _block(self.patched, "void p2b_coop_df_kernel(", "template <int BITS, int CB>\nstatic void launch_moe_batched(")
        self.assertIn("atom.acq_rel.gpu.global.add.s32", self.patched)
        self.assertIn("red.release.gpu.global.add.s32", self.patched)
        self.assertIn("ld.acquire.gpu.global.b32", self.patched)
        self.assertEqual(k.count("grid.sync();"), 2)
        # The task index is re-read before the barrier that precedes thread 0's overwrite.
        gu = _block(k, "            // Count the tile", "        } else {")
        self.assertLess(gu.index("const int t2 = *static_cast<volatile int*>(s_df);"), gu.index("__syncthreads();"))
        self.assertIn("s_df[1] = p2b_df_atom_add_acq_rel(ctl + P2B_DF_ACT + u * hb_n + hb, 1) == 3;", gu)
        self.assertIn("p2b_df_red_release(ctl + P2B_DF_READY + u, 1);", gu)
        self.assertIn("while (p2b_df_ld_acquire(ctl + P2B_DF_READY + u) < hb_n)", k)

    def test_bench_switch_applies(self) -> None:
        for old, new in self.df.BENCH_SWITCHES:
            self.assertEqual(self.patched.count(old), 1)
            self.assertIn("return g_bench_variant;", new)


class DataflowScheduleTests(unittest.TestCase):
    """The kernel's task list and control words, mirrored in Python."""

    HIDDEN, INTER, TOPK = 5120, 1152, 6

    @staticmethod
    def tasks(units: int, hb_n: int, groups_down: int) -> list[tuple]:
        out = []
        gu_tasks = units * hb_n * 4
        for task in range(gu_tasks + units * groups_down):
            if task < gu_tasks:
                u, r = divmod(task, hb_n * 4)
                out.append(("gu", u, (r >> 2) * 2 + ((r >> 1) & 1), r & 1, r >> 2))
            else:
                u, g = divmod(task - gu_tasks, groups_down)
                out.append(("down", u, g))
        return out

    def test_every_tile_once_and_four_per_epilogue(self) -> None:
        sys.path.insert(0, str(ROOT / "tools"))
        from moe_census import synth_routing

        rng = random.Random(3)
        hb_n, groups_down = self.INTER // 128, self.HIDDEN // 64
        for m in range(1, 9):
            for dup in (0.0, 0.3):
                if m == 1 and dup:
                    continue
                rows = synth_routing(m, self.TOPK, 384, dup, rng)
                units = coop_units([x for r in rows for x in r])
                t = self.tasks(len(units), hb_n, groups_down)
                gu = [x for x in t if x[0] == "gu"]
                self.assertEqual(sorted((u, g, up) for _, u, g, up, _ in gu),
                                 sorted((u, g, up) for u in range(len(units)) for g in range(self.INTER // 64) for up in (0, 1)))
                per_hb = {}
                for _, u, g, up, hb in gu:
                    self.assertEqual(g // 2, hb)
                    per_hb[(u, hb)] = per_hb.get((u, hb), 0) + 1
                self.assertEqual(set(per_hb.values()), {4})  # the epilogue fires on the 4th tile (old == 3)
                down = [x for x in t if x[0] == "down"]
                self.assertEqual(sorted((u, g) for _, u, g in down),
                                 [(u, g) for u in range(len(units)) for g in range(groups_down)])
                self.assertTrue(all(i < t.index(down[0]) for i, x in enumerate(t) if x[0] == "gu"))
                # static first task per block + counter from gridDim.x: every task exactly once
                grid = 144
                claimed = list(range(min(grid, len(t)))) + [grid + n for n in range(max(0, len(t) - grid))]
                self.assertEqual(sorted(claimed), list(range(len(t))))

    def test_control_words_fit_in_accum(self) -> None:
        words = 32 + 64 + 64 * (self.INTER // 128)  # P2B_DF_ACT + P2B_SORT_CAP * hb_n
        self.assertEqual(words, 672)
        for m in range(1, 9):
            self.assertLessEqual(words, m * self.HIDDEN)
        src = (ROOT / "docker/patch/widen_p2b_dataflow.py").read_text()
        self.assertIn("constexpr int P2B_DF_READY = 32;", src)
        self.assertIn("constexpr int P2B_DF_ACT = P2B_DF_READY + P2B_SORT_CAP;", src)
        self.assertIn("return P2B_DF_ACT + P2B_SORT_CAP * (inter / 128);", src)

    def test_output_slots_are_the_coop_scratch_layout(self) -> None:
        # p2b_df_out reads down[(e * m + row) * hidden]: the slot the coop tile stores for pair row * K + e.
        for m in range(1, 9):
            self.assertEqual(sorted(em(p, m, self.TOPK) for p in range(m * self.TOPK)), list(range(m * self.TOPK)))


class DataflowWiringTests(unittest.TestCase):
    def test_dockerfile_e15(self) -> None:
        df = (ROOT / "docker/Dockerfile.e15").read_text()
        self.assertIn("FROM dsv41-flash-exl3-sm121:canonical-e13\n", df)
        ref = re.search(r"ARG VLLM_EXL3_REF=(\w+)", (ROOT / "docker/Dockerfile.e13").read_text()).group(1)
        self.assertIn(f"ARG VLLM_EXL3_REF={ref}\n", df)
        order = [df.index(f"widen_p2b_{n}.py /opt/vllm-exl3") for n in CHAIN + ("coop", "dataflow")]
        self.assertEqual(order, sorted(order))
        self.assertLess(order[-1], df.index("pip install --no-build-isolation --no-deps"))
        needle = _load("decode_levers").P2B_DATAFLOW_NEEDLE
        self.assertIn(f"b'{needle}' in so", df)
        self.assertTrue(_load("widen_p2b_dataflow").LOG_ENGAGED.startswith("dsv41: " + needle))
        label = re.search(r'LABEL dsv41.recipe.patches="([^"]+)"', df).group(1).split(",")
        self.assertEqual(label[label.index("coop") + 1], "dataflow")
        self.assertNotIn("widen_p2b_dataflow", (ROOT / "docker/Dockerfile").read_text())

    def test_lever_check_warns_on_extension_without_dataflow(self) -> None:
        import tempfile

        dl = _load("decode_levers")
        from engagement_audit import audit

        with tempfile.TemporaryDirectory() as td:
            so = Path(td) / "vllm_exl3_c.so"
            spec = types.SimpleNamespace(origin=str(so))
            with mock.patch.object(importlib.util, "find_spec", lambda name: spec):
                so.write_bytes(b"\x7fELF" + b"\0" * 4096 + b"DSV41_P2B_SRC_SORT\0DSV41_P2B_COOP\0")  # review-e14
                with redirect_stdout(StringIO()) as out:
                    dl.install({"DSV41_P2B_COOP": "2"})
                line = out.getvalue()
                self.assertIn("decode lever p2b_coop_dataflow:", line)
                self.assertIn("Rebuild docker/Dockerfile.e15", line)
                self.assertNotIn("decode lever p2b_coop:", line)
                self.assertTrue(any("p2b_coop_dataflow" in x for x in audit({"head": line}, {})))
                so.write_bytes(b"\x7fELF" + b"\0" * 4096 + b"DSV41_P2B_COOP\0" + dl.P2B_DATAFLOW_NEEDLE.encode() + b"\0")
                with redirect_stdout(StringIO()) as out:
                    dl.install({"DSV41_P2B_COOP": "2"})
                    dl.install({"DSV41_P2B_COOP": "1"})
                    dl.install({})
                self.assertEqual(out.getvalue(), "")

    def test_audit_expects_the_engaged_line_only_for_mode_2(self) -> None:
        import engagement_audit as ea

        d = _load("widen_p2b_dataflow")
        self.assertEqual(ea.PATCHES["widen_p2b_dataflow.py"], {"DSV41_P2B_COOP": "2"})
        expected, disarm = ea.expectations({"DSV41_P2B_COOP": "2"})
        self.assertIn(("widen_p2b_dataflow.py", d.LOG_ENGAGED), expected)
        self.assertIn(d.LOG_DISARMED, disarm)
        self.assertNotIn(("widen_p2b_dataflow.py", d.LOG_ENGAGED), ea.expectations({"DSV41_P2B_COOP": "1"})[0])
        mine = lambda problems: [x for x in problems if "p2b coop dataflow" in x]  # noqa: E731
        self.assertEqual(len(mine(ea.audit({"worker": "nothing\n"}, {"DSV41_P2B_COOP": "2"}))), 1)
        self.assertEqual(mine(ea.audit({"worker": d.LOG_ENGAGED + "\n"}, {"DSV41_P2B_COOP": "2"})), [])
        self.assertEqual(mine(ea.audit({"worker": "nothing\n"}, {"DSV41_P2B_COOP": "1"})), [])
        off = f"{d.LOG_DISARMED} (K=3 mcg=1 m=4 top_k=6)\n"
        problems = mine(ea.audit({"worker": d.LOG_ENGAGED + "\n" + off}, {"DSV41_P2B_COOP": "2"}))
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("lever is OFF", problems[0])

    def test_env_reaches_both_ranks(self) -> None:
        import run_sh_harness as h

        res = h.dry_run(DSV41_P2B_COOP="2")
        self.assertEqual(res["returncode"], 0, res["stdout"] + res["stderr"])
        for role in ("head", "worker"):
            self.assertEqual(h.container_env(res[role]).get("DSV41_P2B_COOP"), "2", role)


class DataflowBenchSourceTests(unittest.TestCase):
    def test_bench_builds_the_image_chain(self) -> None:
        mb = _load("make_bench_r3", ROOT / "kernel_study/p2b_coop/make_bench_r3.py")
        self.assertEqual(mb.CHAIN, CHAIN + ("coop",))
        self.assertEqual(mb.R3_PATCHES, ("dataflow",))
        srcs = mb.build()
        chain = PIN_CU.read_text()
        for name in CHAIN + ("coop", "dataflow"):
            chain = _load(f"widen_p2b_{name}").patch_cu(chain)
        self.assertEqual(srcs["chain_r3.cu"], chain)
        bench = srcs["bench_r3.cu"]
        self.assertIn("g_bench_variant == 1 && m * e <= P2B_SORT_CAP", bench)
        self.assertIn("static int p2b_coop_mode()\n{\n    return g_bench_variant;\n", bench)
        self.assertIn('mod.def("stream", &p2b_stream', bench)
        ts = srcs["bench_r3_ts.cu"]
        self.assertNotIn("p2b_stream", ts)
        self.assertEqual(ts.count("P2B_GRID_SYNC();"), bench.count("grid.sync();"))
        self.assertIn("DF_ENTRY();", ts)
        self.assertIn("DF_DUMP();", ts)
        # round 4: prologue sub-stamps (stamped build only) and the flat read-ceiling probe
        self.assertEqual(ts.count("P2B_TS(56);"), 1)
        self.assertEqual(ts.count("P2B_TS(57);"), 1)
        self.assertNotIn("P2B_TS(56);", bench)
        self.assertIn('mod.def("flat", &p2b_flat', bench)
        # the bench keeps the image's host conditions: only DSV41_P2B_COOP=2 takes the dataflow path
        self.assertEqual(bench.count("p2b_coop_mode() == 2"), chain.count("p2b_coop_mode() == 2"))


if __name__ == "__main__":
    unittest.main()
