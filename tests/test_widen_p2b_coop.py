#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import random
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN_CU = ROOT / "tests/fixtures/p2b_moe.pin.cu"
# docker/Dockerfile.e13 p2b chain order (the live canonical-e13 code); coop goes last.
CHAIN = ("shapes", "mrow", "cfg1", "codebook", "fshift", "srcsort")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_widen_p2b_srcsort import sorted_item_pairs  # noqa: E402


def _load(name: str, path: Path | None = None):
    path = path or ROOT / "docker/patch" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def coop_units(ids: list[int], rows: int = 8) -> list[tuple[int, list[int]]]:
    """Python mirror of p2b_sort_build + p2b_coop_build: [(src, member pairs)] per chunk."""
    pairs = len(ids)
    perm = [0] * pairs
    for p in range(pairs):
        rank = sum(1 for q in range(pairs) if ids[q] < ids[p] or (ids[q] == ids[p] and q < p))
        perm[rank] = p
    units = []
    for s in range(pairs):
        a = s
        while a > 0 and ids[perm[a - 1]] == ids[perm[s]]:
            a -= 1
        b = s + 1
        while b < pairs and ids[perm[b]] == ids[perm[s]]:
            b += 1
        off = s - a
        if off % rows == 0:
            n = min(rows, b - a - off)
            units.append((ids[perm[s]], [perm[s + r] for r in range(n)]))
    return units


def em(pair: int, m: int, experts: int) -> int:
    """p2b_coop_em: pair (row * experts + e) -> scratch slot e * m + row."""
    return (pair % experts) * m + pair // experts


class CoopPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.coop = _load("widen_p2b_coop")
        src = PIN_CU.read_text()
        for name in CHAIN:
            src = _load(f"widen_p2b_{name}").patch_cu(src)
        cls.chain = src
        cls.patched = cls.coop.patch_cu(src)

    def test_applies_after_srcsort_and_is_idempotent(self) -> None:
        self.assertNotEqual(self.patched, self.chain)
        self.assertEqual(self.coop.patch_cu(self.patched), self.patched)
        self.assertEqual(self.patched.count("grid.sync()"), self.chain.count("grid.sync()"))
        self.assertIn("if constexpr (SORT == 2) p2b_coop_build(m * experts);", self.patched)

    def test_needs_srcsort_first(self) -> None:
        src = PIN_CU.read_text()
        for name in CHAIN[:-1]:
            src = _load(f"widen_p2b_{name}").patch_cu(src)
        with self.assertRaises(SystemExit):
            self.coop.patch_cu(src)

    def test_sort0_and_sort1_source_is_unchanged(self) -> None:
        """Every edit is an insertion or a SORT == 2 ? 0 : <old> guard; undoing them gives the chain back."""
        c = self.coop
        out = self.patched.replace(c.HELPERS.lstrip("\n") + "\n" + c.coop_tile(self.chain) + "\n", "", 1)
        for old, new in ((c.BUILD_OLD, c.BUILD_NEW), (c.GATE_OLD, c.GATE_NEW), (c.DOWN_OLD, c.DOWN_NEW),
                         (c.REDUCE_OLD, c.REDUCE_NEW), (c.LAUNCH_OLD, c.LAUNCH_NEW)):
            self.assertIn(new, out)
            out = out.replace(new, old, 1)
        self.assertEqual(out, self.chain)
        for old, new in ((c.GATE_OLD, c.GATE_NEW), (c.DOWN_OLD, c.DOWN_NEW)):
            expr = old.strip().removeprefix("int total_work = ").removesuffix(";")
            self.assertTrue(new.rstrip().endswith(f"int total_work = SORT == 2 ? 0 : {expr};"))
            self.assertTrue(new.startswith("        if constexpr (SORT == 2) {\n"))
        self.assertTrue(c.REDUCE_NEW.startswith("        if constexpr (SORT == 2) {\n"))
        self.assertTrue(c.BUILD_NEW.startswith(c.BUILD_OLD) and c.LAUNCH_NEW.startswith(c.LAUNCH_OLD))

    def test_coop_tile_is_the_p2b_tile_except_the_row_edits(self) -> None:
        c = self.coop
        base = self.chain[self.chain.index(c.TILE_START) : self.chain.index(c.TILE_END)].rstrip("\n") + "\n"
        tile = c.coop_tile(self.chain)
        for old, new, _ in reversed(c.TILE_EDITS):
            self.assertEqual(tile.count(new), 1, new)
            tile = tile.replace(new, old, 1)
        self.assertEqual(tile, base)
        # Streaming loads, decode and MMA are the p2b ones: same count of each.
        coop = c.coop_tile(self.chain)
        for token in ("__ldcs(", "mma_ab_h(", "dq8_regs_2bits_fs<cb>", "__shfl_sync(", "__syncthreads()"):
            self.assertEqual(coop.count(token), base.count(token), token)
        self.assertNotIn("__ldg(", self.patched)
        self.assertNotIn("atomicAdd", c.REDUCE_NEW)

    def test_default_off_env_k2_mcg_only(self) -> None:
        p = self.patched
        self.assertIn('std::getenv("DSV41_P2B_COOP")', p)
        self.assertIn('std::strcmp(v, "1") == 0', p)
        self.assertIn("if constexpr (BITS == 2 && CB == 1) {", p)
        self.assertIn("if (p2b_coop_enabled() && m * e <= P2B_SORT_CAP)", p)
        self.assertEqual(p.count("p2b_moe_batched_kernel<BITS, CB, 2>"), 1)
        # Host dispatch, checks and scratch allocations are p2b's.
        host = lambda s: s[s.index("at::Tensor p2b_fused_moe_cuda(") :]  # noqa: E731
        self.assertEqual(host(p), host(self.chain))
        self.assertEqual(p.count("at::empty("), self.chain.count("at::empty("))
        self.assertEqual(p.count("cudaLaunchCooperativeKernel("), 1)

    def test_rows_fit_the_mma_and_reduction(self) -> None:
        self.assertEqual(self.coop.ROWS, 8)  # lane / 4 in 0..7: MMA rows a0/a2 only
        self.assertIn("constexpr int P2B_COOP_ROWS = 8;", self.patched)
        self.assertIn("typedef float P2bCoopRed[P2B_COOP_ROWS][64];", self.patched)
        self.assertIn("__shared__ P2bCoopRed s[8];", self.patched)  # WK = 8 at CFG 1
        self.assertIn("run_gemv_tile_coop<BITS, CB, 1>(", self.patched)


class CoopScheduleTests(unittest.TestCase):
    """Index math of p2b_coop_build + the coop work lists, mirrored in Python."""

    def _check(self, ids: list[int], m: int, experts: int) -> list[tuple[int, list[int]]]:
        units = coop_units(ids)
        members = [p for _, mem in units for p in mem]
        # Every (row, expert) pair lands in exactly one chunk, with its own expert.
        self.assertEqual(sorted(members), list(range(len(ids))))
        for src, mem in units:
            self.assertTrue(1 <= len(mem) <= 8)
            self.assertTrue(all(ids[p] == src for p in mem))
        # Scratch slots are p2b's e * m + row, one writer each.
        slots = [em(p, m, experts) for p in members]
        self.assertEqual(sorted(slots), list(range(m * experts)))
        # The chunk members are the srcsort order: a coop chunk is a run of p2b_sort_pair items.
        order = [p for p, sub in sorted_item_pairs(ids, 1)]
        self.assertEqual(members, order)
        return units

    def test_distinct_topk_one_chunk_per_unique_expert(self) -> None:
        rng = random.Random(5)
        sys.path.insert(0, str(ROOT / "tools"))
        from moe_census import synth_routing

        for m in (1, 2, 4, 8):
            for dup in (0.0, 0.3, 0.5):
                if m == 1 and dup:
                    continue
                for _ in range(20):
                    rows = synth_routing(m, 6, 384, dup, rng)
                    ids = [x for r in rows for x in r]
                    units = self._check(ids, m, 6)
                    self.assertEqual(len(units), len(set(ids)))
                    self.assertEqual(len(ids) - len(units), round(dup * m * 6))

    def test_clamped_duplicates_split_into_chunks_of_8(self) -> None:
        # Invalid ids are clamped to 0 by the plugin (weight 0), so one row can repeat an expert.
        ids = [0] * 48
        units = self._check(ids, 8, 6)
        self.assertEqual([len(mem) for _, mem in units], [8] * 6)
        rng = random.Random(9)
        for _ in range(50):
            m = rng.randrange(1, 9)
            self._check([rng.randrange(3) for _ in range(m * 6)], m, 6)

    def test_work_counts(self) -> None:
        # m=4, top-6, dup 0.3 -> 7 repeats -> 17 chunks: 2*17*18 gate/up and 17*80 down items vs 864 / 1920.
        ids = [1, 2, 3, 4, 5, 6, 1, 2, 3, 7, 8, 9, 1, 2, 10, 11, 12, 13, 3, 14, 15, 16, 17, 7]
        units = coop_units(ids)
        self.assertEqual(len(units), 17)
        self.assertEqual((2 * len(units) * (1152 // 64), len(units) * (5120 // 64)), (612, 1360))


class CoopWiringTests(unittest.TestCase):
    def test_dockerfile_e14_builds_coop_on_canonical_e13(self) -> None:
        df = (ROOT / "docker/Dockerfile.e14").read_text()
        self.assertIn("FROM dsv41-flash-exl3-sm121:canonical-e13\n", df)
        e13 = (ROOT / "docker/Dockerfile.e13").read_text()
        ref = re.search(r"ARG VLLM_EXL3_REF=(\w+)", e13).group(1)
        self.assertIn(f"ARG VLLM_EXL3_REF={ref}\n", df)
        order = [df.index(f"widen_p2b_{n}.py /opt/vllm-exl3") for n in CHAIN + ("coop",)]
        self.assertEqual(order, sorted(order))
        self.assertLess(order[-1], df.index("pip install --no-build-isolation --no-deps"))
        self.assertIn("b'DSV41_P2B_COOP' in", df)
        self.assertIn('LABEL dsv41.recipe.patches="', df)
        label = re.search(r'LABEL dsv41.recipe.patches="([^"]+)"', df).group(1).split(",")
        self.assertIn("coop", label)
        self.assertNotIn("widen_p2b_coop", (ROOT / "docker/Dockerfile").read_text())

    def test_env_reaches_both_ranks_only_when_set(self) -> None:
        import run_sh_harness as h

        res = h.dry_run()
        self.assertEqual(res["returncode"], 0, res["stdout"] + res["stderr"])
        for role in ("head", "worker"):
            self.assertNotIn("DSV41_P2B_COOP", h.container_env(res[role]), role)
        res = h.dry_run(DSV41_P2B_COOP="1")
        self.assertEqual(res["returncode"], 0, res["stdout"] + res["stderr"])
        for role in ("head", "worker"):
            self.assertEqual(h.container_env(res[role]).get("DSV41_P2B_COOP"), "1", role)


class CoopMicrobenchSourceTests(unittest.TestCase):
    def test_bench_toggles_coop_without_env(self) -> None:
        mb = _load("make_bench_coop", ROOT / "kernel_study/p2b_coop/make_bench.py")
        srcs = mb.build()
        self.assertEqual(sorted(srcs), ["bench_coop.cu", "chain_coop.cu", "chain_srcsort.cu"])
        self.assertIn("widen_p2b_srcsort", srcs["chain_srcsort.cu"])
        self.assertNotIn("widen_p2b_coop", srcs["chain_srcsort.cu"])
        self.assertEqual(srcs["chain_coop.cu"], _load("widen_p2b_coop").patch_cu(srcs["chain_srcsort.cu"]))
        bench = srcs["bench_coop.cu"]
        self.assertIn("g_bench_coop && m * e <= P2B_SORT_CAP", bench)
        self.assertNotIn("p2b_coop_enabled() && m * e", bench)
        for name in ('"set_coop"', '"occupancy"', '"p2b_fused_moe"'):
            self.assertIn(f"mod.def({name}", bench)

    def test_driver_geometry_matches_the_served_layer(self) -> None:
        text = (ROOT / "kernel_study/p2b_coop/driver.py").read_text()
        self.assertIn("HIDDEN, INTER, EXPERTS, TOPK = 5120, 1152, 384, 6", text)
        self.assertIn("SWIGLU_LIMIT = 10.0", text)
        self.assertIn('ap.add_argument("--dups", default="0,0.3,0.5")', text)
        self.assertIn('ap.add_argument("--m", type=int, default=4', text)
        self.assertIn("2, 2, 2, True, INTER, SWIGLU_LIMIT", text)  # K=2 gate/up/down, MCG
        self.assertTrue((ROOT / "results/2026-09-24-review/campaign/s10-diag-census-skew/census_npy").is_dir())


if __name__ == "__main__":
    unittest.main()
