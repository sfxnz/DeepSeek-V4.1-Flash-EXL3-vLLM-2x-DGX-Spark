#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import random
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN_CU = ROOT / "tests/fixtures/p2b_moe.pin.cu"
# docker/Dockerfile p2b chain order; srcsort goes last.
CHAIN = ("shapes", "mrow", "cfg1", "codebook", "fshift")


def _load(name: str):
    path = ROOT / "docker/patch" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dedent_lines(block: str) -> list[str]:
    return [line.strip() for line in block.strip("\n").splitlines()]


def sorted_item_pairs(ids: list[int], per_pair: int) -> list[tuple[int, int]]:
    """Python mirror of p2b_sort_build + p2b_sort_pair: item -> (pair, sub)."""
    pairs = len(ids)
    perm = [0] * pairs
    for p in range(pairs):
        rank = sum(1 for q in range(pairs) if ids[q] < ids[p] or (ids[q] == ids[p] and q < p))
        perm[rank] = p
    run_start, run_len = [0] * pairs, [0] * pairs
    for s in range(pairs):
        a, b = s, s + 1
        while a > 0 and ids[perm[a - 1]] == ids[perm[s]]:
            a -= 1
        while b < pairs and ids[perm[b]] == ids[perm[s]]:
            b += 1
        run_start[s], run_len[s] = a, b - a
    out = []
    for item in range(pairs * per_pair):
        s = item // per_pair
        s0, n = run_start[s], run_len[s]
        li = item - s0 * per_pair
        out.append((perm[s0 + li % n], li // n))
    return out


class SrcSortPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mrow = _load("widen_p2b_mrow")
        cls.srcsort = _load("widen_p2b_srcsort")
        src = PIN_CU.read_text()
        for name in CHAIN:
            src = _load(f"widen_p2b_{name}").patch_cu(src)
        cls.chain = src
        cls.patched = cls.srcsort.patch_cu(src)

    def test_applies_after_full_chain_and_is_idempotent(self) -> None:
        self.assertNotEqual(self.patched, self.chain)
        self.assertEqual(self.srcsort.patch_cu(self.patched), self.patched)
        self.assertEqual(self.patched.count("grid.sync()"), self.chain.count("grid.sync()"))
        self.assertIn("template <int BITS, int CB, int SORT = 0>", self.patched)
        self.assertIn("if constexpr (SORT) p2b_sort_build(ids, m * experts);", self.patched)

    def test_anchors_are_the_mrow_work_lists(self) -> None:
        gate = _dedent_lines(self.srcsort.GATE_OLD)
        down = _dedent_lines(self.srcsort.DOWN_OLD)
        mrow_gate = _dedent_lines(self.mrow.GEMV_GATE_NEW)
        mrow_down = _dedent_lines(self.mrow.GEMV_DOWN_NEW)
        self.assertEqual(mrow_gate[2 : 2 + len(gate)], gate)
        self.assertEqual(mrow_down[2 : 2 + len(down)], down)

    def test_off_branch_keeps_the_mrow_decode(self) -> None:
        def body(stmt: str) -> str:
            # "int x = expr;" and "x = expr;" are the same decode.
            return re.sub(r"^int (row|e|group|is_up) = ", r"\1 = ", stmt)

        for old, new in (
            (self.srcsort.GATE_OLD, self.srcsort.GATE_NEW),
            (self.srcsort.DOWN_OLD, self.srcsort.DOWN_NEW),
        ):
            off = new.split("} else {", 1)[1]
            self.assertEqual(
                [body(s) for s in _dedent_lines(old)],
                _dedent_lines(off)[: len(_dedent_lines(old))],
            )

    def test_default_off_env_and_cap(self) -> None:
        p = self.patched
        self.assertIn('std::getenv("DSV41_P2B_SRC_SORT")', p)
        self.assertIn('std::strcmp(v, "1") == 0', p)
        self.assertIn("p2b_src_sort_enabled() && m * e <= P2B_SORT_CAP", p)
        self.assertIn(": (void*) p2b_moe_batched_kernel<BITS, CB, 0>;", p)
        # Only the kernel pointer changes on the host; the dispatch stays.
        self.assertEqual(p.count("launch_moe_batched<2, 1>("), self.chain.count("launch_moe_batched<2, 1>("))
        # 8 rows x top-8 still fits the shared arrays.
        self.assertIn("constexpr int P2B_SORT_CAP = 64;", p)

    def test_loads_stay_streaming(self) -> None:
        # Sorting relies on in-flight reuse, not L2 retention (evidence/p2b-ldg).
        self.assertEqual(self.patched.count("__ldcs("), self.chain.count("__ldcs("))
        self.assertNotIn("__ldg(", self.patched)

    def test_needs_mrow_first(self) -> None:
        shaped = _load("widen_p2b_shapes").patch_cu(PIN_CU.read_text())
        with self.assertRaises(SystemExit):
            self.srcsort.patch_cu(shaped)


class SrcSortWiringTests(unittest.TestCase):
    def test_dockerfile_applies_srcsort_after_fshift(self) -> None:
        df = (ROOT / "docker/Dockerfile").read_text()
        fshift = df.index("widen_p2b_fshift.py /opt/vllm-exl3")
        srcsort = df.index("widen_p2b_srcsort.py /opt/vllm-exl3")
        self.assertLess(fshift, srcsort)
        self.assertLess(srcsort, df.index("pip install --no-build-isolation --no-deps /opt/vllm-exl3"))

    def test_env_reaches_both_ranks_only_when_set(self) -> None:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import run_sh_harness as h

        res = h.dry_run()
        self.assertEqual(res["returncode"], 0, res["stdout"] + res["stderr"])
        for role in ("head", "worker"):
            self.assertNotIn("DSV41_P2B_SRC_SORT", h.container_env(res[role]), role)
        res = h.dry_run(DSV41_P2B_SRC_SORT="1")
        self.assertEqual(res["returncode"], 0, res["stdout"] + res["stderr"])
        for role in ("head", "worker"):
            self.assertEqual(h.container_env(res[role]).get("DSV41_P2B_SRC_SORT"), "1", role)


class MicrobenchSourceTests(unittest.TestCase):
    def test_bench_toggles_sort_without_env(self) -> None:
        spec = importlib.util.spec_from_file_location("make_bench", ROOT / "kernel_study/p2b_srcsort/make_bench.py")
        assert spec is not None and spec.loader is not None
        mb = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mb)
        srcs = mb.build()
        self.assertEqual(sorted(srcs), ["bench_srcsort.cu", "chain_base.cu", "chain_srcsort.cu"])
        self.assertNotIn("widen_p2b_srcsort", srcs["chain_base.cu"])
        self.assertEqual(srcs["chain_srcsort.cu"], _load("widen_p2b_srcsort").patch_cu(srcs["chain_base.cu"]))
        bench = srcs["bench_srcsort.cu"]
        self.assertIn("g_bench_sort && m * e <= P2B_SORT_CAP", bench)
        self.assertNotIn("p2b_src_sort_enabled() && m * e", bench)
        self.assertIn('mod.def("set_sort"', bench)


class SortedItemOrderTests(unittest.TestCase):
    """Index math of p2b_sort_pair, mirrored in Python."""

    def _check(self, ids: list[int], per_pair: int) -> list[tuple[int, int]]:
        order = sorted_item_pairs(ids, per_pair)
        # Bijection onto every (pair, sub) the row-major list covers.
        self.assertEqual(sorted(order), [(p, s) for p in range(len(ids)) for s in range(per_pair)])
        return order

    def test_bijection_random_routing(self) -> None:
        rng = random.Random(3)
        for m in (1, 2, 4, 8):
            for _ in range(50):
                ids = [rng.randrange(12) for _ in range(m * 6)]
                self._check(ids, 36)
                self._check(ids, 80)

    def test_duplicates_are_adjacent_items(self) -> None:
        # m=2, top-3: expert 7 in both rows, pairs 0 and 4.
        ids = [7, 1, 2, 5, 7, 3]
        order = self._check(ids, 4)
        where = {(p, s): i for i, (p, s) in enumerate(order)}
        for sub in range(4):
            self.assertEqual(abs(where[(0, sub)] - where[(4, sub)]), 1)

    def test_no_duplicates_is_a_pair_permutation(self) -> None:
        ids = [9, 3, 5, 1]
        order = self._check(ids, 3)
        self.assertEqual([p for p, _ in order[::3]], [3, 1, 2, 0])
        self.assertEqual([s for _, s in order[:3]], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
