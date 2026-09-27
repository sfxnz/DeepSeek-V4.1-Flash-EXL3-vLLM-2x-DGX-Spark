#!/usr/bin/env python3
"""CPU tests for tools/requant_full.py host logic (the encode runs on the GPU)."""
from __future__ import annotations

import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import requant_full as rf  # noqa: E402

UNITS = {"00003": "model-00003-of-00048.safetensors", "00004": "model-00004-of-00048.safetensors",
         "00041": "model-00041-of-00048.safetensors", "00042": "model-00042-of-00048.safetensors"}


def bash_store(root: Path) -> rf.SshStore:
    """SshStore whose 'ssh host' is a local bash: exercises the real remote scripts."""
    return rf.SshStore(["bash", "-c"], root)


def write_shard(path: Path, hdr: dict, payload: bytes = b"") -> None:
    raw = json.dumps(hdr).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


class PureLogicTests(unittest.TestCase):
    def test_units_are_routed_expert_shards_only(self) -> None:
        wm = {
            "layers.0.ffn.experts.0.w1.weight": UNITS["00003"],
            "layers.0.ffn.experts.0.w1.scale": UNITS["00003"],
            "layers.0.attn.wq_a.weight": UNITS["00003"],
            "layers.39.ffn.experts.7.w2.weight": UNITS["00042"],
            "layers.0.ffn.shared_experts.w1.weight": "model-00001-of-00048.safetensors",
            "mtp.0.ffn.experts.0.w1.weight": "model-00044-of-00048.safetensors",
            "embed.weight": "model-00001-of-00048.safetensors",
        }
        self.assertEqual(rf.units_from_index(wm), {"00003": UNITS["00003"], "00042": UNITS["00042"]})
        with self.assertRaises(ValueError):
            rf.units_from_index({"layers.0.ffn.experts.0.w1.weight": "pytorch_model.bin"})

    def test_order(self) -> None:
        self.assertEqual(rf.ordered(UNITS, "asc"), ["00003", "00004", "00041", "00042"])
        self.assertEqual(rf.ordered(UNITS, "desc"), ["00042", "00041", "00004", "00003"])
        with self.assertRaises(ValueError):
            rf.ordered(UNITS, "random")

    def test_seed_is_stable_and_per_tensor(self) -> None:
        a = rf.tensor_seed("layers.0.ffn.experts.0.w1")
        self.assertEqual(a, rf.tensor_seed("layers.0.ffn.experts.0.w1"))
        self.assertEqual(a, 0xFFFFFFFF & a)
        seeds = {rf.tensor_seed(f"layers.{l}.ffn.experts.{e}.w{k}") for l in range(3) for e in range(64)
                 for k in (1, 2, 3)}
        self.assertEqual(len(seeds), 3 * 64 * 3)

    def test_layout_problems(self) -> None:
        ref = {"__metadata__": {"x": "y"},
               "a.trellis": {"dtype": "I16", "shape": [320, 144, 32], "data_offsets": [0, 2949120]},
               "a.suh": {"dtype": "F16", "shape": [5120], "data_offsets": [2949120, 2959360]}}
        self.assertEqual(rf.layout_problems(json.loads(json.dumps(ref)), ref), [])
        self.assertEqual(rf.layout_problems({k: v for k, v in ref.items() if k != "__metadata__"}, ref), [])
        for field, val in (("dtype", "BF16"), ("shape", [144, 320, 32]), ("data_offsets", [8, 2949128])):
            new = json.loads(json.dumps(ref))
            new["a.trellis"][field] = val
            probs = rf.layout_problems(new, ref)
            self.assertEqual(len(probs), 1)
            self.assertIn(f"a.trellis.{field}", probs[0])
        new = json.loads(json.dumps(ref))
        new["b.svh"] = new.pop("a.suh")
        self.assertEqual(len(rf.layout_problems(new, ref)), 2)

    def test_summary_and_gate_fixed_before_run(self) -> None:
        self.assertEqual(rf.GATE_RATIO, 0.97)
        s = rf.summarize([3, 1, 2, 4])
        self.assertEqual((s["n"], s["min"], s["max"], s["mean"]), (4, 1.0, 4.0, 2.5))
        self.assertEqual(rf.summarize([]), {"n": 0})
        # s1 probe (exllamav3 1.5.1): viterbi+refit 0.2616 vs stock 0.3773.
        rows = [{"tensor": "t.w1", "final": 0.2616, "stock": 0.3773, "viterbi": 0.2617, "refit": True}]
        g = rf.unit_gate(rows)
        self.assertTrue(g["ok"])
        self.assertAlmostEqual(g["mse_ratio"], 0.4807, places=3)
        self.assertFalse(rf.unit_gate([dict(rows[0], final=0.3700)])["ok"])  # only 2% better
        self.assertFalse(rf.unit_gate([dict(rows[0], final=float("nan"))])["ok"])
        self.assertFalse(rf.unit_gate([])["ok"])
        st = rf.unit_stats(rows + [dict(rows[0], tensor="t.w2", refit=False, final=0.2620)])
        self.assertEqual(st["refit_kept"], 1)
        self.assertAlmostEqual(st["final_w2_mean"], 0.2620)
        self.assertIsNone(st["final_w3_mean"])
        self.assertIsNone(st["t_enc_mean"])
        timed = [dict(rows[0], t_enc=1.8, t_all=2.0), dict(rows[0], t_enc=2.0, t_all=2.4)]
        self.assertAlmostEqual(rf.unit_stats(timed)["t_all_mean"], 2.2)

    def test_eta(self) -> None:
        self.assertAlmostEqual(rf.eta_seconds(1152, {"spark1": 0.5}), 2304.0)
        self.assertAlmostEqual(rf.eta_seconds(1152, {"spark1": 0.5, "spark2": 0.5}), 1152.0)
        self.assertIsNone(rf.eta_seconds(10, {}))

    def test_serve_format_constants(self) -> None:
        self.assertEqual((rf.K, rf.CODEBOOK, rf.REF_REV, rf.IMAGE), (2, "mcg", "2.0bpw-mcg", "dsv41-quant151"))
        src = (ROOT / "tools/requant_full.py").read_text()
        self.assertIn('os.environ.get("DSV41_PACK_PF_G8", "0") == "1"', src)
        self.assertIn("torch.manual_seed(tensor_seed(stem))", src)
        self.assertIn("refit_identity(w, q.float())", src)
        self.assertIn("_quantize_fast([w.clone()]", src)  # regularize mutates fp32 CUDA input
        self.assertNotIn("greedy=True", src)

    def test_quantize_fast_tile_chunk_defaults_to_history(self) -> None:
        import inspect

        import quantize_experts_exl3 as q

        self.assertEqual(inspect.signature(q._quantize_fast).parameters["tile_chunk"].default, 256)
        self.assertIn("chunk = int(tile_chunk)", inspect.getsource(q._quantize_fast))


class StoreTests(unittest.TestCase):
    def _exercise(self, store: rf.LocalStore) -> None:
        self.assertTrue(store.mkdir("claims"))
        self.assertTrue(store.mkdir("claims/00003"))
        self.assertFalse(store.mkdir("claims/00003"))
        store.write("claims/00003/owner.json", '{"node": "spark1"}\n')
        store.write("claims/00003/owner.json", '{"node": "spark2"}\n')
        self.assertEqual(json.loads(store.read("claims/00003/owner.json")), {"node": "spark2"})
        self.assertIsNone(store.read("claims/00003/none.json"))
        self.assertEqual(store.ls("claims"), ["00003"])
        self.assertEqual(store.ls("claims/00003"), ["owner.json"])  # no leftover temp files
        self.assertEqual(store.ls("nope"), [])
        store.write("text", "a b\n'c' $d\n")
        self.assertEqual(store.read("text"), "a b\n'c' $d\n")

    def test_local(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self._exercise(rf.LocalStore(Path(d)))

    def test_remote_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "with space"
            root.mkdir()
            self._exercise(bash_store(root))

    def test_remote_mkdir_without_parent_raises(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                bash_store(Path(d) / "missing").mkdir("claims/00003")

    def test_ssh_failure_raises(self) -> None:
        def runner(argv, **kw):
            return subprocess.CompletedProcess(argv, 255, "", "Connection refused")

        with self.assertRaisesRegex(RuntimeError, "ssh failed"):
            rf.SshStore(["ssh", "spark1"], Path("/x"), runner=runner).read("units.json")

    def test_ssh_argv_uses_mounted_config(self) -> None:
        argv = rf.ssh_argv("spark1", Path("/home/sfxnz/.ssh"))
        self.assertEqual(argv[:3], ["ssh", "-F", "/home/sfxnz/.ssh/config"])
        self.assertIn("UserKnownHostsFile=/home/sfxnz/.ssh/known_hosts", argv)
        self.assertIn("BatchMode=yes", argv)
        self.assertEqual(argv[-1], "spark1")


class ClaimTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.root = self.out / "state"
        self.root.mkdir()
        self.local = rf.LocalStore(self.root)
        self.remote = bash_store(self.root)  # spark2's view of the same dir
        rf.init_state(self.local, UNITS)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_init_state_refuses_other_units(self) -> None:
        rf.init_state(self.remote, UNITS)
        with self.assertRaises(SystemExit):
            rf.init_state(self.local, {"00003": UNITS["00003"]})

    def test_two_nodes_meet_without_overlap(self) -> None:
        got = {"spark1": [], "spark2": []}
        for _ in range(3):
            for node, store, order in (("spark1", self.local, "asc"), ("spark2", self.remote, "desc")):
                u, resumed = rf.pick_unit(UNITS, order, store, node)
                if u is not None:
                    self.assertFalse(resumed)
                    got[node].append(u)
                    store.write(f"done/{u}.json", json.dumps({"node": node}))
        self.assertEqual(got, {"spark1": ["00003", "00004"], "spark2": ["00042", "00041"]})
        self.assertEqual(rf.pick_unit(UNITS, "asc", self.local, "spark1"), (None, None))
        owner = json.loads(self.local.read("claims/00042/owner.json"))
        self.assertEqual(owner["node"], "spark2")

    def test_resume_own_claim_skip_foreign_claim(self) -> None:
        self.assertEqual(rf.pick_unit(UNITS, "asc", self.local, "spark1"), ("00003", False))
        self.assertEqual(rf.pick_unit(UNITS, "desc", self.remote, "spark2"), ("00042", False))
        # Both workers die mid-unit and restart: each resumes its own claim only.
        self.assertEqual(rf.pick_unit(UNITS, "asc", self.local, "spark1"), ("00003", True))
        self.assertEqual(rf.pick_unit(UNITS, "desc", self.remote, "spark2"), ("00042", True))
        self.assertEqual(rf.pick_unit(UNITS, "desc", self.local, "spark1"), ("00041", False))

    def test_can_run_filter(self) -> None:
        # spark2 holds only source shards 23-42.
        u = rf.pick_unit(UNITS, "asc", self.remote, "spark2", can_run=lambda x: int(x) >= 23)
        self.assertEqual(u, ("00041", False))

    def test_manifest_and_done_marker(self) -> None:
        def rec(u, node, mean):
            return {"unit": u, "file": UNITS[u], "node": node, "sha256": "ab" * 32, "size": 7,
                    "sec_per_tensor": 2.0, "rows": [1, 2],
                    "stats": {"final": {"mean": mean}, "stock": {"mean": 0.3773}}}

        for i, u in enumerate(list(UNITS)[:3]):
            self.local.write(f"done/{u}.json", json.dumps(rec(u, "spark1" if i < 2 else "spark2", 0.2616)))
        man = rf.refresh_manifest(self.local, UNITS)
        self.assertFalse(man["complete"])
        self.assertIsNone(self.local.read("DONE"))
        self.assertNotIn("rows", man["units"]["00003"])
        self.remote.write("done/00042.json", json.dumps(rec("00042", "spark2", 0.2616)))
        man = rf.refresh_manifest(self.remote, UNITS)
        self.assertTrue(man["complete"])
        done = json.loads(self.local.read("DONE"))
        self.assertAlmostEqual(done["mse_ratio_vs_stock"], (0.2616 / 0.3773) ** 2)
        first = self.local.read("DONE")
        rf.refresh_manifest(self.local, UNITS)
        self.assertEqual(self.local.read("DONE"), first)
        self.assertEqual(json.loads(self.local.read("manifest.json"))["units"]["00042"]["node"], "spark2")

    def test_status_eta(self) -> None:
        now = time.time()
        self.local.write("done/00003.json", json.dumps({"node": "spark1", "sec_per_tensor": 2.5, "wall_s": 3456,
                                                        "tensors": 1152,
                                                        "stats": {"final": {"mean": 0.26}, "stock": {"mean": 0.38}}}))
        for u, node, age, k in (("00004", "spark1", 30, 576), ("00042", "spark2", 3600, 100)):
            self.local.mkdir(f"claims/{u}")
            self.local.write(f"claims/{u}/owner.json", json.dumps({"node": node}))
            hb_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - age))
            self.local.write(f"claims/{u}/heartbeat.json", json.dumps(
                {"time": hb_time, "tensors_done": k, "tensors": 1152, "sec_per_tensor": 2.0}))
        rep = rf.status_report(self.local, now=now)
        self.assertEqual(rep["units_done"], 1)
        self.assertFalse(rep["inflight"]["00004"]["stale"])
        self.assertTrue(rep["inflight"]["00042"]["stale"])
        self.assertEqual(rep["active_nodes"], ["spark1"])
        # remaining: 00041 whole + 00004 half + 00042 minus its 100.
        self.assertEqual(rep["remaining_tensors"], 1152 + 576 + 1052)
        self.assertAlmostEqual(rep["eta_s"], (1152 + 576 + 1052) * 2.0)
        with redirect_stdout(io.StringIO()) as out:
            rf.main(["status", "--out", str(self.out)])
        self.assertIn("units 1/4 done {'spark1': 1}", out.getvalue())
        self.assertIn("00042 on spark2", out.getvalue())
        self.assertIn("STALE", out.getvalue())
        self.assertIn("ETA 20", out.getvalue())


    def test_rate_from_done_wall_time(self) -> None:
        for u, wall in (("00003", 3456), ("00004", 2304), ("00041", 4608)):
            self.local.write(f"done/{u}.json", json.dumps({"node": "spark1", "sec_per_tensor": 1.0, "wall_s": wall,
                                                           "tensors": 1152}))
        rep = rf.status_report(self.local)
        self.assertAlmostEqual(rep["rates_tensors_per_s"]["spark1"], 1 / 3.0)  # median wall/tensor
        self.assertEqual(rep["active_nodes"], [])
        self.assertIsNone(rep["eta_s"])


class LaunchTests(unittest.TestCase):
    def test_spark1_local_state(self) -> None:
        a = rf.launch_argv("spark1", "asc", gpu_lock="/tmp/l.lock")
        s = " ".join(a)
        for want in ("docker run -d --name dsv41-requant --restart no --gpus all", "--network none",
                     "/home/sfxnz/.cache/huggingface:/cache/huggingface:ro", "--node spark1 --order asc",
                     "/home/sfxnz/projects/data/dsv41-requant-viterbi/code:/repo:ro", "--gpu-lock /tmp/l.lock",
                     "--user 1000:1000", "dsv41-quant151 -S /repo/tools/requant_full.py run"):
            self.assertIn(want, s)
        self.assertNotIn("--state-host", s)
        self.assertNotIn("/usr/bin/ssh", s)

    def test_spark2_claims_over_ssh(self) -> None:
        s = " ".join(rf.launch_argv("spark2", "desc", state_host="spark1"))
        for want in ("--network host", "/usr/bin/ssh:/usr/bin/ssh:ro", "/home/sfxnz/.ssh:/home/sfxnz/.ssh:ro",
                     "--node spark2 --order desc", "--state-host spark1 --ssh-dir /home/sfxnz/.ssh"):
            self.assertIn(want, s)

    def test_cli_prints_the_same_command(self) -> None:
        with redirect_stdout(io.StringIO()) as out:
            rf.main(["launch-cmd", "--node", "spark2", "--order", "desc", "--state-host", "spark1"])
        self.assertEqual(out.getvalue().split(), rf.launch_argv("spark2", "desc", state_host="spark1"))


class AssembleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.out, self.hub = t / "out", t / "hub"
        snaps = self.hub / rf.PACK_REPO / "snapshots"
        ref, base = snaps / rf.REF_REV, snaps / rf.BASE_REV
        ref.mkdir(parents=True)
        base.mkdir()
        wm = {}
        for i in (1, 3, 4, 43):
            f = f"model-{i:05d}-of-00048.safetensors"
            key = f"layers.{i}.ffn.experts.0.w1.trellis" if i in (3, 4) else f"t{i}"
            write_shard(ref / f, {key: {"dtype": "I16", "shape": [1], "data_offsets": [0, 2]}}, b"\0\0")
            wm[key] = f
        (ref / "config.json").write_text("{}")
        for p in ref.iterdir():
            if p.name != "model-00043-of-00048.safetensors":
                os.symlink(f"../{rf.REF_REV}/{p.name}", base / p.name)
        # lm_head MXFP8 shard 43 is a real file in the base revision.
        write_shard(base / "model-00043-of-00048.safetensors",
                    {"lm_head.weight": {"dtype": "F8_E4M3", "shape": [1], "data_offsets": [0, 1]}}, b"\0")
        wm.pop("t43")
        wm["lm_head.weight"] = "model-00043-of-00048.safetensors"
        (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
        self.units = {}
        (self.out / "shards").mkdir(parents=True)
        (self.out / "state").mkdir()
        for u in ("00003", "00004"):
            f = self.out / "shards" / f"model-{u}-of-00048.safetensors"
            write_shard(f, {f"layers.{int(u)}.ffn.experts.0.w1.trellis":
                            {"dtype": "I16", "shape": [1], "data_offsets": [0, 2]}}, b"\1\1")
            self.units[u] = {"unit": u, "file": f.name, "node": "spark1", "size": f.stat().st_size,
                             "sha256": rf.sha256_file(f),
                             "stats": {"final": {"mean": 0.2616}, "stock": {"mean": 0.3773}}}
        st = rf.LocalStore(self.out / "state")
        rf.init_state(st, {u: r["file"] for u, r in self.units.items()})
        for u, r in self.units.items():
            st.write(f"done/{u}.json", json.dumps(r))
        # A stale manifest (one unit behind) must not drive the assembly.
        st.write("manifest.json", json.dumps({"units": {"00003": self.units["00003"]}}))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _assemble(self) -> str:
        with redirect_stdout(io.StringIO()) as out:
            rf.main(["assemble", "--out", str(self.out), "--node", "spark1", "--hub", str(self.hub)])
        return out.getvalue()

    def test_requires_done(self) -> None:
        with self.assertRaisesRegex(SystemExit, "DONE"):
            self._assemble()

    def test_builds_linked_snapshot(self) -> None:
        rf.LocalStore(self.out / "state").write("DONE", "{}")
        self.assertIn("2 re-encoded shards", self._assemble())
        snap = self.hub / rf.PACK_REPO / "snapshots" / rf.NEW_REV
        for u in ("00003", "00004"):
            f = f"model-{u}-of-00048.safetensors"
            self.assertFalse((snap / f).is_symlink())
            self.assertEqual((snap / f).stat().st_ino, (self.out / "shards" / f).stat().st_ino)
        self.assertEqual(os.readlink(snap / "model-00001-of-00048.safetensors"),
                         f"../{rf.REF_REV}/model-00001-of-00048.safetensors")
        self.assertEqual(os.readlink(snap / "model-00043-of-00048.safetensors"),
                         f"../{rf.BASE_REV}/model-00043-of-00048.safetensors")
        self.assertFalse((snap / "model.safetensors.index.json").is_symlink())
        self.assertTrue((snap / "requant-viterbi-manifest.json").is_file())
        self.assertIn("2 re-encoded shards", self._assemble())  # idempotent

    def test_requires_every_unit(self) -> None:
        st = rf.LocalStore(self.out / "state")
        st.write("DONE", "{}")
        os.unlink(self.out / "state" / "done" / "00004.json")
        with self.assertRaisesRegex(SystemExit, "1/2 units"):
            self._assemble()

    def test_rejects_tampered_unit(self) -> None:
        rf.LocalStore(self.out / "state").write("DONE", "{}")
        f = self.out / "shards" / "model-00004-of-00048.safetensors"
        f.write_bytes(f.read_bytes()[:-1] + b"\2")
        with self.assertRaisesRegex(SystemExit, "sha256"):
            self._assemble()


class ReuseTests(unittest.TestCase):
    def test_reuse_needs_matching_verified_output(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "units").mkdir()
            (out / "shards").mkdir()
            f = out / "shards" / UNITS["00003"]
            f.write_bytes(b"x" * 100)
            rec = {"written": True, "gate": {"ok": True}, "size": 100, "sha256": rf.sha256_file(f), "rows": [1]}
            (out / "units" / "00003.json").write_text(json.dumps(rec))
            got = rf.reuse_local(out, "00003", UNITS["00003"])
            self.assertEqual(got["sha256"], rec["sha256"])
            self.assertNotIn("rows", got)
            (out / "units" / "00003.json").write_text(json.dumps(dict(rec, gate={"ok": False})))
            self.assertIsNone(rf.reuse_local(out, "00003", UNITS["00003"]))
            (out / "units" / "00003.json").write_text(json.dumps(rec))
            f.write_bytes(b"y" * 100)
            self.assertIsNone(rf.reuse_local(out, "00003", UNITS["00003"]))
            self.assertIsNone(rf.reuse_local(out, "00004", UNITS["00004"]))


if __name__ == "__main__":
    unittest.main()
