"""pm_qos: env parsing, the held request, run.sh device wiring (CPU only)."""

from __future__ import annotations

import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "docker" / "patch"))
sys.path.insert(0, str(ROOT / "tests"))

import pm_qos as pq  # noqa: E402


class ParseTests(unittest.TestCase):
    def test_values(self):
        self.assertIsNone(pq.requested_us({}))
        self.assertIsNone(pq.requested_us({pq.ENV: ""}))
        self.assertIsNone(pq.requested_us({pq.ENV: "  "}))
        self.assertEqual(pq.requested_us({pq.ENV: "20"}), 20)
        self.assertEqual(pq.requested_us({pq.ENV: "0"}), 0)
        for bad in ("-1", "2001", "fast"):
            with self.assertRaises(ValueError):
                pq.requested_us({pq.ENV: bad})


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dev = Path(self.tmp.name) / "cpu_dma_latency"
        self.dev.write_bytes(b"")
        self.addCleanup(self._release)

    def _release(self):
        while pq._held:
            os.close(pq._held.pop())

    def test_off_touches_nothing(self):
        logs = []
        self.assertEqual(pq.install({}, logs.append, str(self.dev)), "off")
        self.assertEqual(self.dev.read_bytes(), b"")
        self.assertEqual(pq._held, [])
        self.assertEqual(logs, [])

    def test_armed_writes_an_int32_and_holds_the_fd(self):
        logs = []
        self.assertEqual(pq.install({pq.ENV: "20"}, logs.append, str(self.dev)), "armed")
        self.assertEqual(self.dev.read_bytes(), struct.pack("i", 20))
        self.assertEqual(len(pq._held), 1)
        os.fstat(pq._held[0])  # still open
        self.assertTrue(logs[0].startswith(pq.LOG_ENGAGED))
        # a second install in the same process keeps the one request
        self.assertEqual(pq.install({pq.ENV: "20"}, logs.append, str(self.dev)), "armed")
        self.assertEqual(len(pq._held), 1)

    def test_missing_device_disarms(self):
        logs = []
        state = pq.install({pq.ENV: "20"}, logs.append, str(Path(self.tmp.name) / "absent"))
        self.assertEqual(state, "disarmed")
        self.assertTrue(logs[0].startswith(pq.LOG_DISARMED))
        self.assertIn("--device /dev/cpu_dma_latency", logs[0])
        self.assertEqual(pq._held, [])

    def test_bad_value_disarms(self):
        logs = []
        self.assertEqual(pq.install({pq.ENV: "-5"}, logs.append, str(self.dev)), "disarmed")
        self.assertTrue(logs[0].startswith(pq.LOG_DISARMED))


class WiringTests(unittest.TestCase):
    def test_sitecustomize_installs_it(self):
        site = (ROOT / "docker/patch/sitecustomize.py").read_text()
        self.assertIn('_patch("pm_qos", _p_pm_qos)', site)
        self.assertIn("pm_qos.install()", site)

    def test_default_off_in_run_sh(self):
        run = (ROOT / "run.sh").read_text()
        self.assertIn("  DSV41_PM_QOS_US=\n", run)

    def test_device_passed_to_both_ranks_only_when_set(self):
        import run_sh_harness as h

        res = h.dry_run()
        self.assertEqual(res["returncode"], 0, res["stdout"] + res["stderr"])
        for role in ("head", "worker"):
            self.assertNotIn("/dev/cpu_dma_latency", res[role], role)
        res = h.dry_run(DSV41_PM_QOS_US="20")
        self.assertEqual(res["returncode"], 0, res["stdout"] + res["stderr"])
        for role in ("head", "worker"):
            argv = res[role]
            i = argv.index("/dev/cpu_dma_latency")
            self.assertEqual(argv[i - 1], "--device", role)
            self.assertEqual(h.container_env(argv).get("DSV41_PM_QOS_US"), "20", role)


if __name__ == "__main__":
    unittest.main()
