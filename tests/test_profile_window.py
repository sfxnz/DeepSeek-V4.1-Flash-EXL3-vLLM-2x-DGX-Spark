#!/usr/bin/env python3
"""tools/profile_window.sh with curl/docker/ssh/free shims (no serve, no GPU)."""
from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/profile_window.sh"

CURL = r'''#!/usr/bin/env bash
echo "curl $*" >>"$SHIM_LOG"
case "$*" in
  *start_profile*|*stop_profile*) echo ok ;;
  *'"stream":true'*) printf 'data: {"choices":[{"delta":{"content":"x"}}]}\n\ndata: {"choices":[],"usage":{"completion_tokens":512}}\n\ndata: [DONE]\n' ;;
  *) echo '{"usage":{"completion_tokens":512}}' ;;
esac
'''
LOGGER = '#!/usr/bin/env bash\necho "{name} $*" >>"$SHIM_LOG"\n'
DOCKER = LOGGER.format(name="docker") + '[[ "$1" == cp ]] && mkdir -p "$3"\nexit 0\n'


class ProfileWindowTests(unittest.TestCase):
    def test_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def _run(self, ssh_shim: str) -> tuple[subprocess.CompletedProcess, list[str]]:
        with tempfile.TemporaryDirectory() as d:
            bin_dir = Path(d) / "bin"
            bin_dir.mkdir()
            (bin_dir / "curl").write_text(CURL)
            (bin_dir / "docker").write_text(DOCKER)
            (bin_dir / "ssh").write_text(ssh_shim)
            (bin_dir / "free").write_text(LOGGER.format(name="free"))
            for f in bin_dir.iterdir():
                f.chmod(0o755)
            log = Path(d) / "calls.log"
            env = {
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "SHIM_LOG": str(log),
                "FLUSH_S": "0",
                "OUT_DIR": str(Path(d) / "out"),
                "WORKER_HOST": "spark2",
            }
            r = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=60)
            return r, log.read_text().splitlines()

    def test_worker_without_trace_dir_still_copies_rank0(self) -> None:
        # No trace dir on rank 1: the real `docker exec ls` AND `docker cp` both fail there.
        ssh = LOGGER.format(name="ssh") + '[[ "$*" == *"docker exec"* || "$*" == *"docker cp"* ]] && exit 2\nexit 0\n'
        r, calls = self._run(ssh)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("WARN: no rank-1 trace dir", r.stdout)
        self.assertIn("WARN: rank-1 copy failed", r.stdout)
        self.assertTrue(any(c.startswith("docker cp dsv41-flash-exl3:/tmp/dsv41-traces") for c in calls))
        cp1 = next(i for i, c in enumerate(calls) if c.startswith("ssh spark2") and "docker cp" in c)
        self.assertTrue(any(c.startswith("free") for c in calls[cp1:]), "MemAvail after cp still read")
        self.assertIn("Traces saved", r.stdout)

    def test_copies_both_ranks_and_never_stops_the_serve(self) -> None:
        r, calls = self._run(LOGGER.format(name="ssh"))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("completion_tokens': 512", r.stdout)
        stop = next(i for i, c in enumerate(calls) if "stop_profile" in c)
        head_cp = next(i for i, c in enumerate(calls) if c.startswith("docker cp dsv41-flash-exl3:/tmp/dsv41-traces"))
        worker_ls = next(i for i, c in enumerate(calls) if c.startswith("ssh spark2 docker exec") and "ls -la" in c)
        worker_cp = next(i for i, c in enumerate(calls) if c.startswith("ssh spark2") and "docker cp" in c)
        self.assertLess(stop, worker_ls)
        self.assertLess(stop, head_cp)
        self.assertLess(stop, worker_cp)
        self.assertIn("/rank1", calls[worker_cp])
        self.assertFalse([c for c in calls if "stop.sh" in c or " rm " in c or " stop " in c])


if __name__ == "__main__":
    unittest.main()
