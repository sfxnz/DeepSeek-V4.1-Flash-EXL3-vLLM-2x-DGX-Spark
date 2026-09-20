"""CPU-only checks: host archive uses git; container build does not."""

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class BuildScriptTests(unittest.TestCase):
    def test_build_sh_does_not_invoke_git(self):
        text = (ROOT / "build.sh").read_text()
        self.assertNotRegex(text, r"(?m)^[^#\n]*\bgit\b")

    def test_archive_upstream_refuses_nonempty_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out"
            dest.mkdir()
            (dest / "stale").write_text("x")
            proc = subprocess.run(
                ["bash", str(ROOT / "archive_upstream.sh"), tmp, str(dest)],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("nonempty", proc.stderr)

    def test_build_sh_copies_tree_without_git_when_nvcc_is_stubbed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            upstream = root / "upstream"
            (upstream / "exllamav3" / "exllamav3_ext" / "quant").mkdir(parents=True)
            (upstream / "exllamav3" / "exllamav3_ext" / "quant" / "keep.h").write_text(
                "// fixture\n"
            )
            bin_dir = root / "bin"
            bin_dir.mkdir()
            nvcc = bin_dir / "nvcc"
            nvcc.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "out=\n"
                "while [ $# -gt 0 ]; do\n"
                '  case "$1" in\n'
                "    -o) out=$2; shift 2 ;;\n"
                "    *) shift ;;\n"
                "  esac\n"
                "done\n"
                'printf stub > "$out"\n'
            )
            nvcc.chmod(nvcc.stat().st_mode | stat.S_IEXEC)
            work = root / "work"
            work.mkdir()
            env = os.environ.copy()
            env["NVCC"] = str(nvcc)
            env["PATH"] = f"{bin_dir}:/usr/bin:/bin"
            proc = subprocess.run(
                ["bash", str(ROOT / "build.sh"), str(upstream), str(work)],
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            self.assertTrue((work / "cooperative_moe.so").is_file())
            self.assertTrue((work / "runtime.py").is_file())


if __name__ == "__main__":
    unittest.main()
