#!/usr/bin/env python3
"""Run driver.py against the prebuilt extension (no source regeneration, so no JIT rebuild).

  python3 kernel_study/p2b_coop/run_prebuilt.py <driver.py args...>

make_bench.main() rewrites build/*.cu on every call, which bumps their mtime and makes
torch's JIT rebuild bench_coop.cu inside the GPU window. The sources were generated and
compiled beforehand (CPU-only container); this wrapper only loads them.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import make_bench  # noqa: E402

make_bench.main = lambda: 0

import driver  # noqa: E402

sys.argv = ["driver.py", *sys.argv[1:]]
raise SystemExit(driver.main())
