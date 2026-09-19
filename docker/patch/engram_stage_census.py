#!/usr/bin/env python3
"""Census instrumentation for the Engram disk stager (diagnostic only).

The L.A.I.L decode step carries a 15-28 ms inter-step gap that is almost
entirely `EngramDiskStager.stage`: hash D2H sync, NVMe row preads, CPU
dequant, H2D. This patch times each phase and logs the row counts so the
next patch can target the real bottleneck. No behavior change.

Enable with DSV41_ENGRAM_CENSUS=1. Idempotent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

CENSUS_MARKER = "# --- engram-stage-census ---"


def apply(vllm_root: Path) -> None:
    disk = vllm_root / "models" / "deepseek_v4_1" / "common" / "engram_disk.py"
    text = disk.read_text()
    if CENSUS_MARKER in text:
        return

    cls_anchor = "class DiskEngramTable:"
    if cls_anchor not in text:
        raise SystemExit("engram_stage_census: class anchor missing")
    helpers = (
        CENSUS_MARKER
        + "\n"
        "import os as _os\n"
        "import time as _time\n"
        "\n"
        "_ENG_CENSUS = _os.environ.get(\"DSV41_ENGRAM_CENSUS\", \"0\") == \"1\"\n"
        "_ENG_CENSUS_N = int(_os.environ.get(\"DSV41_ENGRAM_CENSUS_EVERY\", \"32\"))\n"
        "_ENG_CENSUS_SEEN = [0, 0.0, 0.0, 0.0, 0]\n"
        "_ENG_PF_HITS = [0, 0, 0]\n"
        "\n"
        "\n"
    )
    text = text.replace(cls_anchor, helpers + cls_anchor, 1)

    read_w = (
        "        self._read_rows(\n"
        "            self.w_fd, self.w_off, rel_l, self.dim, memoryview(w.numpy()).cast(\"B\")\n"
        "        )"
    )
    read_s = (
        "        self._read_rows(\n"
        "            self.s_fd, self.s_off, rel_l, self.sb, memoryview(s.numpy()).cast(\"B\")\n"
        "        )"
    )
    if read_w not in text or read_s not in text:
        raise SystemExit("engram_stage_census: _read_rows anchors missing")
    text = text.replace(
        read_w,
        "        _c0 = _time.perf_counter()\n" + read_w + "\n        _c1 = _time.perf_counter()",
        1,
    )
    text = text.replace(
        read_s,
        "        _c2 = _time.perf_counter()\n"
        + read_s
        + "\n        _c3 = _time.perf_counter()",
        1,
    )

    old_ret = "        out[~owned] = 0\n        return out.to(torch.bfloat16)"
    if old_ret not in text:
        raise SystemExit("engram_stage_census: return anchor missing")
    new_ret = (
        "        out[~owned] = 0\n"
        "        out = out.to(torch.bfloat16)\n"
        "        _c4 = _time.perf_counter()\n"
        "        if _ENG_CENSUS:\n"
        "            _st = _ENG_CENSUS_SEEN\n"
        "            _pf = _ENG_PF_HITS\n"
        "            _st[0] += 1\n"
        "            _st[1] += _c1 - _c0\n"
        "            _st[2] += _c3 - _c2\n"
        "            _st[3] += _c4 - _c3\n"
        "            _st[4] += r\n"
        "            _exp = getattr(self, \"_pf_expected\", None)\n"
        "            if _exp is not None:\n"
        "                _got = set(rel_l)\n"
        "                _pf = _ENG_PF_HITS\n"
        "                _pf[0] += len(_got & _exp)\n"
        "                _pf[1] += len(_got)\n"
        "                _pf[2] += 1\n"
        "                self._pf_expected = None\n"
        "                _dmp = getattr(self, \"_pf_dump\", None)\n"
        "                if _dmp is not None and max(_dmp[0][:4]) > 16:\n"
        "                    print(\n"
        "                        \"[pf-dump] actual=%s pred=%s chunk=%s \"\n"
        "                        \"cpos=%s hist=%s\"\n"
        "                        % (\n"
        "                            sorted(_got)[:12],\n"
        "                            _dmp[3],\n"
        "                            _dmp[0],\n"
        "                            _dmp[1],\n"
        "                            _dmp[2],\n"
        "                        ),\n"
        "                        flush=True,\n"
        "                    )\n"
        "                    self._pf_dump = None\n"
        "            if _st[0] % _ENG_CENSUS_N == 0:\n"
        "                _n = _st[0]\n"
        "                _pfr = (\n"
        "                    \" pf_hit=%.0f%%(%d)\" % (100.0 * _pf[0] / _pf[1], _pf[2])\n"
        "                    if _pf[1] > 0 and _pf[2] > 0\n"
        "                    else \"\"\n"
        "                )\n"
        "                print(\n"
        "                    \"[engram-census] calls=%d rows/call=%d \"\n"
        "                    \"read_w=%.2fms read_s=%.2fms dequant=%.2fms%s\"\n"
        "                    % (\n"
        "                        _n,\n"
        "                        _st[4] // _n,\n"
        "                        1000.0 * _st[1] / _n,\n"
        "                        1000.0 * _st[2] / _n,\n"
        "                        1000.0 * _st[3] / _n,\n"
        "                        _pfr,\n"
        "                    ),\n"
        "                    flush=True,\n"
        "                )\n"
        "                _st[1] = _st[2] = _st[3] = 0.0\n"
        "                _pf[0] = _pf[1] = _pf[2] = 0\n"
        "        return out"
    )
    text = text.replace(old_ret, new_ret, 1)
    disk.write_text(text)
    print("dsv41: engram disk gather census instrumented")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("vllm_root", type=Path)
    args = p.parse_args()
    apply(args.vllm_root)


if __name__ == "__main__":
    main()
