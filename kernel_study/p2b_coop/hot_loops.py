#!/usr/bin/env python3
"""Inner MMA loops of a p2b kernel from `cuobjdump -sass -fun <kernel>` output (CPU only).

  hot_loops.py p2b_sort0.sass coop_sort2.sass

Prints, per kernel, the innermost backward-branch loops that contain HMMA
(gate/up tile, down tile): instruction count, HMMA, local-memory spill
loads/stores and streaming B loads (LDG.E.EF). widen_p2b_mma lost on GB10
through register pressure, so coop's inner loops should not spill more.
"""

from __future__ import annotations

import re
import sys

INS = re.compile(r"\s*/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;")
BRA = re.compile(r"BRA 0x([0-9a-f]+)")


def inner_loops(path: str) -> list[dict]:
    ins = [(int(m.group(1), 16), m.group(2)) for m in map(INS.match, open(path).read().splitlines()) if m]
    loops = []
    for addr, op in ins:
        m = BRA.search(op)
        if m and int(m.group(1), 16) < addr:
            body = [o for a, o in ins if int(m.group(1), 16) <= a <= addr]
            # Loops that span EXIT are warp-divergence fallback jumps, not loops.
            if any("HMMA" in o for o in body) and not any(o.split()[-1] == "EXIT" for o in body):
                loops.append({"range": (int(m.group(1), 16), addr), "instr": len(body),
                              "hmma": sum("HMMA" in o for o in body), "ldl": sum("LDL" in o for o in body),
                              "stl": sum("STL" in o for o in body), "ldg_ef": sum("LDG.E.EF" in o for o in body)})
    # innermost = contains no other HMMA loop
    return [lp for lp in loops if not any(o is not lp and lp["range"][0] <= o["range"][0] and o["range"][1] <= lp["range"][1]
                                          for o in loops)]


def main(paths: list[str]) -> int:
    for path in paths:
        for lp in inner_loops(path):
            lo, hi = lp["range"]
            print(f"{path}: loop 0x{lo:x}-0x{hi:x} instr {lp['instr']} HMMA {lp['hmma']} "
                  f"LDL {lp['ldl']} STL {lp['stl']} LDG.E.EF {lp['ldg_ef']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
