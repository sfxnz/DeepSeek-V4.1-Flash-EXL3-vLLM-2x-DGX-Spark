#!/usr/bin/env python3
"""Compare p2b kernel .text sections of two sm_121a cubins (stdlib ELF read).

  text_identity.py base.cubin srcsort.cubin

Exit 0 when every <BITS, CB> kernel of base has byte-identical machine code
in srcsort's SORT=0 instantiation (DSV41_P2B_SRC_SORT off cannot regress).
"""

from __future__ import annotations

import re
import struct
import sys

KERNEL = re.compile(r"\.text\..*p2b_moe_batched_kernelILi(\d)ELi(\d)(?:ELi(\d))?E")


def sections(path: str) -> dict[str, bytes]:
    d = open(path, "rb").read()
    if d[:4] != b"\x7fELF" or d[4] != 2:
        raise SystemExit(f"{path}: not an ELF64 cubin")
    shoff, = struct.unpack_from("<Q", d, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", d, 0x3A)
    hdrs = [struct.unpack_from("<IIQQQQIIQQ", d, shoff + i * shentsize) for i in range(shnum)]
    names = hdrs[shstrndx][4]

    def name(off: int) -> str:
        return d[names + off : d.index(b"\0", names + off)].decode()

    return {name(h[0]): d[h[4] : h[4] + h[5]] for h in hdrs}


def kernels(path: str) -> dict[tuple[str, str, str], bytes]:
    out = {}
    for sec, body in sections(path).items():
        m = KERNEL.match(sec)
        if m:
            out[(m.group(1), m.group(2), m.group(3) or "0")] = body
    return out


def main(base: str, srcsort: str) -> int:
    a, b = kernels(base), kernels(srcsort)
    if not a:
        raise SystemExit(f"{base}: no p2b_moe_batched_kernel .text sections")
    bad = 0
    for key in sorted(a):
        same = a[key] == b.get(key)
        bad += not same
        print(f"BITS={key[0]} CB={key[1]} SORT=0: {len(a[key])} vs {len(b.get(key, b''))} bytes "
              f"{'IDENTICAL' if same else 'DIFFERENT'}")
    for key in sorted(k for k in b if k[2] == "1"):
        print(f"BITS={key[0]} CB={key[1]} SORT=1: {len(b[key])} bytes")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))
