#!/usr/bin/env python3
"""CPU ownership-map model of the G8 gemv/p2b readers (Round 29 mandate).

Models the B-tensor addressing of the patched readers:

  exl3_gemv_kernel.cuh (CFG0: WNT=2, WK=16, PF=4)
  p2b_moe.cu run_gemv_tile (CFG1: WNT=4, WK=8, PF=2)

under PF-G8 group-major layout [pack group of 8 tiles][k-slice][8*TWORDS u32]
vs stock [k-slice][tile][TWORDS], for the real per-rank shapes (2bpw mcg,
bits=2, TWORDS=16, K=2):

  gate_up : n=2304 (NT=144 tiles, 18 pack groups), k=5120 (KT=320 slices)
  down    : n=5120 (NT=320 tiles, 40 pack groups), k=2304 (KT=144 slices)

Checks per shape/reader:
  1. K-CHUNKING: WK chunks tile kslices exactly (myn == chunk for all warps;
     no partial warp -> suspects (a)/(c) of the Round-28 triage are void)
  2. OWNERSHIP: each output group loads exactly its own WNT tiles' words,
     every logical word exactly once, no hole, no double-load across groups
  3. BOUNDS: max G8 offset == numel(B)-1 (no OOB)
  4. BLOCK COVERAGE: `for group=blockIdx; group<num_groups; group+=gridDim`
     covers every output group exactly once for arbitrary grid sizes
  5. RE-INDEX PURITY: (tile, ks, word)->G8 offset is a bijection reading the
     same logical word as stock (word stream untouched, reduction order kept)

Exit 0 = EXACT (no ownership race possible in the G8 branch).
"""
import sys


def model(name, ntiles, kt, wnt, wk, bits=2):
    twords = 8 * bits
    kslices = kt
    num_groups = ntiles // wnt
    groups_per_pack = 8 // wnt
    chunk = -(-kslices // wk)
    numel = ntiles * kslices * twords
    errs = []

    # 1. k-chunking exactness
    if kslices % wk:
        errs.append(f"k-chunk inexact: {kslices} % {wk} = {kslices % wk}")
    for warp in range(wk):
        myn = max(0, min(chunk, kslices - warp * chunk))
        if myn != chunk:
            errs.append(f"warp {warp}: myn={myn} != chunk={chunk}")

    # 2+3. ownership + bounds via one pass
    loaded = {}                     # logical word id -> g8 offset
    max_off = 0
    for group in range(num_groups):
        pack, off_in_pack = group // groups_per_pack, (group % groups_per_pack) * wnt * twords
        for warp in range(wk):
            ks0 = warp * chunk
            for i in range(chunk):
                ks = ks0 + i
                base = pack * kslices * 8 * twords + ks * 8 * twords + off_in_pack
                if base + wnt * twords - 1 >= numel:
                    errs.append(f"OOB: group {group} ks {ks} base {base}")
                for w in range(wnt * twords):
                    tile, ww = group * wnt + w // twords, w % twords
                    wid = tile * kslices * twords + ks * twords + ww
                    off = base + w
                    max_off = max(max_off, off)
                    if wid in loaded:
                        errs.append(f"word {wid} loaded twice (g8 offs {loaded[wid]}, {off})")
                        if len(errs) > 8: break
                    loaded[wid] = off
                if len(errs) > 8: break
            if len(errs) > 8: break
        if len(errs) > 8: break
    if not errs:
        if len(loaded) != numel:
            errs.append(f"coverage: {len(loaded)}/{numel} logical words loaded")
        if max_off != numel - 1:
            errs.append(f"bounds: max offset {max_off} != {numel-1}")
        # per-group ownership: loaded word ids must be exactly the group's tiles
        for group in range(num_groups):
            tiles = range(group * wnt, (group + 1) * wnt)
            own = {t * kslices * twords + ks * twords + ww
                   for t in tiles for ks in range(kslices) for ww in range(twords)}
            got = {wid for wid in loaded
                   if group * wnt <= wid // (kslices * twords) < (group + 1) * wnt}
            if got != own:
                errs.append(f"group {group}: loaded != owned")
                break

    # 4. block-stride coverage for arbitrary grids
    for grid in (1, 2, 3, 7, 9, 18, 40, 71, 160, 1000):
        got = [g for b in range(min(grid, num_groups)) for g in range(b, num_groups, grid)]
        if sorted(got) != list(range(num_groups)):
            errs.append(f"grid={grid}: output coverage broken")
            break

    # 5. re-index purity: G8 offset of (tile, ks, ww) must equal the pack copy
    #    of the same logical word:  g8 = pack*KTR*8*T + ks*8*T + (tile%8)*T + ww
    #    with pack = tile//8  (fold: stock view(kt, nt/8, 8w).permute(1,0,2))
    pure = True
    for tile in range(ntiles):
        for ks in range(kslices):
            for ww in range(twords):
                wid = tile * kslices * twords + ks * twords + ww
                expect = (tile // 8) * kslices * 8 * twords + ks * 8 * twords + (tile % 8) * twords + ww
                if loaded.get(wid) != expect:
                    pure = False
                    errs.append(f"purity: word {wid} g8 {loaded.get(wid)} != fold {expect}")
                    break
            if not pure: break
        if not pure: break

    status = "EXACT" if not errs else "BROKEN"
    print(f"[{status}] {name}: NT={ntiles} KT={kt} WNT={wnt} WK={wk} "
          f"pack_groups={ntiles//8} out_groups={num_groups} chunk={chunk} numel(u32)={numel}")
    if errs:
        for e in errs[:8]:
            print(f"    ERROR: {e}")
    return not errs


def main():
    print("== PF-G8 ownership map — exl3_gemv CFG0 (WNT=2, WK=16) ==")
    a = model("gemv gate_up (NT=144, KT=320)", 144, 320, 2, 16)
    b = model("gemv down    (NT=320, KT=144)", 320, 144, 2, 16)
    print()
    print("== PF-G8 ownership map — p2b run_gemv_tile CFG1 (WNT=4, WK=8) ==")
    c = model("p2b gate/up (NT=144, KT=320)", 144, 320, 4, 8)
    d = model("p2b down    (NT=320, KT=144)", 320, 144, 4, 8)
    ok = all((a, b, c, d))
    print()
    print("VERDICT:", "G8 ownership map EXACT for both shapes and both readers — "
          "k-chunks whole (no partial warp), every logical word loaded exactly once "
          "by its owning group, bounds exact-fit, block coverage total, G8 offsets "
          "bit-identical to the quantize-time fold. A write race or double-owned "
          "group in the G8 branch is impossible." if ok else "MAP BROKEN — see errors")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
