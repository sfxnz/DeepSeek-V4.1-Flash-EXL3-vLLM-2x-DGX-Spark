#!/usr/bin/env python3
"""GPU bit-exactness gate for the G8 kernel PORT (audit R2) — v3 (Round 29).

Two-process design (2026-09-22): the image's HOST-side shape checks and
LinearEXL3.K derive read `DSV41_LOAD_PF_G8` via pfg8::env_value()
(static-cached per process) — pybind `pf_g8_set` only drives the DEVICE
constants. So the linear-kernel A/B must be two container invocations:

  docker run --rm --gpus all --network none --entrypoint python3 \
    -e EXLLAMAV3_TUNE_CACHE=/g/tune.bin \
    -v $PWD/results/2026-09-21-pfg8:/g -w /g <canonical-g8> /g/pfg8_bitexact_gate.py ref
  docker run --rm --gpus all --network none -e DSV41_LOAD_PF_G8=1 \
    -e EXLLAMAV3_TUNE_CACHE=/g/tune.bin --entrypoint python3 \
    -v $PWD/results/2026-09-21-pfg8:/g -w /g <canonical-g8> /g/pfg8_bitexact_gate.py g8

  ref : stock-layout trellis, stock kernels -> saves seeded inputs+outputs /g/gate_ref.pt
  g8  : G8-layout trellis, env-gated kernels -> compares torch.equal vs saved refs
        (also runs the in-process p2b_fused_moe A/B via pf_g8_set, which needs
        no env: explicit dims + device-constant flag only)

ROUND-29 FIXES (root cause of the Round-28 "nondeterministic NT=144 gemv
mismatch" — see results/2026-09-22-endgame2/ownership_map_g8.py):

1. mcg MARKER PASSED. v2 built LinearEXL3 with mcg=None -> host cb=0 ->
   exl3_gemv_cfg() returns -1 for K=2 (`K != 4 && cb == 0`) -> the "gemv"
   points actually ran the AUTOTUNED REGULAR GEMM. The autotuner times
   candidates on the real B tensor; stock-layout and G8-layout B stream
   differently (that is the point of G8), so near-tie shapes (down
   m=1/2/4; note the autotune key uses MAX(size_m,2), coupling m=1 and
   m=2) flipped winners between processes -> different split-k
   accumulation order -> sub-ulp diffs (observed max 1.5e-4 on ~94% of
   elements = global order noise, not tile-local garbage). The serving
   pack is 2.0bpw-mcg (cb=1): decode m<=8 takes the deterministic QTIP
   gemv path — which v2 never tested. v3 passes a non-zero mcg marker so
   both legs take the REAL serving path.
2. SHARED TUNE CACHE. EXLLAMAV3_TUNE_CACHE=/g/tune.bin (same file, rw
   mount): the ref leg tunes+stores, the g8 leg cache-hits the identical
   launch config -> the prefill m=64..512 GEMM points compare pure
   addressing, not autotuner luck. Delete tune.bin before each ref run.
3. p2b SCALE LENGTHS. v2 passed ones_n (2304) as gu/uu/dv, which the
   kernel indexes with (w*128) % hidden = % 5120 -> out-of-bounds reads
   -> the Round-28 "illegal memory access" (it crashed in the STOCK leg:
   layout-independent harness bug, not a port defect). Correct lengths:
   gu/uu/dv hidden(5120), gv/uv/du inter(2304).

Kernels covered through the REAL serving entry (vllm_exl3.exl3.make_linear_exl3
-> LinearEXL3.forward -> BC_LinearEXL3 dispatch):
  - exl3_gemm  (prefill GEMM;  m in {64,128,256,512})   [exllamav3_ext]
  - exl3_gemv  (QTIP small-m; m in {1,2,4,8})           [exllamav3_ext]
  - p2b_fused_moe (decode fused MoE, m in {1,2,4,8})     [vllm_exl3_c]
"""
import os
import sys

import torch

# module-scope imports (an in-function `from vllm_exl3 import exl3` proved
# fragile inside this container — failed after heavy CUDA allocs)
from vllm_exl3 import exl3 as vexl3  # noqa: E402
import exllamav3.modules.quant.exl3 as _exl3mod  # noqa: E402

REF_PT = "/g/gate_ref.pt"

# ---- layout helpers (must match tools/quantize_experts_exl3.py pfg8_fold) ----

def fold_g8(trellis: torch.Tensor) -> torch.Tensor:
    """stock [KT][NT][W] -> G8 [NT/8][KT][8*W]"""
    kt, nt, w = trellis.shape
    assert nt % 8 == 0
    return (
        trellis.view(kt, nt // 8, 8 * w)
        .permute(1, 0, 2)
        .contiguous()
        .view(nt // 8, kt, 8 * w)
    )

SHAPES = [
    # (name, k, n) — real pack shapes per rank (TP=2, 2bpw mcg)
    ("gate_up", 5120, 2304),   # stock (320,144,32) per rank
    ("down", 2304, 5120),      # stock (144,320,32) per rank
]

DECODE_M = [1, 2, 4, 8]
PREFILL_M = [64, 128, 256, 512]

failures = []


def check(name: str, a: torch.Tensor, b: torch.Tensor) -> bool:
    ok = torch.equal(a, b)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {'bit-exact' if ok else 'MISMATCH'}")
    if not ok:
        failures.append(name)
        d = (a.float() - b.float()).abs()
        print(f"       max abs diff {d.max().item():.6f}  n_mismatch {(d > 0).sum().item()}")
    return ok


def gen_inputs(dev):
    """Deterministic inputs — identical draw order in BOTH processes."""
    torch.manual_seed(20260921)
    data = {}
    for sname, k, n in SHAPES:
        kt, nt = k // 16, n // 16
        data[sname] = {
            "k": k, "n": n,
            "stock": torch.randint(-32768, 32767, (kt, nt, 32),
                                   dtype=torch.int16, device=dev),
            "x": {m: torch.randn(m, k, dtype=torch.float16, device=dev)
                  for m in PREFILL_M + DECODE_M},
        }
    return data


def run_linear(trellis, suh, svh, mcg, x):
    # Force the trellis-kernel path for ALL m: above AUTO_RECONSTRUCT_THRESHOLD
    # (144) stock would route to reconstruct_hgemm — a different algorithm —
    # which is not the A/B under test (serving sets no_reconstruct for experts).
    # mcg: non-zero marker tensor = the 2.0bpw-mcg serving codebook (host
    # cb=1) — WITHOUT it the decode points silently fall to the autotuned
    # regular GEMM (Round-28 harness bug; see module docstring).
    _exl3mod.AUTO_RECONSTRUCT_THRESHOLD = 10 ** 9
    inner = vexl3.make_linear_exl3(
        trellis, suh, svh, mcg, None, out_dtype=torch.float16,
    )
    return inner.forward(x.contiguous().half(), {}, out_dtype=torch.float32).clone()


def stage_ref(dev) -> int:
    data = gen_inputs(dev)
    mcg = torch.ones(1, dtype=torch.float16, device=dev)
    refs = {}
    for sname, k, n in SHAPES:
        d = data[sname]
        suh = torch.ones(k, dtype=torch.float16, device=dev)
        svh = torch.ones(n, dtype=torch.float16, device=dev)
        for m in PREFILL_M:
            refs[f"gemm {sname} m={m}"] = run_linear(d["stock"], suh, svh, mcg, d["x"][m])
            torch.cuda.synchronize()
        for m in DECODE_M:
            refs[f"gemv {sname} m={m}"] = run_linear(d["stock"], suh, svh, mcg, d["x"][m])
            torch.cuda.synchronize()
    torch.save({"refs": refs}, REF_PT)
    print(f"ref stage: saved {len(refs)} reference outputs + inputs to {REF_PT}")
    return 0


def stage_g8(dev) -> int:
    import exllamav3_ext as ext
    import vllm_exl3_c
    assert ext.pf_g8_get() == 1 and vllm_exl3_c.pf_g8_get() == 1, \
        "g8 stage must run with DSV41_LOAD_PF_G8=1"

    data = gen_inputs(dev)
    saved = torch.load(REF_PT, map_location=dev, weights_only=True)
    refs = saved["refs"]
    mcg = torch.ones(1, dtype=torch.float16, device=dev)

    # ---------------- exl3_gemm + exl3_gemv via the real LinearEXL3 path ----
    for sname, k, n in SHAPES:
        d = data[sname]
        g8 = fold_g8(d["stock"])
        suh = torch.ones(k, dtype=torch.float16, device=dev)
        svh = torch.ones(n, dtype=torch.float16, device=dev)
        for m in PREFILL_M:
            out = run_linear(g8, suh, svh, mcg, d["x"][m])
            torch.cuda.synchronize()
            check(f"exl3_gemm {sname} m={m}", refs[f"gemm {sname} m={m}"], out)
        for m in DECODE_M:
            out = run_linear(g8, suh, svh, mcg, d["x"][m])
            torch.cuda.synchronize()
            check(f"exl3_gemv {sname} m={m}", refs[f"gemv {sname} m={m}"], out)

    # ---------------- p2b_fused_moe (in-process pf_g8_set A/B) --------------
    # hidden = 5120 (x width), inter = 2304 (per-rank intermediate). The kernel
    # indexes gu/uu/dv with (w*128) % hidden and gv/uv/du with (w*128) % inter:
    # scale tables MUST be those lengths (v2 passed 2304-long tensors for
    # gu/uu/dv -> OOB reads -> the Round-28 illegal memory access).
    k, n = 5120, 2304
    kt, nt = k // 16, n // 16
    torch.manual_seed(999)
    stock_g = torch.randint(-32768, 32767, (kt, nt, 32), dtype=torch.int16, device=dev)
    stock_d = torch.randint(-32768, 32767, (nt, kt, 32), dtype=torch.int16, device=dev)
    g8_g, g8_d = fold_g8(stock_g), fold_g8(stock_d)
    ones_hidden = torch.ones(k, dtype=torch.float16, device=dev)   # 5120
    ones_inter = torch.ones(n, dtype=torch.float16, device=dev)    # 2304

    def ptrs(*ts):
        return torch.tensor([int(t.data_ptr()) for t in ts],
                            dtype=torch.int64, device=dev)

    for m in DECODE_M:
        x = torch.randn(m, k, dtype=torch.float16, device=dev)
        ids = torch.arange(2, dtype=torch.int32, device=dev).repeat(m, 1).contiguous()
        rw = torch.full((m, 2), 0.5, dtype=torch.float16, device=dev)

        def run(layout: str):
            g, d = (g8_g, g8_d) if layout == "g8" else (stock_g, stock_d)
            out = torch.zeros_like(x)
            vllm_exl3_c.p2b_fused_moe(
                x, out, ptrs(g), ptrs(ones_hidden), ptrs(ones_inter),
                ptrs(g), ptrs(ones_hidden), ptrs(ones_inter),
                ptrs(d), ptrs(ones_inter), ptrs(ones_hidden),
                ids, rw, 2, 2, 2, True, n, 0.0,
            )
            torch.cuda.synchronize()
            return out

        ext.pf_g8_set(0); vllm_exl3_c.pf_g8_set(0)
        ref = run("stock")
        ext.pf_g8_set(1); vllm_exl3_c.pf_g8_set(1)
        out = run("g8")
        check(f"p2b_fused_moe m={m}", ref, out)

    # restore device flags (process exits anyway)
    ext.pf_g8_set(1)
    vllm_exl3_c.pf_g8_set(1)

    if failures:
        print(f"\nVERDICT: FAIL ({len(failures)} mismatches): {failures}")
        return 1
    print("\nVERDICT: PASS — G8 port is bit-exact on every reachable reader")
    return 0


def main() -> int:
    assert torch.cuda.is_available(), "needs a GPU window"
    dev = torch.device("cuda:0")
    mode = sys.argv[1] if len(sys.argv) > 1 else "ref"
    if mode == "ref":
        return stage_ref(dev)
    if mode == "g8":
        return stage_g8(dev)
    print(f"unknown mode {mode!r} (use 'ref' | 'g8')")
    return 2


if __name__ == "__main__":
    sys.exit(main())
