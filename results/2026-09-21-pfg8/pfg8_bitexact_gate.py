#!/usr/bin/env python3
"""GPU bit-exactness gate for the G8 kernel PORT (audit R2).

Two-process design (2026-09-22): the image's HOST-side shape checks and
LinearEXL3.K derive read `DSV41_LOAD_PF_G8` via pfg8::env_value()
(static-cached per process) — pybind `pf_g8_set` only drives the DEVICE
constants. So the linear-kernel A/B must be two container invocations:

  docker run --rm --gpus all --network none --entrypoint python3 \
    -v $PWD/results/2026-09-21-pfg8:/g -w /g <canonical-g8> /g/pfg8_bitexact_gate.py ref
  docker run --rm --gpus all --network none -e DSV41_LOAD_PF_G8=1 \
    --entrypoint python3 -v $PWD/results/2026-09-21-pfg8:/g -w /g \
    <canonical-g8> /g/pfg8_bitexact_gate.py g8

  ref : stock-layout trellis, stock kernels -> saves seeded inputs+outputs /g/gate_ref.pt
  g8  : G8-layout trellis, env-gated kernels -> compares torch.equal vs saved refs
        (also runs the in-process p2b_fused_moe A/B via pf_g8_set, which needs
        no env: explicit dims + device-constant flag only)

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


def run_linear(trellis, suh, svh, x):
    # Force the trellis-kernel path for ALL m: above AUTO_RECONSTRUCT_THRESHOLD
    # (144) stock would route to reconstruct_hgemm — a different algorithm —
    # which is not the A/B under test (serving sets no_reconstruct for experts).
    _exl3mod.AUTO_RECONSTRUCT_THRESHOLD = 10 ** 9
    inner = vexl3.make_linear_exl3(
        trellis, suh, svh, None, None, out_dtype=torch.float16,
    )
    return inner.forward(x.contiguous().half(), {}, out_dtype=torch.float32).clone()


def stage_ref(dev) -> int:
    data = gen_inputs(dev)
    refs = {}
    for sname, k, n in SHAPES:
        d = data[sname]
        suh = torch.ones(k, dtype=torch.float16, device=dev)
        svh = torch.ones(n, dtype=torch.float16, device=dev)
        for m in PREFILL_M:
            refs[f"gemm {sname} m={m}"] = run_linear(d["stock"], suh, svh, d["x"][m])
            torch.cuda.synchronize()
        for m in DECODE_M:
            refs[f"gemv {sname} m={m}"] = run_linear(d["stock"], suh, svh, d["x"][m])
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

    # ---------------- exl3_gemm + exl3_gemv via the real LinearEXL3 path ----
    for sname, k, n in SHAPES:
        d = data[sname]
        g8 = fold_g8(d["stock"])
        suh = torch.ones(k, dtype=torch.float16, device=dev)
        svh = torch.ones(n, dtype=torch.float16, device=dev)
        for m in PREFILL_M:
            out = run_linear(g8, suh, svh, d["x"][m])
            torch.cuda.synchronize()
            check(f"exl3_gemm {sname} m={m}", refs[f"gemm {sname} m={m}"], out)
        for m in DECODE_M:
            out = run_linear(g8, suh, svh, d["x"][m])
            torch.cuda.synchronize()
            check(f"exl3_gemv {sname} m={m}", refs[f"gemv {sname} m={m}"], out)

    # ---------------- p2b_fused_moe (in-process pf_g8_set A/B) --------------
    k, n = 5120, 2304
    kt, nt = k // 16, n // 16
    torch.manual_seed(999)
    stock_g = torch.randint(-32768, 32767, (kt, nt, 32), dtype=torch.int16, device=dev)
    stock_d = torch.randint(-32768, 32767, (nt, kt, 32), dtype=torch.int16, device=dev)
    g8_g, g8_d = fold_g8(stock_g), fold_g8(stock_d)
    ones_k = torch.ones(k, dtype=torch.float16, device=dev)
    ones_n = torch.ones(n, dtype=torch.float16, device=dev)

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
                x, out, ptrs(g), ptrs(ones_n), ptrs(ones_n),
                ptrs(g), ptrs(ones_k), ptrs(ones_n),
                ptrs(d), ptrs(ones_k), ptrs(ones_n),
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
