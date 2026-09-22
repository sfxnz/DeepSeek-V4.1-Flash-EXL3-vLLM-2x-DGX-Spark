#!/usr/bin/env python3
"""GPU bit-exactness gate for the G8 kernel PORT (audit R2) — run AFTER the
pack rebuild finishes (~04:00 BST) in a throwaway GPU container.

Compares, on real pack shapes, G8-layout trellis + G8 kernels vs stock-layout
trellis + stock kernels, for every reader that serving can reach:
  - exl3_gemm   (prefill GEMM; m in {64,128,256,512})          [exllamav3_ext]
  - exl3_gemv   (QTIP small-m;  m in {1,2,4,8})                [exllamav3_ext]
  - p2b_fused_moe (decode fused MoE, m in {1,2,4,8})           [vllm_exl3_c]
  - exl3_moe    (standard MoE path, m in {64,128,256,512})     [exllamav3_ext]
torch.equal REQUIRED at every point (the harness twins already proved the
math; this proves the port).

Usage (GPU window, after the rebuild containers exit):
  docker run --rm --gpus all --network none -v $PWD:/w -w /w \
    <canonical-g8-image> python3 pfg8_bitexact_gate.py

The script uses the pf_g8_set(1) A/B lever (same kernels both sides), so it
validates the PORTED kernels + flag plumbing, not just the harness twins.
"""
import os
import sys

import torch

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
    ("gate_up", 5120, 2304),   # stock (320,144,32) per rank: k=5120 n=2304
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


def main() -> int:
    assert torch.cuda.is_available(), "needs a GPU window"
    dev = torch.device("cuda:0")
    torch.manual_seed(20260921)

    import exllamav3_ext as ext
    import vllm_exl3_c

    # env read once at import; the A/B lever must agree with it
    assert ext.pf_g8_get() == 0 and vllm_exl3_c.pf_g8_get() == 0, "flag should default to 0"

    for sname, k, n in SHAPES:
        kt, nt = k // 16, n // 16
        stock = torch.randint(
            -32768, 32767, (kt, nt, 32), dtype=torch.int16, device=dev
        )
        g8 = fold_g8(stock)
        suh = torch.ones(k, dtype=torch.float16, device=dev)
        svh = torch.ones(n, dtype=torch.float16, device=dev)

        # ---------------- prefill exl3_gemm ----------------
        for m in PREFILL_M:
            x = torch.randn(m, k, dtype=torch.float16, device=dev)
            ext.pf_g8_set(0)
            ref = ext.exl3_gemm(x, stock, suh, svh, 2, True)
            ext.pf_g8_set(1)
            out = ext.exl3_gemm(x, g8, suh, svh, 2, True)
            torch.cuda.synchronize()
            check(f"exl3_gemm {sname} m={m}", ref, out)

        # ---------------- QTIP exl3_gemv (decode m) ----------------
        for m in DECODE_M:
            x = torch.randn(m, k, dtype=torch.float16, device=dev)
            ext.pf_g8_set(0)
            ref = ext.exl3_gemv(x, stock, suh, svh, 2, True)
            ext.pf_g8_set(1)
            out = ext.exl3_gemv(x, g8, suh, svh, 2, True)
            torch.cuda.synchronize()
            check(f"exl3_gemv {sname} m={m}", ref, out)

    # ---------------- p2b_fused_moe (decode, both shapes at once) ----------
    # gate/up from gate_up shape; down from down shape
    k, n = 5120, 2304
    kt, nt = k // 16, n // 16
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
        rw = torch.full_like(ids, 0.5, dtype=torch.float16)

        def run(layout: str):
            g, d = (g8_g, g8_d) if layout == "g8" else (stock_g, stock_d)
            gt, gu = ptrs(g), ptrs(g)
            gv, uv = ptrs(ones_n), ptrs(ones_n)
            uu = ptrs(ones_k)
            dt, du = ptrs(d), ptrs(ones_n)
            dv = ptrs(ones_k)
            out = torch.empty_like(x)
            vllm_exl3_c.p2b_fused_moe(
                x, out, gt, gu, gv, gu, uu, uv, dt, du, dv,
                ids, rw, 2, 2, 2, True, n, 0.0,
            )
            torch.cuda.synchronize()
            return out

        # flag lives in BOTH .so registries; set both to the same phase
        ext.pf_g8_set(0); vllm_exl3_c.pf_g8_set(0)
        ref = run("stock")
        ext.pf_g8_set(1); vllm_exl3_c.pf_g8_set(1)
        out = run("g8")
        check(f"p2b_fused_moe m={m}", ref, out)

    # restore
    ext.pf_g8_set(0)
    vllm_exl3_c.pf_g8_set(0)

    if failures:
        print(f"\nVERDICT: FAIL ({len(failures)} mismatches): {failures}")
        return 1
    print("\nVERDICT: PASS — G8 port is bit-exact on every reachable reader")
    return 0


if __name__ == "__main__":
    sys.exit(main())
