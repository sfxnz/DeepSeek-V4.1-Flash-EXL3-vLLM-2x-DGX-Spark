#!/usr/bin/env python3
"""p2b GEMV bench driver.

Builds bench.cu (deployed-kernel clone + PFMUL variants), checks bitwise
equality against the installed stock vllm_exl3_c, then times variants.

Usage: driver.py [--e 6|12|30] [--iters 200] [--warmup 20] [--check-only]
"""
import argparse
import os
import sys

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")
import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.join(HERE, "..", "vllm-exl3-patched", "csrc")
EXL = "/usr/local/lib/python3.12/dist-packages/exllamav3/exllamav3_ext"


def build_ext():
    return load(
        name="bench_p2b",
        sources=[os.path.join(HERE, "bench.cu")],
        extra_include_paths=[CSRC, EXL, os.path.join(EXL, "quant")],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e", type=int, default=6)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--variants", type=str, default="0,1,2,3")
    args = ap.parse_args()

    ext = build_ext()
    import vllm_exl3_c as stock

    dev = "cuda"
    torch.manual_seed(7)
    E, hidden, inter = 32, 5120, 1152
    K = hidden // 16   # 320 k-slices for gate/up
    N = inter // 16    # 72 n-tiles for gate/up

    def rand_tr():
        return torch.randint(-32768, 32767, (K, N, 32), dtype=torch.int16, device=dev)

    def rand_tr_t():
        return torch.randint(-32768, 32767, (N, K, 32), dtype=torch.int16, device=dev)

    gate_t = [rand_tr() for _ in range(E)]
    up_t = [rand_tr() for _ in range(E)]
    down_t = [rand_tr_t() for _ in range(E)]
    gate_suh = [torch.randn(hidden, dtype=torch.half, device=dev) for _ in range(E)]
    gate_svh = [torch.randn(inter, dtype=torch.half, device=dev) for _ in range(E)]
    up_suh = [torch.randn(hidden, dtype=torch.half, device=dev) for _ in range(E)]
    up_svh = [torch.randn(inter, dtype=torch.half, device=dev) for _ in range(E)]
    down_suh = [torch.randn(inter, dtype=torch.half, device=dev) for _ in range(E)]
    down_svh = [torch.randn(hidden, dtype=torch.half, device=dev) for _ in range(E)]

    def ptrs(ts):
        return torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device=dev)

    tables = [ptrs(x) for x in (gate_t, gate_suh, gate_svh, up_t, up_suh, up_svh, down_t, down_suh, down_svh)]
    gt, gu, gv, ut, uu, uv, dt, du, dv = tables

    x = torch.randn(1, hidden, dtype=torch.half, device=dev)
    # Unique expert IDs for the bitwise checks: duplicate IDs make the final
    # atomicAdd accumulation order nondeterministic (half-ULP fp differences).
    ids = torch.randperm(E, device=dev)[: args.e].to(torch.int32).contiguous()
    rw = torch.randn(args.e, dtype=torch.half, device=dev)
    rw = (rw / rw.abs().max()).half()

    # --- bitwise correctness vs installed stock ---
    # The stock kernel's final expert accumulation is atomicAdd-order
    # nondeterministic for e>1 with arbitrary weights (observed 1 half-ULP
    # run-to-run). One-hot routing weights make the adds order-exact
    # (0.0 + x in any order), so bitwise gating stays valid while every
    # expert still runs the full GEMV path.
    rw_check = torch.zeros_like(rw)
    rw_check[0] = 1.0
    out_ref = torch.empty_like(x)
    stock.p2b_fused_moe(x, out_ref, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw_check, 2, 2, 2, True, inter, 7.0)
    out_v0 = torch.empty_like(x)
    ext.bench(x, out_v0, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw_check, 2, 2, 2, True, inter, 7.0, 0, 1, 0)
    ok0 = torch.equal(out_ref, out_v0)
    print(f"[check] variant0 vs installed stock: {'BIT-EXACT' if ok0 else 'MISMATCH'}")
    if not ok0:
        d = (out_ref.float() - out_v0.float()).abs()
        print("  max abs diff:", d.max().item(), "ref sample:", out_ref[0, :8].tolist())
        sys.exit(1)
    for v in [int(s) for s in args.variants.split(",") if int(s) != 0]:
        out_v = torch.empty_like(x)
        ext.bench(x, out_v, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw_check, 2, 2, 2, True, inter, 7.0, v, 1, 0)
        ok = torch.equal(out_ref, out_v)
        print(f"[check] variant{v} vs stock: {'BIT-EXACT' if ok else 'MISMATCH'}")
        if not ok:
            sys.exit(1)
    # Determinism band of the stock kernel itself with realistic weights
    # (documents the atomicAdd nondeterminism; informational).
    o1, o2 = torch.empty_like(x), torch.empty_like(x)
    stock.p2b_fused_moe(x, o1, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, 2, 2, 2, True, inter, 7.0)
    stock.p2b_fused_moe(x, o2, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, 2, 2, 2, True, inter, 7.0)
    print(f"[check] stock self-determinism (realistic rw): max diff {(o1.float()-o2.float()).abs().max().item()}")
    if args.check_only:
        return

    # --- timing (also stock via events for cross-check) ---
    weights_per_call = 3 * args.e * hidden * inter
    bytes_per_call = weights_per_call * 2 / 8  # trellis stream bytes

    out_s = torch.empty_like(x)
    for _ in range(args.warmup):
        stock.p2b_fused_moe(x, out_s, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, 2, 2, 2, True, inter, 7.0)
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(args.iters):
        stock.p2b_fused_moe(x, out_s, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, 2, 2, 2, True, inter, 7.0)
    ev1.record()
    torch.cuda.synchronize()
    ms_stock = ev0.elapsed_time(ev1) / args.iters
    print(f"[time] stock-installed e={args.e}: {ms_stock*1000:.1f} us/call  "
          f"{weights_per_call/(ms_stock*1e-3)/1e9:.1f} Gw/s  {bytes_per_call/(ms_stock*1e-3)/1e9:.1f} GB/s stream")

    for v in [int(s) for s in args.variants.split(",")]:
        ext.bench(x, out_s, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, 2, 2, 2, True, inter, 7.0, v, args.iters, args.warmup)
        ms = ext.last_ms()
        print(f"[time] variant{v} (PFMUL={1 << v if v else 1}) e={args.e}: {ms*1000:.1f} us/call  "
              f"{weights_per_call/(ms*1e-3)/1e9:.1f} Gw/s  {bytes_per_call/(ms*1e-3)/1e9:.1f} GB/s stream  "
              f"occupancy={ext.last_occupancy()} blocks/SM")


if __name__ == "__main__":
    main()
