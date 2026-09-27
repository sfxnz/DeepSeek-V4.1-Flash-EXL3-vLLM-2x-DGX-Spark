#!/usr/bin/env python3
"""bench2 driver: decode-variant correctness + warm/cold timing.

Modes:
  --mode check   bit-exact gates for vdec 1,2 vs stock (one-hot rw)
  --mode warm    fixed ids, in-C++ timed loop
  --mode cold    fresh random ids per call, python-event timed (L2-cold)
"""
import argparse
import os
import statistics
import sys

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")
import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.join(HERE, "..", "vllm-exl3-patched", "csrc")
EXL = "/usr/local/lib/python3.12/dist-packages/exllamav3/exllamav3_ext"


def build_ext():
    return load(
        name="bench_p2b2",
        sources=[os.path.join(HERE, "bench2.cu")],
        extra_include_paths=[CSRC, EXL, os.path.join(EXL, "quant")],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e", type=int, default=30)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--cold-iters", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--mode", type=str, default="all", choices=["all", "check", "warm", "cold"])
    ap.add_argument("--vdecs", type=str, default="0,1,2")
    args = ap.parse_args()

    ext = build_ext()
    import vllm_exl3_c as stock

    dev = "cuda"
    torch.manual_seed(7)
    E, hidden, inter = args.experts, 5120, 1152
    K, N = hidden // 16, inter // 16

    def mk(): return torch.randint(-32768, 32767, (K, N, 32), dtype=torch.int16, device=dev)
    def mkt(): return torch.randint(-32768, 32767, (N, K, 32), dtype=torch.int16, device=dev)

    gate_t = [mk() for _ in range(E)]
    up_t = [mk() for _ in range(E)]
    down_t = [mkt() for _ in range(E)]
    gate_suh = [torch.randn(hidden, dtype=torch.half, device=dev) for _ in range(E)]
    gate_svh = [torch.randn(inter, dtype=torch.half, device=dev) for _ in range(E)]
    up_suh = [torch.randn(hidden, dtype=torch.half, device=dev) for _ in range(E)]
    up_svh = [torch.randn(inter, dtype=torch.half, device=dev) for _ in range(E)]
    down_suh = [torch.randn(inter, dtype=torch.half, device=dev) for _ in range(E)]
    down_svh = [torch.randn(hidden, dtype=torch.half, device=dev) for _ in range(E)]

    def ptrs(ts): return torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device=dev)
    tables = [ptrs(x) for x in (gate_t, gate_suh, gate_svh, up_t, up_suh, up_svh, down_t, down_suh, down_svh)]
    gt, gu, gv, ut, uu, uv, dt, du, dv = tables

    x = torch.randn(1, hidden, dtype=torch.half, device=dev)
    ids_u = torch.randperm(E, device=dev)[: args.e].to(torch.int32).contiguous()
    rw = torch.randn(args.e, dtype=torch.half, device=dev)
    rw = (rw / rw.abs().max()).half()
    rw_check = torch.zeros_like(rw)
    rw_check[0] = 1.0
    vdecs = [int(s) for s in args.vdecs.split(",")]

    weights_per_call = 3 * args.e * hidden * inter
    bytes_per_call = weights_per_call * 2 / 8

    def call(variant, ids, iters, warmup):
        return ext.bench2(x, torch.empty_like(x), gt, gu, gv, ut, uu, uv, dt, du, dv,
                          ids, rw, 2, 2, 2, True, inter, 7.0, variant, 1, iters, warmup)

    if args.mode in ("all", "check"):
        out_ref = torch.empty_like(x)
        stock.p2b_fused_moe(x, out_ref, gt, gu, gv, ut, uu, uv, dt, du, dv, ids_u, rw_check, 2, 2, 2, True, inter, 7.0)
        for v in vdecs:
            ext.bench2(x, out_ref, gt, gu, gv, ut, uu, uv, dt, du, dv, ids_u, rw_check, 2, 2, 2, True, inter, 7.0, v, 1, 1, 0)
            o = ext.bench2(x, torch.empty_like(x), gt, gu, gv, ut, uu, uv, dt, du, dv, ids_u, rw_check, 2, 2, 2, True, inter, 7.0, v, 1, 1, 0)
            # compare against stock again (out_ref was reused above; recompute)
            ref = torch.empty_like(x)
            stock.p2b_fused_moe(x, ref, gt, gu, gv, ut, uu, uv, dt, du, dv, ids_u, rw_check, 2, 2, 2, True, inter, 7.0)
            ok = torch.equal(ref, o)
            print(f"[check] vdec{v} vs stock: {'BIT-EXACT' if ok else 'MISMATCH'}")
            if not ok:
                d = (ref.float() - o.float()).abs()
                print("  max abs diff:", d.max().item())
                sys.exit(1)

    if args.mode in ("all", "warm"):
        for v in vdecs:
            call(v, ids_u, args.iters, args.warmup)
            ms = ext.last_ms()
            print(f"[warm] vdec{v} e={args.e}: {ms*1000:.1f} us/call  "
                  f"{weights_per_call/(ms*1e-3)/1e9:.1f} Gw/s  {bytes_per_call/(ms*1e-3)/1e9:.1f} GB/s")

    if args.mode in ("all", "cold"):
        # Fresh ids per call; time each call with events; report median.
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        out = torch.empty_like(x)
        for v in vdecs:
            for _ in range(5):
                ext.bench2(x, out, gt, gu, gv, ut, uu, uv, dt, du, dv,
                           torch.randperm(E, device=dev)[: args.e].to(torch.int32).contiguous(),
                           rw, 2, 2, 2, True, inter, 7.0, v, 1, 1, 0)
            torch.cuda.synchronize()
            times = []
            id_pool = [torch.randperm(E, device=dev)[: args.e].to(torch.int32).contiguous()
                       for _ in range(args.cold_iters)]
            for ids in id_pool:
                ev0.record()
                ext.bench2(x, out, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, 2, 2, 2, True, inter, 7.0, v, 1, 1, 0)
                ev1.record()
                torch.cuda.synchronize()
                times.append(ev0.elapsed_time(ev1))
            med = statistics.median(times)
            print(f"[cold] vdec{v} e={args.e}: median {med*1000:.1f} us/call  "
                  f"{weights_per_call/(med*1e-3)/1e9:.1f} Gw/s  {bytes_per_call/(med*1e-3)/1e9:.1f} GB/s")


if __name__ == "__main__":
    main()
