#!/usr/bin/env python3
"""bench5 driver: group-major (warp-contiguous) trellis layout A/B.

vdec0 reads standard [k][n][words] tensors; vdec5 reads permuted
[group][k][128-uint16] tensors holding identical logical weights.
Correctness: vdec5 output must be bit-exact vs installed stock on the
standard data. Timing: cold-rotating ids.
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
        name="bench_p2b5",
        sources=[os.path.join(HERE, "bench5.cu")],
        extra_include_paths=[CSRC, EXL, os.path.join(EXL, "quant")],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e", type=int, default=30)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--cold-iters", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--mode", default="all", choices=["all", "check", "warm", "cold"])
    args = ap.parse_args()

    ext = build_ext()
    import vllm_exl3_c as stock

    dev = "cuda"
    torch.manual_seed(7)
    E, hidden, inter = args.experts, 5120, 1152
    KT, NT = hidden // 16, inter // 16      # 320, 72 (gate/up)
    G = NT // 4                              # groups of 4 tiles

    def std_gate(): return torch.randint(-32768, 32767, (KT, NT, 32), dtype=torch.int16, device=dev)
    def std_down(): return torch.randint(-32768, 32767, (NT, KT, 32), dtype=torch.int16, device=dev)

    def perm_gate(t):  # [KT, NT, 32] -> [G, KT, 128]
        return t.view(KT, G, 128).permute(1, 0, 2).contiguous()
    def perm_down(t):  # [NT, KT, 32] -> [KT//4, NT, 128]
        return t.view(NT, KT // 4, 128).permute(1, 0, 2).contiguous()

    gate_std = [std_gate() for _ in range(E)]
    up_std = [std_gate() for _ in range(E)]
    down_std = [std_down() for _ in range(E)]
    gate_pm = [perm_gate(t) for t in gate_std]
    up_pm = [perm_gate(t) for t in up_std]
    down_pm = [perm_down(t) for t in down_std]

    gate_suh = [torch.randn(hidden, dtype=torch.half, device=dev) for _ in range(E)]
    gate_svh = [torch.randn(inter, dtype=torch.half, device=dev) for _ in range(E)]
    up_suh = [torch.randn(hidden, dtype=torch.half, device=dev) for _ in range(E)]
    up_svh = [torch.randn(inter, dtype=torch.half, device=dev) for _ in range(E)]
    down_suh = [torch.randn(inter, dtype=torch.half, device=dev) for _ in range(E)]
    down_svh = [torch.randn(hidden, dtype=torch.half, device=dev) for _ in range(E)]

    def ptrs(ts): return torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device=dev)

    tab_std = [ptrs(x) for x in (gate_std, gate_suh, gate_svh, up_std, up_suh, up_svh, down_std, down_suh, down_svh)]
    tab_pm = [ptrs(x) for x in (gate_pm, gate_suh, gate_svh, up_pm, up_suh, up_svh, down_pm, down_suh, down_svh)]

    x = torch.randn(1, hidden, dtype=torch.half, device=dev)
    ids_u = torch.randperm(E, device=dev)[: args.e].to(torch.int32).contiguous()
    rw = torch.randn(args.e, dtype=torch.half, device=dev)
    rw = (rw / rw.abs().max()).half()
    rw_check = torch.zeros_like(rw)
    rw_check[0] = 1.0

    weights_per_call = 3 * args.e * hidden * inter
    bytes_per_call = weights_per_call * 2 / 8

    if args.mode in ("all", "check"):
        ref = torch.empty_like(x)
        stock.p2b_fused_moe(x, ref, *tab_std, ids_u, rw_check, 2, 2, 2, True, inter, 7.0)
        o = ext.bench5(x, torch.empty_like(x), *tab_pm, ids_u, rw_check, 2, 2, 2, True, inter, 7.0, 5, 1, 0)
        ok = torch.equal(ref, o)
        print(f"[check] vdec5 (group-major) vs stock(std): {'BIT-EXACT' if ok else 'MISMATCH'}")
        if not ok:
            d = (ref.float() - o.float()).abs()
            print("  max abs diff:", d.max().item(), " n_diff:", (d > 0).sum().item())
            sys.exit(1)

    if args.mode in ("all", "warm"):
        for v, tab in ((0, tab_std), (5, tab_pm)):
            ext.bench5(x, torch.empty_like(x), *tab, ids_u, rw, 2, 2, 2, True, inter, 7.0, v, args.iters, args.warmup)
            ms = ext.last_ms()
            print(f"[warm] vdec{v} e={args.e}: {ms*1000:.1f} us/call  "
                  f"{weights_per_call/(ms*1e-3)/1e9:.1f} Gw/s  {bytes_per_call/(ms*1e-3)/1e9:.1f} GB/s")

    if args.mode in ("all", "cold"):
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        out = torch.empty_like(x)
        for v, tab in ((0, tab_std), (5, tab_pm)):
            for _ in range(5):
                ids = torch.randperm(E, device=dev)[: args.e].to(torch.int32).contiguous()
                ext.bench5(x, out, *tab, ids, rw, 2, 2, 2, True, inter, 7.0, v, 1, 0)
            torch.cuda.synchronize()
            times = []
            pool = [(torch.randperm(E, device=dev)[: args.e].to(torch.int32).contiguous()) for _ in range(args.cold_iters)]
            for ids in pool:
                ev0.record()
                ext.bench5(x, out, *tab, ids, rw, 2, 2, 2, True, inter, 7.0, v, 1, 0)
                ev1.record()
                torch.cuda.synchronize()
                times.append(ev0.elapsed_time(ev1))
            med = statistics.median(times)
            print(f"[cold] vdec{v} e={args.e}: median {med*1000:.1f} us/call  "
                  f"{weights_per_call/(med*1e-3)/1e9:.1f} Gw/s  {bytes_per_call/(med*1e-3)/1e9:.1f} GB/s")


if __name__ == "__main__":
    main()
