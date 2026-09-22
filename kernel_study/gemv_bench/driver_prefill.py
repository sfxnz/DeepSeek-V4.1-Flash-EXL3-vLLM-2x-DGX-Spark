#!/usr/bin/env python3
"""Prefill no-regression harness: stock vs PF-G8 group-major trellis layout.

Gate harness for the group-major pack-permutation lever (RESULTS.md round 9/10:
+6.0% cold p2b decode, bit-exact, bench5.cu DEC5). This extends the driver5
pattern to the PREFILL side: the exllamav3 exl3_gemm tiled kernels at
m in {64, 128, 256, 512} rows per expert.

Chunked prefill at 8192 tokens / 384 local experts averages ~21 rows/expert,
but fat experts take far more; the sweep measures the fat tail that dominates
layer time.

Matrices (TP=2 local dims, same as driver5):
  gate/up: [m, 5120] x [5120, 2304]  trellis [320][144][32], GEMM shapes 2,3
           (2304 % 256 == 0, 2304 % 512 != 0 -> shape 4 not compatible)
  down:    [m, 2304] x [2304, 5120]  trellis [144][320][32], GEMM shapes 2,3,4

Layouts:
  stock: trellis [KT][NT][32] uint16 words ([k][n][16*K])
  PF-G8: trellis [NT/8][KT][256] — group of 8 n-tiles major. Per-k-block rows
         are 512B contiguous for the GEMM and every aligned 4-tile run stays
         contiguous for the p2b decode warp stream (8 % 4 == 0), so ONE
         permuted pack serves both readers.

Correctness: same logical weights in both layouts; PF-G8 run must be
BIT-EXACT vs stock run (identical kernels, only B-load addresses differ).

Run inside the bench/serve image in a GPU maintenance window (build first
with build_prefill.sh, which compiles CPU-side in a no-GPU container):
  python3 driver_prefill.py                 # check + sweep + layer estimate
  python3 driver_prefill.py --mode check
  python3 driver_prefill.py --reps 5
"""
import argparse
import os
import statistics
import sys

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

HIDDEN, INTER = 5120, 2304  # full local dims; trellis n-tiles 144 / 320

# (name, k, n, compatible GEMM shape indices)
MATRICES = [
    ("gate/up", HIDDEN, INTER, (2, 3)),
    ("down", INTER, HIDDEN, (2, 3, 4)),
]

LOCAL_EXPERTS = 384
ROUTED_LAYERS = 37


def build_ext():
    from torch.utils.cpp_extension import load
    return load(
        name="bench_prefill",
        sources=[os.path.join(HERE, "build_prefill", "bench_prefill.cu")],
        extra_include_paths=[
            os.path.join(HERE, "build_prefill"),
            "/usr/local/lib/python3.12/dist-packages/exllamav3/exllamav3_ext",
            "/usr/local/lib/python3.12/dist-packages/exllamav3/exllamav3_ext/quant",
        ],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def make_stock(kt, nt, dev):
    return torch.randint(-32768, 32767, (kt, nt, 32),
                         dtype=torch.int16, device=dev)


def perm_g8(t):
    """[KT][NT][32] -> [NT/8][KT][256]: group of 8 n-tiles major."""
    kt, nt, w = t.shape
    return t.view(kt, nt // 8, 8 * w).permute(1, 0, 2).contiguous()


def run_gemm(ext, x, t, k, dev, shape, iters, warmup):
    suh = torch.ones(k, dtype=torch.half, device=dev)
    n = t.numel() // (k // 16) // 32 * 16
    svh = torch.ones(n, dtype=torch.half, device=dev)
    return ext.gemm(x, t, suh, svh, shape, iters, warmup)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="64,128,256,512", help="comma list of m")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--reps", type=int, default=3, help="timing repetitions (median)")
    ap.add_argument("--mode", default="all", choices=["all", "check", "bench"])
    args = ap.parse_args()

    dev = "cuda"
    ext = build_ext()
    torch.manual_seed(7)
    ms = [int(v) for v in args.m.split(",")]

    torch.cuda.synchronize()
    data = {}
    for name, k, n, shapes in MATRICES:
        t_std = make_stock(k // 16, n // 16, dev)
        data[name] = (t_std, perm_g8(t_std))

    all_ok = True
    if args.mode in ("all", "check"):
        # Bit-exact: stock layout vs G8 layout, every m, every matrix,
        # every compatible shape (checked on its smallest shape per matrix
        # plus shape 4 on down).
        for name, k, n, shapes in MATRICES:
            t_std, t_g8 = data[name]
            for m in ms:
                x = torch.randn(m, k, dtype=torch.half, device=dev)
                for s in shapes:
                    ext.set_layout(0)
                    ref = run_gemm(ext, x, t_std, k, dev, s, 1, 0)
                    ext.set_layout(1)
                    out = run_gemm(ext, x, t_g8, k, dev, s, 1, 0)
                    ext.set_layout(0)
                    ok = torch.equal(ref, out)
                    all_ok &= ok
                    tag = "BIT-EXACT" if ok else "MISMATCH"
                    print(f"[check] {name} m={m} shape{s}: {tag}")
                    if not ok:
                        d = (ref.float() - out.float()).abs()
                        print(f"  max abs diff {d.max().item()}  "
                              f"n_diff {(d > 0).sum().item()} / {d.numel()}")
        if not all_ok:
            sys.exit(1)

    if args.mode in ("all", "bench"):
        print(f"\n{'matrix':>8} {'m':>5} {'shape':>5} {'stock us':>10} "
              f"{'g8 us':>10} {'delta':>8}")
        results = {}  # (m) -> {layout: layer_us}
        for name, k, n, shapes in MATRICES:
            t_std, t_g8 = data[name]
            for m in ms:
                x = torch.randn(m, k, dtype=torch.half, device=dev)
                for s in shapes:
                    times = {0: [], 1: []}
                    for layout, t in ((0, t_std), (1, t_g8)):
                        ext.set_layout(layout)
                        for _ in range(args.reps):
                            run_gemm(ext, x, t, k, dev, s, args.iters, args.warmup)
                            times[layout].append(ext.last_ms() * 1000.0)
                    ext.set_layout(0)
                    s_med = statistics.median(times[0])
                    g_med = statistics.median(times[1])
                    delta = (s_med - g_med) / s_med * 100.0
                    print(f"{name:>8} {m:>5} {s:>5} {s_med:>10.1f} "
                          f"{g_med:>10.1f} {delta:>+7.1f}%")
                    # best shape per (matrix, m, layout) for the layer estimate
                    best = results.setdefault((name, m), {})
                    for layout, us in ((0, s_med), (1, g_med)):
                        best[layout] = min(best.get(layout, 1e30), us)

        print(f"\n[layer estimate] 2x gate/up + 1x down per expert (best shape), "
              f"x{LOCAL_EXPERTS} experts x{ROUTED_LAYERS} routed layers")
        for m in ms:
            tot = {0: 0.0, 1: 0.0}
            for name, k, n, shapes in MATRICES:
                mult = 2 if name == "gate/up" else 1
                for layout in (0, 1):
                    tot[layout] += mult * results[(name, m)][layout]
            layer_s = {l: tot[l] * LOCAL_EXPERTS * ROUTED_LAYERS / 1e6 for l in (0, 1)}
            d = (layer_s[0] - layer_s[1]) / layer_s[0] * 100.0
            print(f"  m={m}: stock {layer_s[0]:.2f} s  g8 {layer_s[1]:.2f} s  ({d:+.1f}%)")


if __name__ == "__main__":
    main()
