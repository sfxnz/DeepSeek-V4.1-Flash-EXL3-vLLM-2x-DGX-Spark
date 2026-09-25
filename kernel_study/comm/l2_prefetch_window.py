#!/usr/bin/env python3
"""Can the DRAM-idle window around a decode all-reduce warm L2 for the next GEMM?

Question (k3 comm, "overlap the AR with independent work"). After each MoE
all-reduce the serve runs AR (22.7 us, NCCL LL, 5 CTAs spinning) -> mhc_post ->
mhc_pre (DeepGEMM tf32 prenorm + TileLang pre) -> mxfp8 act quant -> qkv_a b12x
GEMM (9.46 MB of weights at m=4) -> q/kv norm -> wq_b (21.6 MB). The window before
qkv_a moves ~2.3 MB of DRAM traffic in ~55 us, so DRAM is mostly idle. If a side
stream forked at the AR start prefetches qkv_a's weights (and maybe the head of wq_b)
into L2 (24 MiB on GB10), the GEMMs read them from L2. Nothing here changes any math.

Per replay, one CUDA graph. L2 is emptied first by a streaming read of 2x L2
(l2_cold_probe.py: this makes qkv_a exactly as cold as 8 rotating weight copies, 45.5
vs 47.0 us, and as the serve's 47.6 us; a memset flush leaves dirty lines whose
write-back doubles the GEMM time). No arm may use an L2 evict_last policy: such lines
survive the flush and warm the next arm's GEMM (runs v1-v3 were contaminated this way):
  e0 | spin(ar_us, 5 CTAs) | e_ar | mhc_post | e_post | mhc_pre | e_pre | act quant | e1 |
     qkv_a | e2 | act quant ; wq_b | e3
Arms (alternated replay by replay in one process), see ARMS: no prefetch, a ceiling
(qkv_a weights read into L2 before e0), and prefetch variants forked on a side
stream at e0: TMA cp.async.bulk.prefetch.L2 of a fraction of qkv_a (weights then
scales), optionally followed by the head of wq_b.
Kernels are the serve's: TileLang mHC, DeepGEMM prenorm, flashinfer mxfp8 quant,
b12x mm_mxfp8 (backend auto with the repo's sitecustomize mounted, as run.sh does).
Weights are random MXFP8 of the serve's shapes; every arm's GEMM outputs are
compared bitwise with arm 'none'.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

HIDDEN, HC = 5120, 4
QKV_A = (5120, 1792)  # K, N per rank: fused wq_a + wkv
WQ_B = (1280, 16384)
# arm -> list of (method, tensor 'a' (qkv_a) / 'b' (wq_b), fraction of its bytes, CTAs);
# method 'bulk' = TMA prefetch issued by CTAs x 128 threads, 'ld' = plain loads by CTAs x 256.
# An arm whose name ends in '_after' forks the prefetch after the AR spin, not before it.
ARMS = {
    "none": [],
    "warm": "ceiling",
    "bulk100": [("bulk", "a", 1.0, 1)],
    "bulk60": [("bulk", "a", 0.6, 1)],
    "bulk40": [("bulk", "a", 0.4, 1)],
    "bulk100_b25": [("bulk", "a", 1.0, 1), ("bulk", "b", 0.25, 1)],
    "ld100": [("ld", "a", 1.0, 16)],
    "bulk100_c8": [("bulk", "a", 1.0, 8)],
    "bulk100_c48": [("bulk", "a", 1.0, 48)],
    "bulk60_c8": [("bulk", "a", 0.6, 8)],
    "bulk100_after": [("bulk", "a", 1.0, 8)],
}
CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

__global__ void spin_kernel(long long cycles) {
  long long t0 = clock64();
  while (clock64() - t0 < cycles) { }
}

__global__ void bulk_prefetch_kernel(const char* p, long long bytes, int chunk) {
  long long n = (bytes + chunk - 1) / chunk;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n;
       i += (long long)gridDim.x * blockDim.x) {
    long long off = i * (long long)chunk;
    long long rem = bytes - off;
    unsigned sz = (unsigned)(rem < chunk ? rem : chunk) & ~15u;
    if (sz) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p + off), "r"(sz) : "memory");
  }
}

template <bool LAST>
__global__ void read_kernel(const int4* p, long long n16, int* sink) {
  uint64_t pol = 0;
  if (LAST) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  int acc = 0;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n16;
       i += (long long)gridDim.x * blockDim.x) {
    int a, b, c, d;
    if (LAST)
      asm volatile("ld.global.L2::cache_hint.v4.s32 {%0,%1,%2,%3}, [%4], %5;"
                   : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(p + i), "l"(pol));
    else
      asm volatile("ld.global.v4.s32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(p + i));
    acc ^= a ^ b ^ c ^ d;
  }
  if (acc == 0x7f7f7f7f) *sink = acc;
}

void spin(double cycles, int ctas) {
  spin_kernel<<<ctas, 32, 0, at::cuda::getCurrentCUDAStream()>>>((long long)cycles);
}
void bulk_prefetch(torch::Tensor t, long long bytes, int chunk, int ctas) {
  bulk_prefetch_kernel<<<ctas, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const char*)t.data_ptr(), bytes, chunk);
}
void read_l2(torch::Tensor t, long long bytes, int ctas, bool last, torch::Tensor sink) {
  auto s = at::cuda::getCurrentCUDAStream();
  if (last) read_kernel<true><<<ctas, 256, 0, s>>>((const int4*)t.data_ptr(), bytes / 16, (int*)sink.data_ptr());
  else read_kernel<false><<<ctas, 256, 0, s>>>((const int4*)t.data_ptr(), bytes / 16, (int*)sink.data_ptr());
}
"""
CPP_SRC = """
void spin(double cycles, int ctas);
void bulk_prefetch(torch::Tensor t, long long bytes, int chunk, int ctas);
void read_l2(torch::Tensor t, long long bytes, int ctas, bool last, torch::Tensor sink);
"""
EVENTS = ("e0", "ar", "post", "pre", "e1", "e2", "e3")
SPANS = {  # name: (from event, to event)
    "window": ("e0", "e1"),
    "w_ar": ("e0", "ar"),
    "w_mhc_post": ("ar", "post"),
    "w_mhc_pre": ("post", "pre"),
    "w_act_quant": ("pre", "e1"),
    "qkv_a": ("e1", "e2"),
    "wq_b": ("e2", "e3"),
    "total": ("e0", "e3"),
}


def build_ext(build_dir: str):
    from torch.utils.cpp_extension import load_inline

    os.makedirs(build_dir, exist_ok=True)
    return load_inline(
        name="l2pf_ext3",
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        functions=["spin", "bulk_prefetch", "read_l2"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_121a,code=sm_121a"],
        build_directory=build_dir,
        verbose=False,
    )


def stats(xs):
    s = sorted(xs)
    q = lambda f: s[min(len(s) - 1, int(round(f * (len(s) - 1))))]  # noqa: E731
    return {"median": round(statistics.median(s), 2), "p10": round(q(0.1), 2), "p90": round(q(0.9), 2), "n": len(s)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-dir", default="/repo/kernel_study/comm/.l2pf_build3")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--ar-us", type=float, default=22.7)
    ap.add_argument("--sm-mhz", type=float, default=2190.0, help="spin calibration (spark2 SM clock)")
    ap.add_argument("--replays", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--json")
    args = ap.parse_args()
    if args.compile_only:
        build_ext(args.build_dir)
        print("compiled", args.build_dir)
        return 0

    import torch
    from vllm.model_executor.kernels.mhc import tilelang as mhc_tl
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
        swizzle_mxfp8_scale,
    )
    from vllm.utils import deep_gemm as vdg
    from vllm.utils import flashinfer as vfi

    ext = build_ext(args.build_dir)
    vdg._lazy_init()
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    props = torch.cuda.get_device_properties(0)
    l2 = int(props.L2_cache_size)
    m = args.m

    def mx_weight(k, n):
        w = torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02
        w8, sc = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=False)
        return w8, swizzle_mxfp8_scale(sc, M=n, K=k).contiguous()

    wa, sa = mx_weight(*QKV_A)
    wb, sb = mx_weight(*WQ_B)
    k_hc, mix = HC * HIDDEN, HC * (HC + 2)
    fn = torch.randn(mix, k_hc, device=dev, dtype=torch.float32) * 0.02
    hc_scale = torch.rand(3, device=dev, dtype=torch.float32) + 0.5
    hc_base = torch.randn(mix, device=dev, dtype=torch.float32) * 0.1
    norm_w = (torch.rand(HIDDEN, device=dev) + 0.5).to(torch.bfloat16)
    residual = torch.randn(m, HC, HIDDEN, device=dev, dtype=torch.bfloat16)
    pre_mix = torch.softmax(torch.randn(m, HC, device=dev), -1).float().contiguous()
    post_mix, res_mix, _, _ = mhc_tl.mhc_pre_delayed_tilelang(
        residual, fn, hc_scale, hc_base, 1e-6, 1e-6, 1e-6, 2.0, 20, pre_mix=pre_mix,
        norm_weight=norm_w, norm_eps=1e-6)
    x_ar = torch.randn(m, HIDDEN, device=dev, dtype=torch.bfloat16)  # the AR output
    xb_in = torch.randn(m, WQ_B[0], device=dev, dtype=torch.bfloat16)
    flush = torch.empty(2 * l2, dtype=torch.uint8, device=dev)
    sink = torch.zeros(1, dtype=torch.int32, device=dev)
    tensors = {"a": (wa, sa), "b": (wb, sb)}
    side = torch.cuda.Stream()
    arms = args.arms.split(",")
    ev = {a: {e: torch.cuda.Event(enable_timing=True, external=True) for e in EVENTS} for a in arms}
    outs = {}

    def prefetch(spec):
        for method, key, frac, ctas in spec:
            for t in tensors[key]:
                nbytes = int(t.numel() * t.element_size() * frac) & ~15
                if method == "bulk":
                    ext.bulk_prefetch(t, nbytes, args.chunk, ctas)
                else:
                    ext.read_l2(t, nbytes, ctas, False, sink)

    def body(arm):
        spec = ARMS[arm]

        def b():
            ext.read_l2(flush, flush.numel(), 48, False, sink)  # clean cold L2
            if spec == "ceiling":
                for t in tensors["a"]:
                    ext.read_l2(t, t.numel() * t.element_size() & ~15, 48, False, sink)
            e = ev[arm]
            cur = torch.cuda.current_stream()
            e["e0"].record()
            fork = bool(spec) and spec != "ceiling"
            after = arm.endswith("_after")
            if fork and not after:
                side.wait_stream(cur)
                with torch.cuda.stream(side):
                    prefetch(spec)
            ext.spin(args.ar_us * args.sm_mhz, 5)
            e["ar"].record()
            if fork and after:
                side.wait_stream(cur)
                with torch.cuda.stream(side):
                    prefetch(spec)
            res2 = mhc_tl.mhc_post_tilelang(x_ar, residual, post_mix, res_mix)
            e["post"].record()
            _, _, x, _ = mhc_tl.mhc_pre_delayed_tilelang(
                res2, fn, hc_scale, hc_base, 1e-6, 1e-6, 1e-6, 2.0, 20, pre_mix=pre_mix,
                norm_weight=norm_w, norm_eps=1e-6)
            e["pre"].record()
            q, s = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
            e["e1"].record()
            oa = vfi.mm_mxfp8(q, wa.t(), s, sa, out_dtype=torch.bfloat16, backend="auto")
            e["e2"].record()
            qb, sbq = mxfp8_e4m3_quantize(xb_in, is_sf_swizzled_layout=True)
            ob = vfi.mm_mxfp8(qb, wb.t(), sbq, sb, out_dtype=torch.bfloat16, backend="auto")
            e["e3"].record()
            if fork:
                cur.wait_stream(side)
            outs[arm] = (oa, ob)

        return b

    graphs = {}
    for a in arms:
        b = body(a)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            b()
            b()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            b()
        graphs[a] = g
    torch.cuda.synchronize()

    res = {a: {k: [] for k in SPANS} for a in arms}
    for i in range(args.warmup + args.replays):
        for a in arms:
            graphs[a].replay()
            torch.cuda.synchronize()
            if i < args.warmup:
                continue
            e = ev[a]
            for k, (x, y) in SPANS.items():
                res[a][k].append(e[x].elapsed_time(e[y]) * 1e3)
    summary = {}
    ref = outs.get("none")
    for a in arms:
        summary[a] = {k: stats(v) for k, v in res[a].items()}
        if ref is not None:
            summary[a]["bitwise_equal_to_none"] = bool(
                torch.equal(outs[a][0], ref[0]) and torch.equal(outs[a][1], ref[1]))
    out = {
        "what": "L2 warm-up of the next GEMM's weights during the AR + mHC window "
                "(single GPU, AR emulated by a 5-CTA clock spin, 2x-L2 streaming-read flush per replay)",
        "device": props.name,
        "l2_bytes": l2,
        "m": m,
        "ar_us": args.ar_us,
        "replays": args.replays,
        "warmup": args.warmup,
        "chunk": args.chunk,
        "qkv_a_bytes": wa.numel() + sa.numel(),
        "wq_b_bytes": wb.numel() + sb.numel(),
        "arms": {a: ARMS[a] for a in arms},
        "summary": summary,
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(json.dumps(out, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
