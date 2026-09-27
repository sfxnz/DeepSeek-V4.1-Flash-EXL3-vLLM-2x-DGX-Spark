#!/usr/bin/env python3
"""CUDA graph host-node latency on GB10: what the first NCCL collective of a replay waits on.

In the r3 serve trace the draft graph is launched ~44.7 ms before the GPU reaches it,
yet the device idles 235 us (rank 0) / 427 us (rank 1) right before the graph's first
all-reduce: NCCL puts every captured network collective behind a host node
(hostStreamPlanCallback uploads its proxy ops), and a host node only runs once the GPU
reaches the graph. This measures that latency without NCCL, on one GPU:

  main: stamp(g0) ; spin(T) ; stamp(g1) ; wait(side) ; stamp(g2)
  side: root host node (no dependencies, like NCCL's strong-stream capture) -> event
GPU delay caused by the host node beyond the spin = g2 - g1 (globaltimer). For T below
the host-node latency L, g2 - g1 ~ L - T. Also a chain of N root-chained host nodes
gating N kernels spaced by `layer_us` (the target graph: 82 nodes, ~650 us apart), and
the same with `spinners` busy host threads in the process (CPU contention).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include <time.h>

__global__ void spin_kernel(long long cycles) {
  long long t0 = clock64();
  while (clock64() - t0 < cycles) { }
}
__global__ void stamp_kernel(long long* out) {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  *out = t;
}
static void CUDART_CB host_cb(void* p) {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  *(volatile long long*)p = ts.tv_sec * 1000000000LL + ts.tv_nsec;
}
void spin(double cycles) {
  spin_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>((long long)cycles);
}
void stamp(torch::Tensor out, long long idx) {
  stamp_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>((long long*)out.data_ptr<int64_t>() + idx);
}
void host_node(torch::Tensor pinned, long long idx) {
  C10_CUDA_CHECK(cudaLaunchHostFunc(at::cuda::getCurrentCUDAStream(), host_cb, pinned.data_ptr<int64_t>() + idx));
}
void capture_root() {  // the current stream's next captured node gets no dependencies
  C10_CUDA_CHECK(cudaStreamUpdateCaptureDependencies(at::cuda::getCurrentCUDAStream(), nullptr, nullptr, 0,
                                                     cudaStreamSetCaptureDependencies));
}
static void CUDART_CB noop_cb(void*) {}
int set_flags(int flags) { return (int)cudaSetDeviceFlags((unsigned)flags); }
int get_flags() { unsigned f = 0; cudaGetDeviceFlags(&f); return (int)f; }
static std::atomic<bool> g_hb_stop{false};
static std::thread g_hb;
void heartbeat(double period_us) {  // host funcs on a private stream every period_us (0 stops)
  g_hb_stop = true;
  if (g_hb.joinable()) g_hb.join();
  if (period_us <= 0) return;
  g_hb_stop = false;
  g_hb = std::thread([period_us] {
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    while (!g_hb_stop.load()) {
      cudaLaunchHostFunc(s, noop_cb, nullptr);
      std::this_thread::sleep_for(std::chrono::microseconds((long)period_us));
    }
    cudaStreamSynchronize(s);
    cudaStreamDestroy(s);
  });
}
static std::atomic<bool> g_stop{false};
static std::vector<std::thread> g_threads;
void spinners(int n) {
  g_stop = true;
  for (auto& t : g_threads) t.join();
  g_threads.clear();
  g_stop = false;
  for (int i = 0; i < n; ++i)
    g_threads.emplace_back([] { volatile unsigned long x = 0; while (!g_stop.load(std::memory_order_relaxed)) x++; });
}
"""
CPP_SRC = """
void spin(double cycles);
void stamp(torch::Tensor out, long long idx);
void host_node(torch::Tensor pinned, long long idx);
void capture_root();
void spinners(int n);
int set_flags(int flags);
int get_flags();
void heartbeat(double period_us);
"""


def build_ext(build_dir: str):
    from torch.utils.cpp_extension import load_inline

    os.makedirs(build_dir, exist_ok=True)
    return load_inline(name="hn_ext", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC,
                       functions=["spin", "stamp", "host_node", "capture_root", "spinners", "set_flags",
                                  "get_flags", "heartbeat"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_121a,code=sm_121a"],
                       build_directory=build_dir, verbose=False)


def stats(xs):
    s = sorted(xs)
    q = lambda f: s[min(len(s) - 1, int(round(f * (len(s) - 1))))]  # noqa: E731
    return {"median": round(statistics.median(s), 2), "p10": round(q(0.1), 2), "p90": round(q(0.9), 2),
            "max": round(s[-1], 2), "n": len(s)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-dir", default="/repo/kernel_study/comm/.hn_build")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--sm-mhz", type=float, default=2190.0)
    ap.add_argument("--replays", type=int, default=200)
    ap.add_argument("--spin-us", type=float, nargs="+", default=[2, 50, 100, 200, 300, 500, 800])
    ap.add_argument("--chain", type=int, default=82)
    ap.add_argument("--layer-us", type=float, default=650.0)
    ap.add_argument("--spinners", type=int, nargs="+", default=[0, 16])
    ap.add_argument("--gap-ms", type=float, nargs="+", default=[0.0],
                    help="host sleep between replays (the serve replays each graph once per ~63 ms step)")
    ap.add_argument("--device-flags", type=int, default=-1,
                    help="cudaSetDeviceFlags before CUDA init: 0 auto, 1 spin, 2 yield, 4 blocking sync")
    ap.add_argument("--heartbeat-us", type=float, default=0.0,
                    help="a thread enqueues no-op host funcs on a private stream at this period")
    ap.add_argument("--pm-qos-us", type=int, nargs="+", default=[],
                    help="also run every case holding a /dev/cpu_dma_latency request of each N us "
                         "(needs root + --device /dev/cpu_dma_latency); arms cycle none,N1,N2,...")
    ap.add_argument("--qos-reps", type=int, default=2)
    ap.add_argument("--json")
    args = ap.parse_args()
    if args.compile_only:
        build_ext(args.build_dir)
        print("compiled", args.build_dir)
        return 0
    import torch

    ext = build_ext(args.build_dir)
    flags_rc = ext.set_flags(args.device_flags) if args.device_flags >= 0 else None
    stamps = torch.zeros(4 * (args.chain + 2), dtype=torch.int64, device="cuda")
    device_flags = ext.get_flags()
    ext.heartbeat(args.heartbeat_us)
    host = torch.zeros(args.chain + 2, dtype=torch.int64).pin_memory()
    main_s, side = torch.cuda.Stream(), torch.cuda.Stream()
    cyc = lambda us: us * args.sm_mhz  # noqa: E731

    def single(t_us):
        def body():
            ext.stamp(stamps, 0)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                ext.capture_root()
                ext.host_node(host, 0)
            ext.spin(cyc(t_us))
            ext.stamp(stamps, 1)
            torch.cuda.current_stream().wait_stream(side)
            ext.stamp(stamps, 2)
        return body

    def chain(n):
        # NCCL-like: one host-stream chain of n nodes, all roots of the graph; kernel i waits node i
        def body():
            ext.stamp(stamps, 0)
            side.wait_stream(torch.cuda.current_stream())
            evs = []
            with torch.cuda.stream(side):
                ext.capture_root()
                for i in range(n):
                    ext.host_node(host, i)
                    ev = torch.cuda.Event()
                    ev.record()
                    evs.append(ev)
            for i in range(n):
                ext.spin(cyc(args.layer_us / 2))
                ext.stamp(stamps, 1 + 2 * i)
                torch.cuda.current_stream().wait_event(evs[i])
                ext.stamp(stamps, 2 + 2 * i)
            torch.cuda.current_stream().wait_stream(side)
        return body

    def capture(body):
        with torch.cuda.stream(main_s):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=main_s):
                body()
        torch.cuda.synchronize()
        return g

    out = {"single": {}, "chain": {}}
    qos_fd = None

    def qos(us):
        nonlocal qos_fd
        import struct
        if qos_fd is not None:
            os.close(qos_fd)
            qos_fd = None
        if us is not None:
            qos_fd = os.open("/dev/cpu_dma_latency", os.O_WRONLY)
            os.write(qos_fd, struct.pack("i", us))

    modes = ([None] + list(args.pm_qos_us)) * args.qos_reps if args.pm_qos_us else [None]
    for rep, qos_us in enumerate(modes):
      qos(qos_us)
      time.sleep(0.5)
      tag = f"qos{qos_us}" if qos_us is not None else "noqos"
      for nsp in args.spinners:
        ext.spinners(nsp)
        time.sleep(0.2)
        for gap in args.gap_ms:
            for t in args.spin_us:
                g = capture(single(t))
                delay = []
                for i in range(args.replays + 10):
                    if gap:
                        time.sleep(gap / 1e3)
                    with torch.cuda.stream(main_s):
                        g.replay()
                    torch.cuda.synchronize()
                    if i >= 10:
                        s = stamps[:3].tolist()
                        delay.append((s[2] - s[1]) / 1e3)
                out["single"][f"{tag}_r{rep}_spinners{nsp}_gap{gap:g}ms_T{t:g}us"] = stats(delay)
                print("single", tag, rep, nsp, gap, t, stats(delay), flush=True)
                del g
        g = capture(chain(args.chain))
        per_node, first = [], []
        for i in range(max(20, args.replays // 10) + 3):
            if args.gap_ms[-1]:
                time.sleep(args.gap_ms[-1] / 1e3)
            with torch.cuda.stream(main_s):
                g.replay()
            torch.cuda.synchronize()
            if i >= 3:
                s = stamps[: 2 * args.chain + 1].tolist()
                d = [(s[2 + 2 * k] - s[1 + 2 * k]) / 1e3 for k in range(args.chain)]
                first.append(d[0])
                per_node += d[1:]
        key = f"{tag}_r{rep}_spinners{nsp}_gap{args.gap_ms[-1]:g}ms"
        out["chain"][key] = {"first_node_delay_us": stats(first), "later_nodes_delay_us": stats(per_node),
                             "layer_us": args.layer_us, "nodes": args.chain}
        print("chain", key, out["chain"][key], flush=True)
        del g
    qos(None)
    ext.spinners(0)
    ext.heartbeat(0)
    res = {"what": "CUDA graph host-node latency (GPU delay behind a root host node), single GPU",
           "device": torch.cuda.get_device_name(0), "cpus": os.cpu_count(),
           "device_flags_set": args.device_flags, "set_flags_rc": flags_rc, "device_flags": device_flags,
           "heartbeat_us": args.heartbeat_us, "pm_qos_us": args.pm_qos_us, **out,
           "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
