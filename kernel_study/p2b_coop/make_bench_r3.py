#!/usr/bin/env python3
"""Round-3 coop-moe microbench sources from the committed pin (CPU only).

Writes build_r3/:
  chain_r3.cu     the p2b pin + the docker/Dockerfile.e13 chain + widen_p2b_coop.py
                  + the round-3 patches in R3_PATCHES (what the serve image compiles)
  bench_r3.cu     chain_r3 with a runtime variant switch and a pybind module
  bench_r3_ts.cu  bench_r3 plus %globaltimer stamps around every grid.sync() (phase
                  breakdown only; its timings are not the reported ones)

Variant ids are the DSV41_P2B_COOP values: 0 = p2b (SORT=0, the served kernel),
1 = round-2 coop (SORT=2), 2.. = round-3 variants. The env getters are replaced by
g_bench_variant, so one process can alternate every variant on identical data.
DSV41_P2B_SRC_SORT stays off.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PIN = ROOT / "tests/fixtures/p2b_moe.pin.cu"
CHAIN = ("shapes", "mrow", "cfg1", "codebook", "fshift", "srcsort", "coop")
R3_PATCHES: tuple[str, ...] = ("dataflow",)
OUT = HERE / "build_r3"

SWITCHES = (
    ("p2b_src_sort_enabled() && m * e <= P2B_SORT_CAP", "false && m * e <= P2B_SORT_CAP"),
    ("p2b_coop_enabled() && m * e <= P2B_SORT_CAP", "g_bench_variant == 1 && m * e <= P2B_SORT_CAP"),
)

GLOBALS = r'''
// --- kernel_study/p2b_coop/make_bench_r3: runtime variant switch ---
static int g_bench_variant = 0;
static int g_bench_last_grid = 0;
'''

TS_DEFS = r'''
// --- kernel_study/p2b_coop/make_bench_r3: phase stamps (bench_r3_ts only) ---
__device__ unsigned long long* g_p2b_ts = nullptr;
__device__ __forceinline__ unsigned long long p2b_gtimer()
{
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
#define P2B_TS(slot) do { if (threadIdx.x == 0 && g_p2b_ts) g_p2b_ts[(size_t) blockIdx.x * 64 + (slot)] = p2b_gtimer(); } while (0)
struct P2bEndTs { __device__ ~P2bEndTs() { P2B_TS(63); } };
#define P2B_GRID_SYNC() do { __syncthreads(); P2B_TS(_p2b_ph); grid.sync(); P2B_TS(_p2b_ph + 1); _p2b_ph += 2; } while (0)
// Dataflow tile accounting (lane 0 of every warp, ns): per warp [0] tile entry, [1] first B word in
// hand, [2] main loop end, [3] sum fill, [4] sum loop, [5] sum reduction, [6] tiles, [7] task-loop span.
__device__ unsigned long long* g_p2b_df = nullptr;
__device__ __forceinline__ unsigned long long* p2b_df_acc()
{
    __shared__ unsigned long long s[8][8];
    return &s[0][0];
}
#define DF_INIT() do { if ((threadIdx.x & 31) == 0) for (int k_ = 0; k_ < 8; ++k_) p2b_df_acc()[(threadIdx.x >> 5) * 8 + k_] = 0; } while (0)
#define DF_ENTRY() do { if (lane == 0) p2b_df_acc()[warp * 8 + 0] = p2b_gtimer(); } while (0)
#define DF_FILL(i, v) do { if (lane == 0 && (i) == 0) { asm volatile("" :: "r"(v)); p2b_df_acc()[warp * 8 + 1] = p2b_gtimer(); } } while (0)
#define DF_LOOPEND() do { if (lane == 0) { unsigned long long* a_ = p2b_df_acc() + warp * 8; const unsigned long long t_ = p2b_gtimer(); a_[3] += a_[1] - a_[0]; a_[4] += t_ - a_[1]; a_[2] = t_; } } while (0)
#define DF_EXIT() do { if (lane == 0) { unsigned long long* a_ = p2b_df_acc() + warp * 8; a_[5] += p2b_gtimer() - a_[2]; a_[6] += 1; } } while (0)
#define DF_SPAN0() do { if (lane == 0) p2b_df_acc()[warp * 8 + 7] = p2b_gtimer(); } while (0)
#define DF_SPAN1() do { if (lane == 0) p2b_df_acc()[warp * 8 + 7] = p2b_gtimer() - p2b_df_acc()[warp * 8 + 7]; } while (0)
#define DF_DUMP() do { if (lane == 0 && g_p2b_df) for (int k_ = 0; k_ < 8; ++k_) g_p2b_df[((size_t) blockIdx.x * 8 + warp) * 8 + k_] = p2b_df_acc()[warp * 8 + k_]; } while (0)
'''

# Bench-only memory-ceiling kernels (not in the serve image): the coop tiles' B streams without the
# decode (pattern modes), and a contiguous read of the same experts' bytes (seq mode).
STREAM = r'''

// ---------------------------------------------------------------------------
// kernel_study/p2b_coop/make_bench_r3: B-stream ceiling kernels (bench only)
// ---------------------------------------------------------------------------
// mode 0: tile pattern, task order (u, gate|up, group) then (u, down group), one __syncthreads
//         per task (the coop tiles' K-split: warp j reads k-slices [j*ch, (j+1)*ch) of 4 n-tiles)
// mode 1: tile pattern, no block barrier between tasks
// mode 2: contiguous: every warp reads 256 B steps of one contiguous span (same total bytes)
// mode 4/5 (round 4): N-split tile pattern (below)
__global__ __launch_bounds__(256, 3) void p2b_stream_kernel(
    const int64_t* __restrict__ gt, const int64_t* __restrict__ ut, const int64_t* __restrict__ dt,
    const int32_t* __restrict__ uniq, int U, int hidden, int inter, int mode, uint32_t* __restrict__ sink)
{
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int gtiles = inter / 16, gks = hidden / 16, dtiles = hidden / 16, dks = inter / 16;
    const int ggroups = inter / 64, dgroups = hidden / 64;
    const int per_u = 2 * ggroups + dgroups;
    const int tasks = U * per_u;
    uint32_t acc = 0;
    if (mode == 2) {
        // contiguous: expert e's gate, up, down trellis are each one contiguous span; split every span
        // into 256 B steps round-robin over all warps of the grid.
        const size_t steps_g = (size_t) gks * gtiles * 64 / 256;   // 256 B steps per gate/up matrix
        const size_t steps_d = (size_t) dks * dtiles * 64 / 256;
        const size_t per_e = 2 * steps_g + steps_d;
        const size_t total = per_e * U;
        const size_t nw = (size_t) gridDim.x * 8;
        const size_t gw = (size_t) blockIdx.x * 8 + warp;
        // each warp takes a contiguous run of steps
        const size_t run = (total + nw - 1) / nw;
        size_t s0 = gw * run, s1 = min(total, s0 + run);
        uint32_t pf0 = 0, pf1 = 0, pf2 = 0, pf3 = 0;
        auto addr = [&](size_t st) -> const uint32_t* {
            const size_t e = st / per_e, r = st % per_e;
            const int src = uniq[e];
            if (r < steps_g) return reinterpret_cast<const uint32_t*>(gt[src]) + r * 64;
            if (r < 2 * steps_g) return reinterpret_cast<const uint32_t*>(ut[src]) + (r - steps_g) * 64;
            return reinterpret_cast<const uint32_t*>(dt[src]) + (r - 2 * steps_g) * 64;
        };
        if (s0 < s1) { const uint32_t* a = addr(s0); pf0 = __ldcs(a + lane); pf1 = __ldcs(a + 32 + lane); }
        if (s0 + 1 < s1) { const uint32_t* a = addr(s0 + 1); pf2 = __ldcs(a + lane); pf3 = __ldcs(a + 32 + lane); }
        for (size_t st = s0; st < s1; st += 2) {
            acc ^= pf0 ^ pf1;
            if (st + 2 < s1) { const uint32_t* a = addr(st + 2); pf0 = __ldcs(a + lane); pf1 = __ldcs(a + 32 + lane); }
            if (st + 1 < s1) {
                acc ^= pf2 ^ pf3;
                if (st + 3 < s1) { const uint32_t* a = addr(st + 3); pf2 = __ldcs(a + lane); pf3 = __ldcs(a + 32 + lane); }
            }
        }
    } else if (mode >= 4) {
        // round 4, N-split tile pattern: a task = (u, matrix, K-range j, group g of 8 warp columns);
        // warp w reads the 256-B chunk of warp column 8g + w for k-slices [j*ch, (j+1)*ch), the same
        // per-warp K-range and B stream as a coop tile, but the block's 8 warps read 2 KB contiguous
        // per k-slice. mode 4: column groups fastest (neighbouring tasks share the K-range), mode 5:
        // K-ranges fastest. Partial groups (gate/up: 18 warp columns = 8 + 8 + 2) leave warps idle.
        const int gcg = (gtiles / 4 + 7) / 8, dcg = (dtiles / 4 + 7) / 8;
        const int per_un = 2 * 8 * gcg + 8 * dcg;
        const int tasks_n = U * per_un;
        for (int task = blockIdx.x; task < tasks_n; task += gridDim.x) {
            const int u = task / per_un, r = task % per_un;
            const int src = uniq[u];
            const uint32_t* B;
            int ks, nt, j, g;
            if (r < 16 * gcg) {
                const int mat = r / (8 * gcg), q = r % (8 * gcg);
                j = mode == 4 ? q / gcg : q % 8;
                g = mode == 4 ? q % gcg : q / 8;
                B = reinterpret_cast<const uint32_t*>(mat == 0 ? gt[src] : ut[src]); ks = gks; nt = gtiles;
            } else {
                const int q = r - 16 * gcg;
                j = mode == 4 ? q / dcg : q % 8;
                g = mode == 4 ? q % dcg : q / 8;
                B = reinterpret_cast<const uint32_t*>(dt[src]); ks = dks; nt = dtiles;
            }
            const int wc = g * 8 + warp;
            if (wc * 4 >= nt) continue;
            const int chunk = (ks + 7) / 8, k0 = j * chunk, myn = max(0, min(chunk, ks - k0));
            const size_t stride = (size_t) nt * 16;
            const uint32_t* bp = B + (size_t) k0 * stride + wc * 64 + lane;
            uint32_t pf[2][2];
            for (int d = 0; d < 2; ++d) if (d < myn) { pf[d][0] = __ldcs(bp + d * stride); pf[d][1] = __ldcs(bp + d * stride + 32); }
            for (int ib = 0; ib < myn; ib += 2) {
                #pragma unroll
                for (int d = 0; d < 2; ++d) {
                    const int i = ib + d;
                    if (i >= myn) break;
                    acc ^= pf[d][0] ^ pf[d][1];
                    if (i + 2 < myn) { pf[d][0] = __ldcs(bp + (size_t) (i + 2) * stride); pf[d][1] = __ldcs(bp + (size_t) (i + 2) * stride + 32); }
                }
            }
        }
    } else {
        for (int task = blockIdx.x; task < tasks; task += gridDim.x) {
            const int u = task / per_u, r = task % per_u;
            const int src = uniq[u];
            const uint32_t* B;
            int ks, nt, grp;
            if (r < 2 * ggroups) { B = reinterpret_cast<const uint32_t*>(r < ggroups ? gt[src] : ut[src]); ks = gks; nt = gtiles; grp = r % ggroups; }
            else { B = reinterpret_cast<const uint32_t*>(dt[src]); ks = dks; nt = dtiles; grp = r - 2 * ggroups; }
            const int chunk = (ks + 7) / 8, k0 = warp * chunk, myn = max(0, min(chunk, ks - k0));
            const size_t stride = (size_t) nt * 16;
            const uint32_t* bp = B + (size_t) k0 * stride + grp * 64 + lane;
            uint32_t pf[2][2];
            for (int d = 0; d < 2; ++d) if (d < myn) { pf[d][0] = __ldcs(bp + d * stride); pf[d][1] = __ldcs(bp + d * stride + 32); }
            for (int ib = 0; ib < myn; ib += 2) {
                #pragma unroll
                for (int d = 0; d < 2; ++d) {
                    const int i = ib + d;
                    if (i >= myn) break;
                    acc ^= pf[d][0] ^ pf[d][1];
                    if (i + 2 < myn) { pf[d][0] = __ldcs(bp + (size_t) (i + 2) * stride); pf[d][1] = __ldcs(bp + (size_t) (i + 2) * stride + 32); }
                }
            }
            if (mode == 0) __syncthreads();
        }
    }
    if (acc == 0x12345678u) sink[0] = acc;  // keep the loads
}

// Plain read-bandwidth probe: grid of `blocks` x 256 threads, every warp reads a contiguous run of
// `words` 32-bit words in steps of 32 lanes x VEC words, PF steps in flight in registers, plus an
// optional L2 prefetch `l2d` steps ahead (0 = off).
template <int PF, int VEC>
__global__ void p2b_bw_kernel(const uint32_t* __restrict__ buf, size_t words_per_warp, int l2d, uint32_t* __restrict__ sink)
{
    const int lane = threadIdx.x % 32;
    const size_t gw = (size_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const uint32_t* base = buf + gw * words_per_warp;
    const int steps = (int) (words_per_warp / (32 * VEC));
    typedef typename std::conditional<VEC == 4, uint4, uint32_t>::type V;
    const V* vb = reinterpret_cast<const V*>(base);
    V pf[PF];
    #pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < steps) pf[d] = __ldcs(vb + (size_t) d * 32 + lane);
    uint32_t acc = 0;
    for (int ib = 0; ib < steps; ib += PF) {
        #pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int i = ib + d;
            if (i >= steps) break;
            if constexpr (VEC == 4) acc ^= pf[d].x ^ pf[d].y ^ pf[d].z ^ pf[d].w; else acc ^= pf[d];
            if (i + PF < steps) pf[d] = __ldcs(vb + (size_t) (i + PF) * 32 + lane);
            if (l2d > 0 && i + PF + l2d < steps && lane < VEC)
                asm volatile("prefetch.global.L2 [%0];" :: "l"(base + (size_t) (i + PF + l2d) * 32 * VEC + lane * 32));
        }
    }
    if (acc == 0x12345678u) sink[0] = acc;
}

// Load-instruction probe: VEC=4 (16 B/lane) contiguous warp runs, PF 2, with ld flavour LD:
// 0 ld.global.cs, 1 ld.global (default), 2 ld.global.nc (__ldg), 3 ld.global.L1::no_allocate,
// 4 cp.async.cg 16 B into a per-warp shared ring (4 stages), 5 cp.async.bulk (TMA 1D) 2 KB per
// warp-stage into shared memory, 4 stages, mbarrier completion.
template <int LD>
__global__ __launch_bounds__(256) void p2b_ld_kernel(const uint32_t* __restrict__ buf, size_t words_per_warp, uint32_t* __restrict__ sink)
{
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const size_t gw = (size_t) blockIdx.x * 8 + warp;
    const uint32_t* base = buf + gw * words_per_warp;
    const int steps = (int) (words_per_warp / 128);  // 512 B per warp step
    uint32_t acc = 0;
    if constexpr (LD <= 3) {
        const uint4* vb = reinterpret_cast<const uint4*>(base);
        auto ld = [&](int i) -> uint4 {
            const uint4* a = vb + (size_t) i * 32 + lane;
            uint4 v;
            if constexpr (LD == 0) v = __ldcs(a);
            else if constexpr (LD == 1) v = *a;
            else if constexpr (LD == 2) v = __ldg(a);
            else asm volatile("ld.global.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(a));
            return v;
        };
        uint4 p0 = ld(0), p1 = ld(1);
        for (int i = 0; i < steps; i += 2) {
            acc ^= p0.x ^ p0.y ^ p0.z ^ p0.w;
            if (i + 2 < steps) p0 = ld(i + 2);
            acc ^= p1.x ^ p1.y ^ p1.z ^ p1.w;
            if (i + 3 < steps) p1 = ld(i + 3);
        }
    } else if constexpr (LD == 4) {
        __shared__ uint4 ring[8][4][32];
        auto issue = [&](int i) {
            const uint4* g = reinterpret_cast<const uint4*>(base) + (size_t) i * 32 + lane;
            const unsigned sa = (unsigned) __cvta_generic_to_shared(&ring[warp][i & 3][lane]);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(g));
            asm volatile("cp.async.commit_group;");
        };
        for (int i = 0; i < 3 && i < steps; ++i) issue(i);
        for (int i = 0; i < steps; ++i) {
            if (i + 3 < steps) issue(i + 3); else asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 3;");
            __syncwarp();
            const uint4 v = ring[warp][i & 3][lane];
            acc ^= v.x ^ v.y ^ v.z ^ v.w;
            __syncwarp();
        }
    } else {
        // TMA 1D bulk: lane 0 issues 4 x 512 B per stage? one 2 KB copy per stage (4 steps)
        __shared__ __align__(128) uint4 ring[8][4][128];     // 8 warps x 4 stages x 2 KB
        __shared__ __align__(8) unsigned long long bar[8][4];
        const int stages = steps / 4;
        if (lane == 0)
            for (int s2 = 0; s2 < 4; ++s2) {
                const unsigned ba = (unsigned) __cvta_generic_to_shared(&bar[warp][s2]);
                asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(ba));
            }
        asm volatile("fence.proxy.async.shared::cta;");
        __syncwarp();
        auto issue = [&](int st) {
            if (lane == 0) {
                const unsigned ba = (unsigned) __cvta_generic_to_shared(&bar[warp][st & 3]);
                const unsigned sa = (unsigned) __cvta_generic_to_shared(&ring[warp][st & 3][0]);
                const uint32_t* g = base + (size_t) st * 512;
                asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], 2048;" :: "r"(ba));
                asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 2048, [%2];"
                             :: "r"(sa), "l"(g), "r"(ba) : "memory");
            }
        };
        for (int st = 0; st < 3 && st < stages; ++st) issue(st);
        for (int st = 0; st < stages; ++st) {
            if (st + 3 < stages) issue(st + 3);
            const unsigned ba = (unsigned) __cvta_generic_to_shared(&bar[warp][st & 3]);
            const unsigned ph = (unsigned) ((st >> 2) & 1);
            asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n @!p bra W;\n}" :: "r"(ba), "r"(ph) : "memory");
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint4 v = ring[warp][st & 3][q * 32 + lane];
                acc ^= v.x ^ v.y ^ v.z ^ v.w;
            }
            __syncwarp();
        }
    }
    if (acc == 0x12345678u) sink[0] = acc;
}

static void p2b_ld(const at::Tensor& buf, int64_t blocks, int64_t ldk, at::Tensor& sink)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const size_t warps = (size_t) blocks * 8;
    const size_t wpw = (size_t) buf.numel() / warps / 512 * 512;
    const uint32_t* b = reinterpret_cast<const uint32_t*>(buf.data_ptr<int32_t>());
    uint32_t* s = reinterpret_cast<uint32_t*>(sink.data_ptr<int32_t>());
    switch (ldk) {
    case 0: p2b_ld_kernel<0><<<blocks, 256, 0, stream>>>(b, wpw, s); break;
    case 1: p2b_ld_kernel<1><<<blocks, 256, 0, stream>>>(b, wpw, s); break;
    case 2: p2b_ld_kernel<2><<<blocks, 256, 0, stream>>>(b, wpw, s); break;
    case 3: p2b_ld_kernel<3><<<blocks, 256, 0, stream>>>(b, wpw, s); break;
    case 4: p2b_ld_kernel<4><<<blocks, 256, 0, stream>>>(b, wpw, s); break;
    case 5: p2b_ld_kernel<5><<<blocks, 256, 0, stream>>>(b, wpw, s); break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void p2b_bw(const at::Tensor& buf, int64_t blocks, int64_t pf, int64_t vec, int64_t l2d, at::Tensor& sink)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const size_t warps = (size_t) blocks * 8;
    const size_t wpw = (size_t) buf.numel() / warps / 512 * 512;
    const uint32_t* b = reinterpret_cast<const uint32_t*>(buf.data_ptr<int32_t>());
    uint32_t* s = reinterpret_cast<uint32_t*>(sink.data_ptr<int32_t>());
    #define P2B_BW(P, V) if (pf == P && vec == V) p2b_bw_kernel<P, V><<<blocks, 256, 0, stream>>>(b, wpw, (int) l2d, s)
    P2B_BW(1, 1); P2B_BW(2, 1); P2B_BW(4, 1); P2B_BW(8, 1);
    P2B_BW(1, 4); P2B_BW(2, 4); P2B_BW(4, 4); P2B_BW(8, 4);
    #undef P2B_BW
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void p2b_stream(const at::Tensor& gt, const at::Tensor& ut, const at::Tensor& dt, const at::Tensor& uniq,
                       int64_t hidden, int64_t inter, int64_t mode, at::Tensor& sink, int64_t blocks)
{
    int dev = 0, sms = 0, resident = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, p2b_stream_kernel, 256, 0);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    p2b_stream_kernel<<<blocks > 0 ? (int) blocks : resident * sms, 256, 0, stream>>>(gt.data_ptr<int64_t>(), ut.data_ptr<int64_t>(), dt.data_ptr<int64_t>(),
        uniq.data_ptr<int32_t>(), (int) uniq.numel(), (int) hidden, (int) inter, (int) mode,
        reinterpret_cast<uint32_t*>(sink.data_ptr<int32_t>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Round 4: flat grid-stride read of the same experts' trellis spans (gate, up, down of every unique
// expert, in that order): thread i reads 16-B chunks i, i + stride, ... with 2 in flight,
// ld.global.nc.L1::no_allocate. The best pure-read pattern of the dense-gemv calibration
// (calib2 flat_g48_u2: 236.6 GB/s at 78.6 MB on this GPU).
__device__ __forceinline__ uint4 p2b_ld_na16(const void* p)
{
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}

__global__ void p2b_flat_kernel(const int64_t* __restrict__ gt, const int64_t* __restrict__ ut,
                                const int64_t* __restrict__ dt, const int32_t* __restrict__ uniq, int U,
                                unsigned span16, uint32_t* __restrict__ sink)
{
    const unsigned n16 = (unsigned) U * 3u * span16;
    const unsigned stride = gridDim.x * blockDim.x;
    unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
    auto at = [&](unsigned k) -> const uint4* {
        const unsigned s = k / span16, o = k - s * span16;
        const int src = uniq[s / 3u];
        const int64_t base = (s % 3u == 0u) ? gt[src] : (s % 3u == 1u) ? ut[src] : dt[src];
        return reinterpret_cast<const uint4*>(base) + o;
    };
    uint32_t acc = 0;
    for (; c + stride < n16; c += 2u * stride) {
        const uint4 a = p2b_ld_na16(at(c));
        const uint4 b = p2b_ld_na16(at(c + stride));
        acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w;
    }
    for (; c < n16; c += stride) {
        const uint4 a = p2b_ld_na16(at(c));
        acc ^= a.x ^ a.y ^ a.z ^ a.w;
    }
    if (acc == 0x12345678u) sink[0] = acc;
}

static void p2b_flat(const at::Tensor& gt, const at::Tensor& ut, const at::Tensor& dt, const at::Tensor& uniq,
                     int64_t span_bytes, int64_t blocks, at::Tensor& sink)
{
    TORCH_CHECK(span_bytes % 16 == 0, "span must be a multiple of 16 B");
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    p2b_flat_kernel<<<(int) blocks, 256, 0, stream>>>(gt.data_ptr<int64_t>(), ut.data_ptr<int64_t>(), dt.data_ptr<int64_t>(),
        uniq.data_ptr<int32_t>(), (int) uniq.numel(), (unsigned) (span_bytes / 16),
        reinterpret_cast<uint32_t*>(sink.data_ptr<int32_t>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''

BINDINGS = r'''

// ---------------------------------------------------------------------------
// kernel_study/p2b_coop/make_bench_r3 bindings
// ---------------------------------------------------------------------------
#include <pybind11/pybind11.h>
#include <type_traits>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod)
{
    mod.def("p2b_fused_moe", &p2b_fused_moe_cuda, "p2b fused MoE (chain + coop + round-3 variants)");
    mod.def("set_variant", [](int v) { g_bench_variant = v; });
    mod.def("get_variant", []() { return g_bench_variant; });
    mod.def("last_grid", []() { return g_bench_last_grid; }, "grid size of the last launch");
%%STREAM%%
%s
}
'''

STREAM_BINDING = ('    mod.def("stream", &p2b_stream, "B-stream ceiling kernel (bench only)", pybind11::arg("gt"), pybind11::arg("ut"), pybind11::arg("dt"), pybind11::arg("uniq"), pybind11::arg("hidden"), pybind11::arg("inter"), pybind11::arg("mode"), pybind11::arg("sink"), pybind11::arg("blocks") = 0);\n'
                  '    mod.def("bw", &p2b_bw, "plain read-bandwidth probe (bench only)");\n'
                  '    mod.def("ld", &p2b_ld, "load-instruction bandwidth probe (bench only)");\n'
                  '    mod.def("flat", &p2b_flat, "flat 16-B grid-stride read of the experts\' trellis (bench only)");\n')

TS_BINDING = r'''    mod.def("set_ts", [](uint64_t ptr) {
        unsigned long long* p = reinterpret_cast<unsigned long long*>(ptr);
        cudaMemcpyToSymbol(g_p2b_ts, &p, sizeof(p));
    }, "device buffer [grid][64] u64 for phase stamps (0 disables)");
    mod.def("set_df", [](uint64_t ptr) {
        unsigned long long* p = reinterpret_cast<unsigned long long*>(ptr);
        cudaMemcpyToSymbol(g_p2b_df, &p, sizeof(p));
    }, "device buffer [grid][8 warps][8] u64 for dataflow tile accounting (0 disables)");'''

LAUNCH = "    cuda_check(cudaLaunchCooperativeKernel(kernel, dim3(grid), dim3(256), args, 0, stream));\n"


def _patcher(name: str):
    path = ROOT / "docker/patch" / f"widen_p2b_{name}.py"
    spec = importlib.util.spec_from_file_location(f"widen_p2b_{name}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sub(src: str, old: str, new: str, count: int = 1) -> str:
    if src.count(old) != count:
        raise SystemExit(f"make_bench_r3: expected {count} of {old!r}, found {src.count(old)}")
    return src.replace(old, new)


def chain() -> str:
    src = PIN.read_text()
    for name in CHAIN + R3_PATCHES:
        src = _patcher(name).patch_cu(src)
    return src


def bench(src: str, stamps: bool) -> str:
    for old, new in SWITCHES:
        src = _sub(src, old, new)
    for name in R3_PATCHES:
        for old, new in getattr(_patcher(name), "BENCH_SWITCHES", ()):
            src = _sub(src, old, new)
    src = _sub(src, "namespace cg = cooperative_groups;\n", "namespace cg = cooperative_groups;\n" + GLOBALS)
    # Every launch path records its grid (the launchers keep a local `grid` int).
    src = src.replace(LAUNCH, "    g_bench_last_grid = grid;\n" + LAUNCH)
    if stamps:
        n = src.count("auto grid = cg::this_grid();\n")
        if n < 1:
            raise SystemExit("make_bench_r3: no cooperative kernel found")
        src = src.replace("auto grid = cg::this_grid();\n",
                          "auto grid = cg::this_grid();\n    int _p2b_ph = 1; P2bEndTs _p2b_end; P2B_TS(0);\n")
        src = src.replace("grid.sync();", "P2B_GRID_SYNC();")
        src = _sub(src, "namespace cg = cooperative_groups;\n", "namespace cg = cooperative_groups;\n" + TS_DEFS)
        src = df_stamps(src)
    src = explore(src)
    # the stamped module carries only the p2b kernels
    bindings = (BINDINGS % (TS_BINDING if stamps else "")).replace("%STREAM%\n", "" if stamps else STREAM_BINDING)
    return src + ("" if stamps else STREAM) + bindings


DF_REGIONS = {
    # region start, region end, edits (old, new) applied once inside the region
    "tile": ("__device__ __forceinline__ void p2b_df_tile(", "// L2 prefetch of the first n k-slices", (
        ("    const half2 hzero = __half2half2(__ushort_as_half(0));\n",
         "    const half2 hzero = __half2half2(__ushort_as_half(0));\n    DF_ENTRY();\n"),
        ("                bench_fshift::dq8_regs_2bits_fs<1>(awv, bwv, lane << 3, f0, f1);\n",
         "                bench_fshift::dq8_regs_2bits_fs<1>(awv, bwv, lane << 3, f0, f1);\n                if (t == 0) DF_FILL(i, bwv);\n"),
        ("    // Warp reduction\n", "    DF_LOOPEND();\n    // Warp reduction\n"),
        ("        C[p2b_coop_em(mem[r], m, experts) * (size_t) (ntiles * 16) + group * COLS + c] = __float2half_rn(sum);\n    }\n    __syncthreads();\n}\n",
         "        C[p2b_coop_em(mem[r], m, experts) * (size_t) (ntiles * 16) + group * COLS + c] = __float2half_rn(sum);\n    }\n    __syncthreads();\n    DF_EXIT();\n}\n"),
    )),
    "df": ("void p2b_coop_df_kernel(", "template <int BITS, int CB>\nstatic void launch_moe_batched(", (
        ("    __shared__ int s_df[2];\n", "    __shared__ int s_df[2];\n    DF_INIT();\n"),
        # prologue sub-stamps (slots 56, 57): sort/chunk build done, first-task prefetch issued
        ("    p2b_df_build(ids, m * experts);\n", "    p2b_df_build(ids, m * experts);\n    P2B_TS(56);\n"),
        ("    // Input Hadamard for gate and up (p2b phase 1)\n",
         "    P2B_TS(57);\n    // Input Hadamard for gate and up (p2b phase 1)\n"),
        ("    for (;;) {\n        const int task = *static_cast<volatile int*>(s_df);\n",
         "    DF_SPAN0();\n    for (;;) {\n        const int task = *static_cast<volatile int*>(s_df);\n"),
        ("    // Out of tasks: pull the svh scales", "    DF_SPAN1();\n    DF_DUMP();\n    // Out of tasks: pull the svh scales"),
    )),
}


def df_stamps(src: str) -> str:
    for name, (a_txt, b_txt, edits) in DF_REGIONS.items():
        if a_txt not in src:
            continue
        a = src.index(a_txt)
        b = src.index(b_txt, a)
        region = src[a:b]
        for old, new in edits:
            if region.count(old) != 1:
                raise SystemExit(f"make_bench_r3: region {name}: expected 1 of {old!r}, found {region.count(old)}")
            region = region.replace(old, new)
        src = src[:a] + region + src[b:]
    return src


# ----------------------------------------------------------------------------- round-4 exploration
# Bench-only dataflow variants (ids >= 3): ONE kernel, p2b_df_x_kernel<2, 1>, derived textually from
# the production p2b_df_tile / p2b_df_prefetch_tile / p2b_coop_df_kernel of the (stamped or plain)
# bench source, with every knob path compiled in and the knob passed as an extra kernel argument. All
# exploration arms therefore run identical machine code: arm 3 (knob = production behaviour) against
# v2 measures the binary's own cost, arm k against arm 3 the knob's effect alone. chain_r3.cu (the
# image TU) never carries any of it.
#   tailn  the last tailn x grid 64-column down tiles run as two 32-column tiles each (every column
#          keeps its K-split, fold and cross-warp order: same bits); 255 = every down tile
#   tailp  the last tailp x grid down tasks L2-prefetch their warp's whole K-range at tile start
#   pro    k-slices of the static first task prefetched in the prologue (production 2 + PFL2 = 4)
#   early  1: the warp's first input-Hadamard task issues its pointer loads before the sort/chunk
#          build and its x / suh loads before the first-task prefetch burst (same values, op for op);
#          2: the same for both of the warp's rounds (m*6*40 <= 2 x 1152 warps up to m = 9), and only
#          when m >= 3 (a shorter sort does not hide the pointer loads: arm 10 lost 1.6% at m=1)
X_VARIANTS: dict[int, tuple[int, int, int, int]] = {
    3: (0, 0, 4, 0),
    4: (1, 0, 4, 0),
    5: (2, 0, 4, 0),
    6: (255, 0, 4, 0),
    7: (0, 1, 4, 0),
    8: (0, 0, 8, 0),
    9: (0, 0, 12, 0),
    10: (0, 0, 4, 1),
    11: (0, 0, 8, 1),
    12: (0, 0, 4, 2),
}

X_TILE = ("__device__ __forceinline__ void p2b_df_tile(\n", "// L2 prefetch of the first n k-slices")
X_PF = ("__device__ __forceinline__ void p2b_df_prefetch_tile(", "// Output (row, 128-block hb)")
X_KER = ("template <int BITS, int CB>\n__global__ __launch_bounds__(256, 3)\nvoid p2b_coop_df_kernel(\n",
         "template <int BITS, int CB>\nstatic void launch_moe_batched(")
X_OCC = "    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, 256, 0);\n    const int grid = std::max(1, resident * sms);\n"
X_ARGS = "(void*)&inter, (void*)&swiglu_limit\n    };\n"
# early Hadamard (knob bit 24): pointer loads before the build, x / suh loads right after it (before
# the ctl zeroing and the first-task prefetch burst), the first task's Hadamard from those registers
# with had_hf_r_128_inner<true, false>'s operations (pre-scale __hmul2, the same transform and rounding)
X_EARLY_PTRS = """    const int hmode = (xknob >> 24) & 3;
    const int hrounds = hmode == 1 ? 1 : (hmode == 2 && m >= 3) ? 2 : 0;
    const int hw0 = warp + (int) (blockDim.x / 32) * (int) blockIdx.x;
    const int hgw = (int) gridDim.x * (int) (blockDim.x / 32), hpe = experts * (hidden / 128);
    int64_t hpg[2] = {0, 0}, hpu[2] = {0, 0};
    #pragma unroll
    for (int q = 0; q < 2; ++q)
        if (q < hrounds && hw0 + q * hgw < m * hpe) {
            const int hw = hw0 + q * hgw;
            const int src = ids[(hw / hpe) * experts + (hw % hpe) / (hidden / 128)];
            hpg[q] = gu_ptrs[src];
            hpu[q] = uu_ptrs[src];
        }
"""
X_EARLY_LOADS = """    half4 hx[2], hsg[2], hsu[2];
    #pragma unroll
    for (int q = 0; q < 2; ++q)
        if (q < hrounds && hw0 + q * hgw < m * hpe) {
            const int hw = hw0 + q * hgw;
            const int c = ((hw % hpe) % (hidden / 128)) * 128 + lane * 4;
            hx[q] = *reinterpret_cast<const half4*>(x + (size_t) (hw / hpe) * hidden + c);
            hsg[q] = *reinterpret_cast<const half4*>(reinterpret_cast<const half*>(hpg[q]) + c);
            hsu[q] = *reinterpret_cast<const half4*>(reinterpret_cast<const half*>(hpu[q]) + c);
        }
"""
X_EARLY_FIRST = """        #pragma unroll
        for (int q = 0; q < 2; ++q)
            if (q < hrounds && this_warp < total_warps) {
                const int e = (this_warp % hpe) / warps_per_exp, w = (this_warp % hpe) % warps_per_exp;
                const size_t o = ((size_t) e * m + this_warp / hpe) * hidden + w * 128 + lane * 4;
                *reinterpret_cast<half4*>(had_gate + o) = p2b_df_had(half4(__hmul2(hx[q].x, hsg[q].x), __hmul2(hx[q].y, hsg[q].y)), lane);
                *reinterpret_cast<half4*>(had_up + o) = p2b_df_had(half4(__hmul2(hx[q].x, hsu[q].x), __hmul2(hx[q].y, hsu[q].y)), lane);
                this_warp += grid_warps;
            }
"""


def _cut(src: str, span: tuple[str, str]) -> str:
    a = src.index(span[0])
    return src[a:src.index(span[1], a)]


def explore(src: str) -> str:
    tile = _cut(src, X_TILE)
    ntile = "template <int WNT_>\n" + _sub(tile, "void p2b_df_tile(\n", "void p2b_df_tile_n(\n")
    ntile = _sub(ntile, "    constexpr int WNT = 4;\n", "    constexpr int WNT = WNT_;\n")
    ntile = _sub(ntile, "    constexpr int LOADS = 2;\n", "    constexpr int LOADS = WNT_ / 2;\n")
    pf = _cut(src, X_PF)
    npf = "template <int WNT_>\n" + _sub(pf, "void p2b_df_prefetch_tile(", "void p2b_df_prefetch_tile_n(")
    npf = _sub(npf, "    if (lane < 2) {\n", "    if (lane < WNT_ / 2) {\n")
    npf = _sub(npf, " + group * 64 + lane * 32;\n", " + group * (WNT_ * 16) + lane * 32;\n")
    ker = _cut(src, X_KER)
    ker = _sub(ker, X_KER[0], "template <int BITS, int CB>\n__global__ __launch_bounds__(256, 3)\nvoid p2b_df_x_kernel(\n")
    ker = _sub(ker, "    float swiglu_limit)\n{\n", "    float swiglu_limit,\n    int xknob)\n{\n")
    # the static first task: gate/up only (top-6 always has >= 216 gate/up tasks > the 144-block grid)
    ker = _sub(ker, "        if (task < gu_tasks + units * num_groups_down) {\n", "        if (task < gu_tasks) {\n")
    ker = _sub(ker, "                                 lane, 2 + P2B_DF_PFL2);\n", "                                 lane, (xknob >> 16) & 0xff);\n")
    # task bounds once per block into shared memory (visible after the grid barrier); the loop re-reads
    # them the way the production loop re-derives everything, so nothing extra stays live across a tile
    ker = _sub(ker, "    p2b_df_build(ids, m * experts);\n",
               X_EARLY_PTRS +
               "    p2b_df_build(ids, m * experts);\n"
               "    __shared__ int s_x[3];  // task count, first narrow down task (of the down tasks), first tail-prefetch task\n"
               "    if (threadIdx.x == 0) {\n"
               "        const int dwide = coop[2 * P2B_SORT_CAP] * num_groups_down;\n"
               "        const int nsplit = min(dwide, (xknob & 0xff) * (int) gridDim.x);\n"
               "        s_x[0] = coop[2 * P2B_SORT_CAP] * hb_n * 4 + dwide + nsplit;\n"
               "        s_x[1] = dwide - nsplit;\n"
               "        s_x[2] = ((xknob >> 8) & 0xff) > 0 ? s_x[0] - ((xknob >> 8) & 0xff) * (int) gridDim.x : 0x7fffffff;\n"
               "    }\n" + X_EARLY_LOADS)
    ker = _sub(ker, "        for (; this_warp < total_warps; this_warp += grid_warps) {\n",
               X_EARLY_FIRST + "        for (; this_warp < total_warps; this_warp += grid_warps) {\n")
    ker = _sub(ker, "        if (task >= gu_tasks + coop[2 * P2B_SORT_CAP] * num_groups_down)\n            break;\n",
               "        if (task >= *static_cast<volatile int*>(s_x))\n            break;\n")
    ker = _sub(ker, """        } else {
            const int u = (task - gu_tasks) / num_groups_down;
            if (threadIdx.x == 0)
                while (p2b_df_ld_acquire(ctl + P2B_DF_READY + u) < hb_n)
                    __nanosleep(64);
            __syncthreads();
            const int* mem = p2b_df_sort_smem() + coop[u];
            p2b_df_tile(reinterpret_cast<const uint32_t*>(dt_ptrs[ids[mem[0]]]), had_down, down, kslices_down, inter,
                        task - gu_tasks - u * num_groups_down, ntiles_down, warp, lane, p2b_df_red(), mem,
                        coop[P2B_SORT_CAP + u], m, experts);
        }
""", """        } else {
            // down task dtk: wide tiles [0, nfirst), then two narrow halves per remaining wide tile
            const int dtk = task - gu_tasks;
            const int nfirst = *static_cast<volatile int*>(s_x + 1);
            const int nrel = dtk - nfirst;
            const bool narrow = nrel >= 0;
            const int wtk = narrow ? nfirst + (nrel >> 1) : dtk;
            const int u = wtk / num_groups_down;
            const int grp = wtk - u * num_groups_down;
            if (threadIdx.x == 0)
                while (p2b_df_ld_acquire(ctl + P2B_DF_READY + u) < hb_n)
                    __nanosleep(64);
            __syncthreads();
            const int* mem = p2b_df_sort_smem() + coop[u];
            const uint32_t* bdn = reinterpret_cast<const uint32_t*>(dt_ptrs[ids[mem[0]]]);
            const bool tailp = task >= *static_cast<volatile int*>(s_x + 2);
            if (narrow) {
                if (tailp)
                    p2b_df_prefetch_tile_n<2>(bdn, kslices_down, ntiles_down, grp * 2 + (nrel & 1), warp, lane, 64);
                p2b_df_tile_n<2>(bdn, had_down, down, kslices_down, inter, grp * 2 + (nrel & 1), ntiles_down, warp,
                                 lane, p2b_df_red(), mem, coop[P2B_SORT_CAP + u], m, experts);
            } else {
                if (tailp)
                    p2b_df_prefetch_tile(bdn, kslices_down, ntiles_down, grp, warp, lane, 64);
                p2b_df_tile(bdn, had_down, down, kslices_down, inter, grp, ntiles_down, warp, lane, p2b_df_red(), mem,
                            coop[P2B_SORT_CAP + u], m, experts);
            }
        }
""")
    knobs = "".join(f"    case {v}: return {a | (b << 8) | (c << 16) | (d << 24)};\n" for v, (a, b, c, d) in sorted(X_VARIANTS.items()))
    pick = ("static int p2b_df_x_knob(int v)\n{\n    switch (v) {\n" + knobs
            + "    }\n    TORCH_CHECK(false, \"make_bench_r3: unknown dataflow bench variant \", v);\n    return 0;\n}\n\n")
    src = _sub(src, X_KER[1], "// --- make_bench_r3 round-4 exploration (bench only) ---\n" + ntile + npf + ker + pick + X_KER[1])
    src = _sub(src, X_OCC, "    int xknob = 0;\n    if constexpr (BITS == 2 && CB == 1) {\n"
               "        if (g_bench_variant >= 3 && m * e <= P2B_SORT_CAP && p2b_df_ctl_words(inter) <= m * hidden) {\n"
               "            kernel = (void*) p2b_df_x_kernel<BITS, CB>;\n"
               "            xknob = p2b_df_x_knob(g_bench_variant);\n        }\n    }\n" + X_OCC)
    # kernels without the extra parameter ignore the trailing pointer (the driver reads the kernel's own layout)
    src = _sub(src, X_ARGS, "(void*)&inter, (void*)&swiglu_limit, (void*)&xknob\n    };\n")
    # the host path of every dataflow variant: control words zeroed by the kernel, no accum memset
    src = _sub(src, "const bool df = mcg && kg == 2 && p2b_coop_mode() == 2 &&", "const bool df = mcg && kg == 2 && p2b_coop_mode() >= 2 &&")
    src = _sub(src, "    if (p2b_coop_mode() == 2 && !df) {\n", "    if (p2b_coop_mode() >= 2 && !df) {\n")
    return src


def build() -> dict[str, str]:
    src = chain()
    return {"chain_r3.cu": src, "bench_r3.cu": bench(src, False), "bench_r3_ts.cu": bench(src, True)}


def main() -> int:
    OUT.mkdir(exist_ok=True)
    for name, text in build().items():
        path = OUT / name
        if path.exists() and path.read_text() == text:
            print(f"unchanged {path}")
            continue
        path.write_text(text)
        print(f"wrote {path} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
