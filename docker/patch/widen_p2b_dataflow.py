#!/usr/bin/env python3
"""Dataflow coop p2b (DSV41_P2B_COOP=2): the coop tiles as one task list, 2 grid barriers.

The round-2 coop kernel (widen_p2b_coop.py, DSV41_P2B_COOP=1, SORT=2) streams each unique
expert once but keeps p2b's 7 phase barriers. Measured on GB10 (round 3, real layer weights,
census routing, results/2026-09-25-kernels/coop-moe/iterations.txt): at m=4 the gate/up phase
waits 20 us and the down phase 12 us for their slowest block, and the elementwise phases plus
barriers cost another ~20 us of a ~340 us call.

DSV41_P2B_COOP=2 launches p2b_coop_df_kernel<2, 1> instead (K=2 MCG only), 3 blocks per SM
(80 registers, no spills in the MMA loops):
- Prologue: every block sorts the (row, expert) pairs and cuts the coop chunks (as SORT=2),
  L2-prefetches the first k-slices of its first tile (static: task blockIdx.x) and runs its share
  of the input Hadamard. One grid.sync.
- Tasks: gate/up tiles (unit, 128-block, 64-col half, gate|up), then down tiles (unit, 64-col
  group), handed out by one atomic counter. After a gate/up tile, thread 0 counts it with an
  acq_rel RMW (the tile's closing __syncthreads orders the block's stores before it; release is
  cumulative). The 4th tile of (unit, 128-block) runs that block's epilogue for every row of the
  unit: output Hadamard of gate and up, SwiGLU, down input Hadamard (p2b's three elementwise
  phases, operation for operation), then counts the unit's 128-blocks with a release. A down
  tile waits until its unit has all of them (acquire). All gate/up tiles are dispatched before
  any down tile and never wait, so every wait ends.
- A block out of tasks L2-prefetches the svh scales of its output rows. One more grid.sync, then
  the down output Hadamard and the fixed-order weighted slot sum per (row, 128-block).
- The tile is run_gemv_tile_coop's (same B stream, decode, MMA, fold and cross-warp order) plus
  an L2 prefetch 2 k-slices beyond the register ring and the A fragment loaded one k-slice
  ahead; neither changes a value.
So the output is bit-identical to DSV41_P2B_COOP=1 for any routing weights and to p2b for
one-hot weights; deterministic, no atomics on data. Same launch (one cooperative launch), grid
rule (occupancy x SMs), scratch and host checks as p2b; the control words live in the accum
scratch (unused here; zeroed by block 0 before the first barrier, so CUDA-graph replays need no
host reset) and the host skips accum's memset in this mode.

LOG_ENGAGED is printed once per process by the first dataflow launch; LOG_DISARMED once when
DSV41_P2B_COOP=2 meets a call it cannot take (K != 2, cb != 1 or m * K > 64), which then runs
the SORT=0 kernel. Off (unset, "0", "1"): the SORT=0/1/2 kernels, whose code this patch does not
touch. Apply after widen_p2b_coop.py. Idempotent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MARK = "// --- widen_p2b_dataflow"
LOG_ENGAGED = "dsv41: p2b coop dataflow kernel engaged (DSV41_P2B_COOP=2)"
LOG_DISARMED = "dsv41: p2b coop dataflow lever is OFF for this call"

KERNEL = r'''
// --- widen_p2b_dataflow: coop tiles as one dataflow task list (DSV41_P2B_COOP=2) ---
static int p2b_coop_mode()
{
    static const int mode = [] {
        const char* v = std::getenv("DSV41_P2B_COOP");
        return v == nullptr ? 0 : std::atoi(v);
    }();
    return mode;
}

// Control words in accum: [0] task counter, [P2B_DF_READY + u] had_down 128-blocks done for
// unit u, [P2B_DF_ACT + u * (inter / 128) + hb] gate/up tiles done for (unit u, 128-block hb).
constexpr int P2B_DF_READY = 32;
constexpr int P2B_DF_ACT = P2B_DF_READY + P2B_SORT_CAP;
constexpr int P2B_DF_PFL2 = 2;  // L2 prefetch distance beyond the 2-slot register ring

__host__ __device__ __forceinline__ int p2b_df_ctl_words(int inter)
{
    return P2B_DF_ACT + P2B_SORT_CAP * (inter / 128);
}

__device__ __forceinline__ int p2b_df_ld_acquire(const int* p)
{
    int v;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

// Counter RMW with acquire+release semantics at GPU scope: releases the block's stores ordered
// before it by a __syncthreads, returns the old value for last-arriver detection.
__device__ __forceinline__ int p2b_df_atom_add_acq_rel(int* p, int v)
{
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}

__device__ __forceinline__ void p2b_df_red_release(int* p, int v)
{
    asm volatile("red.release.gpu.global.add.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void p2b_df_prefetch_l2(const void* p)
{
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}

// Own shared arrays (same shapes as p2b_sort_smem / p2b_coop_smem / p2b_coop_red), so the SORT=2
// kernel's shared-memory layout, and with it its machine code, stays exactly widen_p2b_coop's.
__device__ __forceinline__ int* p2b_df_sort_smem()
{
    __shared__ int s[3 * P2B_SORT_CAP];
    return s;
}

__device__ __forceinline__ int* p2b_df_chunk_smem()
{
    __shared__ int s[2 * P2B_SORT_CAP + 1];
    return s;
}

__device__ __forceinline__ P2bCoopRed* p2b_df_red()
{
    __shared__ P2bCoopRed s[8];
    return s;
}

// p2b_sort_build + p2b_coop_build on the arrays above: [0, CAP) sorted slot -> pair; chunks of
// <= P2B_COOP_ROWS members of equal src: [0, CAP) start slot, [CAP, 2CAP) length, [2CAP] count.
__device__ __forceinline__ void p2b_df_build(const int32_t* __restrict__ ids, int pairs)
{
    int* perm = p2b_df_sort_smem();
    int* run_start = perm + P2B_SORT_CAP;
    int* run_len = run_start + P2B_SORT_CAP;
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) {
        const int v = ids[p];
        int rank = 0;
        for (int q = 0; q < pairs; ++q) {
            const int w = ids[q];
            rank += (w < v) || (w == v && q < p);
        }
        perm[rank] = p;
    }
    __syncthreads();
    for (int s = threadIdx.x; s < pairs; s += blockDim.x) {
        const int v = ids[perm[s]];
        int a = s, b = s + 1;
        while (a > 0 && ids[perm[a - 1]] == v) --a;
        while (b < pairs && ids[perm[b]] == v) ++b;
        run_start[s] = a;
        run_len[s] = b - a;
    }
    __syncthreads();
    int* chunk = p2b_df_chunk_smem();
    if (threadIdx.x == 0) {
        int u = 0;
        for (int s = 0; s < pairs; ++s) {
            const int off = s - run_start[s];
            if (off % P2B_COOP_ROWS == 0) {
                chunk[u] = s;
                chunk[P2B_SORT_CAP + u] = min(P2B_COOP_ROWS, run_len[s] - off);
                ++u;
            }
        }
        chunk[2 * P2B_SORT_CAP] = u;
    }
    __syncthreads();
}

// had_hf_r_128_inner's transform of the 4 values this lane holds (same float ops and roundings).
__device__ __forceinline__ half4 p2b_df_had(half4 v, int lane)
{
    float v0 = __half2float(__low2half(v.x));
    float v1 = __half2float(__high2half(v.x));
    float v2 = __half2float(__low2half(v.y));
    float v3 = __half2float(__high2half(v.y));
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    float h0 = s0 + s1;
    float h1 = d0 + d1;
    float h2 = s0 - s1;
    float h3 = d0 - d1;
    shuffle_had_f4x32(h0, h1, h2, h3, lane);
    const float r = 0.088388347648f;
    return half4(__floats2half2_rn(h0 * r, h1 * r), __floats2half2_rn(h2 * r, h3 * r));
}

// L2 load of 4 halves another block wrote in this launch.
__device__ __forceinline__ half4 p2b_df_ldcg4(const half* p)
{
    const uint2 u = __ldcg(reinterpret_cast<const uint2*>(p));
    half4 h;
    *reinterpret_cast<uint2*>(&h) = u;
    return h;
}

// The Hadamard pre/post scale: half multiply by 4 scales.
__device__ __forceinline__ half4 p2b_df_scale(half4 v, const half* s)
{
    const half4 sc = *reinterpret_cast<const half4*>(s);
    return half4(__hmul2(v.x, sc.x), __hmul2(v.y, sc.y));
}

__device__ __forceinline__ half p2b_df_swiglu(half gh, half uh, float limit)
{
    // p2b phase 3, element for element.
    float g = __half2float(gh);
    float u = __half2float(uh);
    if (limit > 0.0f) {
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);
    }
    float s = g / (1.0f + expf(-g));
    return __float2half(s * u);
}

// Epilogue of one 128-block hb of a finished gate/up unit, for one chunk member:
// p2b's epilogue Hadamard (gate, up), SwiGLU and down input Hadamard, operation for operation.
__device__ __forceinline__ void p2b_df_act(const half* __restrict__ gate, const half* __restrict__ up,
                                           half* __restrict__ had_down, const half* __restrict__ gv_e,
                                           const half* __restrict__ uv_e, const half* __restrict__ du_e,
                                           size_t em, int inter, int hb, int lane, float swiglu_limit)
{
    const int c = hb * 128 + lane * 4;
    const half4 g = p2b_df_scale(p2b_df_had(p2b_df_ldcg4(gate + em * inter + c), lane), gv_e + c);
    const half4 v = p2b_df_scale(p2b_df_had(p2b_df_ldcg4(up + em * inter + c), lane), uv_e + c);
    const half4 a(p2b_df_swiglu(__low2half(g.x), __low2half(v.x), swiglu_limit),
                  p2b_df_swiglu(__high2half(g.x), __high2half(v.x), swiglu_limit),
                  p2b_df_swiglu(__low2half(g.y), __low2half(v.y), swiglu_limit),
                  p2b_df_swiglu(__high2half(g.y), __high2half(v.y), swiglu_limit));
    *reinterpret_cast<half4*>(had_down + em * inter + c) = p2b_df_had(p2b_df_scale(a, du_e + c), lane);
}

// run_gemv_tile_coop for K=2 MCG, CFG 1 (8 k-split warps x 4 n-tiles, register ring 2, fold 2):
// same loads, decode, MMA, fold and cross-warp reduction order, so the same bits. Also
// prefetches k-slice i + 2 + P2B_DF_PFL2 into L2 at k-slice i and loads the A fragment one
// k-slice ahead.
__device__ __forceinline__ void p2b_df_tile(
    const uint32_t* __restrict__ B32,
    const half* __restrict__ A,
    half* __restrict__ C,
    int kslices,
    int size_k,
    int group,
    int ntiles,
    int warp,
    int lane,
    P2bCoopRed* sh_red,
    const int* __restrict__ mem,
    int len,
    int m,
    int experts)
{
    constexpr int WK = 8;
    constexpr int WNT = 4;
    constexpr int PF = 2;
    constexpr int FOLD = 2;
    constexpr int THREADS = WK * 32;
    constexpr int COLS = WNT * 16;
    constexpr int TWORDS = 16;
    constexpr int LOADS = 2;
    constexpr int LSTRIDE = 32;

    const int chunk = CEIL_DIVIDE(kslices, WK);
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));
    const size_t slice_stride = (size_t) ntiles * TWORDS;

    const int r0 = lane >> 2;
    const bool r0_ok = r0 < len;
    const half2* A2 = reinterpret_cast<const half2*>(A + p2b_coop_em(mem[r0_ok ? r0 : 0], m, experts) * size_k);
    const half2 hzero = __half2half2(__ushort_as_half(0));

    const int x_src_b = lane >> 1;
    const int x_src_a = (x_src_b + 15) & 15;

    const uint32_t* bp = B32 + (size_t) ks0 * slice_stride + group * WNT * TWORDS + lane;
    const uint32_t* lp = B32 + (size_t) ks0 * slice_stride + group * WNT * TWORDS + lane * LSTRIDE;

    uint32_t pf[PF][LOADS];
    #pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < myn)
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                pf[d][l] = __ldcs(bp + (size_t) d * slice_stride + l * LSTRIDE);
    if (lane < LOADS)
        for (int d = PF; d < min(PF + P2B_DF_PFL2, myn); ++d)
            p2b_df_prefetch_l2(lp + (size_t) d * slice_stride);

    half2 an0 = hzero, an2 = hzero;
    if (myn > 0) {
        const size_t c0 = (size_t) ks0 * 8 + (lane & 3);
        an0 = A2[c0];
        an2 = A2[c0 + 4];
    }

    FragC_h ch[WNT][2] = {};
    float2 acc0[WNT][2] = {};

    for (int ib = 0; ib < myn; ib += PF) {
        #pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int i = ib + d;
            if (i >= myn) break;

            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                bw[l] = pf[d][l];

            if (i + PF < myn) {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = __ldcs(bp + (size_t) (i + PF) * slice_stride + l * LSTRIDE);
            }
            if (lane < LOADS && i + PF + P2B_DF_PFL2 < myn)
                p2b_df_prefetch_l2(lp + (size_t) (i + PF + P2B_DF_PFL2) * slice_stride);

            const half2 a0v = an0;
            const half2 a2v = an2;
            if (i + 1 < myn) {
                const size_t cn = (size_t) (ks0 + i + 1) * 8 + (lane & 3);
                an0 = A2[cn];
                an2 = A2[cn + 4];
            }
            FragB a01, a23;
            a01[0] = r0_ok ? a0v : hzero;
            a23[0] = r0_ok ? a2v : hzero;
            a01[1] = hzero;
            a23[1] = hzero;

            #pragma unroll
            for (int t = 0; t < WNT; ++t) {
                FragB f0, f1;
                const uint32_t w = bw[t >> 1];
                const int base = (t & 1) << 4;
                uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                bench_fshift::dq8_regs_2bits_fs<1>(awv, bwv, lane << 3, f0, f1);
                exl3_gemv_ns::mma_ab_h(a01, a23, f0, ch[t][0]);
                exl3_gemv_ns::mma_ab_h(a01, a23, f1, ch[t][1]);
            }

            if ((d + 1) % FOLD == 0 || i + 1 == myn) {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        acc0[t][f].x += __low2float(ch[t][f][0]);
                        acc0[t][f].y += __high2float(ch[t][f][0]);
                        ch[t][f][0] = hzero;
                    }
            }
        }
    }

    // Warp reduction
    if (r0_ok) {
        #pragma unroll
        for (int t = 0; t < WNT; ++t) {
            #pragma unroll
            for (int f = 0; f < 2; ++f) {
                const int col = t * 16 + f * 8 + (lane & 3) * 2;
                sh_red[warp][r0][col + 0] = acc0[t][f].x;
                sh_red[warp][r0][col + 1] = acc0[t][f].y;
            }
        }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < COLS * len; idx += THREADS) {
        const int r = idx / COLS;
        const int c = idx % COLS;
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < WK; ++j)
            sum += sh_red[j][r][c];
        C[p2b_coop_em(mem[r], m, experts) * (size_t) (ntiles * 16) + group * COLS + c] = __float2half_rn(sum);
    }
    __syncthreads();
}

// L2 prefetch of the first n k-slices of warp's K-range of a coop tile (2 lines per slice).
__device__ __forceinline__ void p2b_df_prefetch_tile(const uint32_t* __restrict__ B32, int kslices, int ntiles,
                                                     int group, int warp, int lane, int n)
{
    const int chunk = CEIL_DIVIDE(kslices, 8);
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));
    if (lane < 2) {
        const size_t stride = (size_t) ntiles * 16;
        const uint32_t* lp = B32 + (size_t) ks0 * stride + group * 64 + lane * 32;
        for (int d = 0; d < min(n, myn); ++d)
            p2b_df_prefetch_l2(lp + (size_t) d * stride);
    }
}

// Output (row, 128-block hb): down output Hadamard of every slot and the fixed-order weighted
// slot sum (the round-2 coop sum). One warp.
__device__ __forceinline__ void p2b_df_out(const half* __restrict__ down, const int64_t* __restrict__ dv_ptrs,
                                           const int32_t* __restrict__ ids, const half* __restrict__ rw,
                                           half* __restrict__ out, int row, int hb, int experts, int m, int hidden,
                                           int lane)
{
    const int c = hb * 128 + lane * 4;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (int e = 0; e < experts; ++e) {
        const half* dv_e = reinterpret_cast<const half*>(dv_ptrs[ids[row * experts + e]]);
        const half4 d = p2b_df_scale(p2b_df_had(p2b_df_ldcg4(down + ((size_t) e * m + row) * hidden + c), lane), dv_e + c);
        const float w = __half2float(rw[row * experts + e]);
        s0 = __fadd_rn(s0, __fmul_rn(w, __half2float(__low2half(d.x))));
        s1 = __fadd_rn(s1, __fmul_rn(w, __half2float(__high2half(d.x))));
        s2 = __fadd_rn(s2, __fmul_rn(w, __half2float(__low2half(d.y))));
        s3 = __fadd_rn(s3, __fmul_rn(w, __half2float(__high2half(d.y))));
    }
    *reinterpret_cast<half4*>(out + (size_t) row * hidden + c) =
        half4(__float2half(s0), __float2half(s1), __float2half(s2), __float2half(s3));
}

// p2b_df_out for top-6: all loads first, the 6 Hadamards independent, the same sum order.
__device__ __forceinline__ void p2b_df_out6(const half* __restrict__ down, const int64_t* __restrict__ dv_ptrs,
                                            const int32_t* __restrict__ ids, const half* __restrict__ rw,
                                            half* __restrict__ out, int row, int hb, int m, int hidden, int lane)
{
    constexpr int E = 6;
    const int c = hb * 128 + lane * 4;
    half4 dv[E];
    half4 sc[E];
    float wt[E];
    #pragma unroll
    for (int e = 0; e < E; ++e) {
        dv[e] = p2b_df_ldcg4(down + ((size_t) e * m + row) * hidden + c);
        sc[e] = *reinterpret_cast<const half4*>(reinterpret_cast<const half*>(dv_ptrs[ids[row * E + e]]) + c);
        wt[e] = __half2float(rw[row * E + e]);
    }
    half4 h[E];
    #pragma unroll
    for (int e = 0; e < E; ++e)
        h[e] = p2b_df_had(dv[e], lane);
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    #pragma unroll
    for (int e = 0; e < E; ++e) {
        const half4 d(__hmul2(h[e].x, sc[e].x), __hmul2(h[e].y, sc[e].y));
        s0 = __fadd_rn(s0, __fmul_rn(wt[e], __half2float(__low2half(d.x))));
        s1 = __fadd_rn(s1, __fmul_rn(wt[e], __half2float(__high2half(d.x))));
        s2 = __fadd_rn(s2, __fmul_rn(wt[e], __half2float(__low2half(d.y))));
        s3 = __fadd_rn(s3, __fmul_rn(wt[e], __half2float(__high2half(d.y))));
    }
    *reinterpret_cast<half4*>(out + (size_t) row * hidden + c) =
        half4(__float2half(s0), __float2half(s1), __float2half(s2), __float2half(s3));
}

template <int BITS, int CB>
__global__ __launch_bounds__(256, 3)
void p2b_coop_df_kernel(
    const half* __restrict__ x,
    const int64_t* __restrict__ gt_ptrs,
    const int64_t* __restrict__ gu_ptrs,
    const int64_t* __restrict__ gv_ptrs,
    const int64_t* __restrict__ ut_ptrs,
    const int64_t* __restrict__ uu_ptrs,
    const int64_t* __restrict__ uv_ptrs,
    const int64_t* __restrict__ dt_ptrs,
    const int64_t* __restrict__ du_ptrs,
    const int64_t* __restrict__ dv_ptrs,
    const int32_t* __restrict__ ids,
    const half* __restrict__ rw,
    half* __restrict__ gate,
    half* __restrict__ up,
    half* __restrict__ down,
    half* __restrict__ out,
    half* __restrict__ had_gate,
    half* __restrict__ had_up,
    half* __restrict__ had_down,
    float* __restrict__ accum,
    int experts,
    int m,
    int hidden,
    int inter,
    float swiglu_limit)
{
    auto grid = cg::this_grid();
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int ntiles_gate = inter / 16;
    const int kslices_gate = hidden / 16;
    const int ntiles_down = hidden / 16;
    const int kslices_down = inter / 16;
    const int num_groups_down = hidden / 64;
    const int hb_n = inter / 128;
    const int hb_out = hidden / 128;
    int* ctl = reinterpret_cast<int*>(accum);
    const int* coop = p2b_df_chunk_smem();
    // [0] current task, [1] last-finisher flag
    __shared__ int s_df[2];

    p2b_df_build(ids, m * experts);
    if (blockIdx.x == 0)
        for (int j = threadIdx.x; j < p2b_df_ctl_words(inter); j += blockDim.x)
            ctl[j] = 0;

    // The block's first task is static (blockIdx.x; the counter hands out gridDim.x + n): its
    // first k-slices stream into L2 while the input Hadamard and the grid barrier run.
    if (threadIdx.x == 0)
        s_df[0] = blockIdx.x;
    {
        const int units = coop[2 * P2B_SORT_CAP];
        const int gu_tasks = units * hb_n * 4;
        const int task = blockIdx.x;
        if (task < gu_tasks + units * num_groups_down) {
            const bool gu = task < gu_tasks;
            const int u = gu ? task / (hb_n * 4) : (task - gu_tasks) / num_groups_down;
            const int r = task - u * (hb_n * 4);
            const int src = ids[p2b_df_sort_smem()[coop[u]]];
            const int64_t base = gu ? ((r & 1) ? ut_ptrs[src] : gt_ptrs[src]) : dt_ptrs[src];
            p2b_df_prefetch_tile(reinterpret_cast<const uint32_t*>(base), gu ? kslices_gate : kslices_down,
                                 gu ? ntiles_gate : ntiles_down,
                                 gu ? (r >> 2) * 2 + ((r >> 1) & 1) : task - gu_tasks - u * num_groups_down, warp,
                                 lane, 2 + P2B_DF_PFL2);
        }
    }

    // Input Hadamard for gate and up (p2b phase 1)
    {
        int warps_per_exp = hidden / 128;
        int total_warps = m * experts * warps_per_exp;
        int this_warp = warp + (blockDim.x / 32) * blockIdx.x;
        int grid_warps = gridDim.x * (blockDim.x / 32);

        for (; this_warp < total_warps; this_warp += grid_warps) {
            int row = this_warp / (experts * warps_per_exp);
            int rem = this_warp % (experts * warps_per_exp);
            int e = rem / warps_per_exp;
            int w = rem % warps_per_exp;
            int src = ids[row * experts + e];
            const half* x_row = x + (size_t) row * hidden;
            const half* gu_e = reinterpret_cast<const half*>(gu_ptrs[src]);
            const half* uu_e = reinterpret_cast<const half*>(uu_ptrs[src]);
            half* hg_e = had_gate + ((size_t) e * m + row) * hidden;
            half* hu_e = had_up + ((size_t) e * m + row) * hidden;

            had_hf_r_128_inner<true, false>(x_row + w * 128, hg_e + w * 128, gu_e + (w * 128) % hidden, 0.088388347648f);
            had_hf_r_128_inner<true, false>(x_row + w * 128, hu_e + w * 128, uu_e + (w * 128) % hidden, 0.088388347648f);
        }
        grid.sync();
    }

    // Task loop. Only the task index (shared) and thread 0's claimed index live across a tile;
    // everything else is re-derived from shared memory so the MMA loop keeps its registers.
    for (;;) {
        const int task = *static_cast<volatile int*>(s_df);
        const int gu_tasks = coop[2 * P2B_SORT_CAP] * hb_n * 4;
        if (task >= gu_tasks + coop[2 * P2B_SORT_CAP] * num_groups_down)
            break;
        int nxt = 0;
        if (threadIdx.x == 0)
            nxt = atomicAdd(ctl, 1) + (int) gridDim.x;
        if (task < gu_tasks) {
            {
                const int u = task / (hb_n * 4);
                const int r = task - u * (hb_n * 4);
                const int is_up = r & 1;
                const int* mem = p2b_df_sort_smem() + coop[u];
                const int src = ids[mem[0]];
                p2b_df_tile(reinterpret_cast<const uint32_t*>(is_up ? ut_ptrs[src] : gt_ptrs[src]),
                            is_up ? had_up : had_gate, is_up ? up : gate, kslices_gate, hidden,
                            (r >> 2) * 2 + ((r >> 1) & 1), ntiles_gate, warp, lane, p2b_df_red(), mem,
                            coop[P2B_SORT_CAP + u], m, experts);
            }
            // Count the tile (the tile's closing __syncthreads orders the block's stores before
            // thread 0's acq_rel RMW); the 4th tile of (u, hb) runs that 128-block's epilogue.
            const int t2 = *static_cast<volatile int*>(s_df);
            const int u = t2 / (hb_n * 4);
            const int hb = (t2 - u * (hb_n * 4)) >> 2;
            if (threadIdx.x == 0)
                s_df[1] = p2b_df_atom_add_acq_rel(ctl + P2B_DF_ACT + u * hb_n + hb, 1) == 3;
            __syncthreads();
            if (s_df[1]) {
                const int* mem = p2b_df_sort_smem() + coop[u];
                if (warp < coop[P2B_SORT_CAP + u]) {
                    const int src = ids[mem[0]];
                    p2b_df_act(gate, up, had_down, reinterpret_cast<const half*>(gv_ptrs[src]),
                               reinterpret_cast<const half*>(uv_ptrs[src]), reinterpret_cast<const half*>(du_ptrs[src]),
                               p2b_coop_em(mem[warp], m, experts), inter, hb, lane, swiglu_limit);
                }
                __syncthreads();
                if (threadIdx.x == 0)
                    p2b_df_red_release(ctl + P2B_DF_READY + u, 1);
            }
        } else {
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
        if (threadIdx.x == 0)
            s_df[0] = nxt;
        __syncthreads();
    }

    // Out of tasks: pull the svh scales of this block's output rows into L2 while the other
    // blocks finish their last tiles.
    for (int t = warp + (blockDim.x / 32) * blockIdx.x; t < m * hb_out; t += gridDim.x * (blockDim.x / 32))
        if (lane < 2 * experts)
            p2b_df_prefetch_l2(reinterpret_cast<const half*>(dv_ptrs[ids[(t / hb_out) * experts + lane / 2]])
                               + (t % hb_out) * 128 + (lane & 1) * 64);
    grid.sync();

    // Down output Hadamard and the fixed-order weighted slot sum, per (row, 128-block).
    for (int t = warp + (blockDim.x / 32) * blockIdx.x; t < m * hb_out; t += gridDim.x * (blockDim.x / 32)) {
        if (experts == 6)
            p2b_df_out6(down, dv_ptrs, ids, rw, out, t / hb_out, t % hb_out, m, hidden, lane);
        else
            p2b_df_out(down, dv_ptrs, ids, rw, out, t / hb_out, t % hb_out, experts, m, hidden, lane);
    }
}
'''

KERNEL_ANCHOR = "template <int BITS, int CB>\nstatic void launch_moe_batched("

LAUNCH_OLD = """        if (p2b_coop_enabled() && m * e <= P2B_SORT_CAP)
            kernel = (void*) p2b_moe_batched_kernel<BITS, CB, 2>;
"""
LAUNCH_NEW = LAUNCH_OLD + """        if (p2b_coop_mode() == 2 && m * e <= P2B_SORT_CAP && p2b_df_ctl_words(inter) <= m * hidden) {
            kernel = (void*) p2b_coop_df_kernel<BITS, CB>;
            static bool logged = false;
            if (!logged) {
                logged = true;
                std::fprintf(stderr, "%s\\n", "dsv41: p2b coop dataflow kernel engaged (DSV41_P2B_COOP=2)");
            }
        }
"""

ACCUM_OLD = "    auto accum = at::zeros({m, hidden}, x.options().dtype(at::kFloat));\n"
ACCUM_NEW = """    // widen_p2b_dataflow: the dataflow kernel zeroes its control words itself.
    const bool df = mcg && kg == 2 && p2b_coop_mode() == 2 && m * e <= P2B_SORT_CAP && p2b_df_ctl_words(inter) <= m * hidden;
    if (p2b_coop_mode() == 2 && !df) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::fprintf(stderr, "%s (K=%d mcg=%d m=%d top_k=%d)\\n", "dsv41: p2b coop dataflow lever is OFF for this call",
                         (int) kg, (int) mcg, m, e);
        }
    }
    auto accum = df ? at::empty({m, hidden}, x.options().dtype(at::kFloat))
                    : at::zeros({m, hidden}, x.options().dtype(at::kFloat));
"""

INCLUDE_OLD = "#include <cstring>\n"
INCLUDE_NEW = "#include <cstring>\n#include <cstdio>\n"

# kernel_study/p2b_coop/make_bench_r3.py: the mode comes from the bench's runtime variant.
BENCH_SWITCHES = (
    ("static int p2b_coop_mode()\n{\n", "static int p2b_coop_mode()\n{\n    return g_bench_variant;\n"),
)


def _sub1(src: str, old: str, new: str, label: str) -> str:
    if src.count(old) != 1:
        raise SystemExit(f"widen_p2b_dataflow: {label}: expected 1 match, found {src.count(old)}")
    return src.replace(old, new, 1)


def patch_cu(src: str) -> str:
    if MARK in src:
        return src
    if "// --- widen_p2b_coop" not in src:
        raise SystemExit("widen_p2b_dataflow: apply widen_p2b_coop.py first")
    out = _sub1(src, KERNEL_ANCHOR, KERNEL.lstrip("\n") + "\n" + KERNEL_ANCHOR, "launch_moe_batched anchor")
    out = _sub1(out, LAUNCH_OLD, LAUNCH_NEW, "coop launch selection")
    out = _sub1(out, ACCUM_OLD, ACCUM_NEW, "accum allocation")
    out = _sub1(out, INCLUDE_OLD, INCLUDE_NEW, "includes")
    if out.count("grid.sync()") != src.count("grid.sync()") + 2:
        raise SystemExit("widen_p2b_dataflow: expected exactly the dataflow kernel's 2 grid barriers")
    return out


def apply(root: Path) -> None:
    cu = root / "csrc" / "p2b_moe.cu"
    cu.write_text(patch_cu(cu.read_text()))
    print(f"patched {cu}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    apply(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
