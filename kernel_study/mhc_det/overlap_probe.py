#!/usr/bin/env python3
"""Overlap probe: hide det mHC work under the TP all-reduce that precedes each post (serve image,
spark2, one GPU).

In the serve every mHC post follows a TP all-reduce (NCCL RING_LL over RoCE, ~22.8 us median in
the r3 profile, 80 per step; DRAM nearly idle while it runs). Two ways to use that window:
  pf   fork a side stream right before the AR and L2-prefetch the next sublayer's packed fn
       (1 CTA, cp.async.bulk.prefetch.L2, 16 KiB requests); join before the post.
  ovl  split the pre: layer_input runs right after the post (main stream, mhc_det_norm_li),
       the coefficient half (prenorm GEMM + mhc_det_norm_coef: split sums, sigmoids, sinkhorn)
       is deferred and forked onto a side stream right before the NEXT all-reduce (it feeds only
       the next post and the next pre's layer_input); join before that post.
Same kernels and arithmetic as the det path (docker/patch/mhc_det.cu: mhc_det_norm_li,
mhc_det_norm_coef), so the same bits.

Chain per sublayer (86: 40 target + 3 draft layers x attn/ffn, real mHC weights), one CUDA graph
per arm: [AR stand-in: nb CTAs spin --ar-us on globaltimer, no memory traffic, then
x = bf16(src + y_prev), 8 bf16 per thread step] -> post -> pre -> [work stand-in: streams --work-mb of its own weight
buffer with ld.global.cs on 48x256 threads, y = layer_input]. Arms: base (stand-ins only),
stock, det, pf, ovl. Every output of every sublayer (residual, post_mix, comb_mix, layer_input,
pre_mix) of det / pf / ovl is compared bitwise with stock (graph replay and eager). Timing:
graphs replayed round-robin (order rotates every rep), --iters reps, CUDA events.
mHC us per pass = t(arm) - t(base) of the same rep.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys

import torch

sys.path.insert(0, "/repo/kernel_study/mhc_det")
import bench_path as BP  # noqa: E402
import common as C  # noqa: E402
import mhc_det  # noqa: E402
from mhc_det_rt import Module  # noqa: E402

PROXY_SRC = r"""
typedef unsigned short u16;
extern "C" __global__ void ar_proxy(const uint4* __restrict__ src, const uint4* __restrict__ y, uint4* __restrict__ out,
                                    int n8, unsigned long long ns) {
  if (threadIdx.x == 0) {
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1)); } while (t1 - t0 < ns);
  }
  __syncthreads();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += gridDim.x * blockDim.x) {
    const uint4 a = src[i], b = y[i];
    const unsigned av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
    unsigned o[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float lo = __fadd_rn(__uint_as_float(av[k] << 16), __uint_as_float(bv[k] << 16));
      const float hi = __fadd_rn(__uint_as_float(av[k] & 0xffff0000u), __uint_as_float(bv[k] & 0xffff0000u));
      asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(o[k]) : "f"(hi), "f"(lo));
    }
    out[i] = make_uint4(o[0], o[1], o[2], o[3]);
  }
}
extern "C" __global__ void work_proxy(const uint4* __restrict__ w, int n16, const u16* __restrict__ li,
                                      u16* __restrict__ y, int n, unsigned* __restrict__ sink) {
  unsigned acc = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += gridDim.x * blockDim.x) {
    unsigned a, b, c, d;
    asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(w + i));
    acc ^= a ^ b ^ c ^ d;
  }
  if (acc == 0x9E3779B9u) sink[threadIdx.x] = acc;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) y[i] = li[i];
}
extern "C" __global__ void l2_prefetch(const char* __restrict__ p, unsigned bytes) {
  for (unsigned off = threadIdx.x * 16384u; off < bytes; off += blockDim.x * 16384u) {
    const unsigned n = min(16384u, bytes - off);
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p + off), "r"(n) : "memory");
  }
}
"""



class Stand:
    def __init__(self, args):
        m = Module(PROXY_SRC, "overlap_proxies.cu")
        self.ar, self.work, self.pf = m.function("ar_proxy"), m.function("work_proxy"), m.function("l2_prefetch")
        self.ns = int(args.ar_us * 1000)
        self.nb = args.ar_ctas
        self.sink = torch.zeros(256, dtype=torch.int32, device="cuda")

    def ar_call(self, src, y, out):
        n8 = out.numel() // 8
        self.ar.launch((self.nb,), (128,), 0,
                       [(src.data_ptr(), ctypes.c_void_p), (y.data_ptr(), ctypes.c_void_p),
                        (out.data_ptr(), ctypes.c_void_p), (n8, ctypes.c_int), (self.ns, ctypes.c_ulonglong)])

    def work_call(self, w, li, y):
        self.work.launch((48,), (256,), 0,
                         [(w.data_ptr(), ctypes.c_void_p), (w.numel() // 16, ctypes.c_int),
                          (li.data_ptr(), ctypes.c_void_p), (y.data_ptr(), ctypes.c_void_p),
                          (y.numel(), ctypes.c_int), (self.sink.data_ptr(), ctypes.c_void_p)])

    def prefetch(self, t):
        self.pf.launch((1,), (32,), 0, [(t.data_ptr(), ctypes.c_void_p), (t.numel() * t.element_size(), ctypes.c_uint)])


def norm_args(mixes, sqrsum, hc_scale, hc_base, residual, pre_mix, norm_weight, post, comb, layer_input,
              pre_mix_out, t, rms_numel, rms_eps, hc_pre_eps, sk_eps, post_mult, sk_repeat, norm_eps):
    vp = ctypes.c_void_p
    return [(mixes.data_ptr(), vp), (sqrsum.data_ptr(), vp), (hc_scale.data_ptr(), vp), (hc_base.data_ptr(), vp),
            (residual.data_ptr(), vp), (pre_mix.data_ptr() if pre_mix is not None else 0, vp),
            (norm_weight.data_ptr(), vp), (post.data_ptr(), vp), (comb.data_ptr(), vp),
            (layer_input.data_ptr(), vp), (pre_mix_out.data_ptr(), vp), (t, ctypes.c_int),
            (float(rms_numel), ctypes.c_float), (float(rms_eps), ctypes.c_float), (float(hc_pre_eps), ctypes.c_float),
            (float(sk_eps), ctypes.c_float), (float(post_mult), ctypes.c_float), (int(sk_repeat), ctypes.c_int),
            (float(norm_eps), ctypes.c_float)]


class Arm:
    """mHC ops of one arm plus its hooks at the all-reduce (at_ar: before it, join: after it)."""

    def __init__(self, name, w, dk, packed, stand, side):
        self.name, self.w, self.dk, self.packed, self.stand, self.side = name, w, dk, packed, stand, side
        self.path = BP.Path(w, dk, packed, name != "stock")
        self.pending = None
        self.forked = False
        if dk is not None:
            self.f_li = dk.mod.function("mhc_det_norm_li")
            self.f_coef = dk.mod.function("mhc_det_norm_coef")

    def post(self, x, residual, post_mix, comb):
        return self.path.post(x, residual, post_mix, comb)

    def pre(self, residual, fn_name, prefix, sub, pre_mix, x=None):
        if self.name != "ovl":
            return self.path.pre(residual, fn_name, prefix, sub, pre_mix, x=x)
        w = self.w
        t = residual.shape[0]
        dev = residual.device
        if x is None:
            x = residual.view(t, 4 * C.HIDDEN)
        nxt = torch.empty(t, 4, dtype=torch.float32, device=dev)
        post = torch.empty_like(nxt)
        comb = torch.empty(t, 16, dtype=torch.float32, device=dev)
        li = torch.empty(t, C.HIDDEN, dtype=torch.bfloat16, device=dev)
        mixes = torch.empty(16, t, 24, dtype=torch.float32, device=dev)
        sqr = torch.empty(16, t, dtype=torch.float32, device=dev)
        a = norm_args(mixes, sqr, w[f"{prefix}.hc_{sub}_scale"], w[f"{prefix}.hc_{sub}_base"], residual, pre_mix,
                      w[f"{prefix}.{sub}_norm.weight"], post, comb, li, nxt, t, x.shape[1], BP.RMS_EPS, BP.HC_EPS,
                      BP.HC_EPS, BP.POST_ALPHA, BP.SINKHORN, BP.RMS_EPS)
        self.f_li.launch((t,), (256,), 0, a, pdl=True)
        assert self.pending is None
        self.pending = (x, self.packed[fn_name], mixes, sqr, a, t)
        return post.unsqueeze(-1), comb.view(t, 4, 4), li, nxt

    def at_ar(self, next_fn_name):
        if self.name == "pf" and next_fn_name is not None:
            self.side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.side):
                self.stand.prefetch(self.packed[next_fn_name])
            self.forked = True
        elif self.name == "ovl" and self.pending is not None:
            x, fnp, mixes, sqr, a, t = self.pending
            self.side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.side):
                self.dk.gemm_pk(x, fnp, mixes, sqr, pdl=False)
                self.f_coef.launch((t,), (32,), 0, a, pdl=True)
            self.forked = True

    def join(self):
        if self.forked:
            torch.cuda.current_stream().wait_stream(self.side)
            self.forked = False
            self.pending = None


def fn_name_of(prefix, sub, first):
    return "layers.0.hc_attn_fn_broadcast" if (first and prefix == "layers.0") else f"{prefix}.hc_{sub}_fn"


def run(arm, t, emb, src, xouts, ys, ws, stand, static_li, record=None):
    subs = BP.sublayers()
    state = None
    for i, (prefix, sub, first) in enumerate(subs):
        if i > 0:
            if arm is not None:
                arm.at_ar(fn_name_of(prefix, sub, first))
            stand.ar_call(src[i], ys[i - 1], xouts[i])
            if arm is not None:
                arm.join()
        if arm is None:
            stand.work_call(ws[i], static_li, ys[i])
            continue
        if first and prefix == "layers.0":
            e = emb[:t]
            residual = e.unsqueeze(1).expand(-1, C.HC, -1).contiguous()
            pm, cm, li, pr = arm.pre(residual, fn_name_of(prefix, sub, first), prefix, sub, None, x=e)
        elif first:
            residual = emb[:t].unsqueeze(-2).repeat(1, C.HC, 1)
            pm, cm, li, pr = arm.pre(residual, fn_name_of(prefix, sub, first), prefix, sub, None)
        else:
            rp, pp, cp, prp = state
            residual = arm.post(xouts[i], rp, pp, cp)
            pm, cm, li, pr = arm.pre(residual, fn_name_of(prefix, sub, first), prefix, sub, prp)
        state = (residual, pm, cm, pr)
        if record is not None:
            record.append((residual, pm, cm, li, pr))
        stand.work_call(ws[i], li, ys[i])
    # the all-reduce after the last sublayer (the model's final post consumes its coefficients)
    if arm is not None:
        arm.at_ar(None)
    stand.ar_call(src[0], ys[-1], xouts[0])
    if arm is not None:
        arm.join()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar-us", type=float, default=22.7)
    ap.add_argument("--ar-ctas", type=int, default=5)
    ap.add_argument("--work-mb", type=float, default=4.0)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--T", default="1,4,8")
    ap.add_argument("--tag", default="")
    ap.add_argument("--src", default=None, help="kernel source (default docker/patch/mhc_det.cu)")
    args = ap.parse_args()
    w = BP.load_weights()
    emb = C.embeddings(64)
    fn_names = [k for k in w if k.endswith("_fn") or k.endswith("_broadcast")]
    packed = {k: mhc_det.pack_fn(w[k]) for k in fn_names}
    dk = mhc_det.DetKernels(src_path=args.src)
    stand = Stand(args)
    side = torch.cuda.Stream()
    subs = BP.sublayers()
    nbytes = int(args.work_mb * 2**20) // 16 * 16
    ws = [torch.randint(0, 255, (nbytes,), dtype=torch.uint8, device="cuda") for _ in subs]
    res = {"args": vars(args), "device": torch.cuda.get_device_name(), "rows": []}
    arm_names = ("stock", "det", "pf", "ovl")
    for t in [int(v) for v in args.T.split(",")]:
        g = torch.Generator(device="cuda").manual_seed(9 + t)
        src = [(torch.randn(t, C.HIDDEN, device="cuda", generator=g) * 2).bfloat16() for _ in subs]
        xouts = [torch.empty_like(s) for s in src]
        ys = [torch.empty_like(s) for s in src]
        static_li = (torch.randn(t, C.HIDDEN, device="cuda", generator=g)).bfloat16()
        arms = {n: Arm(n, w, None if n == "stock" else dk, packed, stand, side) for n in arm_names}
        # bitwise: eager, then graph replay, every output of every sublayer vs stock (eager)
        ref = []
        run(arms["stock"], t, emb, src, xouts, ys, ws, stand, static_li, ref)
        ref = [[o.clone() for o in r] for r in ref]
        eq = {}
        for n in arm_names[1:]:
            rec = []
            run(arms[n], t, emb, src, xouts, ys, ws, stand, static_li, rec)
            torch.cuda.synchronize()
            eq[f"{n}_eager"] = sum(int(not bool((BP.ints(a) == BP.ints(b)).all()))
                                   for ra, rb in zip(ref, rec) for a, b in zip(ra, rb))
        graphs = {n: BP._graph(lambda n=n: run(None if n == "base" else arms[n], t, emb, src, xouts, ys, ws, stand,
                                               static_li))
                  for n in ("base",) + arm_names}
        # a recording graph per arm for the replay check (the timing graphs keep no outputs)
        for n in arm_names[1:]:
            rec = []
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                run(arms[n], t, emb, src, xouts, ys, ws, stand, static_li, rec)
            bad = 0
            for _ in range(20):
                for r in rec:
                    r[3].fill_(0)
                gr.replay()
                torch.cuda.synchronize()
                bad += sum(int(not bool((BP.ints(a) == BP.ints(b)).all())) for ra, rb in zip(ref, rec)
                           for a, b in zip(ra, rb))
            eq[f"{n}_graph20"] = bad
            del gr
        print(t, "mismatching outputs (430 per run):", json.dumps(eq), flush=True)
        for _ in range(3):
            for gr in graphs.values():
                gr.replay()
        torch.cuda.synchronize()
        names = list(graphs)
        samples = {k: [] for k in names}
        for it in range(args.iters):
            k0 = it % len(names)
            for k in names[k0:] + names[:k0]:
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
                graphs[k].replay()
                b.record()
                b.synchronize()
                samples[k].append(a.elapsed_time(b) * 1000.0)
        row = {"T": t, "mismatches": eq, "base_us": C.summarize(samples["base"])}
        for n in arm_names:
            row[n] = C.summarize([x - y for x, y in zip(samples[n], samples["base"])])
        for n in ("pf", "ovl"):
            row[f"{n}_minus_det_paired"] = C.summarize([x - y for x, y in zip(samples[n], samples["det"])])
        res["rows"].append(row)
        print(json.dumps({k: (v if not isinstance(v, dict) or "median_us" not in v else
                              {"median": v["median_us"], "p10": v["p10_us"], "p90": v["p90_us"]})
                          for k, v in row.items()}), flush=True)
    with open(f"{BP.OUT}/overlap_probe{args.tag}.json", "w") as fh:
        json.dump(res, fh, indent=1)
    ok = all(v == 0 for r in res["rows"] for v in r["mismatches"].values())
    print("bitwise PASS" if ok else "bitwise FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
