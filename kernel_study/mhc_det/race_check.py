#!/usr/bin/env python3
"""PDL / side-stream race check of the det mHC chain with CHANGING inputs (serve image, spark2).

From the review of 4f661a5 (reviewer's rv_race.py): graph-replay and repeat-run determinism tests
feed identical inputs, so a read-before-write race returns the previous run's identical value
and passes. Here every rep gets new sublayer outputs, new embeddings and new spin delays, and a
PDL-launched proxy before each post (trigger at entry, griddepcontrol.wait, a random 0..20 us spin
(a third are 0), then x_i = bf16(src_i + layer_input_{i-1})) stands in for the all-reduce: every
sublayer depends on the previous layer_input as in the model, and every det kernel may launch as
early as PDL allows. The 43-layer recurrence (86 sublayers, real weights) runs as
  det_graph / det_eager    the fused det path (post, GEMM, fused norm)
  ovl_graph / ovl_eager    DSV41_MHC_DET_OVERLAP through mhc_det_overlap itself: layer_input
                           right after the post, GEMM + coefficient half forked right before the
                           proxy (fork_at_all_reduce), joined before the next post (settle)
  stock_graph              the stock kernels
each compared, every output of every sublayer, with the stock chain run eagerly with a device
sync after every op (no overlap possible).

Expectation, fixed before any run: shipped source -> 0 mismatching outputs in every arm at every
T. Negative controls (--neg) must be detected (> 0 mismatches in each family of arms they target
at every T), which is what shows the check has power: gemm_nowait (GEMM without its
griddepcontrol.wait: det; the overlap GEMM is not PDL-launched),
coef_nowait (the coefficient half without its wait: det, ovl), li_nowait (the split layer_input
without its wait: ovl), ovl_nojoin (settle does not join the side stream: ovl). Exit 0 iff the
expectation holds. Writes results/2026-09-25-kernels/mhc-det/race_check_<neg>.json.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import tempfile
import time

import torch

sys.path.insert(0, "/repo/kernel_study/mhc_det")
sys.path.insert(0, "/repo/docker/patch")
import bench_path as BP  # noqa: E402
import common as C  # noqa: E402
import mhc_det  # noqa: E402
import mhc_det_overlap as O  # noqa: E402
from mhc_det_rt import Module  # noqa: E402

PROXY = r"""
typedef unsigned short u16;
extern "C" __global__ void proxy(const u16* __restrict__ li, const u16* __restrict__ src, u16* __restrict__ out,
                                 int n, const unsigned long long* __restrict__ spin, int idx, int trig) {
  if (trig) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (threadIdx.x == 0) {
    unsigned long long ns = spin[idx], t0, t1;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1)); } while (t1 - t0 < ns);
  }
  __syncthreads();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    float s = __uint_as_float(((unsigned)src[i]) << 16);
    float l = li ? __uint_as_float(((unsigned)li[i]) << 16) : 0.f;
    float v = __fadd_rn(s, l);
    unsigned short r;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(v));
    out[i] = r;
  }
}
"""

# name -> (exact text in docker/patch/mhc_det.cu, replacement); each must occur exactly once
NEG_SRC = {
    "gemm_nowait": ("  pdl_wait();\n  // The producer of x (the post) is complete now", "  // The producer of x"),
    "coef_nowait": ("  const float bsk = hc_base[(lane & 15) + 8];\n  pdl_wait();\n",
                    "  const float bsk = hc_base[(lane & 15) + 8];\n"),
    "li_nowait": ("  if (wait_first) pdl_wait();\n", ""),
}
# The side-stream GEMM of the overlap path is not PDL-launched (full dependency on the fork), so
# its wait is a no-op there: gemm_nowait targets the fused det arms only.
NEG_ARMS = {"gemm_nowait": ("det_",), "coef_nowait": ("det_", "ovl_"), "li_nowait": ("ovl_",),
            "ovl_nojoin": ("ovl_",)}
ARMS = ("det_graph", "stock_graph", "det_eager", "ovl_graph", "ovl_eager")


def ints(t):
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


class OvlPath(BP.Path):
    """The det path with DSV41_MHC_DET_OVERLAP's split pre (mhc_det_overlap.defer)."""

    def pre(self, residual, fn_name, prefix, sub, pre_mix, x=None):
        w = self.w
        args = (w[f"{prefix}.hc_{sub}_scale"], w[f"{prefix}.hc_{sub}_base"], BP.RMS_EPS, BP.HC_EPS, BP.HC_EPS,
                BP.POST_ALPHA, BP.SINKHORN)
        return mhc_det.det_pre_delayed(self.dk, self.packed[fn_name], residual, w[fn_name], *args, pre_mix=pre_mix,
                                       x=x, norm_weight=w[f"{prefix}.{sub}_norm.weight"], norm_eps=BP.RMS_EPS,
                                       defer=O.defer)


def kernel_source(neg: str) -> str | None:
    if neg not in NEG_SRC:
        return None
    src = open("/repo/docker/patch/mhc_det.cu").read()
    old, new = NEG_SRC[neg]
    if src.count(old) != 1:
        raise SystemExit(f"negative control {neg}: anchor found {src.count(old)} times")
    fd, path = tempfile.mkstemp(suffix=f"_{neg}.cu")
    with os.fdopen(fd, "w") as fh:
        fh.write(src.replace(old, new))
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--T", default="1,3,4,6,8,16")
    ap.add_argument("--neg", default="none", choices=("none",) + tuple(NEG_ARMS))
    args = ap.parse_args()
    alt = kernel_source(args.neg)
    res: dict = {"device": torch.cuda.get_device_name(), "neg": args.neg, "arms": {}}
    w = BP.load_weights()
    emb_all = C.embeddings(512, seed=3)
    dk = mhc_det.DetKernels(src_path=alt) if alt else mhc_det.DetKernels()
    O._S.torch = torch
    O._S.armed = O._S.on = True
    settle = O.settle
    if args.neg == "ovl_nojoin":
        def settle_nojoin():
            p = O._S.pending
            if p is not None and p.forked:
                O._S.pending = None  # drop the join: the next post may read unwritten coefficients
                return
            settle()
        O.settle = settle_nojoin
    fn_names = [k for k in w if k.endswith("_fn") or k.endswith("_broadcast")]
    packed = {k: mhc_det.pack_fn(w[k]) for k in fn_names}
    subs = BP.sublayers()
    pk = Module(PROXY, "proxy.cu").function("proxy")
    paths = {"det": BP.Path(w, dk, packed, True), "stock": BP.Path(w, dk, packed, False),
             "ovl": OvlPath(w, dk, packed, True)}
    ts = [int(x) for x in args.T.split(",")]
    spin = torch.zeros(len(subs), dtype=torch.int64, device="cuda")

    def proxy(li, src, out, i, pdl):
        n = out.numel()
        pk.launch((min(48, max(1, n // 256)),), (256,), 0,
                  [(li.data_ptr() if li is not None else 0, ctypes.c_void_p), (src.data_ptr(), ctypes.c_void_p),
                   (out.data_ptr(), ctypes.c_void_p), (n, ctypes.c_int), (spin.data_ptr(), ctypes.c_void_p),
                   (i, ctypes.c_int), (1 if pdl else 0, ctypes.c_int)], pdl=pdl)

    def chain(kind, t, emb_buf, src, xouts, rec, pdl, serial):
        path = paths[kind]
        ovl = kind == "ovl"
        state, li_prev = None, None
        for i, (prefix, sub, first) in enumerate(subs):
            if first and prefix == "layers.0":
                e = emb_buf[:t]
                residual = e.unsqueeze(1).expand(-1, C.HC, -1).contiguous()
                pm, cm, li, pr = path.pre(residual, "layers.0.hc_attn_fn_broadcast", prefix, sub, None, x=e)
            elif first:
                residual = emb_buf[:t].unsqueeze(-2).repeat(1, C.HC, 1)
                pm, cm, li, pr = path.pre(residual, f"{prefix}.hc_{sub}_fn", prefix, sub, None)
            else:
                if ovl:
                    O.fork_at_all_reduce(xouts[i])  # the all-reduce stand-in comes next
                proxy(li_prev, src[i], xouts[i], i, pdl)
                if serial:
                    torch.cuda.synchronize()
                if ovl:
                    O.settle()
                rp, pp, cp, prp = state
                residual = path.post(xouts[i], rp, pp, cp)
                if serial:
                    torch.cuda.synchronize()
                pm, cm, li, pr = path.pre(residual, f"{prefix}.hc_{sub}_fn", prefix, sub, prp)
            if serial:
                torch.cuda.synchronize()
            state, li_prev = (residual, pm, cm, pr), li
            if rec is not None:
                rec.append((residual, pm, cm, li, pr))
        if ovl:
            O.settle()  # the last pre's coefficient half (the model's final post would settle it)
            if O._S.side is not None:  # (no-op normally; ends the capture joined under ovl_nojoin)
                torch.cuda.current_stream().wait_stream(O._S.side)

    gen = torch.Generator(device="cuda").manual_seed(2025)
    for t in ts:
        emb_buf = torch.empty(64, C.HIDDEN, dtype=torch.bfloat16, device="cuda")
        src = [torch.empty(t, C.HIDDEN, dtype=torch.bfloat16, device="cuda") for _ in subs]
        xouts = [torch.empty_like(s) for s in src]

        def refill(r):
            idx = torch.randint(0, emb_all.shape[0], (64,), device="cuda", generator=gen)
            emb_buf.copy_(emb_all[idx])
            for i, s in enumerate(src):
                s.copy_((torch.randn(t, C.HIDDEN, device="cuda", generator=gen) * (0.5 + (i + r) % 5)).bfloat16())
            sp = torch.randint(0, 20000, (len(subs),), device="cuda", generator=gen)
            sp[torch.rand(len(subs), device="cuda", generator=gen) < 0.33] = 0
            spin.copy_(sp)

        refill(0)
        graphs, recs = {}, {}
        for kind in ("det", "stock", "ovl"):
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                chain(kind, t, emb_buf, src, xouts, None, True, False)
            torch.cuda.current_stream().wait_stream(st)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            rec = []
            with torch.cuda.graph(g):
                chain(kind, t, emb_buf, src, xouts, rec, True, False)
            graphs[kind], recs[kind] = g, rec
        bad = {a: 0 for a in ARMS}
        first_bad: dict = {}
        t0 = time.time()
        for r in range(args.reps):
            refill(r + 1)
            torch.cuda.synchronize()
            ref = []
            chain("stock", t, emb_buf, src, xouts, ref, False, True)
            ref = [[o.clone() for o in row] for row in ref]
            outs = {}
            for kind in ("det", "stock", "ovl"):
                graphs[kind].replay()
                torch.cuda.synchronize()
                outs[f"{kind}_graph"] = [[o.clone() for o in row] for row in recs[kind]]
            for kind in ("det", "ovl"):
                rec = []
                chain(kind, t, emb_buf, src, xouts, rec, True, False)
                torch.cuda.synchronize()
                outs[f"{kind}_eager"] = [[o.clone() for o in row] for row in rec]
            for name, o in outs.items():
                for si, (ra, rb) in enumerate(zip(ref, o)):
                    for nm, a, b in zip(("residual", "post_mix", "comb_mix", "layer_input", "pre_mix"), ra, rb):
                        if not bool((ints(a) == ints(b)).all()):
                            bad[name] += 1
                            first_bad.setdefault(name, {"rep": r, "sublayer": si, "out": nm,
                                                        "n_diff": int((ints(a) != ints(b)).sum())})
        res["arms"][f"T{t}"] = {"reps": args.reps, "outputs_per_rep": 5 * len(subs), "mismatching_outputs": bad,
                                "first": first_bad, "secs": round(time.time() - t0, 1)}
        print(t, json.dumps(res["arms"][f"T{t}"]), flush=True)
        del graphs, recs
    res["overlap_forks"], res["overlap_in_place"] = O._S.forks, O._S.in_place
    if args.neg == "none":
        ok = all(v == 0 for a in res["arms"].values() for v in a["mismatching_outputs"].values())
        res["expectation"] = "0 mismatching outputs in every arm"
    else:
        targets = NEG_ARMS[args.neg]
        ok = all(sum(v for k, v in a["mismatching_outputs"].items() if k.startswith(fam)) > 0
                 for a in res["arms"].values() for fam in targets)
        res["expectation"] = f"detected (> 0 mismatches) in the {'/'.join(targets)} arms at every T"
    res["pass"] = ok
    with open(f"{BP.OUT}/race_check_{args.neg}.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print(("PASS: " if ok else "FAIL: ") + res["expectation"], flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
