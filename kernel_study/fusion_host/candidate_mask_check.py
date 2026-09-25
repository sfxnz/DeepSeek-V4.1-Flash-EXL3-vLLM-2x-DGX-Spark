#!/usr/bin/env python3
"""DSV41_CANDIDATE_MASK_BOUNDED: equality with the stock mask and the decode top-k, timing.

Runs in the serving image on a GPU (patch dir mounted at /opt/dsv41-patch).

1. mask: random logits [rows, width], random candidate blocks (valid, -1
   padding, past the width), per-row ends (block multiples, odd, < one block,
   = width); stock apply_candidate_mask vs the bounded one: every column below
   each row's end bit for bit.
2. top-k: the serve's decode sequence, stock mask + top_k_per_row_decode on
   logits A vs bounded mask + top_k_per_row_decode on B = A with NaN / +inf /
   1e30 garbage past each row's end (paged_mqa_logits(clean_logits=False)
   leaves stale values there): the SET of indices per row must be equal.
   top_k_per_row_decode itself is not order-deterministic (two stock runs on
   the same logits return the same set in a different order; counted in
   stock_order_nondeterministic), so raw order cannot be the criterion.
   seq_lens is (B, next_n) like the native spec decode path, rows =
   B * next_n, vis and row_repeat computed exactly as sparse_attn_indexer does.
3. timing: each arm in a CUDA graph at the serve's width (max_model_len
   1048576) and a 64k width, rows 4 and 8, typical decode ends; >= 300
   replays, CUDA events, arms alternating.
Prints one JSON line; exit 1 on any mismatch.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys

import torch

sys.path.insert(0, "/opt/dsv41-patch")
import candidate_mask_bounded as cmb  # noqa: E402

BS = 8  # candidate_block_size
K = 2048  # candidate_topk_blocks
TOPK = 512  # index_topk


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def summarize(v):
    return dict(n=len(v), median=statistics.median(v), p10=pct(v, 0.10), p90=pct(v, 0.90), mean=statistics.fmean(v))


def candidates(rows, ends, width, g, dev):
    out = torch.full((rows, K), -1, dtype=torch.int32)
    for r in range(rows):
        nb = max(1, (int(ends[r]) + BS - 1) // BS)
        n = min(K, nb)
        pick = torch.randperm(nb, generator=g)[:n].to(torch.int32)
        out[r, :n] = pick
        if n > 3:
            out[r, 1] = (width + BS - 1) // BS + 5  # past the width -> clamped to nblocks
            out[r, 2] = -1
    return out.to(dev)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    from vllm import _custom_ops as ops
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import apply_candidate_mask as stock
    from vllm.triton_utils import tl, triton

    dev = torch.device("cuda")
    fk, mk = cmb._build_kernels(tl, triton)
    bounded = cmb.make_apply(torch, triton, stock, fk, mk)
    g = torch.Generator().manual_seed(5)
    bad = []
    n_mask = n_topk = n_nondet = 0

    # 1. mask equality below each row's end
    for width in (1 << 20, 65536, 4096):
        for rows in (1, 4, 8):
            for trial in range(6):
                ends = torch.randint(1, min(width, 60000) + 1, (rows,), generator=g)
                ends[0] = min(width, BS * 37)
                if rows > 1:
                    ends[1] = 5  # below one block
                if rows > 2:
                    ends[2] = width  # full row
                ends_d = ends.to(torch.int32).to(dev)
                cand = candidates(rows, ends, width, g, dev)
                logits = torch.randn(rows, width, generator=g).to(dev)
                a, b = logits.clone(), logits.clone()
                stock(a, None, ends_d, cand, BS, 1)
                bounded(b, None, ends_d, cand, BS, 1)
                torch.cuda.synchronize()
                for r in range(rows):
                    e = int(ends[r])
                    if not torch.equal(a[r, :e].view(torch.int32), b[r, :e].view(torch.int32)):
                        bad.append(f"mask width={width} rows={rows} row={r} end={e}")
                n_mask += 1

    # 2. decode top-k on garbage past the end
    for width in (1 << 20, 65536):
        for batch, next_n in ((1, 4), (2, 4), (1, 1), (2, 3)):
            for trial in range(5):
                rows = batch * next_n
                base = torch.randint(2 * TOPK, min(width, 40000), (batch,), generator=g)
                seq_lens = torch.stack([base - next_n + j + 1 for j in range(next_n)], 1).to(torch.int32)
                if trial == 0:
                    seq_lens[0, :] = torch.arange(next_n, dtype=torch.int32) + 300  # shorter than topk
                seq_lens_d = seq_lens.to(dev)
                vis = seq_lens_d.reshape(-1)[:rows]
                row_repeat = next_n if vis.numel() != rows else 1
                ends = vis.cpu()
                cand = candidates(rows, ends, width, g, dev)
                A = torch.randn(rows, width, generator=g).to(dev)
                B = A.clone()
                for r in range(rows):
                    e = int(ends[r])
                    tail = B[r, e:]
                    if tail.numel():
                        tail[0::3] = float("nan")
                        tail[1::3] = float("inf")
                        tail[2::3] = 1e30
                stock(A, None, vis, cand, BS, row_repeat)
                bounded(B, None, vis, cand, BS, row_repeat)
                ia = torch.full((rows, TOPK), -7, dtype=torch.int32, device=dev)
                ib, ia2 = ia.clone(), ia.clone()
                ops.top_k_per_row_decode(A, next_n, seq_lens_d, ia, rows, A.stride(0), A.stride(1), TOPK)
                ops.top_k_per_row_decode(A, next_n, seq_lens_d, ia2, rows, A.stride(0), A.stride(1), TOPK)
                ops.top_k_per_row_decode(B, next_n, seq_lens_d, ib, rows, B.stride(0), B.stride(1), TOPK)
                torch.cuda.synchronize()
                if not torch.equal(ia.sort(1).values, ib.sort(1).values):
                    bad.append(f"topk width={width} B={batch} next_n={next_n} trial={trial}")
                n_nondet += int(not torch.equal(ia, ia2))
                n_topk += 1

    # 3. timing
    timing = {}
    for width in (1 << 20, 65536):
        for rows in (4, 8):
            ends = torch.randint(1500, 20000, (rows,), generator=g)
            ends_d = ends.to(torch.int32).to(dev)
            cand = candidates(rows, ends, width, g, dev)
            logits = torch.randn(rows, width, device=dev)
            graphs = {}
            for name, fn in (("stock", stock), ("bounded", bounded)):
                fn(logits, None, ends_d, cand, BS, 1)
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    fn(logits, None, ends_d, cand, BS, 1)
                graphs[name] = gr
            t = {"stock": [], "bounded": []}
            for it in range(args.warmup + args.iters):
                for name in (("stock", "bounded") if it % 2 == 0 else ("bounded", "stock")):
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    graphs[name].replay()
                    e1.record()
                    torch.cuda.synchronize()
                    if it >= args.warmup:
                        t[name].append(e0.elapsed_time(e1) * 1e3)
            timing[f"width{width}_rows{rows}"] = {k: summarize(v) for k, v in t.items()}
    print(json.dumps({"torch": torch.__version__, "mask_cases": n_mask, "topk_cases": n_topk,
                      "stock_order_nondeterministic": n_nondet, "lever_state": dict(cmb._STATE),
                      "timing_us": timing, "mismatches": bad}))
    return 1 if bad or not cmb._STATE["armed"] else 0


if __name__ == "__main__":
    sys.exit(main())
