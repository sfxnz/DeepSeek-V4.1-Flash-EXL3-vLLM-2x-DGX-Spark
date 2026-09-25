#!/usr/bin/env python3
"""EngramDiskStager.stage(): stock vs DSV41_ENGRAM_NATIVE_STAGE, end to end on a GPU.

Runs in the serving image with a GPU (one rank's view, real tables):

  docker run --rm --gpus all --network none -v <repo>:/repo -v <hf>:/hf:ro \
      -e DSV41_ENGRAM_GATHER_V2=1 -e DSV41_ENGRAM_CENSUS=1 ... \
      --entrypoint python3 <image> /repo/kernel_study/fusion_host/engram_stage_gpu_bench.py \
      --snapshot /hf/hub/.../2.0bpw-mcg-lmhead-mxfp8

The stager is the image's EngramDiskStager with its real stage() (the
engram_stage_fast + gather v2 chain) and the real DiskEngramTable objects;
only the hash module is a stand-in that returns precomputed ids (uniform in
each head's bucket range, fresh every call) from a device tensor. Every
iteration queues a calibrated --busy-us GPU sleep first, so the host runs
ahead and blocks in hashes_ready.synchronize() exactly as in the serve. Then,
between two CUDA events: --chain-ops small kernels standing in for the eager
metadata chain between the early-hash point and stage() (native_early arm:
early hash launch, chain, stage(); other arms: chain, stage()), and stage().
Their elapsed time = chain + hash + DtoH + host gather + H2D, minus whatever
the early arm hides under the chain. The host time from the sleep launch to
the stage() call is recorded for every timed iteration and the run fails if
it reaches the busy time (the GPU would have gone idle waiting for launches).
The arms rotate per iteration, each with its own id sets. --l2 cold writes a
128 MiB buffer (GB10 L2 is 24 MiB) before the sleep. The chain alone is also
timed (chain_gpu_us), which is the hide window the early hash gets.

Correctness: every call's staged device rows are compared bit for bit between
the arms for the same ids (both arms run on the same id set at verify time),
and the native arm must report engaged and armed. Then a sabotage run: the
native gather followed by one flipped bit; the verify must refill the step
with the stock gather (staged rows equal the stock stage's) and disarm, and
the next call must be stock. Prints one JSON line.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "docker" / "patch"))


def load_text_config(snapshot: str):
    with open(os.path.join(snapshot, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    return SimpleNamespace(**cfg.get("text_config", cfg))


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def summarize(v):
    return dict(n=len(v), median=statistics.median(v), p10=pct(v, 0.10), p90=pct(v, 0.90),
                mean=statistics.fmean(v), min=min(v), max=max(v))


class FakeBatch:
    """InputBatch stand-in (a plain dataclass in vLLM: weakref-able)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeHash:
    """Stand-in for NgramHashState: returns the next precomputed ids."""

    def __init__(self, nl, heads):
        self.multipliers = torch.zeros((nl, 4))
        self.next = None

    def ensure_cache(self):
        return True

    def __call__(self, ids, positions, qsl, mask, lookback, lbmask, slot, table):
        return self.next.clone()  # a device copy, like the hash kernel's output


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--tokens", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--busy-us", type=float, default=4000.0)
    ap.add_argument("--hot", action="store_true", help="read every row before timing (else prefetch-like WILLNEED)")
    ap.add_argument("--lead-us", type=float, default=0.0, help="extra host spin after WILLNEED (the busy work already leads)")
    ap.add_argument("--chain-ops", type=int, default=241,
                    help="small eager kernels queued between the early-hash point and stage(): the r3 "
                         "trace (c1-rank0, step 120) has 241 stock (block tables, slot mappings, 21 metadata "
                         "builds, embedding + AR), 122 with DSV41_ATTN_T2R_DEDUP=1, 77 with it and "
                         "DSV41_SWA_META_FUSED=1")
    ap.add_argument("--l2", choices=("warm", "cold"), default="warm")
    args = ap.parse_args()

    from vllm.models.deepseek_v4_1.common import engram as eng
    from vllm.models.deepseek_v4_1.common.engram_disk import DiskEngramTable
    import engram_native_stage as ens

    dev = torch.device("cuda")
    layout = eng.EngramLayout(load_text_config(args.snapshot))
    heads = layout.n_hash_cols
    part = -(-heads // 2)
    h0 = args.rank * part
    h1 = min(h0 + part, heads)
    nl = len(layout.layer_ids)
    dim = layout.head_dim
    max_n = max(args.tokens)

    engrams = []
    for li, layer_id in enumerate(layout.layer_ids):
        disk = DiskEngramTable(args.snapshot, layer_id, dim, 32)
        sizes = [p for order in layout.primes[li] for p in order]
        emb = SimpleNamespace(
            part_n_hash_cols=part, head_start=h0, n_hash_cols=heads, dim=dim,
            vocab_start_idx=sum(sizes[:h0]), vocab_end_idx=sum(sizes[:h0 + part]), disk=disk,
            sizes=sizes, offs=[sum(sizes[:c]) for c in range(heads)],
        )
        emb.disk_file_rows_owned = MethodType(eng.ParallelEngramEmbedding.disk_file_rows_owned, emb)
        staged = torch.zeros((max_n, part, dim), dtype=torch.bfloat16, device=dev)
        e = SimpleNamespace(layer_hash_index=li, embed_tokens=emb, staged=staged)
        e._staged_rows_for_ubatch = (lambda s=staged: s)
        engrams.append(e)

    def make_stager():
        st = object.__new__(eng.EngramDiskStager)
        st.hash_state = FakeHash(nl, heads)
        st.engrams = engrams
        st.local_heads = part
        st.head_start, st.head_end = h0, h1
        st.dim = dim
        st.max_tokens = 8192
        st.hash_host = torch.empty((st.max_tokens, nl, h1 - h0), dtype=torch.int32, pin_memory=True)
        st.rows_host = [torch.empty((st.max_tokens, part, dim), dtype=torch.bfloat16, pin_memory=True) for _ in engrams]
        st.hashes_ready = torch.cuda.Event()
        st.num_staged = 0
        st.prefetch_on = False
        return st

    if getattr(eng, "_ENG_STAGE_POOL", None) is None:
        eng._ENG_STAGE_POOL = ThreadPoolExecutor(max_workers=int(os.environ.get("DSV41_ENGRAM_STAGE_THREADS", "16")))
    stock_stage = eng.EngramDiskStager.stage
    from vllm.models.deepseek_v4_1.common.mm_preprocess import image_sentinel_mask

    native_stage = ens._make_stage(stock_stage, torch, image_sentinel_mask)
    st_stock, st_nat, st_early = make_stager(), make_stager(), make_stager()
    st_nat._eng_native = ens._NativeStager(st_nat, torch)
    st_early._eng_native = ens._NativeStager(st_early, torch)
    import engram_early_hash as eeh
    from vllm.models.deepseek_v4_1.nvidia import model_state as ms_mod
    from vllm.triton_utils import triton as _triton

    early_hook = eeh.EarlyHash(torch, image_sentinel_mask, ms_mod._gather_lookback_kernel, _triton)
    eeh._STATE["verify_left"] = 8

    rng = random.Random(99 + args.rank)

    def make_hashes(n):
        # [n, nl, heads] full-width ids (the stager slices its head span)
        out = torch.empty((n, nl, heads), dtype=torch.int32)
        for li, e in enumerate(engrams):
            emb = e.embed_tokens
            for t in range(n):
                for c in range(heads):
                    out[t, li, c] = emb.offs[c] + rng.randrange(emb.sizes[c])
        return out

    def spans(hashes, n):
        for li, e in enumerate(engrams):
            emb, d = e.embed_tokens, e.embed_tokens.disk
            for x in hashes[:n, li, h0:h1].reshape(-1).tolist():
                if emb.vocab_start_idx <= x < emb.vocab_end_idx:
                    for fd, off, ln in ((d.w_fd, d.w_off + x * d.dim, d.dim), (d.s_fd, d.s_off + x * d.sb, d.sb)):
                        a = off & ~4095
                        yield fd, a, ((off + ln + 4095) & ~4095) - a

    ids_dev = torch.zeros(max_n, dtype=torch.int64, device=dev)
    pos_dev = torch.arange(max_n, dtype=torch.int64, device=dev)
    qsl_dev = torch.tensor([0, max_n], dtype=torch.int32, device=dev)
    look_dev = torch.full((1, 3), 5, dtype=torch.int32, device=dev)
    # Busy work: a GPU spin long enough that the host launches everything
    # (early hash, chain, stage) before the GPU reaches it, as in the serve
    # where the host runs ~40 ms ahead. Calibrated cycles per us.
    torch.cuda._sleep(1000)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    torch.cuda._sleep(20_000_000)
    e1.record()
    torch.cuda.synchronize()
    cycles_per_us = 20_000_000 / (e0.elapsed_time(e1) * 1e3)
    busy_cycles = int(args.busy_us * cycles_per_us)

    ev0 = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    ev1 = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    arms = [("stock", st_stock, stock_stage), ("native", st_nat, native_stage),
            ("native_early", st_early, native_stage)]
    ms_fake = SimpleNamespace(lookback_token_ids=torch.full((2, 3), -1, dtype=torch.int32, device=dev), rope_state=None)
    rs_fake = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=torch.randint(0, 1000, (2, 4096), dtype=torch.int32, device=dev)),
        num_computed_tokens=SimpleNamespace(gpu=torch.full((2,), 100, dtype=torch.int32, device=dev)),
    )
    small = torch.zeros(64, device=dev)
    flush = torch.empty(128 << 20, dtype=torch.uint8, device=dev) if args.l2 == "cold" else None
    idx_mapping = torch.zeros(1, dtype=torch.int32, device=dev)

    def chain():
        for _ in range(args.chain_ops):
            small.add_(1.0)

    out = {"args": vars(args), "torch": torch.__version__, "cycles_per_us": round(cycles_per_us, 1), "results": {}}
    mismatches = host_bound = 0
    for n in args.tokens:
        gpu_us = {a: [] for a, _, _ in arms}
        host_us = {a: [] for a, _, _ in arms}
        enq_us = {a: [] for a, _, _ in arms}
        for it in range(args.warmup + args.iters):
            order = arms[it % 3:] + arms[: it % 3]
            host_t = [0.0, 0.0, 0.0]
            enq_t = [0.0, 0.0, 0.0]
            verify_ids = make_hashes(n) if it % 25 == 0 else None
            for k, (name, st, fn) in enumerate(order):
                hashes = verify_ids if verify_ids is not None else make_hashes(n)
                if args.hot:
                    for fd, a, ln in spans(hashes, n):
                        os.pread(fd, ln, a)
                else:
                    for fd, a, ln in spans(hashes, n):
                        os.posix_fadvise(fd, a, ln, os.POSIX_FADV_DONTNEED)
                    for fd, a, ln in spans(hashes, n):
                        os.posix_fadvise(fd, a, ln, os.POSIX_FADV_WILLNEED)
                st.hash_state.next = hashes.to(dev)
                ib = FakeBatch(input_ids=ids_dev, positions=pos_dev, query_start_loc=qsl_dev,
                               num_reqs=1, num_tokens=n, idx_mapping=idx_mapping)
                torch.cuda.synchronize()
                if not args.hot:
                    tl = time.perf_counter() + args.lead_us * 1e-6
                    while time.perf_counter() < tl:
                        pass
                if flush is not None:
                    flush.fill_(it & 0xFF)
                torch.cuda._sleep(busy_cycles)
                hs = time.perf_counter_ns()
                ev0[k].record()
                h0t = time.perf_counter_ns()
                if name == "native_early":
                    early_hook.launch(st, ms_fake, ib, rs_fake)
                    chain()
                    st._early_cur = ib
                    ens._EARLY_HOOK[0] = early_hook
                    enq_t[k] = (time.perf_counter_ns() - hs) / 1e3
                    fn(st, ib.input_ids, ib.positions, ib.query_start_loc[:2], ms_fake.lookback_token_ids, n)
                    ens._EARLY_HOOK[0] = None
                    st._early_cur = None
                else:
                    chain()
                    enq_t[k] = (time.perf_counter_ns() - hs) / 1e3
                    fn(st, ids_dev[:n], pos_dev[:n], qsl_dev, look_dev, n)
                h1t = time.perf_counter_ns()
                ev1[k].record()
                host_t[k] = (h1t - h0t) / 1e3
            torch.cuda.synchronize()
            for k, (name, st, fn) in enumerate(order):
                if it >= args.warmup:
                    gpu_us[name].append(ev0[k].elapsed_time(ev1[k]) * 1e3)
                    host_us[name].append(host_t[k])
                    enq_us[name].append(enq_t[k])
                    host_bound += enq_t[k] >= args.busy_us
            if verify_ids is not None:
                # re-run both arms on the same ids and compare the staged device rows
                ref = []
                for name, st, fn in arms:
                    st.hash_state.next = verify_ids.to(dev)
                    fn(st, ids_dev[:n], pos_dev[:n], qsl_dev, look_dev, n)
                    torch.cuda.synchronize()
                    ref.append([e.staged[:n].clone() for e in engrams])
                for other in ref[1:]:
                    for x, y in zip(ref[0], other):
                        if not torch.equal(x.view(torch.int16), y.view(torch.int16)):
                            mismatches += 1
        out["results"][f"n{n}"] = {
            "gpu_gap_us": {k: summarize(v) for k, v in gpu_us.items()},
            "host_stage_us": {k: summarize(v) for k, v in host_us.items()},
            "host_enqueue_us": {k: summarize(v) for k, v in enq_us.items()},
        }
    # the hide window: the chain alone on the GPU, behind the same sleep
    chain_t = []
    c0, c1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for it in range(args.warmup + 200):
        torch.cuda.synchronize()
        if flush is not None:
            flush.fill_(it & 0xFF)
        torch.cuda._sleep(busy_cycles)
        c0.record()
        chain()
        c1.record()
        torch.cuda.synchronize()
        if it >= args.warmup:
            chain_t.append(c0.elapsed_time(c1) * 1e3)
    out["chain_gpu_us"] = summarize(chain_t)
    out["mismatches"] = mismatches
    out["host_bound_samples"] = host_bound
    out["native_armed"] = ens._STATE["armed"]
    out["native_engaged"] = ens._STATE["engaged"]
    out["early_armed"] = eeh._STATE["armed"]
    out["early_engaged"] = eeh._STATE["engaged"]
    ok = not mismatches and not host_bound and ens._STATE["armed"] and ens._STATE["engaged"]

    # Sabotage: the native gather followed by one flipped bf16 bit in the
    # pinned rows. The verify must catch it, refill that step with the stock
    # gather (staged device rows == the stock stage's for the same ids) and
    # disarm; the next call must take the stock stage.
    st_sab = make_stager()
    st_sab._eng_native = ens._NativeStager(st_sab, torch)
    real_gather = st_sab._eng_native.gather

    def bad_gather(stager, n):
        real_gather(stager, n)
        stager.rows_host[0].view(torch.int16).view(-1)[:1].bitwise_xor_(1)

    st_sab._eng_native.gather = bad_gather
    ens._STATE.update(armed=True, verify_left=2, engaged=True)
    sab = {}
    n = min(args.tokens)
    for label in ("verify", "after_disarm"):
        ids = make_hashes(n)
        got = []
        for st, fn in ((st_sab, native_stage), (st_stock, stock_stage)):
            st.hash_state.next = ids.to(dev)
            fn(st, ids_dev[:n], pos_dev[:n], qsl_dev, look_dev, n)
            torch.cuda.synchronize()
            got.append([e.staged[:n].clone() for e in engrams])
        sab[f"{label}_rows_equal_stock"] = all(
            torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(*got)
        )
        if label == "verify":
            sab["disarmed"] = not ens._STATE["armed"]
    out["sabotage"] = sab
    ok = ok and all(sab.values())
    print(json.dumps(out))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
