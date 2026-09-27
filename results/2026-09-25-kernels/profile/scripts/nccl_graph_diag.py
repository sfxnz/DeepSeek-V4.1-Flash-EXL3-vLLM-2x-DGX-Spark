"""Diagnose graph-mode NCCL: event-timed vs profiled replays of a gapped graph (AR 40 KB)."""
import sys, os, statistics, time
import torch, torch.distributed as dist
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from torch.profiler import profile, ProfilerActivity
rank = int(sys.argv[1]); master = sys.argv[2]; tag = sys.argv[3]
torch.cuda.set_device(0); dev = torch.device("cuda:0")
dist.init_process_group("gloo", init_method=f"tcp://{master}", rank=rank, world_size=2)
comm = PyNcclCommunicator(group=dist.group.WORLD, device=dev)
x = torch.randn(20480, dtype=torch.bfloat16, device=dev); out = torch.empty_like(x)
flush = torch.empty(48 << 20, dtype=torch.uint8, device=dev)
G = 20; R = 100
def body(call=True):
    def b():
        for _ in range(G):
            flush.zero_()
            if call: comm.all_reduce(x, out)
    return b
def cap(b):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s): b()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): b()
    torch.cuda.synchronize(); return g
gw = cap(body()); gr = cap(body(False))
def ev_pass(g, tagp):
    torch.cuda.synchronize(); dist.barrier()
    ps = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(R)]
    t0 = time.perf_counter()
    for a, b in ps: a.record(); g.replay(); b.record()
    th = time.perf_counter() - t0
    torch.cuda.synchronize(); tw = time.perf_counter() - t0
    t = sorted(a.elapsed_time(b) * 1e3 for a, b in ps)
    ia = [ps[i][0].elapsed_time(ps[i+1][0]) * 1e3 for i in range(R - 1)]
    print(f"{tag} r{rank} {tagp}: replay med {t[R//2]:.0f} p10 {t[R//10]:.0f} p90 {t[9*R//10]:.0f} us; start-to-start med {statistics.median(ia):.0f}; host enqueue {th/R*1e6:.0f} us/replay; wall {tw/R*1e6:.0f} us/replay", flush=True)
def prof_pass(g, tagp):
    torch.cuda.synchronize(); dist.barrier()
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(R): g.replay()
        torch.cuda.synchronize()
    tw = time.perf_counter() - t0
    ks = sorted((e.time_range.start, e.time_range.end, "nccl" in e.name.lower()) for e in prof.events() if e.device_type.name == "CUDA")
    per = len(ks) // R
    reps = [ks[i:i+per] for i in range(0, len(ks), per)]
    span = [r[-1][1] - r[0][0] for r in reps]
    gaps = [reps[i][0][0] - reps[i-1][-1][1] for i in range(1, len(reps))]
    nk = [e - s for s, e, n in ks if n]
    fk = [e - s for s, e, n in ks if not n]
    # per replay: flush durations by position, and gaps flush->AR and AR->next flush
    fpos = [statistics.median(r[j][1]-r[j][0] for r in reps) for j in range(0, per, 2 if per == 2*G else 1)][:6]
    g1 = [r[j+1][0]-r[j][1] for r in reps for j in range(0, per-1, 2)] if per == 2*G else [0]
    g2 = [r[j+1][0]-r[j][1] for r in reps for j in range(1, per-1, 2)] if per == 2*G else [0]
    print(f"{tag} r{rank} {tagp}: flush kern med {statistics.median(fk):.1f} p10 {sorted(fk)[len(fk)//10]:.1f} p90 {sorted(fk)[9*len(fk)//10]:.1f}; by position {[round(v) for v in fpos]}; gap flush->ar {statistics.median(g1):.1f}; gap ar->flush {statistics.median(g2):.1f}", flush=True)
    print(f"{tag} r{rank} {tagp}: kernels/replay {per}; span med {statistics.median(span):.0f}; inter-replay gap med {statistics.median(gaps):.1f} p90 {sorted(gaps)[9*len(gaps)//10]:.1f}; nccl kern med {statistics.median(nk) if nk else 0:.1f}; wall {tw/R*1e6:.0f} us/replay", flush=True)
for _ in range(10): gw.replay(); gr.replay()
prof_pass(gw, "prof-1 with")
prof_pass(gr, "prof-1 flushonly")
import threading
# host-bound check: enqueue then sleep so no CPU work overlaps replays
torch.cuda.synchronize(); dist.barrier()
t0=time.perf_counter(); gw.replay(); th=time.perf_counter()-t0; torch.cuda.synchronize(); tw=time.perf_counter()-t0
print(f"{tag} r{rank} single replay: host launch {th*1e6:.0f} us, wall {tw*1e6:.0f} us", flush=True)
comm.destroy(); dist.destroy_process_group()
