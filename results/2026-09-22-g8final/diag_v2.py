#!/usr/bin/env python3
"""diag v2: REAL iterator semantics — safetensors safe_open (mmap), many
layers, CUDA dests. If RSS grows linearly -> reproduced; then bisect by
swapping get_tensor with pread (FORCE_PREAD=1)."""
import json, os, struct, sys, threading, time

import torch

sys.path.insert(0, "/opt/dsv41-patch")
PACK = os.environ["PACK"]
G8 = os.environ.get("DSV41_LOAD_PF_G8", "0") == "1"
FORCE_PREAD = os.environ.get("FORCE_PREAD", "0") == "1"
N_LAYERS = int(os.environ.get("N_LAYERS", "10"))
N_EXPERTS, HIDDEN, INTER = 384, 5120, 1152

from safetensors import safe_open
import vllm_exl3.exl3 as exl3

print(f"leg={'g8' if G8 else 'stock'} pread={FORCE_PREAD} layers={N_LAYERS}", flush=True)


class Watcher(threading.Thread):
    interval = 0.5

    def __init__(self):
        super().__init__(daemon=True)
        self.peak = 0
        self.hist = []
        self.stop_evt = threading.Event()

    def run(self):
        while not self.stop_evt.is_set():
            rss = swap = 0
            with open("/proc/self/status") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        rss = int(line.split()[1])
                    elif line.startswith("VmSwap:"):
                        swap = int(line.split()[1])
            self.hist.append((time.time(), rss // 2**20, swap // 2**20))
            self.peak = max(self.peak, rss + swap)
            time.sleep(self.interval)

    def stop(self):
        self.stop_evt.set()
        self.join(timeout=2)


def pread_tensor(path, name, meta):
    nbytes = meta["data_offsets"][1] - meta["data_offsets"][0]
    with open(path, "rb") as fh0:
        hn = struct.unpack("<Q", fh0.read(8))[0]
    with open(path, "rb") as fh:
        fh.seek(8 + hn + meta["data_offsets"][0])
        raw = fh.read(nbytes)
    td = {"I16": torch.int16, "F16": torch.float16, "I32": torch.int32}[meta["dtype"]]
    return torch.frombuffer(bytearray(raw), dtype=td).reshape(meta["shape"])


def _pick(layer, name):
    parts = name.split(".")
    suffix, wname = parts[-1], parts[-2]
    return getattr(
        layer,
        {"w1": "w13_trellis", "w3": "w13_trellis", "w2": "w2_trellis"}[wname]
        if suffix == "trellis"
        else {"w1": "w13_suh", "w3": "w13_suh", "w2": "w2_suh"}[wname]
        if suffix == "suh"
        else {"w1": "w13_svh", "w3": "w13_svh", "w2": "w2_svh"}[wname]
        if suffix == "svh"
        else {"w1": "w13_mcg", "w3": "w13_mcg", "w2": "w2_mcg"}[wname],
    )


class FakeLayer(torch.nn.Module):
    tp_rank = 1
    tp_size = 2
    layer_name = "layers.3.ffn"

    def _map_global_expert_id_to_local_expert_id(self, gid):
        return gid


torch.cuda.set_device(0)
method = exl3.Exl3MoEMethod.__new__(exl3.Exl3MoEMethod)
method.bits = 2
method._logged = False
method.quant_config = type("QC", (), {"bits": 2})()
layer = FakeLayer()
with torch.device("cuda"):
    method.create_weights(layer, N_EXPERTS, HIDDEN, INTER, torch.float16)

w = Watcher()
w.start()

idx = json.load(open(os.path.join(PACK, "model.safetensors.index.json")))["weight_map"]
want_layers = [str(i) for i in range(1, N_LAYERS + 1)]
by_file = {}
for k, f in idx.items():
    p = k.split(".")
    if len(p) > 5 and p[0] == "layers" and p[3] == "experts" and p[1] in want_layers:
        by_file.setdefault(f, []).append(k)

t0 = time.time()
n = 0
for f, keys in sorted(by_file.items()):
    path = os.path.join(PACK, f)
    if FORCE_PREAD:
        with open(path, "rb") as fh:
            hn = struct.unpack("<Q", fh.read(8))[0]
            fh.seek(8)
            hdr = json.loads(fh.read(hn))
        for name in sorted(keys):
            t = pread_tensor(path, name, hdr[name])
            ok = method._load_exl3(param=_pick(layer, name), loaded_weight=t,
                                   weight_name=name, shard_id=name.split(".")[-2],
                                   expert_id=int(name.split(".")[-3]), return_success=True)
            assert ok
            n += 1
            del t
    else:
        with safe_open(path, framework="pt") as sf:
            hdr_keys = set(sf.keys())
            for name in sorted(k for k in keys if k in hdr_keys):
                t = sf.get_tensor(name)
                ok = method._load_exl3(param=_pick(layer, name), loaded_weight=t,
                                       weight_name=name, shard_id=name.split(".")[-2],
                                       expert_id=int(name.split(".")[-3]), return_success=True)
                assert ok
                n += 1
                del t
    rss = w.hist[-1][1] if w.hist else 0
    print(f"  shard {f}: cum tensors={n} rss={rss}GiB t={time.time()-t0:.0f}s", flush=True)

time.sleep(1)
w.stop()
print(json.dumps({
    "leg": "g8" if G8 else "stock", "pread": FORCE_PREAD, "layers": N_LAYERS,
    "n_tensors": n, "peak_gib": w.peak / 2**20,
    "cuda_alloc_mib": torch.cuda.memory_allocated() // 2**20,
    "secs": time.time() - t0,
}, indent=2), flush=True)
