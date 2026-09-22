#!/usr/bin/env python3
"""diag v3: FULL-model dest topology — 43 layers × create_weights on CUDA
(like the real model), then the real iterator order (per-shard, all layers'
tensors interleaved exactly as checkpoint order), then per-layer
process_weights_after_loading. Closest yet to the real boot."""
import json, os, sys, threading, time

import torch

sys.path.insert(0, "/opt/dsv41-patch")
PACK = os.environ["PACK"]
G8 = os.environ.get("DSV41_LOAD_PF_G8", "0") == "1"
from safetensors import safe_open
import vllm_exl3.exl3 as exl3

N_EXPERTS, HIDDEN, INTER = 384, 5120, 1152
# layer id -> num experts (routed layers 0..39 × 384; checkpoint uses 0-based)
LAYER_EXPERTS = {str(i): 384 for i in range(0, 40)}

print(f"leg={'g8' if G8 else 'stock'}", flush=True)


class Watcher(threading.Thread):
    interval = 0.5

    def __init__(self):
        super().__init__(daemon=True)
        self.peak = 0
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
            self.peak = max(self.peak, rss + swap)
            time.sleep(self.interval)

    def stop(self):
        self.stop_evt.set()
        self.join(timeout=2)


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
    layer_name = "x"

    def _map_global_expert_id_to_local_expert_id(self, gid):
        return gid


torch.cuda.set_device(0)
methods, layers = [], []
with torch.device("cuda"):
    for lid, ne in LAYER_EXPERTS.items():
        m = exl3.Exl3MoEMethod.__new__(exl3.Exl3MoEMethod)
        m.bits = 2
        m._logged = True
        m.quant_config = type("QC", (), {"bits": 2})()
        L = FakeLayer()
        m.create_weights(L, ne, HIDDEN, INTER, torch.float16)
        methods.append(m)
        layers.append(L)
print(f"dest pools: {sum(p.numel()*p.element_size() for L in layers for p in L.parameters())/2**30:.1f} GiB cuda", flush=True)

w = Watcher()
w.start()
idx = json.load(open(os.path.join(PACK, "model.safetensors.index.json")))["weight_map"]
layer_of = {}
for k, f in idx.items():
    p = k.split(".")
    if p[0] == "layers" and len(p) > 3 and p[3] == "experts":
        layer_of.setdefault(f, {})[k] = p[1]

t0 = time.time()
n = 0
for f in sorted(layer_of):
    path = os.path.join(PACK, f)
    with safe_open(path, framework="pt") as sf:
        keys = set(sf.keys())
        for name in sorted(k for k in layer_of[f] if k in keys):
            lid = layer_of[f][name]
            L = layers[int(lid)]
            m = methods[int(lid)]
            t = sf.get_tensor(name)
            ok = m._load_exl3(param=_pick(L, name), loaded_weight=t,
                              weight_name=name, shard_id=name.split(".")[-2],
                              expert_id=int(name.split(".")[-3]), return_success=True)
            assert ok, name
            n += 1
            del t
    rss = w.peak / 2**20
    print(f"  {f}: cum={n} rss_peak={rss:.1f}GiB t={time.time()-t0:.0f}s", flush=True)

# per-layer process_weights_after_loading (real)
for L, m in zip(layers, methods):
    m.process_weights_after_loading(L)
print(f"pwaf done t={time.time()-t0:.0f}s", flush=True)
w.stop()
print(json.dumps({
    "leg": "g8" if G8 else "stock", "n_tensors": n,
    "peak_gib": w.peak / 2**20,
    "cuda_alloc_mib": torch.cuda.memory_allocated() // 2**20,
    "cuda_reserved_mib": torch.cuda.memory_reserved() // 2**20,
    "secs": time.time() - t0,
}, indent=2), flush=True)
