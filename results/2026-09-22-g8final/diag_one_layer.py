#!/usr/bin/env python3
"""GPU diag: one REAL MoE layer load on the REAL image, G8 vs stock.

Runs inside the container (both ranks of the path are single-process here):
  create_weights (real) -> per-expert load loop over real shard tensors
  (real _load_exl3) -> process_weights_after_loading (REAL, incl.
  make_linear_exl3 -> exllamav3_ext BC_LinearEXL3 + build_exl3_fused_state).

Samples VmRSS/VmSwap + torch.cuda allocated/reserved every 0.5 s.
Env: DSV41_LOAD_PF_G8 (leg), PACK (snapshot dir), LAYER_SHARDS (files).
"""
import json, os, struct, sys, threading, time

import torch

sys.path.insert(0, "/opt/dsv41-patch")

PACK = os.environ["PACK"]
G8 = os.environ.get("DSV41_LOAD_PF_G8", "0") == "1"
OUT = os.environ.get("DIAG_OUT", "/tmp/diag.json")
N_EXPERTS = 384
HIDDEN = 5120
INTER = 1152

import vllm_exl3.exl3 as exl3

print(f"leg={'g8' if G8 else 'stock'} pack={PACK} exl3={exl3.__file__}", flush=True)


class RSSWatcher(threading.Thread):
    def __init__(self, interval=0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self.snap = []
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
            try:
                ca = torch.cuda.memory_allocated() // 2**20
                cr = torch.cuda.memory_reserved() // 2**20
            except Exception:
                ca = cr = -1
            self.snap.append((time.time(), rss // 1024, swap // 1024, ca, cr))
            self.peak = max(self.peak, rss + swap)
            time.sleep(self.interval)

    def stop(self):
        self.stop_evt.set()
        self.join(timeout=2)


def read_tensor(path, offsets, shape, dtype):
    nbytes = 1
    for s in shape:
        nbytes *= s
    nbytes *= {"I16": 2, "F16": 2, "BF16": 2, "I32": 4, "U8": 1, "F32": 4, "F8_E4M3": 1, "F8_E8M0": 1}[dtype]
    # data_offsets are relative to the end of the safetensors header
    with open(path, "rb") as fh0:
        hn = struct.unpack("<Q", fh0.read(8))[0]
        base = 8 + hn
    with open(path, "rb") as fh:
        fh.seek(base + offsets[0])
        raw = fh.read(nbytes)
    td = {"I16": torch.int16, "F16": torch.float16, "BF16": torch.bfloat16,
          "I32": torch.int32, "U8": torch.uint8, "F32": torch.float32,
          "F8_E4M3": torch.uint8, "F8_E8M0": torch.uint8}[dtype]
    return torch.frombuffer(bytearray(raw), dtype=td).reshape(shape)


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
with torch.device("cuda"):  # real boot: initialize_model under `with target_device:`
    method.create_weights(layer, N_EXPERTS, HIDDEN, INTER, torch.float16)
print(
    f"dest on cuda: w13={layer.w13_trellis.device} {layer.w13_trellis.numel()*2/2**30:.2f} GiB",
    flush=True,
)

w = RSSWatcher()
w.start()
t0 = time.time()

# find the shard file(s) holding layer 3 experts and load ALL of layer 3
idx = json.load(open(os.path.join(PACK, "model.safetensors.index.json")))["weight_map"]
by_file = {}
for k, f in idx.items():
    if ".ffn.experts." in k and k.split(".")[1] == "3":
        by_file.setdefault(f, []).append(k)
n_tensors = 0
for f, keys in sorted(by_file.items()):
    path = os.path.join(PACK, f)
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    for name in sorted(keys):
        meta = hdr[name]
        t = read_tensor(path, meta["data_offsets"], meta["shape"], meta["dtype"])
        parts = name.split(".")
        suffix, wname, e = parts[-1], parts[-2], int(parts[-3])
        param = getattr(
            layer,
            {"w1": "w13_trellis", "w3": "w13_trellis", "w2": "w2_trellis"}[wname]
            if suffix == "trellis"
            else {"w1": "w13_suh", "w3": "w13_suh", "w2": "w2_suh"}[wname]
            if suffix == "suh"
            else {"w1": "w13_svh", "w3": "w13_svh", "w2": "w2_svh"}[wname]
            if suffix == "svh"
            else {"w1": "w13_mcg", "w3": "w13_mcg", "w2": "w2_mcg"}[wname],
        )
        ok = method._load_exl3(
            param=param, loaded_weight=t,
            weight_name=f"experts.{e}.{wname}.{suffix}",
            shard_id=wname, expert_id=e, return_success=True,
        )
        assert ok, name
        n_tensors += 1
        del t
t_load = time.time() - t0
print(f"expert loop done: {n_tensors} tensors in {t_load:.1f}s", flush=True)

# REAL process_weights_after_loading (LinearEXL3 + fused state build)
method.process_weights_after_loading(layer)
t_pw = time.time() - t0
print(f"process_weights_after_loading done at {t_pw:.1f}s", flush=True)

time.sleep(2)
w.stop()
peak_gib = w.peak / 2**20
mx = max(w.snap, key=lambda s: s[1] + s[2])
res = {
    "leg": "g8" if G8 else "stock",
    "n_tensors": n_tensors,
    "peak_rss_swap_gib": peak_gib,
    "peak_rss_gib": mx[1] / 1024,
    "peak_swap_gib": mx[2] / 1024,
    "peak_cuda_alloc_mib": max(s[3] for s in w.snap),
    "peak_cuda_reserved_mib": max(s[4] for s in w.snap),
    "loop_s": t_load,
    "total_s": t_pw,
}
print(json.dumps(res, indent=2), flush=True)
json.dump({**res, "snap": w.snap[::4]}, open(OUT, "w"))
