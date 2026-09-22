#!/usr/bin/env python3
"""diag v5: FULL GPU topology, wrapper-in-loop — the closest CPU/GPU repro
of the real boot short of serving. On the REAL GPU (spark2):

  1. create_weights for all 40 routed layers on CUDA (real dest pool ~86GiB)
  2. REAL iterator semantics: per-shard safe_open -> get_tensor per key ->
     real VL mapper.apply -> sorted() RETAINED list (vl_model.py:344)
  3. per-key consumption through the REAL patched shard fns + dest.copy_
     (CUDA dests: every copy_ is an H2D like the real boot)
  4. per-layer process_weights_after_loading (make_linear_exl3 ->
     exllamav3_ext BC_LinearEXL3 -> build_exl3_fused_state) with a census
     after EVERY layer — the untested 49,536-ctor + pointer-table path.

Census: gc Tensor count, file/anon view split via /proc/self/maps,
smaps_rollup (Rss/Anon/Swap), torch.cuda allocated/reserved, MemAvail.
Legs: LEG=g8 (DSV41_LOAD_PF_G8=1, pack 2.0bpw-mcg-g8) | stock.
"""
import gc
import inspect
import json
import os
import sys
import threading
import time

import torch

sys.path.insert(0, "/opt/dsv41-patch")
LEG = os.environ.get("LEG", "g8")
G8 = LEG == "g8"
PACK = "/packs/snapshots/" + ("2.0bpw-mcg-g8" if G8 else "2.0bpw-mcg")
N_LAYERS = int(os.environ.get("N_LAYERS", "40"))
N_EXPERTS, HIDDEN, INTER = 384, 5120, 1152
if G8:
    os.environ["DSV41_LOAD_PF_G8"] = "1"

# chain-apply phase-2 (alloc) onto the image's phase-1 exl3.py, then exec
import vllm_exl3.exl3 as _exl3_mod  # noqa: E402
from pfg8_loader_reindex import patch as _pfg8_patch  # noqa: E402

_src = inspect.getsource(_exl3_mod)
_patched = _pfg8_patch(_src)
_ns = {"__name__": "vllm_exl3.exl3_patched"}
exec(compile(_patched, "exl3_patched.py", "exec"), _ns)
exl3 = type(sys)("exl3_patched")
for _k in dir(_exl3_mod):
    if not _k.startswith("__"):
        setattr(exl3, _k, getattr(_exl3_mod, _k))
for _k, _v in _ns.items():
    setattr(exl3, _k, _v)

from safetensors import safe_open  # noqa: E402
import importlib.util  # noqa: E402
from vllm.model_executor.models.utils import WeightsMapper  # noqa: E402

_VL_PATH = "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/nvidia/vl_model.py"
_MOD_PATH = "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/nvidia/model.py"
import re as _re  # noqa: E402

_g = {"WeightsMapper": WeightsMapper, "re": _re}
_srcmod = open(_MOD_PATH).read()
_srcvl = open(_VL_PATH).read()
_base = _re.search(r"^def _make_deepseek_v4_weights_mapper\(.*?\n    return WeightsMapper\(.*?\)\n", _srcmod, _re.S | _re.M)
_fn = _re.search(r"^def _make_deepseek_v4_vl_weights_mapper\(.*?\n    return WeightsMapper\(.*?\)\n", _srcvl, _re.S | _re.M)
assert _base and _fn, "mapper source slice failed"
exec(_base.group(0), _g)
exec(_fn.group(0), _g)
mapper = _g["_make_deepseek_v4_vl_weights_mapper"]("fp4", "weight_scale")

print(f"diag_v5 leg={LEG} pack={PACK} layers={N_LAYERS}", flush=True)

_file_ranges = []


def refresh_file_ranges():
    global _file_ranges
    rng = []
    with open("/proc/self/maps") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 6 or parts[1][0] != "r":
                continue
            if parts[5].endswith(".safetensors"):
                a, b = parts[0].split("-")
                rng.append((int(a, 16), int(b, 16)))
    _file_ranges = rng


def rollup():
    r = {}
    with open("/proc/self/smaps_rollup") as fh:
        for line in fh:
            if line.startswith(("Rss:", "Anonymous:", "Swap:")):
                k, v = line.split(":", 1)
                r[k] = int(v.strip().split()[0]) // 1024
    return r


def memavail_mib():
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    return 0


def census(tag):
    refresh_file_ranges()
    tensors = [o for o in gc.get_objects() if isinstance(o, torch.Tensor)]
    n_file = sum(1 for t in tensors if t.data_ptr() and any(a <= t.data_ptr() < b for a, b in _file_ranges))
    r = rollup()
    try:
        ca = torch.cuda.memory_allocated() // 2**20
        cr = torch.cuda.memory_reserved() // 2**20
    except Exception:
        ca = cr = -1
    print(
        f"[{tag}] gc_T={len(tensors)} file={n_file} anon={len(tensors)-n_file} | "
        f"MiB Rss={r.get('Rss',0)} Anon={r.get('Anonymous',0)} Swap={r.get('Swap',0)} | "
        f"cuda alloc={ca} res={cr} | Avail={memavail_mib()}",
        flush=True,
    )
    return r.get("Anonymous", 0)


class FakeLayer(torch.nn.Module):
    tp_rank = 1
    tp_size = 2
    layer_name = "layers.0.ffn"

    def _map_global_expert_id_to_local_expert_id(self, gid):
        return gid


torch.cuda.set_device(0)
methods, layers = [], {}
with torch.device("cuda"):
    for i in range(N_LAYERS):
        m = exl3.Exl3MoEMethod.__new__(exl3.Exl3MoEMethod)
        m.bits = 2
        m._logged = True
        m.quant_config = type("QC", (), {"bits": 2})()
        L = FakeLayer()
        m.create_weights(L, N_EXPERTS, HIDDEN, INTER, torch.float16)
        methods.append(m)
        layers[str(i)] = L
dest = sum(p.numel() * p.element_size() for L in layers.values() for p in L.parameters())
print(f"dest pool: {dest/2**30:.1f} GiB on {next(iter(layers.values())).w13_trellis.device}", flush=True)
census("dest alloc")

PICK = {
    ("w1", "trellis"): ("w13_trellis", 0), ("w3", "trellis"): ("w13_trellis", 1),
    ("w2", "trellis"): ("w2_trellis", None),
    ("w1", "suh"): ("w13_suh", 0), ("w3", "suh"): ("w13_suh", 1), ("w2", "suh"): ("w2_suh", None),
    ("w1", "svh"): ("w13_svh", 0), ("w3", "svh"): ("w13_svh", 1), ("w2", "svh"): ("w2_svh", None),
    ("w1", "mcg"): ("w13_mcg", 0), ("w3", "mcg"): ("w13_mcg", 1), ("w2", "mcg"): ("w2_mcg", None),
}
shard_col, shard_row = exl3.shard_exl3_col, exl3.shard_exl3_row

idx = json.load(open(os.path.join(PACK, "model.safetensors.index.json")))["weight_map"]
want = {str(i) for i in range(N_LAYERS)}
by_file = {}
for k, f in idx.items():
    p = k.split(".")
    if p[0] == "layers" and len(p) > 3 and p[3] == "experts" and p[1] in want:
        by_file.setdefault(f, []).append(k)
n_total = sum(len(v) for v in by_file.values())
print(f"expert keys {n_total} over {len(by_file)} shards", flush=True)

t0 = time.time()
mapped = []
for f in sorted(by_file):
    with safe_open(os.path.join(PACK, f), framework="pt") as sf:
        keys = set(sf.keys())
        for name in sorted(k for k in by_file[f] if k in keys):
            mapped.append((name, sf.get_tensor(name)))
    if int(f.split("-")[1]) % 8 == 0 or f == sorted(by_file)[-1]:
        census(f"feed {f} ({len(mapped)} keys) t={time.time()-t0:.0f}s")
mapped = sorted(mapper.apply(mapped), key=lambda x: x[0])
census(f"feed+map+sorted ({len(mapped)}) t={time.time()-t0:.0f}s")

n = 0
for name, t in mapped:
    parts = name.split(".")
    lid, sh, suffix = parts[3], parts[-2], parts[-1]
    pname, sidx = PICK[(sh, suffix)]
    param = getattr(layers[lid], pname)
    eid = int(parts[-3])
    if suffix in ("mcg", "mul1"):
        d = param.data[eid] if sidx is None else param.data[eid, sidx]
        d.fill_(int(t.reshape(-1)[0].item()) if t.numel() else 0)
        n += 1
        continue
    loaded = t.detach().contiguous()
    sharded = shard_col(loaded, suffix, 1, 2) if sh in ("w1", "w3") else shard_row(loaded, suffix, 1, 2)
    d = param.data[eid] if sidx is None else param.data[eid, sidx]
    d.copy_(sharded)
    n += 1
    del loaded, sharded
    if n % 46080 == 0:
        census(f"consume n={n} t={time.time()-t0:.0f}s")
    if memavail_mib() < 4000:
        print(f"ABORT avail={memavail_mib()} at n={n}", flush=True)
        break
census(f"consume done n={n} t={time.time()-t0:.0f}s")
del mapped
gc.collect()
census("post-consume gc")

for i, (L, m) in enumerate(zip(layers.values(), methods)):
    m.process_weights_after_loading(L)
    if i < 5 or i % 5 == 4:
        census(f"pwaf layer {i} t={time.time()-t0:.0f}s")
    if memavail_mib() < 4000:
        print(f"ABORT avail={memavail_mib()} after layer {i}", flush=True)
        break
census(f"pwaf done t={time.time()-t0:.0f}s")
print("DIAG_V5 DONE", flush=True)
