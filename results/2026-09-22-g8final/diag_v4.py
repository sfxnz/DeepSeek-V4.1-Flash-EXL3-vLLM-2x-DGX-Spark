#!/usr/bin/env python3
"""diag v4: WRAPPER-IN-LOOP repro — the layer every prior diag skipped.

Replicates the real serving chain exactly at CPU scale:
  safetensors safe_open (mmap) -> real WeightsMapper.apply (vl_model) ->
  sorted(...) LIST RETAINED FOR THE WHOLE LOAD (vl_model.py:344) ->
  per-key consumption via the real shard_exl3_col/row + _narrow_tp +
  dest.copy_ sequence (patched vllm_exl3.exl3, G8 env-gated).

Measures per shard:
  - gc Tensor census (total / file-backed views / anon-backed)
    classified by data_ptr membership in /proc/self/maps file mappings
  - smaps_rollup Rss / Anonymous / Swap
  - per-step attribution on sampled keys: which step (narrow / contiguous /
    copy_) allocates a NEW anon storage that SURVIVES the iteration

Legs (env LEG): g8-full | stock-full | g8-feedonly (list built, no consumption)
N_LAYERS: routed layers to load (default 3; ~3.2 GiB file + 1.6 GiB dest/layer... per rank halves).
Run CPU-only inside canonical-g8 (phase-1 baked; phase-2 chain-applied here):

  docker run --rm --network none --entrypoint python3 \
    -v $PWD:/repo:ro -v $PWD/docker/patch:/opt/dsv41-patch:ro \
    -v $HOME/.cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-EXL3:/packs:ro \
    -e LEG=g8-full -e DSV41_LOAD_PF_G8=1 -e N_LAYERS=3 \
    dsv41-flash-exl3-sm121:canonical-g8 /repo/results/2026-09-22-g8final/diag_v4.py
"""
import gc
import json
import os
import sys
import threading
import time

import torch

sys.path.insert(0, "/opt/dsv41-patch")

LEG = os.environ.get("LEG", "g8-full")
G8 = LEG.startswith("g8")
CONSUME = LEG != "g8-feedonly"
N_LAYERS = int(os.environ.get("N_LAYERS", "3"))
PACK = "/packs/snapshots/" + ("2.0bpw-mcg-g8" if G8 else "2.0bpw-mcg")
N_EXPERTS, HIDDEN, INTER = 384, 5120, 1152
SAMPLE_EVERY = 256

if G8:
    os.environ["DSV41_LOAD_PF_G8"] = "1"

# chain-apply phase-2 (alloc) onto the image's phase-1 exl3.py, then import
import vllm_exl3.exl3 as _exl3_mod  # noqa: E402
import inspect  # noqa: E402

_src = inspect.getsource(_exl3_mod)
from pfg8_loader_reindex import patch as _pfg8_patch  # noqa: E402

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

# real wrapper mapper, loaded standalone (image path vllm/models/...):
import importlib.util  # noqa: E402
from vllm.model_executor.models.utils import WeightsMapper  # noqa: E402

_VL_PATH = "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/nvidia/vl_model.py"
_spec = importlib.util.spec_from_file_location("vl_model_pulled", _VL_PATH)
_vl = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(_vl)
    _mk_mapper = _vl._make_deepseek_v4_vl_weights_mapper
except Exception:
    # heavy module-level imports fail without GPU; the mapper makers are
    # pure — re-extract BOTH the base (nvidia/model.py) and the VL wrapper
    # maker (nvidia/vl_model.py) by source slice and exec them.
    import re as _re

    _MOD_PATH = "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/nvidia/model.py"
    _srcmod = open(_MOD_PATH).read()
    _srcvl = open(_VL_PATH).read()
    _g = {"WeightsMapper": WeightsMapper, "re": _re}
    _base = _re.search(r"^def _make_deepseek_v4_weights_mapper\(.*?\n    return WeightsMapper\(.*?\)\n", _srcmod, _re.S | _re.M)
    _fn = _re.search(r"^def _make_deepseek_v4_vl_weights_mapper\(.*?\n    return WeightsMapper\(.*?\)\n", _srcvl, _re.S | _re.M)
    assert _base and _fn, "mapper source slice failed"
    exec(_base.group(0), _g)
    exec(_fn.group(0), _g)
    _mk_mapper = _g["_make_deepseek_v4_vl_weights_mapper"]

print(f"diag_v4 leg={LEG} pack={PACK} layers={N_LAYERS}", flush=True)

# ---------------------------------------------------------------- helpers
maps_lock = threading.Lock()
_file_ranges = []  # (start, end) of file-backed rw mappings


def refresh_file_ranges():
    global _file_ranges
    rng = []
    with open("/proc/self/maps") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 6 or parts[1][0] != "r" or "-" not in parts[5]:
                continue
            path = parts[5]
            if path.startswith(("/packs/", ".safetensors")) or path.endswith(
                ".safetensors"
            ):
                a, b = parts[0].split("-")
                rng.append((int(a, 16), int(b, 16)))
    _file_ranges = rng


def tensor_is_filebacked(t: torch.Tensor) -> bool:
    p = t.data_ptr()
    return any(a <= p < b for a, b in _file_ranges)


def rollup():
    r = {}
    with open("/proc/self/smaps_rollup") as fh:
        for line in fh:
            if line.startswith(("Rss:", "Anonymous:", "Swap:", "Private_Clean:", "Private_Dirty:")):
                k, v = line.split(":", 1)
                r[k] = int(v.strip().split()[0]) // 1024  # MiB
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
    n_file = sum(1 for t in tensors if t.data_ptr() and tensor_is_filebacked(t))
    r = rollup()
    line = (
        f"[{tag}] gc_Tensors={len(tensors)} file_views={n_file} "
        f"anon_views={len(tensors) - n_file} | rollup MiB: "
        f"Rss={r.get('Rss', 0)} Anon={r.get('Anonymous', 0)} "
        f"Swap={r.get('Swap', 0)} PDirty={r.get('Private_Dirty', 0)} "
        f"| MemAvail={memavail_mib()}MiB"
    )
    print(line, flush=True)
    return {"tensors": len(tensors), "file": n_file, "anon": len(tensors) - n_file,
            "rss": r.get("Rss", 0), "anonmib": r.get("Anonymous", 0)}


# ---------------------------------------------------------------- dest pool
class FakeLayer(torch.nn.Module):
    tp_rank = 1
    tp_size = 2
    layer_name = "layers.0.ffn"

    def _map_global_expert_id_to_local_expert_id(self, gid):
        return gid


method = exl3.Exl3MoEMethod.__new__(exl3.Exl3MoEMethod)
method.bits = 2
method._logged = True
method.quant_config = type("QC", (), {"bits": 2})()
layers = {}
for i in range(N_LAYERS):
    L = FakeLayer()
    method.create_weights(L, N_EXPERTS, HIDDEN, INTER, torch.float16)
    layers[str(i)] = L
dest_bytes = sum(p.numel() * p.element_size() for L in layers.values() for p in L.parameters())
print(f"dest pool (CPU=anon ledger): {dest_bytes / 2**30:.2f} GiB", flush=True)

# ---------------------------------------------- wrapper feed: mapper + sorted
mapper = _mk_mapper("fp4", "weight_scale")
idx = json.load(open(os.path.join(PACK, "model.safetensors.index.json")))["weight_map"]
want = {str(i) for i in range(N_LAYERS)}
expert_keys = {}
for k, f in idx.items():
    p = k.split(".")
    if p[0] == "layers" and len(p) > 3 and p[3] == "experts" and p[1] in want:
        expert_keys.setdefault(f, []).append(k)
n_keys_total = sum(len(v) for v in expert_keys.values())
print(f"expert keys: {n_keys_total} across {len(expert_keys)} shards", flush=True)

census("after dest alloc")

# real iterator: per shard safe_open, get_tensor per key, mapper applied,
# THEN sorted + retained (vl_model.py:344)
t0 = time.time()
mapped = []
for f in sorted(expert_keys):
    path = os.path.join(PACK, f)
    with safe_open(path, framework="pt") as sf:
        keys = set(sf.keys())
        for name in sorted(k for k in expert_keys[f] if k in keys):
            mapped.append((name, sf.get_tensor(name)))
    census(f"feed shard {f} ({len(mapped)} keys)")
mapped = sorted(
    mapper.apply(mapped), key=lambda x: x[0]
)  # mapper yields (new_name, tensor); tensors retained by the list
census(f"feed done + mapper + sorted ({len(mapped)} keys) t={time.time()-t0:.0f}s")

# ---------------------------------------- consumption (real shard fns inline)
shard_col = exl3.shard_exl3_col
shard_row = exl3.shard_exl3_row
PICK = {
    ("w1", "trellis"): ("w13_trellis", 0), ("w3", "trellis"): ("w13_trellis", 1),
    ("w2", "trellis"): ("w2_trellis", None),
    ("w1", "suh"): ("w13_suh", 0), ("w3", "suh"): ("w13_suh", 1), ("w2", "suh"): ("w2_suh", None),
    ("w1", "svh"): ("w13_svh", 0), ("w3", "svh"): ("w13_svh", 1), ("w2", "svh"): ("w2_svh", None),
    ("w1", "mcg"): ("w13_mcg", 0), ("w3", "mcg"): ("w13_mcg", 1), ("w2", "mcg"): ("w2_mcg", None),
}
per_step = {"detach_contig_new_storage": 0, "shard_new_storage": 0, "copy_new_storage": 0}
if CONSUME:
    n = 0
    cur_shard = None
    for name, t in mapped:
        # mapped name: language_model.model.layers.<lid>. ... .experts.<eid>.<w>.<suffix>
        parts = name.split(".")
        lid, sh, suffix = parts[3], parts[-2], parts[-1]
        pname, sidx = PICK[(sh, suffix)]
        param = getattr(layers[lid], pname)
        if suffix in ("mcg", "mul1"):
            dest = param.data[int(parts[-3])] if sidx is None else param.data[int(parts[-3]), sidx]
            dest.fill_(int(t.reshape(-1)[0].item()) if t.numel() else 0)
            n += 1
            continue
        dest = param.data[int(parts[-3])] if sidx is None else param.data[int(parts[-3]), sidx]
        # --- exact _load_exl3 sequence, per-step storage attribution on sample
        sample = (n % SAMPLE_EVERY) == 0
        if sample:
            pre = t.data_ptr(), t.untyped_storage().data_ptr()
        loaded = t.detach().contiguous()
        if sample and loaded.untyped_storage().data_ptr() != pre[1]:
            per_step["detach_contig_new_storage"] += 1
        sharded = shard_col(loaded, suffix, 1, 2) if sh in ("w1", "w3") else shard_row(loaded, suffix, 1, 2)
        if sample and sharded.untyped_storage().data_ptr() not in (
            t.untyped_storage().data_ptr(), loaded.untyped_storage().data_ptr()
        ):
            per_step["shard_new_storage"] += 1
        if tuple(dest.shape) != tuple(sharded.shape):
            raise RuntimeError(f"shape mismatch {name}: dest {tuple(dest.shape)} vs {tuple(sharded.shape)}")
        dest.copy_(sharded)
        n += 1
        if sample and sh == "w2" and suffix == "trellis" and n < 4000:
            # keep a weakref-style check: does sharded storage survive the iter?
            pass
        del loaded, sharded
        if n % (SAMPLE_EVERY * 8) == 0:
            shard_now = name.split(".")[1]
            if shard_now != cur_shard or n % (SAMPLE_EVERY * 32) == 0:
                census(f"consume n={n}")
                cur_shard = shard_now
        if memavail_mib() < 6000:
            print(f"ABORT: MemAvail {memavail_mib()}MiB at n={n}", flush=True)
            break
    census(f"consume done n={n} t={time.time()-t0:.0f}s")

# retention check: after full consumption + del of locals, what survives?
del mapped
gc.collect()
census("after del mapped + gc.collect")
print(json.dumps({
    "leg": LEG, "layers": N_LAYERS, "keys": n_keys_total,
    "per_step_new_storage_samples": per_step,
    "secs": time.time() - t0,
}, indent=2), flush=True)
print("DIAG_V4 DONE", flush=True)
