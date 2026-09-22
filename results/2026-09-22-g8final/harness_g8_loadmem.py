#!/usr/bin/env python3
"""G8 vs stock load-path memory harness (CPU-only, Phase A evidence).

Simulates ONE rank's (TP=2, rank1) walk of the Exl3MoEMethod load sequence
for the routed-expert tensors of the real packs, using the REAL loader code
(vllm_exl3.exl3 pulled from the image, phase-1+2 patch chain-applied), with
real per-tensor reads from the real shards (safetensors headers + pread, no
mmap retention), driving the REAL _load_exl3 / create_weights / (stubbed)
process_weights_after_loading.

Legs:
  stock : DSV41_LOAD_PF_G8 unset, pack 2.0bpw-mcg,       stock dims
  g8    : DSV41_LOAD_PF_G8=1,     pack 2.0bpw-mcg-g8,    narrow dims swapped

Measures: peak VmRSS+VmSwap (kB) via /proc/self/status sampled every 0.25 s
in a background thread; tracemalloc peak for torch-python attribution;
torch allocator reserved for the dest Parameter pool.

PASS criterion: g8 peak - stock peak < 2 GiB (dest bytes are identical, so
any larger delta = a retention bug the fix must remove).
"""
import importlib.util
import json
import os
import struct
import sys
import threading
import time

import torch

REPO = os.path.expanduser(
    "~/projects/ai-lab/recipes/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark"
)
HERE = os.path.join(REPO, "results", "2026-09-22-g8final")
HUB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-EXL3/snapshots"
)

NUM_EXPERTS = 384  # per MoE layer (43 routed layers; 3 layers have 128)
HIDDEN = 5120
INTER = 1152  # intermediate_size_per_partition: 2304 col-parallel / TP2


def read_tensor(path, offsets, shape, dtype):
    """Real per-tensor disk read (no mmap): exactly what safe_open+get_tensor
    materializes, an 8+n header skip + pread of the tensor bytes."""
    nbytes = 1
    for s in shape:
        nbytes *= s
    nbytes *= {"I16": 2, "F16": 2, "BF16": 2, "I32": 4, "U8": 1, "F32": 4}[dtype]
    with open(path, "rb") as fh:
        fh.seek(offsets[0])
        raw = fh.read(nbytes)
    td = {"I16": torch.int16, "F16": torch.float16, "BF16": torch.bfloat16,
          "I32": torch.int32, "U8": torch.uint8, "F32": torch.float32}[dtype]
    return torch.frombuffer(bytearray(raw), dtype=td).reshape(shape)


class RSSWatcher(threading.Thread):
    def __init__(self, interval=0.25):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self.peak_anon = 0
        self.peak_swap = 0
        self.samples = []
        self.stop_evt = threading.Event()

    def _read(self):
        rss = swap = 0
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1])
                elif line.startswith("VmSwap:"):
                    swap = int(line.split()[1])
        return rss, swap

    def run(self):
        while not self.stop_evt.is_set():
            rss, swap = self._read()
            self.samples.append((time.time(), rss, swap))
            if rss + swap > self.peak:
                self.peak = rss + swap
                self.peak_anon = rss
                self.peak_swap = swap
            time.sleep(self.interval)

    def stop(self):
        self.stop_evt.set()
        self.join(timeout=2)


def load_exl3_module(env_g8):
    """Load the image's exl3.py with vllm imports stubbed, patch applied."""
    if env_g8:
        os.environ["DSV41_LOAD_PF_G8"] = "1"
    else:
        os.environ.pop("DSV41_LOAD_PF_G8", None)

    src = os.path.join(HERE, "exl3_chain_ref.py")
    patched = os.path.join(HERE, "exl3_patched_for_harness.py")
    txt = open(src).read()
    sys.path.insert(0, os.path.join(REPO, "docker", "patch"))
    import pfg8_loader_reindex

    out = pfg8_loader_reindex.patch(txt)
    assert out != txt or "pfg8-loader-reindex-alloc" in out
    open(patched, "w").write(out)

    # stub the vllm imports the module does at import time
    import types

    vllm = types.ModuleType("vllm")
    vllm_logger = types.ModuleType("vllm.logger")

    def init_logger(name):
        import logging

        return logging.getLogger(name)

    vllm_logger.init_logger = init_logger
    fme = types.ModuleType("vllm.model_executor")
    layers = types.ModuleType("vllm.model_executor.layers")
    fm = types.ModuleType("vllm.model_executor.layers.fused_moe")
    fmconf = types.ModuleType("vllm.model_executor.layers.fused_moe.config")

    class FusedMoEQuantConfig:
        pass

    fmconf.FusedMoEQuantConfig = FusedMoEQuantConfig
    fmbase = types.ModuleType("vllm.model_executor.layers.fused_moe.fused_moe_method_base")

    class FusedMoEMethodBase:
        pass

    fmbase.FusedMoEMethodBase = FusedMoEMethodBase
    lin = types.ModuleType("vllm.model_executor.layers.linear")

    class LinearMethodBase:
        pass

    class UnquantizedLinearMethod:
        pass

    lin.LinearMethodBase = LinearMethodBase
    lin.UnquantizedLinearMethod = UnquantizedLinearMethod
    qbase = types.ModuleType("vllm.model_executor.layers.quantization.base_config")

    class QuantizationConfig:
        pass

    class QuantizeMethodBase:
        pass

    qbase.QuantizationConfig = QuantizationConfig
    qbase.QuantizeMethodBase = QuantizeMethodBase
    qmod = types.ModuleType("vllm.model_executor.layers.quantization")
    qmod.register_quantization_config = lambda *a, **k: None
    utils_me = types.ModuleType("vllm.model_executor.utils")
    utils_me.set_weight_attrs = lambda param, attrs: None
    dist = types.ModuleType("vllm.distributed")
    dist.get_tensor_model_parallel_rank = lambda: 1
    dist.get_tensor_model_parallel_world_size = lambda: 2
    exllamav3 = types.ModuleType("exllamav3")
    exllamav3_ext = types.ModuleType("exllamav3_ext")

    mods = {
        "vllm": vllm,
        "vllm.logger": vllm_logger,
        "vllm.model_executor": fme,
        "vllm.model_executor.layers": layers,
        "vllm.model_executor.layers.fused_moe": fm,
        "vllm.model_executor.layers.fused_moe.config": fmconf,
        "vllm.model_executor.layers.fused_moe.fused_moe_method_base": fmbase,
        "vllm.model_executor.layers.linear": lin,
        "vllm.model_executor.layers.quantization": qmod,
        "vllm.model_executor.layers.quantization.base_config": qbase,
        "vllm.model_executor.utils": utils_me,
        "vllm.distributed": dist,
        "exllamav3": exllamav3,
        "exllamav3_ext": exllamav3_ext,
    }
    saved = {k: sys.modules.get(k) for k in mods}
    sys.modules.update(mods)
    try:
        spec = importlib.util.spec_from_file_location("exl3_harness_mod", patched)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod


class FakeOwner:
    tp_rank = 1
    tp_size = 2
    moe_tp_size = None
    _exl3_tp_size = None

    def _map_global_expert_id_to_local_expert_id(self, gid):
        return gid  # EP off: physical == local


class FakeLayer(torch.nn.Module):
    tp_rank = 1
    tp_size = 2

    def _map_global_expert_id_to_local_expert_id(self, gid):
        return gid  # EP off: physical == local


def run_leg(pack, env_g8, shard_ids):
    print(f"\n=== leg {'g8' if env_g8 else 'stock'} pack={pack} ===", flush=True)
    mod = load_exl3_module(env_g8)
    P = os.path.join(HUB, pack)

    # --- create_weights (real) ---
    method = mod.Exl3MoEMethod.__new__(mod.Exl3MoEMethod)
    method.bits = 2

    class FakeLayer(torch.nn.Module):
        tp_rank = 1
        tp_size = 2

        def _map_global_expert_id_to_local_expert_id(self, gid):
            return gid  # EP off: physical == local

    layer = FakeLayer()
    method.create_weights(layer, NUM_EXPERTS, HIDDEN, INTER, torch.float16)
    dest_bytes = sum(
        p.numel() * p.element_size()
        for p in layer.parameters()
    )
    print(f"create_weights dest pool: {dest_bytes/2**30:.2f} GiB")

    # real per-shard walk: replicate _exl3_routed_experts_loader's mapping loop
    watcher = RSSWatcher()
    watcher.start()
    import tracemalloc

    tracemalloc.start()
    t0 = time.time()
    n_tensors = 0
    for sid in shard_ids:
        fname = f"model-{sid:05d}-of-00048.safetensors"
        path = os.path.join(P, fname)
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for name in hdr:
            if name == "__metadata__" or ".ffn.experts." not in name:
                continue
            meta = hdr[name]
            t = read_tensor(path, meta["data_offsets"], meta["shape"], meta["dtype"])
            # drive the real per-expert loader (mapping resolution simplified
            # to the 3 known expert param names, exactly as the loader does)
            parts = name.split(".")
            # ...ffn.experts.<e>.<wX>.<suffix> — rsplit for safety
            suffix = parts[-1]
            w = parts[-2]
            e = int(parts[-3])
            shard_id = w  # w1/w2/w3
            param = getattr(
                layer,
                {"w1": "w13_trellis", "w3": "w13_trellis", "w2": "w2_trellis"}[w]
                if suffix == "trellis"
                else {"w1": "w13_suh", "w3": "w13_suh", "w2": "w2_suh"}[w]
                if suffix == "suh"
                else {"w1": "w13_svh", "w3": "w13_svh", "w2": "w2_svh"}[w]
                if suffix == "svh"
                else {"w1": "w13_mcg", "w3": "w13_mcg", "w2": "w2_mcg"}[w],
            )
            ok = method._load_exl3(
                param=param,
                loaded_weight=t,
                weight_name=f"experts.{e}.{w}.{suffix}",
                shard_id=shard_id,
                expert_id=e,
                return_success=True,
            )
            assert ok, name
            n_tensors += 1
            del t
            if n_tensors % 4608 == 0:
                rss, swap = watcher._read()
                print(
                    f"  [tick] tensors={n_tensors} rss={rss/2**20:.2f} swap={swap/2**20:.2f}",
                    flush=True,
                )
    dt = time.time() - t0
    _, tm_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    watcher.stop()
    print(
        f"tensors={n_tensors} time={dt:.1f}s peak RSS+swap={watcher.peak/2**20:.2f} GiB "
        f"(rss {watcher.peak_anon/2**20:.2f} + swap {watcher.peak_swap/2**20:.2f}) "
        f"tracemalloc-peak(py)={tm_peak/2**20:.2f} GiB"
    )
    return {
        "leg": "g8" if env_g8 else "stock",
        "pack": pack,
        "n_tensors": n_tensors,
        "peak_gib": watcher.peak / 2**20,
        "peak_rss_gib": watcher.peak_anon / 2**20,
        "peak_swap_gib": watcher.peak_swap / 2**20,
        "tracemalloc_py_gib": tm_peak / 2**20,
        "dest_pool_gib": dest_bytes / 2**30,
        "secs": dt,
    }


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("stock", "g8"):
        # single-leg mode (separate process for clean RSS attribution)
        leg = sys.argv[1] == "g8"
        shards = [int(x) for x in sys.argv[2:]] or [23, 24, 25]
        r = run_leg("2.0bpw-mcg-g8" if leg else "2.0bpw-mcg", leg, shards)
        json.dump([r], open(os.path.join(HERE, f"harness_{sys.argv[1]}.json"), "w"), indent=2)
        sys.exit(0)
    # real geometry: 40 MoE layers × 384 experts + 3 dense-layer MTP-ish × 128.
    # Simulate rank1's FULL pass over one mixed set: 3 shards cover ~6 layers
    # (~2300 experts, ~19 GiB trellis read) — enough to expose per-tensor
    # retention if any exists (the observed balloon was ~94% of full pass).
    shards = [int(x) for x in sys.argv[1:]] or [23, 24, 25]
    import subprocess

    results = []
    for legname in ("stock", "g8"):
        p = subprocess.run(
            [sys.executable, __file__, legname, *map(str, shards)],
            capture_output=True, text=True,
        )
        print(p.stdout, end="")
        if p.returncode != 0:
            print(p.stderr[-3000:])
            sys.exit(1)
        results.append(json.load(open(os.path.join(HERE, f"harness_{legname}.json")))[0])
    json.dump(results, open(os.path.join(HERE, "harness_result.json"), "w"), indent=2)
    s, g = results
    delta = g["peak_gib"] - s["peak_gib"]
    print(f"\nDELTA g8-stock = {delta:.2f} GiB over {len(shards)} shards")
    print("VERDICT:", "LEAN (no retention bug at this scale)" if delta < 1.0 else "BALLOON")
