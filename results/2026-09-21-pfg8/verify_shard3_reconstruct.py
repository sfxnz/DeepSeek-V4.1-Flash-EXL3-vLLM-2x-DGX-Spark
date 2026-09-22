import os, sys, json, struct
os.environ["DSV41_PACK_PF_G8"] = "1"
sys.path.insert(0, "/repo/tools")
import numpy as np
import torch
from pathlib import Path
from safetensors.torch import safe_open
from exllamav3.ext import exllamav3_ext as ext
from quantize_experts_exl3 import _dequant_t, _load_index

SRC = Path("/cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277")
STOCK = "/cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-EXL3/snapshots/2.0bpw-mcg/model-00003-of-00048.safetensors"
G8 = "/cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-EXL3/snapshots/2.0bpw-mcg-g8/model-00003-of-00048.safetensors"
SH = "model-00003-of-00048.safetensors"
DEV = "cuda:0"
K = 2

idx = _load_index(SRC)
names = sorted(n for n, f in idx["weight_map"].items() if f == SH)

DT = {"I16": np.uint16, "I32": np.int32, "F16": np.float16}

def read_tensor(path, key):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    b = 8 + n
    t = {k: v for k, v in hdr.items() if k != "__metadata__"}[key]
    s, e = t["data_offsets"]
    f.seek(b + s)
    raw = f.read(e - s)
    return torch.from_numpy(np.frombuffer(raw, dtype=DT[t["dtype"]]).reshape(t["shape"]).copy())

def unfold(t):  # (nt/8, kt, 8w) -> (kt, nt, w)
    ng, kt, gw = t.shape
    w = gw // 8
    return (t.view(ng, kt, 8, w).permute(1, 0, 2, 3).reshape(kt, ng * 8, w)).contiguous()

def relerr(a, b):
    return (torch.linalg.vector_norm(a.double() - b.double()) / torch.linalg.vector_norm(b.double())).item()

with safe_open(SRC / SH, framework="pt") as fh:
    sel = {}
    for kind in ("w1", "w2", "w3"):
        cand = [n for n in names if n.endswith(f".{kind}.weight")]
        sel[kind] = cand[0]
    src_ws, scales = {}, {}
    for kind, wn in sel.items():
        w = fh.get_tensor(wn)
        s = fh.get_tensor(wn[: -len(".weight")] + ".scale")
        src_ws[kind] = _dequant_t(w, s, DEV)
        scales[kind] = (wn[: -len(".weight")], wn)

print(f"{'kind':4} {'relerr G8-unfolded':>18} {'relerr stock':>12} {'rec-vs-rec':>10}")
worst_ratio = 0.0
for kind in ("w1", "w2", "w3"):
    stem = scales[kind][0]
    src = src_ws[kind]  # (in, out) float32 on GPU
    in_f, out_f = src.shape
    # stock
    t_stock = read_tensor(STOCK, stem + ".trellis").to(DEV)
    suh_s = read_tensor(STOCK, stem + ".suh").to(torch.float16).to(DEV)
    svh_s = read_tensor(STOCK, stem + ".svh").to(torch.float16).to(DEV)
    w_stock = torch.empty((in_f, out_f), dtype=torch.half, device=DEV)
    ext.reconstruct_had_slice(w_stock, t_stock, suh_s, svh_s, K, True, False, 0)
    src = src.to(DEV)
    # g8 unfolded (with the G8 pack's own scales)
    t_g8 = unfold(read_tensor(G8, stem + ".trellis").to(DEV))
    suh_g = read_tensor(G8, stem + ".suh").to(torch.float16).to(DEV)
    svh_g = read_tensor(G8, stem + ".svh").to(torch.float16).to(DEV)
    assert t_g8.shape == t_stock.shape, (t_g8.shape, t_stock.shape)
    w_g8 = torch.empty((in_f, out_f), dtype=torch.half, device=DEV)
    ext.reconstruct_had_slice(w_g8, t_g8, suh_g, svh_g, K, True, False, 0)
    e_g8 = relerr(w_g8, src)
    e_st = relerr(w_stock, src)
    rr = relerr(w_g8, w_stock)
    worst_ratio = max(worst_ratio, e_g8 / e_st)
    print(f"{kind:4} {e_g8:18.5f} {e_st:12.5f} {rr:10.5f}")

ok = worst_ratio < 1.10
print(f"worst G8/stock relerr ratio: {worst_ratio:.4f} (threshold 1.10)")
print("VERDICT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
