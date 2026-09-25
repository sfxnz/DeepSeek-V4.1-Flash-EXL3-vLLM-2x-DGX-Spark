import sys, time, torch
sys.path.insert(0, "/repo/tools")
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3_lib.quantize import (finalize_capture_H, pack_trellis, quantize_tiles, regularize, tensor_core_perm)
from safetensors.torch import safe_open
from quantize_experts_exl3 import _dequant_t, _meta_h, quant_args_for
from mxfp4 import dequant_mxfp4
SRC = "/cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277/model-00003-of-00048.safetensors"
dev = torch.device("cuda:0")
print("scratch", ext.quantize_tiles_scratch(0, 2, True, False, 256), "SMs", torch.cuda.get_device_properties(0).multi_processor_count)
qa = quant_args_for(2, "cuda:0", "mcg")
perm = tensor_core_perm(dev)
def sync(): torch.cuda.synchronize()
with safe_open(SRC, framework="pt") as fh:
    for t in ("layers.0.ffn.experts.5.w1", "layers.0.ffn.experts.5.w2"):
        t0 = time.time(); w8 = fh.get_tensor(t + ".weight"); s8 = fh.get_tensor(t + ".scale"); t_read = time.time() - t0
        t0 = time.time(); wc = _dequant_t(w8, s8, "cuda:0"); t_cpu = time.time() - t0
        sync(); t0 = time.time(); wg = dequant_mxfp4(w8.to(dev), s8.to(dev)).float().t().contiguous(); sync(); t_gpu = time.time() - t0
        print(t, tuple(wc.shape), f"read {t_read:.3f}s cpu_dequant {t_cpu:.3f}s gpu_dequant {t_gpu:.3f}s equal {torch.equal(wc.to(dev), wg)}")
        torch.manual_seed(1)
        wf = wg
        h = _meta_h(wf.shape[0], "cuda:0", {})
        qf, _H, _L, su, Hd = finalize_capture_H(h, qa, False)
        sv = (torch.randn(wf.shape[1], device=dev).sign() + 1e-5).sign().float().unsqueeze(0)
        sync(); t0 = time.time()
        _a, wr, _g, su, sv = regularize(wf, su.to(dev), sv, dict(qa), False, Hd, None, skip_g_scale=True, q_fallback=qf)
        sync(); print(f"  regularize {time.time()-t0:.3f}s")
        k, n = wr.shape
        tiles = wr.reshape(k//16, 16, n//16, 16).permute(0, 2, 1, 3).reshape(-1, 256).contiguous()[:, perm]
        ref = None
        for chunk in (256, 1024, 4608, tiles.shape[0], 256):
            sync(); t0 = time.time()
            idx = [quantize_tiles(tiles[i:i+chunk], qa)[1] for i in range(0, tiles.shape[0], chunk)]
            qi = torch.cat(idx, 0); sync(); dt = time.time() - t0
            if ref is None: ref = qi
            print(f"  chunk {chunk:6d} {dt:.3f}s  bitwise_eq_256 {torch.equal(qi, ref)}")
        sync(); t0 = time.time(); tr = pack_trellis(qi.view(k//16, n//16, 256), qa); sync(); print(f"  pack {time.time()-t0:.3f}s", tuple(tr.shape), tr.dtype)
