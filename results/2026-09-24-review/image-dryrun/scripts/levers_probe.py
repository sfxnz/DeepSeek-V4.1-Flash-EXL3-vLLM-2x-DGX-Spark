"""In-image CPU probe: do the B-scenario levers reach the objects the serve uses?"""
import inspect
import os

print("env", {k: os.environ.get(k) for k in (
    "DSV41_ENGRAM_WILLNEED", "DSV41_ENGRAM_GATHER_V2_MAX_ROWS", "DSV41_MHC_DECODE_SPLITS",
    "DSV41_DSPARK_SPARSE_MARKOV", "DSV41_WOA_PREPACK")})

import vllm.models.deepseek_v4_1.common.engram_disk as ed
print("gv2 WILLNEED", ed._ENG_GV2_WILLNEED, "MIN", ed._ENG_GV2_WILLNEED_MIN, "MAX_ROWS", ed._ENG_GV2_MAX_ROWS,
      "GATHER_V2", ed._ENG_GATHER_V2)
print("gv2 has _gv2_willneed", hasattr(ed.DiskEngramTable, "_gv2_willneed"))

from vllm.model_executor.kernels.mhc import warmup as wu
f = wu.compute_mhc_pre_num_splits
print("mhc patched fn", f.__module__, f.__qualname__)
for k, t in ((16384, 1), (16384, 4), (16384, 6), (16384, 64), (1024, 4)):
    print(f"  splits(K={k}, T={t}) =", f(k, t))
tl_src = inspect.getsource(__import__("vllm.model_executor.kernels.mhc.tilelang", fromlist=["x"]))
print("tilelang imports compute_mhc_pre_num_splits at call time:",
      "        compute_mhc_pre_num_splits," in tl_src)

from vllm.models.deepseek_v4_1.nvidia.dspark import DSparkDeepseekV4ForCausalLM
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
print("DeepSeek apply_markov_bias_gathered:", hasattr(DSparkDeepseekV4ForCausalLM, "apply_markov_bias_gathered"))
print("DSparkSpeculator.__init__ wrapped by:", DSparkSpeculator.__init__.__module__)
print("DSparkSpeculator has _sample_sequential_topk:", hasattr(DSparkSpeculator, "_sample_sequential_topk"))

import torch
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
# Gathered bias == dense bias on the candidates (CPU, tiny, no quant).
V, r, B, k = 64, 8, 2, 5
w2 = torch.randn(V, r)
emb = torch.randn(B, r)
base = torch.randn(B, V)
vals, idx = base.topk(k, dim=-1)
head = DSparkMarkovHead.__new__(DSparkMarkovHead)
torch.nn.Module.__init__(head)
head.markov_w2 = torch.nn.Module()
head.markov_w2.weight = w2
out = torch.full((B, V), float("-inf"))
head.apply_bias_gathered(emb, out, vals, idx, 1.0)
dense = base + emb @ w2.T
print("gathered == dense on top-k:", torch.allclose(out.gather(1, idx), dense.gather(1, idx), atol=1e-5))
