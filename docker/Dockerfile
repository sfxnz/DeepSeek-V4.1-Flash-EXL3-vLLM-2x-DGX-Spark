# DeepSeek-V4.1-Flash EXL3 runtime for GB10 (sm_121).
# Official day-0 image plus Engram-on-disk, vllm-exl3, and ExLlamaV3 kernels.
FROM vllm/vllm-openai:deepseekv41-flash-0909@sha256:d84a123255b822fc22508635218000187221794f59c0694c33b0650d1e377d58

COPY patch /opt/dsv41-patch
RUN python3 /opt/dsv41-patch/apply_engram_disk.py \
      --engram /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/engram.py \
      --weights /usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py \
 && python3 /opt/dsv41-patch/apply_engram_prestage.py \
      --engram /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/engram.py \
      --model-state /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/nvidia/model_state.py

# Plugin source is cloned at build so the pin is in the RUN line, not a floating latest.
# Native p2b CUDA is parameterized by hidden/inter in 128-wide tiles. The
# host gate used to allow only 4096 x {1024,2048}; widen_p2b_shapes.py also
# allows V4.1 TP=2 5120x1152.
ARG VLLM_EXL3_REF=d3cfd394920360d69f820d2dc96f8292a9e10283
ARG EXLLAMAV3_REF=5be886578ec80324c2c715269387be2058724b6e
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git libcusparse-dev-13-0 libcusolver-dev-13-0 \
 && git clone https://github.com/turboderp-org/exllamav3.git /opt/exllamav3 \
 && git -C /opt/exllamav3 checkout "$EXLLAMAV3_REF" \
 && python3 /opt/dsv41-patch/patch_exllamav3_aarch64.py /opt/exllamav3/exllamav3/exllamav3_ext \
 && TORCH_CUDA_ARCH_LIST=12.1a python3 -m pip install --no-build-isolation --no-deps /opt/exllamav3 \
 && python3 -c "import torch, exllamav3, exllamav3_ext; assert hasattr(exllamav3_ext, 'exl3_moe'); print('exllamav3 ok')" \
 && git clone https://github.com/vcruz305/vllm-exl3.git /opt/vllm-exl3 \
 && git -C /opt/vllm-exl3 checkout "$VLLM_EXL3_REF" \
 && python3 /opt/dsv41-patch/fix_vllm_exl3_setup.py /opt/vllm-exl3/setup.py \
 && python3 /opt/dsv41-patch/widen_p2b_shapes.py /opt/vllm-exl3 \
 && python3 /opt/dsv41-patch/widen_p2b_mrow.py /opt/vllm-exl3 \
 && python3 /opt/dsv41-patch/widen_p2b_cfg1.py /opt/vllm-exl3 \
 && python3 /opt/dsv41-patch/widen_p2b_codebook.py /opt/vllm-exl3 \
 && python3 /opt/dsv41-patch/widen_p2b_fshift.py /opt/vllm-exl3 \
 && TORCH_CUDA_ARCH_LIST=12.1a python3 -m pip install --no-build-isolation --no-deps /opt/vllm-exl3 \
 && python3 -c "import torch, vllm_exl3, vllm_exl3_c; assert hasattr(vllm_exl3_c, 'p2b_fused_moe'); print('vllm_exl3 cuda ok')" \
 && rm -rf /opt/exllamav3 /opt/vllm-exl3 /var/lib/apt/lists/*

ENV VLLM_PLUGINS=vllm_exl3
ENV VLLM_EXL3_MOE_KERNEL=native
ENV DSV41_ENGRAM_DISK=1
COPY patch/sitecustomize.py /usr/lib/python3.12/sitecustomize.py
# E10: (16,64) tiles for decode-sized m in the flashinfer b12x dense GEMM
# (2x CTAs at n<=8192; no K-split exists in that kernel), and a boot log of
# the o_proj wo_a runtime dtype (pack stores F8; loader hands over bf16 ->
# torch.bmm -> sm_80 WMMA, the next decode target).
RUN python3 /opt/dsv41-patch/widen_b12x_smalls.py /usr/local/lib/python3.12/dist-packages/flashinfer/gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py \
 && python3 /opt/dsv41-patch/probe_wo_a.py /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/ops/o_proj.py \
 && python3 /opt/dsv41-patch/fix_o_proj_woa_fp8.py /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/ops/o_proj.py
# NOTE: the optional `pip install b12x` extra is deliberately NOT installed:
# with the package present, vLLM routes dense MXFP8 projections through the
# b12x package kernels and prose decode measured 5-9% slower than the
# flashinfer cute-dsl path at equal acceptance (2026-09-19 A/B, two 5-run
# batches per image). Revisit only with a proper kernel-level A/B.
# run.sh warns when IMAGE lacks this label (e.g. a stale local :latest).
LABEL dsv41.recipe.patches="shapes,mrow,cfg1,codebook,fshift,b12x_smalls,fix_o_proj_woa_fp8"
ENTRYPOINT ["vllm", "serve"]
