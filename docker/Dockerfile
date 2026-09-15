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
 && TORCH_CUDA_ARCH_LIST=12.1a python3 -m pip install --no-build-isolation --no-deps /opt/vllm-exl3 \
 && python3 -c "import torch, vllm_exl3, vllm_exl3_c; assert hasattr(vllm_exl3_c, 'p2b_fused_moe'); print('vllm_exl3 cuda ok')" \
 && rm -rf /opt/exllamav3 /opt/vllm-exl3 /var/lib/apt/lists/*

ENV VLLM_PLUGINS=vllm_exl3
ENV VLLM_EXL3_MOE_KERNEL=native
ENV DSV41_ENGRAM_DISK=1
COPY patch/sitecustomize.py /usr/lib/python3.12/sitecustomize.py
# Native SM120 grouped MXFP8 BMM / WO projection (CuTe DSL). The extra is a
# py3-none-any wheel; cutlass-dsl is already in the FlashInfer b12x Q/O path.
RUN python3 -m pip install --no-deps "b12x==1.3.0" \
 && python3 -c "import b12x; print('b12x', getattr(b12x, '__version__', 'ok'))"
ENTRYPOINT ["vllm", "serve"]
