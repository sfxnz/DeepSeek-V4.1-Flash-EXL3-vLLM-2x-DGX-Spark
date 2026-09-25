#!/usr/bin/env python3
"""Round-3 lever probe for the CPU dry-run (no GPU). sitecustomize (run.sh's mount) already ran at
this interpreter's start with the arm's env; report, per lever, whether its install step wrapped
its target (the flag attribute the patch sets), import every round-3 patch module, and read the
kernel markers in the image's vllm_exl3_c. Writes one JSON object to argv[1].

A wrap flag is expected True with the lever's env on and False with it off; dryrun.py compares
against the arm's env. Nothing here touches CUDA."""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys

MODULES = (
    "widen_p2b_dataflow", "dense_gemv", "mhc_det", "mhc_det_rt", "mhc_det_overlap",
    "engram_native_stage", "engram_early_hash", "attn_t2r_dedup", "moe_prep_fused",
    "candidate_mask_bounded", "indexer_wp_gemv", "swa_meta_fused", "engram_wkv_tp",
    "nccl_eager_twin", "pm_qos", "ar_l2_prefetch", "l2pf_kernel", "decode_levers",
)

# lever -> (env gate {name: value}, callable returning the wrapped state as bool)
def _attr(mod: str, path: str, flag: str):
    def get() -> bool:
        obj = importlib.import_module(mod)
        for part in path.split("."):
            obj = getattr(obj, part)
        return bool(getattr(obj, flag, False))
    return get


def _module_fn_is(mod: str, name: str, owner: str):
    def get() -> bool:
        return getattr(importlib.import_module(mod), name).__module__ == owner
    return get


def _file_has(rel: str, needle: str):
    def get() -> bool:
        spec = importlib.util.find_spec("vllm")
        root = os.path.dirname(spec.origin)
        with open(os.path.join(root, rel)) as fh:
            return needle in fh.read()
    return get


def _pm_qos_held() -> bool:
    import pm_qos

    return bool(pm_qos._held)


def _mixing_off() -> bool:
    return os.environ.get("NCCL_GRAPH_MIXING_SUPPORT") == "0"


LEVERS = {
    "dense_gemv/flashinfer.py": ({"DSV41_DENSE_GEMV": "1"},
                                 _file_has("model_executor/kernels/linear/mxfp8/flashinfer.py", "_dsv41_gemv_")),
    "dense_gemv/o_proj.py": ({"DSV41_DENSE_GEMV": "1"},
                             _file_has("models/deepseek_v4/nvidia/ops/o_proj.py", "_dsv41_gemv_")),
    "mhc_det/model.mhc_post": ({"DSV41_MHC_DET_SPLITS": "16"},
                               _module_fn_is("vllm.models.deepseek_v4_1.nvidia.model", "mhc_post_tilelang", "mhc_det")),
    "mhc_det/dspark.mhc_post": ({"DSV41_MHC_DET_SPLITS": "16"},
                                _module_fn_is("vllm.models.deepseek_v4_1.nvidia.dspark", "mhc_post_tilelang", "mhc_det")),
    "mhc_det_overlap/all_reduce": ({"DSV41_MHC_DET_SPLITS": "16", "DSV41_MHC_DET_OVERLAP": "1"},
                                   _attr("vllm.distributed.parallel_state", "GroupCoordinator.all_reduce", "_dsv41_mhc_ovl")),
    "engram_native_stage": ({"DSV41_ENGRAM_NATIVE_STAGE": "1"},
                            _attr("vllm.models.deepseek_v4_1.common.engram", "EngramDiskStager.stage", "_dsv41_native")),
    "attn_t2r_dedup": ({"DSV41_ATTN_T2R_DEDUP": "1"},
                       _attr("vllm.v1.attention.backend", "CommonAttentionMetadata.token_to_req_indices", "_dsv41_t2r_dedup")),
    "moe_prep_fused": ({"DSV41_MOE_PREP_FUSED": "1"},
                       _attr("vllm_exl3.exl3", "_apply_native_fused_moe", "_dsv41_moe_prep")),
    "candidate_mask_bounded": ({"DSV41_CANDIDATE_MASK_BOUNDED": "1"},
                               _attr("vllm.model_executor.layers.sparse_attn_indexer", "_apply_candidate_mask", "_dsv41_bounded")),
    "indexer_wp_gemv": ({"DSV41_INDEXER_WP_GEMV": "1"},
                        _attr("vllm.models.deepseek_v4_1.attention", "DeepseekV4Indexer.__init__", "_dsv41_wp_gemv")),
    "swa_meta_fused": ({"DSV41_SWA_META_FUSED": "1"},
                       _attr("vllm.v1.attention.backends.mla.sparse_swa", "DeepseekSparseSWAMetadataBuilder.build", "_dsv41_swa_fused")),
    "engram_wkv_tp": ({"DSV41_ENGRAM_WKV_TP": "1"},
                      _attr("vllm.models.deepseek_v4_1.common.engram", "Engram.__init__", "_dsv41_wkv_tp")),
    "ar_l2_prefetch": ({"DSV41_AR_L2_PREFETCH": "1"},
                       _attr("vllm.models.deepseek_v4_1.nvidia.model", "DeepseekV4Model.forward", "_dsv41_l2pf")),
    "nccl_eager_twin/init": ({"DSV41_NCCL_EAGER_TWIN": "1"},
                             _attr("vllm.distributed.device_communicators.cuda_communicator",
                                   "CudaCommunicator.__init__", "_dsv41_eager_twin")),
    "nccl_eager_twin/mixing_off": ({"DSV41_NCCL_EAGER_TWIN": "1"}, _mixing_off),
    "pm_qos/held": ({"DSV41_PM_QOS_US": "20"}, _pm_qos_held),
}

SO_NEEDLES = ("DSV41_P2B_SRC_SORT", "DSV41_P2B_COOP", "p2b coop dataflow kernel engaged",
              "p2b coop dataflow lever is OFF", "p2b_coop_df_kernel", "p2b_moe_batched_kernel")


def main() -> int:
    out: dict = {"levers": {}, "modules": {}, "so": {}}
    for name, (gate, get) in LEVERS.items():
        expected = all(os.environ.get(k, "") == v for k, v in gate.items())
        try:
            got = get()
            out["levers"][name] = {"expected": expected, "wrapped": got, "ok": got == expected}
        except Exception as exc:  # noqa: BLE001 - report, never raise
            out["levers"][name] = {"expected": expected, "error": repr(exc)[:300], "ok": False}
    for m in MODULES:
        try:
            importlib.import_module(m)
            out["modules"][m] = "ok"
        except Exception as exc:  # noqa: BLE001
            out["modules"][m] = "FAIL " + repr(exc)[:300]
    try:
        so = importlib.util.find_spec("vllm_exl3_c").origin
        data = open(so, "rb").read()
        out["so"] = {"path": so, "bytes": len(data), **{n: data.count(n.encode()) for n in SO_NEEDLES}}
    except Exception as exc:  # noqa: BLE001
        out["so"] = {"error": repr(exc)[:300]}
    out["summary"] = {
        "levers_ok": sum(v["ok"] for v in out["levers"].values()),
        "levers": len(out["levers"]),
        "module_import_failures": sum(not v.startswith("ok") for v in out["modules"].values()),
    }
    with open(sys.argv[1], "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
