#!/usr/bin/env python3
"""Decode dense MXFP8 GEMM microbench: b12x vs deep_gemm fp8_gemm_nt vs split-K.

Shapes are per rank at TP=2 (K = in_features, N = out_features), from trace3:
qkv_a 5120x1792, wq_b 1280x16384, wo_b 4096x5120, shared gate_up 5120x2304,
shared down 1152x5120 at M=4 (DSpark k=3 verify), and draft main_proj
15360x5120 at M=3.

wq_b is benched as a diagnostic only and never promoted or counted in the
ms/step estimate: in the serve its activation quant is fused into the q/kv
RMSNorm (fused_q_kv_rmsnorm_quant), so it receives a QuantizedActivation and
the dense deep_gemm hook falls through to b12x. Its e2e rows are not like for
like with the serve either (the serve pays no separate b12x quant for it).

Backends:
  b12x  live path: mxfp8_e4m3_quantize (swizzled) + vllm mm_mxfp8 backend=auto
  dg    deep_gemm fp8_gemm_nt, recipe (1,1,32), weight scales packed once
        (the exact dg_mm/pack_weight_scale that docker/patch/dense_mxfp8_deepgemm.py
        runs in the serve). The deep_gemm host heuristic picks the in-kernel
        split-K factor; set DG_PRINT_CONFIGS=1 to log it.
  skS   explicit split-K S (diagnostic, qkv_a and shared gate_up only): the
        weight is re-laid out as [S, N, K/S] and run as the grouped fp8 einsum
        with fp32 partials, then summed. Not wireable without a second weight
        layout. It only answers whether more CTAs help these two shapes.

Each backend is timed two ways: e2e (bf16 activation in, including its
activation quant, as the serve runs it) and gemm (pre-quantized activation).
L2 is kept cold by rotating >= --rotate-mib of weight copies through one
CUDA graph (the capture also proves graph safety). Errors are measured against
a bf16 reference: bf16 x @ dequant(W)^T in fp32.

Run inside the image with the serve DOWN (exclusive GPU):
  docker run --rm --gpus all --ipc host --entrypoint python3 \
    -v $PWD:/w -e DG_PRINT_CONFIGS=1 dsv41-flash-exl3-sm121:canonical-e12 \
    /w/kernel_study/dense_gemm/bench_dense_gemm.py \
    --out /w/results/2026-09-24-review/dense-gemm/microbench.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docker/patch"))

# name, K, N, M list
SHAPES = [
    ("qkv_a", 5120, 1792, (4,)),
    ("wq_b", 1280, 16384, (4,)),
    ("wo_b", 4096, 5120, (4,)),
    ("shared_gate_up", 5120, 2304, (4,)),
    ("shared_down", 1152, 5120, (4,)),
    ("main_proj", 15360, 5120, (3,)),
]
SPLITK_SHAPES = {"qkv_a", "shared_gate_up"}
# trace3 calls/step. The shared expert runs beside the routing chain, so its
# savings may not reach step time (verify-dense-gemm-bw impact lens).
CALLS_PER_STEP = {
    "qkv_a": 45.8,
    "wq_b": 47.8,
    "wo_b": 47.8,
    "shared_gate_up": 42.8,
    "shared_down": 42.8,
    "main_proj": 1.0,
}
SHARED = {"shared_gate_up", "shared_down"}
# Shapes the serve hook cannot reach (see module docstring).
NOT_WIREABLE = {"wq_b": "fused_q_kv_rmsnorm_quant feeds a QuantizedActivation"}
WIN_GATE = 0.10  # promote a shape to the serve arm only at >= 10% e2e


def _err(out, ref) -> dict:
    d = (out.float() - ref).abs()
    return {
        "max_abs": float(d.max()),
        "max_rel": float(d.max() / ref.abs().max().clamp_min(1e-12)),
        "rel_fro": float((out.float() - ref).norm() / ref.norm().clamp_min(1e-12)),
    }


def _time_graph(torch, fn, nrot: int, reps: int, trials: int) -> list[float]:
    """us per call; one graph holds nrot calls, each on a different weight copy."""
    for i in range(nrot):
        fn(i)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(nrot):
            fn(i)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            graph.replay()
        end.record()
        end.synchronize()
        out.append(start.elapsed_time(end) * 1000.0 / (reps * nrot))
    del graph
    return out


def bench_shape(torch, name, K, N, M, args) -> list[dict]:
    from dense_mxfp8_deepgemm import dg_mm, pack_weight_scale
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
        per_token_group_quant_fp8_packed_for_deepgemm,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        dequant_mxfp8_to_bf16,
        mxfp8_e4m3_quantize,
        swizzle_mxfp8_scale,
    )
    from vllm.utils import flashinfer as vllm_flashinfer
    from vllm.utils.deep_gemm import fp8_einsum, fp8_gemm_nt

    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(1234 + K + N)
    w_bf16 = torch.randn(N, K, generator=gen, device=dev, dtype=torch.bfloat16) * 0.02
    w8, wsc = mxfp8_e4m3_quantize(w_bf16, is_sf_swizzled_layout=False)
    del w_bf16
    x = torch.randn(M, K, generator=gen, device=dev, dtype=torch.bfloat16)
    ref = x.float() @ dequant_mxfp8_to_bf16(w8, wsc).float().t()

    weight_bytes = N * K + N * (K // 32)
    nrot = max(2, min(64, math.ceil(args.rotate_mib * 2**20 / weight_bytes)))
    ws = [w8] + [w8.clone() for _ in range(nrot - 1)]
    wscs = [wsc] + [wsc.clone() for _ in range(nrot - 1)]
    rows: list[dict] = []

    def record(backend, e2e, gemm, extra=None):
        row = {"shape": name, "K": K, "N": N, "M": M, "backend": backend, "nrot": nrot}
        try:
            out = e2e(0)
            row.update(_err(out, ref))
            t_e2e = _time_graph(torch, e2e, nrot, args.reps, args.trials)
            t_gemm = _time_graph(torch, gemm, nrot, args.reps, args.trials)
            med = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
            row.update(
                us_e2e=med(t_e2e),
                us_e2e_min=min(t_e2e),
                us_gemm=med(t_gemm),
                us_gemm_min=min(t_gemm),
                gbps_e2e=weight_bytes / med(t_e2e) / 1e3,
                gbps_gemm=weight_bytes / med(t_gemm) / 1e3,
                capture_ok=True,
            )
        except Exception as exc:  # noqa: BLE001 - record and continue
            row.update(error=repr(exc)[:400], capture_ok=False)
        row.update(extra or {})
        rows.append(row)
        print(json.dumps(row), flush=True)

    # b12x (live)
    sw = [swizzle_mxfp8_scale(s, M=N, K=K).contiguous() for s in wscs]
    xq_b, xs_b = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)

    def b12x_e2e(i):
        q, s = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
        return vllm_flashinfer.mm_mxfp8(q, ws[i].t(), s, sw[i], out_dtype=torch.bfloat16, backend="auto")

    def b12x_gemm(i):
        return vllm_flashinfer.mm_mxfp8(xq_b, ws[i].t(), xs_b, sw[i], out_dtype=torch.bfloat16, backend="auto")

    record("b12x", b12x_e2e, b12x_gemm)
    del sw

    # deep_gemm fp8_gemm_nt, scales packed once
    try:
        sfs = [pack_weight_scale(w, s) for w, s in zip(ws, wscs)]
        xq_d, xs_d = per_token_group_quant_fp8_packed_for_deepgemm(x, group_size=32, use_ue8m0=True)
        out_d = torch.empty(M, N, device=dev, dtype=torch.bfloat16)

        def dg_e2e(i):
            return dg_mm(x, ws[i], sfs[i])

        def dg_gemm(i):
            fp8_gemm_nt((xq_d, xs_d), (ws[i], sfs[i]), out_d, recipe=(1, 1, 32), is_deep_gemm_e8m0_used=True)
            return out_d

        record("dg", dg_e2e, dg_gemm)
        del sfs
    except Exception as exc:  # noqa: BLE001
        rows.append({"shape": name, "K": K, "N": N, "M": M, "backend": "dg", "error": repr(exc)[:400]})

    # explicit split-K via grouped einsum (diagnostic)
    if name in SPLITK_SHAPES or args.splitk_all:
        for S in args.splitk:
            if K % (S * 128):
                continue
            try:
                kk = K // S
                wp, sp = [], []
                for w, s in zip(ws, wscs):
                    w3 = w.view(N, S, kk).permute(1, 0, 2).contiguous()
                    s3 = s.view(N, S, kk // 32).permute(1, 0, 2).contiguous()
                    w3, sf3 = deepgemm_post_process_fp8_weight_block(
                        wq=w3.view(S * N, kk),
                        ws=s3.view(S * N, kk // 32),
                        quant_block_shape=(1, 32),
                        use_e8m0=False,
                        is_bmm=True,
                        bmm_batch_size=S,
                    )
                    wp.append(w3)
                    sp.append(sf3)
                part = torch.empty(M, S, N, device=dev, dtype=torch.float32)
                xq_s, xs_s = per_token_group_quant_fp8_packed_for_deepgemm(x, group_size=32, use_ue8m0=True)
                a_pre = (xq_s.view(M, S, kk), xs_s.view(M, S, kk // 128))

                def sk_run(i, a):
                    fp8_einsum("bhr,hdr->bhd", a, (wp[i], sp[i]), part, recipe=(1, 1, 32))
                    return part.sum(dim=1).to(torch.bfloat16)

                def sk_e2e(i):
                    q, s = per_token_group_quant_fp8_packed_for_deepgemm(x, group_size=32, use_ue8m0=True)
                    return sk_run(i, (q.view(M, S, kk), s.view(M, S, kk // 128)))

                record(f"sk{S}", sk_e2e, lambda i: sk_run(i, a_pre), {"wireable": False})
                del wp, sp
            except Exception as exc:  # noqa: BLE001
                rows.append({"shape": name, "K": K, "N": N, "M": M, "backend": f"sk{S}", "error": repr(exc)[:400]})
    del ws, wscs
    torch.cuda.empty_cache()
    return rows


def summarize(rows: list[dict]) -> dict:
    by = {(r["shape"], r["M"], r["backend"]): r for r in rows}
    target_m = {name: ms[0] for name, _, _, ms in SHAPES}
    shapes_ok, per_shape = [], {}
    d_serial = d_shared = 0.0
    for name, K, N, _ in SHAPES:
        m = target_m[name]
        base = by.get((name, m, "b12x"), {})
        entry = {"M": m, "b12x_us": base.get("us_e2e")}
        for (s, mm, be), r in by.items():
            if s != name or mm != m or be == "b12x" or "us_e2e" not in r or "us_e2e" not in base:
                continue
            gain = base["us_e2e"] / r["us_e2e"] - 1.0
            err_ok = r["rel_fro"] <= 1.5 * base["rel_fro"] + 1e-3
            entry[be] = {"us": r["us_e2e"], "gain": gain, "err_ok": err_ok}
        dg = entry.get("dg")
        if name in NOT_WIREABLE:
            entry["not_wireable"] = NOT_WIREABLE[name]
        elif dg and dg["gain"] >= WIN_GATE and dg["err_ok"]:
            shapes_ok.append(f"{K}x{N}")
            dms = (entry["b12x_us"] - dg["us"]) * CALLS_PER_STEP[name] / 1000.0
            if name in SHARED:
                d_shared += dms
            else:
                d_serial += dms
        per_shape[name] = entry
    return {
        "win_gate": WIN_GATE,
        "per_shape": per_shape,
        "DSV41_DENSE_DG_SHAPES": ",".join(shapes_ok),
        "est_ms_per_step_serial": -d_serial,
        "est_ms_per_step_shared_maybe_hidden": -d_shared,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", default="", help="comma list of shape names")
    ap.add_argument("--ms", default="", help="override M list for every shape, e.g. 1,4,8")
    ap.add_argument("--splitk", default="2,4")
    ap.add_argument("--splitk-all", action="store_true", help="split-K on every shape")
    ap.add_argument("--rotate-mib", type=float, default=256.0)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--trials", type=int, default=5)
    args = ap.parse_args()
    args.splitk = [int(s) for s in args.splitk.split(",") if s]

    import torch

    only = {s for s in args.only.split(",") if s}
    ms_override = tuple(int(m) for m in args.ms.split(",") if m)
    rows: list[dict] = []
    for name, K, N, ms in SHAPES:
        if only and name not in only:
            continue
        for M in ms_override or ms:
            rows.extend(bench_shape(torch, name, K, N, M, args))
    summary = summarize(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print(f"DSV41_DENSE_DG_SHAPES={summary['DSV41_DENSE_DG_SHAPES']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
