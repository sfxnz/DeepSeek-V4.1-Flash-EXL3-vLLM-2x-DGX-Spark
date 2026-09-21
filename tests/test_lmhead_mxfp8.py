"""Offline chain test for the lm_head MXFP8 patch: install() swap, weight
loading through a row-sharded loader, process_weights_after_loading swizzle,
and apply() numerics — all against a minimal fake vllm package on CPU.

Also drives tools/quantize_lmhead_mxfp8.py end-to-end on a synthetic
two-shard snapshot COPY (head.weight bf16 -> lm_head.weight/.weight_scale),
verifying index rewrite, tensor values, and idempotent refusal.

Run: python3 -m unittest tests.test_lmhead_mxfp8 -v
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
from torch.nn.parameter import Parameter

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "docker/patch/lmhead_mxfp8.py"
REENC = ROOT / "tools/quantize_lmhead_mxfp8.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _swizzle(sf: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Verbatim copy of vllm swizzle_mxfp8_scale (mxfp8_utils.py) for CPU."""
    factor = 128
    num_m_tiles = (M + 127) // 128
    num_k_tiles = (K + factor - 1) // factor
    m_padded = num_m_tiles * 128
    k_scale_padded = num_k_tiles * 4
    scale_cols = K // 32
    sf_padded = torch.zeros((m_padded, k_scale_padded), dtype=sf.dtype)
    sf_padded[:M, :scale_cols] = sf
    return (
        sf_padded.view(num_m_tiles, 4, 32, num_k_tiles, 4)
        .transpose(1, 3)
        .contiguous()
        .view(-1)
    )


def _fake_vllm() -> types.ModuleType:
    """Minimal vllm package exposing the symbols lmhead_mxfp8 imports."""
    vllm = types.ModuleType("vllm")

    pkg_me = types.ModuleType("vllm.model_executor")
    pkg_layers = types.ModuleType("vllm.model_executor.layers")
    pkg_q = types.ModuleType("vllm.model_executor.layers.quantization")

    base_config = types.ModuleType(
        "vllm.model_executor.layers.quantization.base_config"
    )

    class QuantizeMethodBase:
        def create_weights(self, *a, **k):  # pragma: no cover
            raise NotImplementedError

        def process_weights_after_loading(self, layer):  # pragma: no cover
            raise NotImplementedError

        def apply(self, *a, **k):  # pragma: no cover
            raise NotImplementedError

    base_config.QuantizeMethodBase = QuantizeMethodBase

    utils_mod = types.ModuleType("vllm.model_executor.utils")

    def set_weight_attrs(weight, attrs):
        for k, v in attrs.items():
            setattr(weight, k, v)

    utils_mod.set_weight_attrs = set_weight_attrs

    mxfp8_utils = types.ModuleType(
        "vllm.model_executor.layers.quantization.utils.mxfp8_utils"
    )
    mxfp8_utils.swizzle_mxfp8_scale = _swizzle

    def mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=False, alignment=0):
        # reference torch impl (mxfp8_utils._mxfp8_e4m3_quantize_torch)
        M, K = x.shape
        xb = x.view(M, K // 32, 32)
        amax = xb.abs().amax(dim=-1).clamp(min=torch.finfo(torch.float32).tiny)
        biased = (torch.ceil(torch.log2(amax / 448.0)) + 127.0).clamp(0, 254)
        scales = biased.to(torch.uint8).view(M, -1)
        xq = (xb / torch.exp2(biased - 127.0).unsqueeze(-1)).view(M, K).to(
            torch.float8_e4m3fn
        )
        if is_sf_swizzled_layout:
            scales = _swizzle(scales, M, K)
        return xq, scales

    mxfp8_utils.mxfp8_e4m3_quantize = mxfp8_e4m3_quantize

    fi_mod = types.ModuleType("vllm.utils.flashinfer")

    class _FI:
        @staticmethod
        def mm_mxfp8(a, b, a_scale, b_scale, out_dtype, backend):
            # Emulate: dequant both sides, fp32 GEMM. b is weight.t() of the
            # e4m3 values; b_scale is the swizzled [N,K//32] scales.
            af = a.to(torch.float32)
            af = af.view(af.shape[0], af.shape[1] // 32, 32) * torch.exp2(
                a_scale.to(torch.float32).view(af.shape[0], -1, 1) - 127.0
            )
            af = af.view(*a.shape)
            bf = b.t().to(torch.float32)  # [N, K] values
            bsc = b_scale.t()  # un-swizzle is identity here: we pass the
            # row-major scales through a side channel in the test below by
            # pre-unswizzling — see apply test note.
            raise AssertionError("unused when row-major scales are passed")

    fi_mod.vllm_flashinfer = _FI
    vllm.__dict__["utils"] = types.ModuleType("vllm.utils")
    vllm.utils.flashinfer = fi_mod

    nvidia = types.ModuleType("vllm.model_executor.models.deepseek_v4_1.nvidia")
    model_mod = types.ModuleType(
        "vllm.model_executor.models.deepseek_v4_1.nvidia.model"
    )

    class DeepseekV41LLMForCausalLM(torch.nn.Module):
        def __init__(self, *, vllm_config, prefix: str = ""):
            super().__init__()
            if getattr(vllm_config, "pp_last", True):
                self.lm_head = _FakeLMHead(
                    vllm_config.vocab_size, vllm_config.hidden_size
                )
            else:
                self.lm_head = None  # PPMissingLayer stand-in

    model_mod.DeepseekV41LLMForCausalLM = DeepseekV41LLMForCausalLM

    class _FakeLMHead(torch.nn.Module):
        def __init__(self, V, H):
            super().__init__()
            self.weight = Parameter(
                torch.empty(V, H, dtype=torch.bfloat16), requires_grad=False
            )
            # row-sharded loader replica (vocab_parallel_embedding.py:459+)
            self.tp_rank = 0
            self.tp_size = 1

            def weight_loader(param, loaded_weight):
                sharded = self._shard(loaded_weight)
                param.data.copy_(sharded)

            self.weight_loader = weight_loader
            set_weight_attrs(
                self.weight, {"input_dim": 1, "output_dim": 0}
            )

        def _shard(self, w):
            per = w.shape[0] // self.tp_size
            return w[self.tp_rank * per : (self.tp_rank + 1) * per]

    def register(parent, name, mod):
        sys.modules[f"{parent.__name__}.{name}"] = mod
        parent.__dict__[name] = mod
        return mod

    register(vllm, "model_executor", pkg_me)
    register(pkg_me, "layers", pkg_layers)
    register(pkg_layers, "quantization", pkg_q)
    register(pkg_q, "base_config", base_config)
    register(pkg_me, "utils", utils_mod)
    qutils = types.ModuleType("vllm.model_executor.layers.quantization.utils")
    register(pkg_q, "utils", qutils)
    register(qutils, "mxfp8_utils", mxfp8_utils)
    vutils = types.ModuleType("vllm.utils")
    sys.modules["vllm.utils"] = vutils
    register(vutils, "flashinfer", fi_mod)
    models = types.ModuleType("vllm.model_executor.models")
    ds41 = types.ModuleType("vllm.model_executor.models.deepseek_v4_1")
    register(pkg_me, "models", models)
    register(models, "deepseek_v4_1", ds41)
    register(ds41, "nvidia", nvidia)
    register(nvidia, "model", model_mod)
    sys.modules["vllm"] = vllm
    return vllm, model_mod.DeepseekV41LLMForCausalLM, mxfp8_utils


class _MMRecorder:
    """Capture mm_mxfp8 args; emulate the GEMM with row-major scales."""


class TestLMHeadMxfp8(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for mod in list(sys.modules):
            if mod.startswith("vllm"):
                del sys.modules[mod]
        cls.vllm, cls.Cls, cls.mxfp8_utils = _fake_vllm()
        cls.mod = _load(PATCH, "lmhead_mxfp8_test")

    def test_env_gate(self):
        self.assertFalse(self.mod.enabled_from_env({}))
        self.assertTrue(self.mod.enabled_from_env({"DSV41_LMHEAD_MXFP8": "1"}))

    def test_index_probe(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"head.weight": "model-1.safetensors"}})
            )
            self.assertFalse(self.mod.snapshot_has_lmhead_mxfp8(td))
            wm = {
                "lm_head.weight": "model-1.safetensors",
                "lm_head.weight_scale": "model-1.safetensors",
            }
            (td / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": wm})
            )
            self.assertTrue(self.mod.snapshot_has_lmhead_mxfp8(td))
            self.assertFalse(self.mod.snapshot_has_lmhead_mxfp8(td / "missing"))

    def test_swap_and_apply(self):
        V, H = 512, 512
        cfg = types.SimpleNamespace(
            vocab_size=V,
            hidden_size=H,
            head_dtype=None,
            model=types.SimpleNamespace(path=None),
        )
        vllm_config = types.SimpleNamespace(
            model_config=cfg, pp_last=True, vocab_size=V, hidden_size=H
        )
        # Route the snapshot probe at a prepared dir.
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            wm = {
                "lm_head.weight": "model-1.safetensors",
                "lm_head.weight_scale": "model-1.safetensors",
            }
            (td / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": wm})
            )
            cfg.model = types.SimpleNamespace(path=str(td))

            installed = self.mod.install()
            self.assertTrue(installed)
            model = self.Cls(vllm_config=vllm_config)
            lm = model.lm_head
            self.assertEqual(lm.weight.dtype, torch.float8_e4m3fn)
            self.assertEqual(lm.weight_scale.dtype, torch.uint8)
            self.assertEqual(tuple(lm.weight.shape), (V, H))
            self.assertEqual(tuple(lm.weight_scale.shape), (V, H // 32))
            self.assertNotIsInstance(
                lm.quant_method, type(None)
            )

            # Load a bf16-derived quantized weight through the swapped
            # params' weight_loader (sharded copy path).
            torch.manual_seed(7)
            w_bf16 = torch.randn(V, H).to(torch.bfloat16)
            values, scales = self.mod.quantize_weight_mxfp8(w_bf16)
            lm.weight_loader(lm.weight, values)
            lm.weight_loader(lm.weight_scale, scales)

            lm.quant_method.process_weights_after_loading(lm)
            # swizzled scale is a flat vector of padded tiles
            n_m = (V + 127) // 128
            self.assertEqual(
                lm.weight_scale.numel(), n_m * 128 * ((H // 32 + 3) // 4) * 4
            )

            # apply(): swap in a recording mm_mxfp8 that emulates with the
            # ORIGINAL row-major scales (we recover them by un-swizzling).
            x = torch.randn(6, H).to(torch.bfloat16)

            def fake_mm(a, b, a_scale, b_scale, out_dtype, backend):
                self.assertEqual(backend, "auto")

                def unswizzle(flat: torch.Tensor, M: int, K: int):
                    n_m = (M + 127) // 128
                    n_k = (K // 32 + 3) // 4
                    sf = flat.view(n_m, 4, 32, n_k, 4).transpose(1, 3)
                    return sf.reshape(n_m * 128, -1)[:M, : K // 32]

                # un-swizzle b_scale back to [N, K//32]
                N, K = b.shape[1], b.shape[0]
                sf = unswizzle(b_scale, N, K)
                asf = unswizzle(a_scale, a.shape[0], a.shape[1])
                af = a.to(torch.float32)
                af = af.view(a.shape[0], -1, 32) * torch.exp2(
                    asf.to(torch.float32).unsqueeze(-1) - 127.0
                )
                af = af.view(*a.shape)
                bf = b.t().to(torch.float32)
                bf = bf.view(N, -1, 32) * torch.exp2(
                    sf.to(torch.float32).unsqueeze(-1) - 127.0
                )
                bf = bf.view(N, K)
                return (af @ bf.t()).to(out_dtype)

            fi = sys.modules["vllm.utils.flashinfer"]
            fi.vllm_flashinfer.mm_mxfp8 = staticmethod(fake_mm)

            out = lm.quant_method.apply(lm, x)
            self.assertEqual(out.dtype, torch.bfloat16)
            self.assertEqual(tuple(out.shape), (6, V))

            # Numerics: compare against the bf16 GEMM (normwise).
            ref = x.to(torch.float32) @ w_bf16.to(torch.float32).t()
            err = (out.to(torch.float32) - ref).norm() / ref.norm()
            self.assertLess(float(err), 0.06)

    def test_self_disarm_without_tensor(self):
        V, H = 128, 128
        cfg = types.SimpleNamespace(
            vocab_size=V,
            hidden_size=H,
            head_dtype=None,
            model=types.SimpleNamespace(path=None),
        )
        vllm_config = types.SimpleNamespace(
            model_config=cfg, pp_last=True, vocab_size=V, hidden_size=H
        )
        # install() is already applied; construct with a snapshot dir that
        # lacks the marker keys -> weight stays bf16.
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"head.weight": "x.safetensors"}})
            )
            cfg.model = types.SimpleNamespace(path=td)
            model = self.Cls(vllm_config=vllm_config)
            self.assertEqual(model.lm_head.weight.dtype, torch.bfloat16)

    def test_reencode_script(self):
        mod = _load(REENC, "quantize_lmhead_mxfp8_test")
        from safetensors.torch import load_file, save_file

        with tempfile.TemporaryDirectory() as td:
            snap = Path(td) / "snap"
            snap.mkdir()
            V, H = 256, 512  # small stand-in; math identical
            head = torch.randn(V, H).to(torch.bfloat16)
            save_file(
                {"head.weight": head, "other.weight": torch.zeros(4)},
                str(snap / "model-00043-of-00048.safetensors"),
            )
            save_file(
                {"norm.weight": torch.zeros(8)},
                str(snap / "model-00044-of-00048.safetensors"),
            )
            (snap / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "head.weight": "model-00043-of-00048.safetensors",
                            "other.weight": "model-00043-of-00048.safetensors",
                            "norm.weight": "model-00044-of-00048.safetensors",
                        }
                    }
                )
            )
            # Dry run.
            import contextlib, io

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = _run_main(mod, ["--snapshot", str(snap), "--dry-run"])
            self.assertEqual(rc, 0)
            wm = json.loads(
                (snap / "model.safetensors.index.json").read_text()
            )["weight_map"]
            self.assertIn("head.weight", wm)
            # Real run.
            with contextlib.redirect_stdout(buf):
                rc = _run_main(mod, ["--snapshot", str(snap)])
            self.assertEqual(rc, 0)
            t = load_file(str(snap / "model-00043-of-00048.safetensors"))
            self.assertNotIn("head.weight", t)
            self.assertIn("lm_head.weight", t)
            self.assertEqual(t["lm_head.weight"].dtype, torch.float8_e4m3fn)
            self.assertEqual(t["lm_head.weight_scale"].dtype, torch.uint8)
            self.assertEqual(tuple(t["lm_head.weight"].shape), (V, H))
            self.assertEqual(tuple(t["lm_head.weight_scale"].shape), (V, H // 32))
            self.assertTrue(torch.equal(t["other.weight"], torch.zeros(4)))
            values, scales = self.mod.quantize_weight_mxfp8(head)
            self.assertTrue(torch.equal(t["lm_head.weight"], values))
            self.assertTrue(torch.equal(t["lm_head.weight_scale"], scales))
            wm = json.loads(
                (snap / "model.safetensors.index.json").read_text()
            )["weight_map"]
            self.assertEqual(
                wm["lm_head.weight_scale"], "model-00043-of-00048.safetensors"
            )
            self.assertEqual(wm["norm.weight"], "model-00044-of-00048.safetensors")
            # Idempotent refusal.
            with self.assertRaises(SystemExit):
                _run_main(mod, ["--snapshot", str(snap)])
            # Patch side sees the marker.
            self.assertTrue(self.mod.snapshot_has_lmhead_mxfp8(snap))


def _run_main(mod, argv):
    sys.argv = [mod.__name__] + argv
    return mod.main()


if __name__ == "__main__":
    unittest.main()
