#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(rel: str, name: str):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PackMetaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.meta = _load("tools/pack_meta.py", "pack_meta")

    def test_build_quantization_config_is_exl3(self) -> None:
        cfg = self.meta.build_quantization_config(bits=2)
        self.assertEqual(cfg["quant_method"], "exl3")
        self.assertEqual(cfg["bits"], 2)
        self.assertEqual(cfg["codebook"], "mcg")
        self.assertEqual(cfg["mtp_experts"], "source")
        self.assertEqual(cfg["mtp_experts_start_layer"], 40)
        self.assertNotIn("non_routed_dtype_policy", cfg)
        self.assertEqual(
            cfg["non_routed_quantization"]["quant_method"], "deepseek_v4_fp8"
        )
        self.assertEqual(cfg["non_routed_quantization"]["weight_block_size"], [32, 32])
        self.assertEqual(cfg["non_routed_quantization"]["scale_fmt"], "ue8m0")
        self.assertEqual(cfg["non_routed_quantization"]["expert_dtype"], "fp4")

    def test_apply_pack_config_keeps_nested_text_config(self) -> None:
        src = {
            "architectures": ["DeepseekV41ForCausalLM"],
            "model_type": "deepseek_v41",
            "text_config": {"hidden_size": 5120, "n_routed_experts": 384},
            "quantization_config": {"quant_method": "fp8"},
        }
        out = self.meta.apply_pack_config(src, self.meta.build_quantization_config())
        self.assertEqual(out["quantization_config"]["quant_method"], "exl3")
        self.assertEqual(out["text_config"]["n_routed_experts"], 384)
        self.assertEqual(out["architectures"], ["DeepseekV41ForCausalLM"])

    def test_routed_expert_names(self) -> None:
        self.assertTrue(self.meta.is_routed_expert_tensor("layers.6.ffn.experts.0.w1.weight"))
        self.assertTrue(self.meta.is_routed_expert_tensor("layers.6.ffn.experts.3.w2.scale"))
        self.assertFalse(self.meta.is_routed_expert_tensor("layers.6.ffn.shared_experts.w1.weight"))
        self.assertFalse(self.meta.is_routed_expert_tensor("layers.1.engram.embed.weight"))
        self.assertFalse(self.meta.is_routed_expert_tensor("mtp.layers.0.ffn.experts.0.w1.weight"))


class AssemblePackTests(unittest.TestCase):
    def test_assemble_pulls_spark2_shards_and_rebuilds_index(self) -> None:
        src = (ROOT / "tools/assemble_pack.sh").read_text()
        self.assertIn("rebuild_index.py", src)
        self.assertIn("seq 23 42", src)
        self.assertIn('quant_method")=="exl3"', src)
        self.assertIn('WORKER:-10.100.8.2', src)
        self.assertIn("rsync -rltD", src)
        self.assertNotIn("chgrp", src)


class QuantizeFastTests(unittest.TestCase):
    def test_default_convert_uses_fast_fallback(self) -> None:
        q = _load("tools/quantize_experts_exl3.py", "quantize_experts_exl3")
        src = (ROOT / "tools/quantize_experts_exl3.py").read_text()
        self.assertIn("def _quantize_fast", src)
        self.assertIn("skip_g_scale=True", src)
        self.assertIn("fast=not args.ldlq", src)
        self.assertTrue(q.convert_shards.__defaults__[-1] is True)


class RebuildIndexTests(unittest.TestCase):
    def test_rebuild_index_from_safetensors_headers(self) -> None:
        rebuild = _load("tools/rebuild_index.py", "rebuild_index")
        with tempfile.TemporaryDirectory() as d:
            dst = Path(d)
            hdr = {
                "layers.6.ffn.experts.0.w1.trellis": {
                    "dtype": "I16",
                    "shape": [2, 2, 32],
                    "data_offsets": [0, 256],
                },
                "__metadata__": {"format": "pt"},
            }
            hb = json.dumps(hdr).encode()
            hb += b" " * ((8 - len(hb) % 8) % 8)
            shard = dst / "model-00003-of-00048.safetensors"
            with shard.open("wb") as fh:
                fh.write(struct.pack("<Q", len(hb)))
                fh.write(hb)
                fh.write(b"\x00" * 256)
            wm = rebuild.rebuild(dst)
            self.assertEqual(
                wm["layers.6.ffn.experts.0.w1.trellis"],
                "model-00003-of-00048.safetensors",
            )
            idx = json.loads((dst / "model.safetensors.index.json").read_text())
            self.assertEqual(idx["weight_map"], wm)


class DownloadOfficialTests(unittest.TestCase):
    def test_enables_xet_and_pins_revision(self) -> None:
        src = (ROOT / "tools/download_official.py").read_text()
        self.assertIn('os.environ.pop("HF_HUB_DISABLE_XET", None)', src)
        self.assertIn("dba1be0a40aa45a94ad051997016db3960a90277", src)
        self.assertIn("deepseek-ai/DeepSeek-V4.1-Flash", src)
        self.assertIn("snapshot_download", src)


class QuantizeHelpersTests(unittest.TestCase):
    def setUp(self) -> None:
        self.q = _load("tools/quantize_experts_exl3.py", "quantize_experts_exl3")

    def test_quant_args_include_out_scales_and_mcg(self) -> None:
        args = self.q.quant_args_for(2, "cuda:0")
        self.assertEqual(args["K"], 2)
        self.assertTrue(args["mcg"])
        self.assertIsNone(args["apply_out_scales"])
        self.assertEqual(args["devices"], ["cuda:0"])

    def test_shard_needs_exl3(self) -> None:
        self.assertTrue(
            self.q.shard_needs_exl3(
                ["layers.6.ffn.experts.0.w1.weight", "layers.6.ffn.gate.weight"]
            )
        )
        self.assertFalse(
            self.q.shard_needs_exl3(
                ["layers.1.engram.embed.weight", "layers.1.engram.embed.scale"]
            )
        )

    def test_link_or_copy_hardlinks_same_fs(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src.bin"
            dst = Path(d) / "dst.bin"
            src.write_bytes(b"hello-engram")
            kind = self.q.link_or_copy(src, dst)
            self.assertEqual(kind, "link")
            self.assertEqual(dst.read_bytes(), b"hello-engram")
            self.assertEqual(src.stat().st_ino, dst.stat().st_ino)

    def test_link_or_copy_resolves_snapshot_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            blob = Path(d) / "blobs" / "abc"
            snap = Path(d) / "snapshots" / "rev"
            dest = Path(d) / "pack" / "out.safetensors"
            blob.parent.mkdir(parents=True)
            snap.mkdir(parents=True)
            blob.write_bytes(b"payload")
            link = snap / "model-00001-of-00048.safetensors"
            link.symlink_to(os.path.relpath(blob, snap))
            kind = self.q.link_or_copy(link, dest)
            self.assertEqual(kind, "link")
            self.assertFalse(dest.is_symlink())
            self.assertEqual(dest.read_bytes(), b"payload")
            self.assertEqual(dest.stat().st_ino, blob.stat().st_ino)


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


class Mxfp4Tests(unittest.TestCase):
    def setUp(self) -> None:
        if not _has_torch():
            self.skipTest("torch not installed on host")
        self.mx = _load("tools/mxfp4.py", "mxfp4")

    def test_unpack_zero(self) -> None:
        import torch

        packed = torch.zeros(2, 16, dtype=torch.uint8)
        got = self.mx.unpack_e2m1(packed)
        self.assertEqual(tuple(got.shape), (2, 32))
        self.assertTrue(torch.all(got == 0))

    def test_dequant_scale_one(self) -> None:
        import torch

        # ue8m0 exponent 127 => scale 1.0
        weight = torch.zeros(1, 16, dtype=torch.uint8)
        scale = torch.full((1, 1), 127, dtype=torch.uint8)
        out = self.mx.dequant_mxfp4(weight, scale, block=32)
        self.assertEqual(tuple(out.shape), (1, 32))
        self.assertTrue(torch.all(out.float() == 0))


class EngramDiskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _load("docker/patch/engram_disk.py", "engram_disk")

    def test_is_engram_embed_tensor(self) -> None:
        self.assertTrue(self.mod.is_engram_embed_tensor("layers.1.engram.embed.weight"))
        self.assertTrue(self.mod.is_engram_embed_tensor("layers.14.engram.embed.scale"))
        self.assertFalse(self.mod.is_engram_embed_tensor("layers.1.engram.wkv.weight"))

    def test_gather_dequant_matches_reference(self) -> None:
        if not _has_torch():
            self.skipTest("torch not installed on host")
        import torch

        r, dim, sb = 64, 256, 8
        torch.manual_seed(0)
        w = torch.randint(0, 256, (r, dim), dtype=torch.uint8)
        s = torch.randint(118, 136, (r, sb), dtype=torch.uint8)
        hdr = {
            "layers.1.engram.embed.weight": {
                "dtype": "F8_E4M3",
                "shape": [r, dim],
                "data_offsets": [0, r * dim],
            },
            "layers.1.engram.embed.scale": {
                "dtype": "F8_E8M0",
                "shape": [r, sb],
                "data_offsets": [r * dim, r * dim + r * sb],
            },
            "__metadata__": {"format": "pt"},
        }
        hb = json.dumps(hdr).encode()
        hb += b" " * ((8 - len(hb) % 8) % 8)
        with tempfile.TemporaryDirectory() as d:
            shard = Path(d) / "model-00001-of-00001.safetensors"
            with shard.open("wb") as fh:
                fh.write(struct.pack("<Q", len(hb)))
                fh.write(hb)
                fh.write(w.numpy().tobytes())
                fh.write(s.numpy().tobytes())
            (Path(d) / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "layers.1.engram.embed.weight": shard.name,
                            "layers.1.engram.embed.scale": shard.name,
                        }
                    }
                )
            )
            table = self.mod.DiskEngramTable(d, 1, dim, 32)
            idx = torch.tensor([0, 3, 7, 63], dtype=torch.int64)
            owned = torch.tensor([True, True, False, True])
            got = table.gather_dequant(idx, owned)
            vals = w[idx].view(torch.float8_e4m3fn).to(torch.float32).view(-1, sb, 32)
            scale = (s[idx].to(torch.int32) << 23).view(torch.float32)
            ref = (vals * scale[:, :, None]).reshape(-1, dim)
            ref[~owned] = 0
            ref = ref.to(torch.bfloat16)
            self.assertEqual(tuple(got.shape), tuple(ref.shape))
            self.assertTrue(torch.allclose(got.float(), ref.float(), equal_nan=True))


class ApplyEngramDiskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.apply = _load("docker/patch/apply_engram_disk.py", "apply_engram_disk")

    def test_disk_lookup_uses_global_file_rows(self) -> None:
        self.assertNotIn("rows - self.vocab_start_idx", self.apply.FWD_NEW)
        self.assertIn("file_rows", self.apply.FWD_NEW)
        self.assertIn("gather_dequant(file_rows, owned)", self.apply.FWD_NEW)

    def test_patch_engram_is_idempotent_and_inserts_disk(self) -> None:
        src = (
            self.apply.IMPORT_OLD
            + "pass\n"
            + self.apply.SIG_OLD
            + "\n        x = 1\n"
            + self.apply.ALLOC_OLD
            + "x)\n"
            + self.apply.LOOKUP_OLD
            + "\n        y = 1\n"
            + self.apply.FWD_OLD
            + "\n        return x\n"
            + self.apply.CTOR_OLD
            + "\n"
        )
        once = self.apply.patch_engram(src)
        self.assertIn("DiskEngramTable", once)
        self.assertIn("_disk_lookup", once)
        self.assertIn("engram_disk_enabled", once)
        self.assertIn("file_rows", once)
        self.assertNotIn("rows - self.vocab_start_idx", once)
        twice = self.apply.patch_engram(once)
        self.assertEqual(once, twice)

    def test_patch_weight_utils_skips_embed(self) -> None:
        src = "\nimport os\n" + self.apply.WU_OLD
        out = self.apply.patch_weight_utils(src)
        self.assertIn("engram.embed.weight", out)
        self.assertIn("DSV41_ENGRAM_DISK", out)
        self.assertEqual(out, self.apply.patch_weight_utils(out))


class FixVllmExl3SetupTests(unittest.TestCase):
    def test_rewrites_absolute_cuda_sources(self) -> None:
        fix = _load("docker/patch/fix_vllm_exl3_setup.py", "fix_vllm_exl3_setup")
        sample = '''foo
                sources=[
                    str(ROOT / "csrc" / "bindings.cpp"),
                    str(ROOT / "csrc" / "exl3_gemv.cu"),
                    str(ROOT / "csrc" / "p2b_batched.cu"),
                    str(ROOT / "csrc" / "p2b_moe.cu"),
                    str(ROOT / "csrc" / "exl3_gemm.cu"),
                    str(ROOT / "csrc" / "exl3_fat_gemm.cu"),
                ],
bar
'''
        out = fix.patch(sample)
        self.assertIn('"csrc/bindings.cpp"', out)
        self.assertNotIn('str(ROOT / "csrc"', out)


if __name__ == "__main__":
    unittest.main()
