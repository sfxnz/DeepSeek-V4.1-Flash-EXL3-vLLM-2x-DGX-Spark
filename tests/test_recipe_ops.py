#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import re
import socket
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINNED_FROM = (
    "vllm/vllm-openai:deepseekv41-flash-0909@"
    "sha256:d84a123255b822fc22508635218000187221794f59c0694c33b0650d1e377d58"
)


def _read(rel: str) -> str:
    return (ROOT / rel).read_text()


def _env(**extra: str) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("FORCE_UNSAFE_CTX", None)
    env.pop("FORCE_UNSAFE_ENGRAM", None)
    env.pop("FORCE_UNSAFE_QUANT", None)
    env.pop("VALIDATE_ONLY", None)
    env.pop("RESOLVE_SNAPSHOT_ONLY", None)
    env.update(extra)
    return env


def _run_sh(**extra: str) -> subprocess.CompletedProcess[str]:
    env = _env(**extra)
    env["VALIDATE_ONLY"] = "1"
    return subprocess.run(
        [str(ROOT / "run.sh")],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
    )


def _func_body(src: str, name: str) -> str:
    m = re.search(rf"^{name}\(\) \{{(.*?)^\}}", src, re.M | re.S)
    if m is None:
        raise AssertionError(f"missing function {name}")
    return m.group(1)


def _load_tool(rel: str, name: str):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {rel}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HUB_MODEL = "sfxnz/DeepSeek-V4.1-Flash-EXL3"
HUB_REV = "2.0bpw-mcg"
HUB_DIRNAME = "models--sfxnz--DeepSeek-V4.1-Flash-EXL3"


def _hub(cache: Path) -> Path:
    return cache / "hub" / HUB_DIRNAME


def _resolve(hf_cache: str, **extra: str) -> subprocess.CompletedProcess[str]:
    env = _env(**extra)
    env.pop("VALIDATE_ONLY", None)
    env["RESOLVE_SNAPSHOT_ONLY"] = "1"
    env["HF_CACHE"] = hf_cache
    env["MODEL"] = extra.get("MODEL", HUB_MODEL)
    env["SNAPSHOT_SHA"] = extra.get("SNAPSHOT_SHA", HUB_REV)
    return subprocess.run(
        [str(ROOT / "run.sh")],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
    )


class RecipeOpsTests(unittest.TestCase):
    def test_stop_sh_ssh_probe_has_else_exit_1(self) -> None:
        stop = _read("stop.sh")
        self.assertRegex(
            stop,
            r"ssh -o BatchMode=yes.*\n(?:.*\n)*?\s+else\n(?:.*\n)*?\s+exit 1",
        )
        self.assertIn(">&2", stop)

    def test_stop_sh_reads_worker_host_state_then_default(self) -> None:
        stop = _read("stop.sh")
        self.assertIn(".run-state/worker_host", stop)
        self.assertLess(stop.find(".run-state/worker_host"), stop.find('WORKER_HOST:-spark2'))

    def test_stop_sh_orchestrate_zero_is_local_only(self) -> None:
        stop = _read("stop.sh")
        self.assertRegex(stop, re.compile(r'ORCHESTRATE" == "0".*exit 0', re.S))
        proc = subprocess.run(
            [str(ROOT / "stop.sh")],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            env=_env(
                ORCHESTRATE="0",
                CONTAINER_NAME="dsv41-ops-test-no-such-container",
            ),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_stop_sh_ssh_probe_fails_loud_when_not_spark2(self) -> None:
        host = socket.gethostname().split(".", 1)[0].lower()
        if host.startswith("spark2"):
            self.skipTest("this host is spark2")
        proc = subprocess.run(
            [str(ROOT / "stop.sh")],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            env=_env(
                ORCHESTRATE="auto",
                WORKER_HOST="no-such-host-xyz",
                CONTAINER_NAME="dsv41-ops-test-no-such-container",
            ),
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertTrue(proc.stderr.strip())

    def test_run_sh_worker_ssh_forwards_snapshot_and_revision(self) -> None:
        run = _read("run.sh")
        ssh_idx = run.find('ssh "$WORKER_HOST"')
        self.assertGreater(ssh_idx, 0)
        ssh_block = run[ssh_idx : ssh_idx + 3500]
        self.assertIn("SNAPSHOT_SHA='$SNAPSHOT_SHA'", ssh_block)
        self.assertIn("HF_CACHE='$HF_CACHE'", ssh_block)
        self.assertIn("MODEL='$MODEL'", ssh_block)
        self.assertIn("QUANTIZATION='$QUANTIZATION'", ssh_block)
        self.assertIn("DSV41_ENGRAM_DISK='$DSV41_ENGRAM_DISK'", ssh_block)
        self.assertIn('--revision "$SNAPSHOT_SHA"', run)
        self.assertIn(".run-state/worker_host", run)
        self.assertNotIn("starting local rank only", run)

    def test_run_sh_does_not_default_disable_xet(self) -> None:
        run = _read("run.sh")
        self.assertNotIn('HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"', run)
        self.assertIn("unset HF_HUB_DISABLE_XET", run)
        self.assertIn("HF_XET_HIGH_PERFORMANCE", run)
        self.assertIn('HF_HUB_CACHE="${HF_CACHE}/hub"', run)
        self.assertIn('--revision "$SNAPSHOT_SHA"', run)
        body = _func_body(run, "resolve_snapshot")
        self.assertIn("/refs/", body)
        self.assertIn("/snapshots/", body)
        self.assertLess(body.find('-f "$refs"'), body.find('-d "$named"'))

    def test_resolve_model_does_not_fall_back_to_hub_id(self) -> None:
        body = _func_body(_read("run.sh"), "resolve_model")
        self.assertNotIn("$MODEL", body)
        self.assertIn("$SNAPSHOT_IN_CONTAINER", body)

    def test_dockerfile_from_is_digest_pinned(self) -> None:
        from_line = ""
        for line in _read("docker/Dockerfile").splitlines():
            if line.startswith("FROM "):
                from_line = line.split(None, 1)[1].strip()
                break
        self.assertEqual(from_line, PINNED_FROM)
        self.assertIn("@sha256:", from_line)
        df = _read("docker/Dockerfile")
        self.assertIn("EXLLAMAV3_REF=5be886578ec80324c2c715269387be2058724b6e", df)
        self.assertIn("patch_exllamav3_aarch64.py", df)
        self.assertIn("sitecustomize.py", df)
        self.assertIn('ENTRYPOINT ["vllm", "serve"]', df)
        self.assertIn("load_general_plugins", _read("docker/patch/sitecustomize.py"))
        self.assertIn(
            "deepseek_v4_sparse_mla_attention_warmup",
            _read("docker/patch/sitecustomize.py"),
        )
        self.assertIn("compile_or_warm_up_model", _read("docker/patch/sitecustomize.py"))
        site = _read("docker/patch/sitecustomize.py")
        self.assertIn("coerce_swa_block_size", site)
        self.assertIn("DeepseekV41Config", site)
        self.assertIn("from sm120_page import", site)
        self.assertTrue((ROOT / "docker/patch/sm120_page.py").is_file())
        self.assertIn("libcusparse-dev-13-0", df)
        self.assertIn("VLLM_EXL3_NO_CUDA=1", df)
        self.assertIn("exl3_moe", df)

    def test_assemble_pack_is_executable_and_wired(self) -> None:
        path = ROOT / "tools/assemble_pack.sh"
        self.assertTrue(path.is_file())
        self.assertTrue(os.access(path, os.X_OK))
        run = _read("README.md")
        self.assertIn("assemble_pack.sh", run)

    def test_head_preflight_before_worker_scp(self) -> None:
        run = _read("run.sh")
        idx = run.find('ORCHESTRATE" == "auto" && "$ROLE" == "head"')
        self.assertGreater(idx, 0)
        block = run[idx:]
        scp = block.find("scp ")
        self.assertGreater(scp, 0)
        self.assertGreater(block.find("refuse_foreign_serve"), -1)
        self.assertGreater(block.find("refuse_busy_port"), -1)
        self.assertLess(block.find("refuse_foreign_serve"), scp)
        self.assertLess(block.find("refuse_busy_port"), scp)
        wait = _func_body(run, "wait_ready")
        self.assertIn("$SERVED_NAME", wait)
        self.assertIn("/health", wait)

    def test_validate_only_defaults_pass(self) -> None:
        proc = _run_sh()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("validate-only", proc.stdout)
        self.assertIn("quant=exl3", proc.stdout)
        self.assertIn("engram_disk=1", proc.stdout)

    def test_validate_only_refuses_max_num_seqs(self) -> None:
        proc = _run_sh(MAX_NUM_SEQS="8")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("exceeds 2", proc.stderr)

    def test_validate_only_refuses_spec_tokens_not_divisible_by_5(self) -> None:
        proc = _run_sh(SPEC="dspark", NUM_SPECULATIVE_TOKENS="3")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("not divisible by 5", proc.stderr)

    def test_validate_only_refuses_26gib_kv_pin(self) -> None:
        proc = _run_sh(KV_CACHE_MEMORY="27917287424")
        self.assertNotEqual(proc.returncode, 0)
        self.assertTrue(proc.stderr.strip())

    def test_validate_only_refuses_window_above_1m(self) -> None:
        proc = _run_sh(MAX_MODEL_LEN="2097152")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("cannot hold --max-model-len", proc.stderr)

    def test_validate_only_refuses_native_quant(self) -> None:
        proc = _run_sh(QUANTIZATION="fp8")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("EXL3", proc.stderr)

    def test_validate_only_refuses_engram_resident(self) -> None:
        proc = _run_sh(DSV41_ENGRAM_DISK="0")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Engram", proc.stderr)

    def test_gitignore_run_state(self) -> None:
        self.assertIn(".run-state/", _read(".gitignore"))

    def test_smoke_requires_content(self) -> None:
        smoke = _read("smoke_chat.py")
        self.assertIn("choices", smoke)
        self.assertIn("content", smoke)
        self.assertIn("thinking", smoke)

    def test_parsers_are_v41(self) -> None:
        run = _read("run.sh")
        self.assertIn("--tokenizer-mode deepseek_v41", run)
        self.assertIn("--tool-call-parser deepseek_v41", run)
        self.assertIn("--reasoning-parser deepseek_v41", run)
        self.assertIn('"thinking":false', run)
        self.assertIn('VLLM_PLUGINS=vllm_exl3', run)
        self.assertIn("--quantization", run)
        self.assertIn("--entrypoint vllm", run)
        serve_idx = run.find("--entrypoint vllm")
        self.assertIn("serve", run[serve_idx : serve_idx + 400])
        self.assertIn('BLOCK_SIZE="${BLOCK_SIZE:-64}"', run)
        self.assertIn('SPEC="${SPEC:-none}"', run)
        self.assertIn('ENFORCE_EAGER="${ENFORCE_EAGER:-1}"', run)
        self.assertIn("enable_flashinfer_autotune", run)
        self.assertIn("enable_jit_warmup", run)

    def test_locator_prefers_refs_commit_over_named_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d)
            hub = _hub(cache)
            commit = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            (hub / "refs").mkdir(parents=True)
            (hub / "refs" / HUB_REV).write_text(commit + "\n")
            named = hub / "snapshots" / HUB_REV
            named.mkdir(parents=True)
            (named / "config.json").write_text("{}")
            snap = hub / "snapshots" / commit
            snap.mkdir(parents=True)
            (snap / "config.json").write_text("{}")
            proc = _resolve(str(cache))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stdout.strip(), str(snap))

    def test_locator_falls_back_to_named_snapshot_when_refs_absent(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d)
            hub = _hub(cache)
            named = hub / "snapshots" / HUB_REV
            named.mkdir(parents=True)
            (named / "config.json").write_text("{}")
            proc = _resolve(str(cache))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stdout.strip(), str(named))

    def test_locator_falls_back_to_named_snapshot_when_commit_dir_missing(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d)
            hub = _hub(cache)
            (hub / "refs").mkdir(parents=True)
            (hub / "refs" / HUB_REV).write_text("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n")
            named = hub / "snapshots" / HUB_REV
            named.mkdir(parents=True)
            (named / "config.json").write_text("{}")
            proc = _resolve(str(cache))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stdout.strip(), str(named))

    def test_resolve_snapshot_only_exits_1_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            proc = _resolve(d)
            self.assertEqual(proc.returncode, 1)
            self.assertTrue(proc.stderr.strip())
            self.assertFalse(proc.stdout.strip())


class PublishPackTests(unittest.TestCase):
    def test_allow_omits_official_non_serve_trees(self) -> None:
        pub = _load_tool("tools/publish_pack.py", "publish_pack")
        for bad in ("inference/", "encoding/", "evaluation/", "assets/", "README.md"):
            self.assertNotIn(bad, pub.ALLOW)
        with tempfile.TemporaryDirectory() as d:
            src = Path(d)
            (src / "config.json").write_text("{}")
            (src / "LICENSE").write_text("MIT")
            (src / "tokenizer.json").write_text("{}")
            (src / "model-00001-of-00048.safetensors").write_bytes(b"x")
            for rel in (
                "README.md",
                "inference/run.py",
                "encoding/enc.bin",
                "evaluation/eval.py",
                "assets/logo.png",
            ):
                path = src / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("official")
            allowed = {p.relative_to(src).as_posix() for p in pub.iter_allowed(src)}
        self.assertEqual(
            allowed,
            {"config.json", "LICENSE", "tokenizer.json", "model-00001-of-00048.safetensors"},
        )


if __name__ == "__main__":
    unittest.main()
