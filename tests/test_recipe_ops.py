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

import yaml

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


def _script(rel: str) -> list[str]:
    return [str(ROOT / rel)]


def _run_sh(**extra: str) -> subprocess.CompletedProcess[str]:
    env = _env(**extra)
    env["VALIDATE_ONLY"] = "1"
    return subprocess.run(
        _script("run.sh"),
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
HUB_PACK_URL = "https://huggingface.co/sfxnz/DeepSeek-V4.1-Flash-EXL3"
DEFAULTS_BEGIN = (
    "<!-- BEGIN generated defaults from recipe.yaml — edit recipe.yaml and run kit/render.py -->"
)
DEFAULTS_END = "<!-- END generated defaults -->"


def _recipe() -> dict:
    return yaml.load((ROOT / "recipe.yaml").read_text(), Loader=yaml.BaseLoader)


def _generated_defaults() -> str:
    text = _read("README.md")
    start = text.find(DEFAULTS_BEGIN)
    stop = text.find(DEFAULTS_END)
    if start < 0 or stop < 0 or stop <= start:
        raise AssertionError("README.md is missing generated defaults markers")
    return text[start:stop]


def _defaults_row(setting: str) -> str:
    prefix = f"| {setting} |"
    for line in _generated_defaults().splitlines():
        if line.startswith(prefix):
            return line
    raise AssertionError(f"missing generated defaults row {setting!r}")


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
        _script("run.sh"),
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
            _script("stop.sh"),
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
            _script("stop.sh"),
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
        self.assertIn("MM_ENCODER_TP_MODE='$MM_ENCODER_TP_MODE'", ssh_block)
        self.assertIn("QUANTIZATION='$QUANTIZATION'", ssh_block)
        self.assertIn("DSV41_ENGRAM_DISK='$DSV41_ENGRAM_DISK'", ssh_block)
        self.assertIn("DSV41_PATCH_DIR='/tmp/dsv41-patch'", ssh_block)
        self.assertIn('scp -q -r "$SCRIPT_DIR/docker/patch"', run)
        self.assertIn("/opt/dsv41-patch:ro", run)
        self.assertIn("/usr/lib/python3.12/sitecustomize.py:ro", run)
        self.assertIn("DSV41_STEP_CENSUS=", run)
        self.assertIn("DSV41_INDEX_TOPK=", run)
        self.assertIn("DSV41_MHC_DECODE_SPLITS=", run)
        self.assertIn("DSV41_ENGRAM_CACHE=", run)
        self.assertIn("DSV41_MHC_NO_DEEPGEMM=", run)
        self.assertIn("DSV41_DSPARK_DRAFT_TOPK=", run)
        self.assertIn("DSV41_DSPARK_TAIL_NGRAM=", run)
        self.assertIn("DSV41_DSPARK_TAIL_NGRAM_POS=", run)
        self.assertIn("DSV41_DSPARK_SOFTMAX_VERIFY=", run)
        self.assertIn("DSV41_DSPARK_REFINE_PASS=", run)
        self.assertIn("DSV41_DSPARK_CONF_GATE=", run)
        self.assertIn("DSV41_MLA_IO_WARPS=", run)
        self.assertIn("DSV41_MLA_CHUNKS_PER_BLOCK=", run)
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
        self.assertIn("prefer_b12x_mxfp8", site)
        self.assertTrue((ROOT / "docker/patch/prefer_b12x_mxfp8.py").is_file())
        self.assertIn('b12x==1.3.0', df)
        self.assertIn("libcusparse-dev-13-0", df)
        self.assertIn("VLLM_EXL3_MOE_KERNEL=native", df)
        self.assertIn("widen_p2b_shapes.py", df)
        self.assertIn("widen_p2b_mrow.py", df)
        self.assertIn("widen_p2b_cfg1.py", df)
        self.assertIn("widen_p2b_codebook.py", df)
        self.assertLess(df.find("widen_p2b_cfg1.py"), df.find("widen_p2b_codebook.py"))
        self.assertNotIn("widen_p2b_fma.py", df)
        self.assertNotIn("widen_p2b_pf4.py", df)
        self.assertNotIn("widen_p2b_nocoop.py", df)
        self.assertNotIn("widen_p2b_cpasync.py", df)
        self.assertNotIn("widen_p2b_ldg.py", df)
        self.assertNotIn("widen_p2b_cp16.py", df)
        self.assertIn("p2b_fused_moe", df)
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

    def test_run_sh_is_tracked_executable(self) -> None:
        proc = subprocess.run(
            ["git", "ls-files", "--stage", "--", "run.sh"],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        self.assertTrue(proc.stdout.startswith("100755 "), proc.stdout)
        self.assertTrue(os.access(ROOT / "run.sh", os.X_OK))

    def test_validate_only_defaults_pass(self) -> None:
        proc = _run_sh()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("validate-only", proc.stdout)
        self.assertIn("quant=exl3", proc.stdout)
        self.assertIn("engram_disk=1", proc.stdout)
        self.assertIn("spec=dspark", proc.stdout)
        self.assertIn("spec_tokens=5", proc.stdout)
        self.assertIn("eager=0", proc.stdout)
        self.assertIn("lm_only=0", proc.stdout)
        self.assertNotIn("--language-model-only", proc.stdout)

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

    def test_smoke_vision_uses_image_url(self) -> None:
        smoke = _read("smoke_vision.py")
        self.assertIn("image_url", smoke)
        self.assertIn("deepseek-ai/DeepSeek-V4.1-Flash", smoke)
        self.assertIn("is not a multimodal model", smoke)
        self.assertNotIn("from PIL", smoke)
        self.assertNotIn("import PIL", smoke)

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
        spec = _recipe()["serve"]["env"]["SPEC"]
        self.assertIn(f'SPEC="${{SPEC:-{spec}}}"', run)
        self.assertEqual(spec, "dspark")
        self.assertIn('NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-5}"', run)
        self.assertIn('ENFORCE_EAGER="${ENFORCE_EAGER:-0}"', run)
        self.assertIn('LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-0}"', run)
        self.assertIn("enable_flashinfer_autotune", run)
        self.assertIn("enable_jit_warmup", run)

    def test_dspark_default_draft_sample_method_is_greedy(self) -> None:
        run = _read("run.sh")
        self.assertIn('"draft_sample_method":"greedy"', run)
        self.assertNotIn('"draft_sample_method":"probabilistic"', run)

    def test_default_language_model_only_is_off(self) -> None:
        self.assertEqual(_recipe()["serve"]["env"]["LANGUAGE_MODEL_ONLY"], "0")
        run = _read("run.sh")
        body = _func_body(run, "start_local")
        self.assertIn('lm_args+=(--language-model-only)', body)
        self.assertIn('LANGUAGE_MODEL_ONLY" == "1"', body)
        self.assertIn('LANGUAGE_MODEL_ONLY=$LANGUAGE_MODEL_ONLY', body)
        self.assertIn('--mm-encoder-tp-mode', body)
        self.assertIn('MM_ENCODER_TP_MODE', body)
        proc = _run_sh()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("lm_only=0", proc.stdout)
        self.assertNotIn("--language-model-only", proc.stdout)

    def test_default_prefill_batch_is_8192_and_hub_rev_stays_mcg(self) -> None:
        self.assertEqual(_recipe()["serve"]["env"]["MAX_NUM_BATCHED_TOKENS"], "8192")
        self.assertEqual(_recipe()["serve"]["env"]["MM_ENCODER_TP_MODE"], "data")
        self.assertEqual(_recipe()["serve"]["env"]["SNAPSHOT_SHA"], HUB_REV)
        self.assertEqual(HUB_REV, "2.0bpw-mcg")
        run = _read("run.sh")
        self.assertIn('MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"', run)
        self.assertIn('MM_ENCODER_TP_MODE="${MM_ENCODER_TP_MODE:-data}"', run)
        row = _defaults_row("`--max-num-batched-tokens`")
        self.assertIn("8192", row)
        self.assertIn("data", _defaults_row("`--mm-encoder-tp-mode`"))

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


class ReadmeHowToTests(unittest.TestCase):
    def test_readme_stranger_serve_path(self) -> None:
        readme = _read("README.md")
        pack = _recipe()["links"]["pack"]
        self.assertEqual(pack, HUB_PACK_URL)
        self.assertIn(pack, readme)
        self.assertIn("git clone", readme)
        self.assertIn("hf download", readme)
        self.assertIn("docker build", readme)
        self.assertIn("./run.sh", readme)
        self.assertIn("python3 smoke_chat.py", readme)
        self.assertIn("python3 smoke_vision.py", readme)
        self.assertIn("python3 bench_decode.py", readme)
        self.assertIn("--codebook mul1", readme)
        self.assertIn("2.0bpw-mul1", readme)
        self.assertIn("pack-only", readme.lower())
        self.assertNotIn("once published", readme)
        self.assertNotIn("docker exec dsv41-quant", readme)

    def test_readme_clone_uses_github_default_main(self) -> None:
        readme = _read("README.md")
        clones = [
            line.strip()
            for line in readme.splitlines()
            if line.strip().startswith("git clone")
        ]
        self.assertEqual(
            clones,
            [
                "git clone https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark.git"
            ],
        )
        self.assertIn("The default branch is `main`.", readme)
        self.assertNotIn("recipe/dsv41-flash-exl3", readme)

    def test_measured_prose_c1_is_a_number_with_evidence(self) -> None:
        rows = _recipe()["measured"]["decode"]["rows"]
        row = next(
            r
            for r in rows
            if r["phase"] == "prose" and str(r["concurrency"]) == "1"
        )
        decode = float(row["decode"])
        self.assertGreater(decode, 0.0)
        evidence = row.get("evidence") or ""
        self.assertTrue(evidence, "measured prose c=1 needs an evidence path")
        self.assertTrue((ROOT / evidence).is_file(), evidence)

    def test_measured_lail_prose_c1_has_evidence(self) -> None:
        rows = _recipe()["measured"]["decode"]["rows"]
        row = next(
            r
            for r in rows
            if r["phase"] == "lail_prose" and str(r["concurrency"]) == "1"
        )
        decode = float(row["decode"])
        self.assertGreater(decode, 0.0)
        evidence = row.get("evidence") or ""
        self.assertTrue(evidence, "measured lail_prose c=1 needs an evidence path")
        self.assertTrue((ROOT / evidence).is_file(), evidence)

    def test_generated_speculative_row_matches_recipe(self) -> None:
        spec = _recipe()["serve"]["env"]["SPEC"]
        tokens = _recipe()["serve"]["env"]["NUM_SPECULATIVE_TOKENS"]
        row = _defaults_row("Speculative")
        self.assertIn(f"`SPEC={spec}`", row)
        if spec == "none":
            self.assertNotIn("DSpark", row)
        elif spec == "dspark":
            self.assertIn("DSpark", row)
            self.assertIn(tokens, row)
        else:
            self.fail(f"unexpected SPEC={spec!r}")


if __name__ == "__main__":
    unittest.main()
