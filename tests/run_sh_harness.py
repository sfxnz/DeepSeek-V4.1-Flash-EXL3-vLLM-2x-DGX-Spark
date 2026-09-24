#!/usr/bin/env python3
"""Dry-run run.sh on the head with docker/ssh/scp stubbed, CPU only.

The ssh stub runs the worker command locally in an empty env (like a real
ssh login), so the worker `docker run` shows exactly what the ssh line
forwards. No real docker, ssh or GPU call is made: the stubs come first on
PATH, and CONTAINER_NAME / WORKER_HOST / PORT are test-only values.

    python3 tests/run_sh_harness.py        # print head + worker docker run argv
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "sfxnz/DeepSeek-V4.1-Flash-EXL3"
CONTAINER = "dsv41-dryrun-harness"

_LOG = r'''#!/usr/bin/env python3
import json, os, sys
with open(os.environ["STUB_LOG"], "a") as fh:
    fh.write(json.dumps({"tool": os.path.basename(sys.argv[0]),
                         "role": os.environ.get("ROLE", "head"),
                         "argv": sys.argv[1:]}) + "\n")
'''

STUBS = {
    "docker": _LOG + r'''
if sys.argv[1:3] == ["image", "inspect"] and "-f" in sys.argv:
    print(os.environ.get("STUB_IMAGE_LABELS", "null"))
elif sys.argv[1:2] == ["run"]:
    print("0" * 64)
''',
    "ssh": _LOG + r'''
import subprocess
args = [a for a in sys.argv[1:] if a != "-q"]
while args and args[0] == "-o":
    args = args[2:]
cmd = args[1] if len(args) > 1 else ""
if cmd.endswith("bash /tmp/dsv41-exl3-run.sh"):
    cmd = cmd[: -len("/tmp/dsv41-exl3-run.sh")] + os.environ["STUB_RUN_SH"]
    env = {k: os.environ[k] for k in ("PATH", "HOME", "STUB_LOG", "STUB_RUN_SH", "STUB_IMAGE_LABELS") if k in os.environ}
    sys.exit(subprocess.run(["bash", "-c", cmd], env=env).returncode)
''',
    "scp": _LOG,
    "hostname": "#!/bin/sh\necho spark1\n",
    "sudo": "#!/bin/sh\nexit 1\n",
    "sleep": "#!/bin/sh\nexit 0\n",
    "ip": "#!/bin/sh\nexit 0\n",
    "curl": "#!/bin/sh\necho '{\"data\":[{\"id\":\"deepseek-ai/DeepSeek-V4.1-Flash\"}]}'\n",
}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _revision() -> str:
    import yaml

    recipe = yaml.load((ROOT / "recipe.yaml").read_text(), Loader=yaml.BaseLoader)
    return recipe["serve"]["env"]["SNAPSHOT_SHA"]


def dry_run(**extra: str) -> dict:
    """Run ./run.sh as the head with stubs; return head/worker docker run argv and scp calls."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        bindir = tmp / "bin"
        bindir.mkdir()
        for name, body in STUBS.items():
            path = bindir / name
            path.write_text(body)
            path.chmod(0o755)
        path_env = f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}"
        for tool in ("docker", "ssh", "scp"):
            if shutil.which(tool, path=path_env) != str(bindir / tool):
                raise RuntimeError(f"{tool} stub is not first on PATH")
        cache = tmp / "hf"
        snap = cache / "hub" / f"models--{MODEL.replace('/', '--')}" / "snapshots" / extra.get("SNAPSHOT_SHA", _revision())
        snap.mkdir(parents=True)
        (snap / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "exl3"}}))
        log = tmp / "calls.jsonl"
        log.touch()
        env = {
            "PATH": path_env,
            "HOME": str(tmp),
            "HF_TOKEN": "",
            "STUB_LOG": str(log),
            "STUB_RUN_SH": str(ROOT / "run.sh"),
            "HF_CACHE": str(cache),
            "CONTAINER_NAME": CONTAINER,
            "WORKER_HOST": "dryrun-no-such-host",
            "PORT": str(_free_port()),
        }
        env.update(extra)
        proc = subprocess.run(
            [str(ROOT / "run.sh")], cwd=str(tmp), env=env, capture_output=True, text=True, timeout=120
        )
        calls = [json.loads(line) for line in log.read_text().splitlines()]
    runs = {c["role"]: c["argv"] for c in calls if c["tool"] == "docker" and c["argv"][:1] == ["run"]}
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "head": runs.get("head"),
        "worker": runs.get("worker"),
        "scp": [c["argv"] for c in calls if c["tool"] == "scp"],
    }


def container_env(argv: list[str]) -> dict[str, str]:
    """-e NAME=VALUE pairs of a docker run argv."""
    out = {}
    for flag, val in zip(argv, argv[1:]):
        if flag == "-e":
            name, _, value = val.partition("=")
            out[name] = value
    return out


def image_and_args(argv: list[str]) -> tuple[str, list[str]]:
    """(image, vllm serve argv) of a docker run argv."""
    i = argv.index("--entrypoint")
    return argv[i + 2], argv[i + 3 :]


def flag_value(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


if __name__ == "__main__":
    res = dry_run()
    if res["returncode"] != 0 or not res["head"] or not res["worker"]:
        sys.stderr.write(res["stdout"] + res["stderr"])
        sys.exit(1)
    for role in ("worker", "head"):
        print(f"# {role}")
        print("docker " + shlex.join(res[role]))
        print()
    sys.stderr.write(res["stderr"])
