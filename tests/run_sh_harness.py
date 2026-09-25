#!/usr/bin/env python3
"""Dry-run run.sh on the head with docker/ssh/scp stubbed, CPU only.

The ssh stub runs the worker command locally in an empty env (like a real
ssh login), so the worker `docker run` shows exactly what the ssh line
forwards. No real docker, ssh or GPU call is made: the stubs come first on
PATH, and CONTAINER_NAME / WORKER_HOST / PORT are test-only values.

run.sh runs from a copy in a temp dir (docker/ and tools/ symlinked), so its
.run-state never lands in the repo. The docker stub keeps per-rank container
state; STUB_* env knobs make a rank die or the API never come up.

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
ROLE = os.environ.get("STUB_ROLE") or os.environ.get("ROLE", "head")
with open(os.environ["STUB_LOG"], "a") as fh:
    fh.write(json.dumps({"tool": os.path.basename(sys.argv[0]),
                         "role": ROLE,
                         "argv": sys.argv[1:]}) + "\n")
'''

STUBS = {
    # Per-rank state: <STUB_STATE>/<role> lists running containers.
    # STUB_HEAD_EXITS=1: the head container is never listed as running.
    # STUB_WORKER_PS_OK=N: only the first N head-side `ssh worker docker ps` list it.
    # STUB_WORKER_NO_IMAGE=1: `docker image inspect` fails on the worker.
    "docker": _LOG + r'''
from pathlib import Path
state = Path(os.environ["STUB_STATE"]) / ROLE
names = state.read_text().split() if state.exists() else []
cmd = sys.argv[1:2]
if sys.argv[1:3] == ["image", "inspect"]:
    if ROLE == "worker" and os.environ.get("STUB_WORKER_NO_IMAGE") == "1":
        sys.exit(1)
    if "-f" in sys.argv:
        print(os.environ.get("STUB_IMAGE_LABELS", "null"))
elif cmd == ["run"]:
    state.write_text(" ".join(names + [sys.argv[sys.argv.index("--name") + 1]]))
    print("0" * 64)
elif cmd == ["ps"]:
    if ROLE == "head" and os.environ.get("STUB_HEAD_EXITS") == "1":
        names = []
    if os.environ.get("STUB_ROLE") == "worker" and "STUB_WORKER_PS_OK" in os.environ:
        counter = state.with_suffix(".ps")
        n = int(counter.read_text()) if counter.exists() else 0
        counter.write_text(str(n + 1))
        if n >= int(os.environ["STUB_WORKER_PS_OK"]):
            names = []
    print("\n".join(names))
elif cmd == ["rm"]:
    state.write_text(" ".join(n for n in names if n not in sys.argv))
elif cmd == ["logs"]:
    print(os.environ.get("STUB_LOGS_" + ROLE.upper(), ""))
''',
    "ssh": _LOG + r'''
import subprocess
args = [a for a in sys.argv[1:] if a != "-q"]
while args and args[0] == "-o":
    args = args[2:]
cmd = args[1] if len(args) > 1 else ""
env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME") or k.startswith("STUB_")}
if cmd.endswith("bash /tmp/dsv41-exl3-run.sh"):
    cmd = cmd[: -len("/tmp/dsv41-exl3-run.sh")] + os.environ["STUB_RUN_SH"]
    sys.exit(subprocess.run(["bash", "-c", cmd], env=env).returncode)
if cmd.startswith("docker "):
    env["STUB_ROLE"] = "worker"
    sys.exit(subprocess.run(["bash", "-c", cmd], env=env).returncode)
''',
    "scp": _LOG,
    "hostname": "#!/bin/sh\necho spark1\n",
    "sudo": "#!/bin/sh\nexit 1\n",
    "sleep": "#!/bin/sh\nexit 0\n",
    "ip": "#!/bin/sh\nexit 0\n",
    # STUB_CURL_FAIL=1: the API never comes up.
    "curl": "#!/bin/sh\n[ \"$STUB_CURL_FAIL\" = 1 ] && exit 7\n"
    "echo '{\"data\":[{\"id\":\"deepseek-ai/DeepSeek-V4.1-Flash\"}]}'\n",
}
# vLLM prints this when the headless worker starts connecting to the head.
WORKER_LAUNCH_LINE = "INFO [serve.py:217] Launching vLLM headless multiproc executor, with head node address"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _revision() -> str:
    import yaml

    recipe = yaml.load((ROOT / "recipe.yaml").read_text(), Loader=yaml.BaseLoader)
    return recipe["serve"]["env"]["SNAPSHOT_SHA"]


def dry_run(make_snapshot: bool = True, **extra: str) -> dict:
    """Run ./run.sh as the head with stubs.

    Returns head/worker docker run argv, all stub calls, and the files run.sh
    left in .run-state. AUDIT and WARMUP default to off here; pass them to test.
    """
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        repo = tmp / "repo"
        repo.mkdir()
        shutil.copy2(ROOT / "run.sh", repo / "run.sh")
        for sub in ("docker", "tools"):
            (repo / sub).symlink_to(ROOT / sub)
        state = tmp / "state"
        state.mkdir()
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
        if make_snapshot:
            snap.mkdir(parents=True)
            (snap / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "exl3"}}))
        log = tmp / "calls.jsonl"
        log.touch()
        env = {
            "PATH": path_env,
            "HOME": str(tmp),
            "HF_TOKEN": "",
            "STUB_LOG": str(log),
            "STUB_RUN_SH": str(repo / "run.sh"),
            "STUB_STATE": str(state),
            "STUB_LOGS_WORKER": WORKER_LAUNCH_LINE,
            "HF_CACHE": str(cache),
            "CONTAINER_NAME": CONTAINER,
            "WORKER_HOST": "dryrun-no-such-host",
            "PORT": str(_free_port()),
            "AUDIT": "off",
            "WARMUP": "0",
        }
        env.update(extra)
        proc = subprocess.run(
            [str(repo / "run.sh")], cwd=str(tmp), env=env, capture_output=True, text=True, timeout=120
        )
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        run_state = repo / ".run-state"
        files = {p.name: p.read_text() for p in run_state.iterdir()} if run_state.is_dir() else {}
    runs = {c["role"]: c["argv"] for c in calls if c["tool"] == "docker" and c["argv"][:1] == ["run"]}
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "head": runs.get("head"),
        "worker": runs.get("worker"),
        "scp": [c["argv"] for c in calls if c["tool"] == "scp"],
        "calls": calls,
        "run_state": files,
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
