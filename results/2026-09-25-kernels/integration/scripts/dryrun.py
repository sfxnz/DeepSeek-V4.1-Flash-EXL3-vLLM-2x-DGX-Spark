#!/usr/bin/env python3
"""CPU strict dry-run of one (node, role, arm) on an image: run.sh's own docker run argv, minus GPU.

The argv comes from tests/run_sh_harness.dry_run (run.sh with docker/ssh/scp stubbed), so the
container gets exactly the -e set and patch mounts run.sh would give that role under the arm's
env (DSV41_PATCH_STRICT=1 included). Changes: no --gpus / --device /dev/infiniband / host
network / IPC / ulimits / HF cache; --network none --memory 12g --cpus 6; the worktree at /repo
(ro) and an output dir at /out; entrypoint = dryrun_driver.sh. /dev/cpu_dma_latency is kept when
run.sh passes it (DSV41_PM_QOS_US set). The harness's worker argv lacks the patch mounts only
because its scp stub copies nothing; the real worker mounts ~/.cache/dsv41-patch the same way,
so both roles get /opt/dsv41-patch and sitecustomize.py from the worktree here.

  dryrun.py --image IMG --role head|worker --arm on|off --out DIR [--node spark2 --remote-repo P]
Writes DIR/{cmd.txt, summary.json, ...driver outputs}. Exit 0 only when every gate holds."""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tests"))
import run_sh_harness as harness  # noqa: E402

ARMS = {
    "off": {},
    "on": {
        "DSV41_P2B_COOP": "2",
        "DSV41_DENSE_GEMV": "1",
        "DSV41_MHC_DET_SPLITS": "16",
        "DSV41_MHC_DET_OVERLAP": "1",
        "DSV41_ENGRAM_NATIVE_STAGE": "1",
        "DSV41_ENGRAM_EARLY_HASH": "1",
        "DSV41_ATTN_T2R_DEDUP": "1",
        "DSV41_MOE_PREP_FUSED": "1",
        "DSV41_CANDIDATE_MASK_BOUNDED": "1",
        "DSV41_INDEXER_WP_GEMV": "1",
        "DSV41_SWA_META_FUSED": "1",
        "DSV41_ENGRAM_WKV_TP": "1",
        "DSV41_NCCL_EAGER_TWIN": "1",
        "DSV41_PM_QOS_US": "20",
        "DSV41_AR_L2_PREFETCH": "1",
    },
}
DROP_WITH_VALUE = {"--name", "--restart", "--gpus", "--network", "--ipc", "--shm-size", "--cap-add", "--ulimit"}
DROP_FLAG = {"-d", "--rm"}
SCRIPTS_REL = "results/2026-09-25-kernels/integration/scripts"


def cpu_argv(argv: list[str], image: str, repo: str, out: str) -> list[str]:
    i = argv.index("--entrypoint")
    assert argv[0] == "run" and argv[i + 2] == image, argv[i : i + 3]
    opts, j = [], 1
    while j < i:
        t = argv[j]
        if t in DROP_FLAG:
            j += 1
        elif t in DROP_WITH_VALUE:
            j += 2
        elif t == "--device":
            if argv[j + 1] == "/dev/cpu_dma_latency":
                opts += [t, argv[j + 1]]
            j += 2
        elif t == "-v":
            j += 2  # patch mounts are re-added below from this node's worktree; HF cache dropped
        elif t == "-e":
            opts += [t, argv[j + 1]]
            j += 2
        else:
            raise SystemExit(f"unexpected docker run option {t!r}: extend dryrun.py")
    return [
        "docker", "run", "--rm", "--network", "none", "--memory", "12g", "--cpus", "6",
        "--name", f"dsv41-integ-dryrun-{os.getpid()}",
        *opts,
        "-e", f"HOST_UID={os.getuid()}", "-e", f"HOST_GID={os.getgid()}",
        "-v", f"{repo}/docker/patch:/opt/dsv41-patch:ro",
        "-v", f"{repo}/docker/patch/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro",
        "-v", f"{repo}:/repo:ro", "-v", f"{out}:/out",
        "--entrypoint", "bash", image, f"/repo/{SCRIPTS_REL}/dryrun_driver.sh",
    ]


def _read(p: Path) -> str:
    return p.read_text(errors="replace") if p.is_file() else ""


def summarize(out: Path, arm: str) -> dict:
    logs = {n: _read(out / n) for n in ("run1.stdout", "run1.stderr", "run2.stdout", "run2.stderr",
                                        "probe.stdout", "probe.stderr")}
    text = "\n".join(logs.values())
    probe = json.loads(_read(out / "probe.json") or "{}")
    num = lambda pat, s: int(m.group(1)) if (m := re.search(pat, s)) else -1  # noqa: E731
    s = {
        "arm": arm,
        "run1_rc": _read(out / "run1.rc").strip(),
        "run2_rc": _read(out / "run2.rc").strip(),
        "probe_rc": _read(out / "probe.rc").strip(),
        "patch_FAIL_lines": [ln for ln in text.splitlines() if "dsv41-patch FAIL" in ln],
        "lever_FAILED_lines": [ln for ln in text.splitlines() if re.search(r"decode lever \S+ FAILED", ln)],
        "py_compile_failures": num(r"py_compile failures: (\d+)", _read(out / "py_compile.txt")),
        "import_failures": num(r"import failures: (\d+)", _read(out / "imports.txt")),
        "disarm_hits": num(r"disarm hits: (\d+)", _read(out / "disarm_scan.txt")),
        "disarm_markers": num(r"markers: (\d+)", _read(out / "disarm_scan.txt")),
        "rewritten_files": len(_read(out / "rewritten.txt").split()),
        "patch_ok_lines": len(re.findall(r"^dsv41-patch ok ", logs["run1.stderr"], re.M)),
        "levers_not_ok": {k: v for k, v in probe.get("levers", {}).items() if not v.get("ok")},
        "levers_ok": probe.get("summary", {}).get("levers_ok"),
        "levers": probe.get("summary", {}).get("levers"),
        "module_import_failures": probe.get("summary", {}).get("module_import_failures"),
        "so": probe.get("so", {}),
        "early_hash_armed": "early hash armed" in text,
        "install_lines": [ln for ln in logs["run1.stdout"].splitlines() if ln.startswith("dsv41: ")],
        "no_gpu": _read(out / "env-check.txt").splitlines()[:2],
    }
    gates = {
        "run1_rc_0": s["run1_rc"] == "0",
        "run2_rc_0": s["run2_rc"] == "0",
        "probe_rc_0": s["probe_rc"] == "0",
        "no_patch_FAIL": not s["patch_FAIL_lines"],
        "no_lever_FAILED": not s["lever_FAILED_lines"],
        "py_compile_0": s["py_compile_failures"] == 0,
        "imports_0": s["import_failures"] == 0,
        "patch_modules_import": s["module_import_failures"] == 0,
        "disarm_hits_0": s["disarm_hits"] == 0 and s["disarm_markers"] > 0,
        "levers_match_arm": s["levers"] is not None and s["levers_ok"] == s["levers"],
        "early_hash_matches_arm": s["early_hash_armed"] == (arm == "on"),
        "so_kernel_markers": all(s["so"].get(k, 0) > 0 for k in (
            "DSV41_P2B_SRC_SORT", "DSV41_P2B_COOP", "p2b coop dataflow kernel engaged", "p2b_coop_df_kernel")),
    }
    s["gates"] = gates
    s["pass"] = all(gates.values())
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--role", choices=("head", "worker"), required=True)
    ap.add_argument("--arm", choices=tuple(ARMS), required=True)
    ap.add_argument("--out", required=True, help="local output dir")
    ap.add_argument("--node", default="", help="ssh host to run on (default: here)")
    ap.add_argument("--remote-repo", default="", help="worktree copy on --node")
    a = ap.parse_args()

    res = harness.dry_run(IMAGE=a.image, **ARMS[a.arm])
    if res["returncode"] != 0 or not res[a.role]:
        sys.stderr.write(res["stdout"] + res["stderr"])
        return 2
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if a.node:
        rout = f"{a.remote_repo}.integ-out/{out.name}"
        cmd = cpu_argv(res[a.role], a.image, a.remote_repo, rout)
        full = ["ssh", a.node, f"rm -rf {shlex.quote(rout)} && mkdir -p {shlex.quote(rout)} && {shlex.join(cmd)}"]
    else:
        cmd = cpu_argv(res[a.role], a.image, str(REPO), str(out))
        full = cmd
    (out / "cmd.txt").write_text(f"# node={a.node or 'local'} role={a.role} arm={a.arm}\n"
                                 f"# run.sh argv ({a.role}): docker {shlex.join(res[a.role])}\n{shlex.join(full)}\n")
    proc = subprocess.run(full, capture_output=True, text=True, timeout=1200)
    (out / "docker.txt").write_text(proc.stdout + proc.stderr + f"\n# rc={proc.returncode}\n")
    if a.node:
        subprocess.run(["rsync", "-a", f"{a.node}:{rout}/", f"{out}/"], check=True)
        subprocess.run(["ssh", a.node, f"rm -rf {shlex.quote(rout)}"], check=True)
    s = summarize(out, a.arm)
    s.update(node=a.node or os.uname().nodename, role=a.role, image=a.image, docker_rc=proc.returncode)
    (out / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
    print(json.dumps({k: s[k] for k in ("node", "role", "arm", "pass", "gates")}))
    return 0 if s["pass"] and proc.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
