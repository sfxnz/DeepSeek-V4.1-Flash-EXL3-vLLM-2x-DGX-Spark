#!/usr/bin/env python3
"""Publish the local EXL3 pack to Hugging Face. Resume-safe.

Run with a Python that has huggingface_hub (the hf CLI venv is enough):

  HF_XET_HIGH_PERFORMANCE=1 python3 tools/publish_pack.py --dry-run
  HF_XET_HIGH_PERFORMANCE=1 python3 tools/publish_pack.py

Re-running skips blobs the Hub already has. Do not write into the live pack.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.pop("HF_HUB_DISABLE_XET", None)
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

REPO = "sfxnz/DeepSeek-V4.1-Flash-EXL3"
REVISION = "2.0bpw-mcg"
DEFAULT_SRC = Path.home() / ".cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-EXL3/snapshots/2.0bpw-mcg"
ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "model-card.md"

# Serve files only. Skip the official DeepSeek README, inference tree, PDF, and assets.
ALLOW = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
    "model-*-of-00048.safetensors",
    "LICENSE",
    ".gitattributes",
]


def _need_hub():
    try:
        from huggingface_hub import HfApi  # noqa: F401
    except ImportError as exc:
        print(
            "huggingface_hub is missing. Run this with the hf CLI venv, for example "
            "/home/sfxnz/.hf-cli/venv/bin/python tools/publish_pack.py",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def iter_allowed(src: Path) -> list[Path]:
    import fnmatch

    files = [p for p in src.rglob("*") if p.is_file()]
    out = []
    for path in files:
        rel = path.relative_to(src).as_posix()
        if any(fnmatch.fnmatch(rel, pat) for pat in ALLOW):
            out.append(path)
    return sorted(out)


def check_pack(src: Path) -> None:
    if not (src / "config.json").is_file():
        raise SystemExit(f"missing {src / 'config.json'}")
    import json

    cfg = json.loads((src / "config.json").read_text())
    q = cfg.get("quantization_config") or {}
    if q.get("quant_method") != "exl3":
        raise SystemExit(f"{src / 'config.json'} is not quant_method=exl3")
    shards = [src / f"model-{i:05d}-of-00048.safetensors" for i in range(1, 49)]
    missing = [p.name for p in shards if not p.is_file()]
    if missing:
        raise SystemExit(f"pack incomplete, missing {len(missing)} shards: {' '.join(missing)}")
    if not CARD.is_file():
        raise SystemExit(f"missing model card {CARD}")


def dry_run(src: Path) -> int:
    files = iter_allowed(src)
    total = sum(p.stat().st_size for p in files)
    print(f"src {src}")
    print(f"repo {REPO} revision {REVISION}")
    print(f"files {len(files)} bytes {total}")
    for path in files:
        rel = path.relative_to(src).as_posix()
        print(f"{path.stat().st_size:15d}  {rel}")
    return 0


def publish(src: Path) -> int:
    from huggingface_hub import HfApi

    api = HfApi()
    print(f"create_repo {REPO} public", flush=True)
    api.create_repo(REPO, repo_type="model", exist_ok=True, private=False)
    print("upload model card to main", flush=True)
    api.upload_file(
        path_or_fileobj=str(CARD),
        path_in_repo="README.md",
        repo_id=REPO,
        repo_type="model",
        revision="main",
        commit_message="Add EXL3 2.0bpw-mcg model card",
    )
    print(f"create_branch {REVISION}", flush=True)
    api.create_branch(REPO, branch=REVISION, repo_type="model", exist_ok=True)
    print(f"upload_folder {src} -> {REPO}@{REVISION}", flush=True)
    api.upload_folder(
        repo_id=REPO,
        folder_path=str(src),
        repo_type="model",
        revision=REVISION,
        allow_patterns=ALLOW,
        commit_message="Upload EXL3 2.0bpw-mcg pack",
    )
    print(f"upload model card to {REVISION}", flush=True)
    api.upload_file(
        path_or_fileobj=str(CARD),
        path_in_repo="README.md",
        repo_id=REPO,
        repo_type="model",
        revision=REVISION,
        commit_message="Add EXL3 2.0bpw-mcg model card",
    )
    print(f"DONE {REPO} revision {REVISION}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish the local EXL3 pack to Hugging Face.")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    src = args.src.expanduser().resolve()
    check_pack(src)
    if args.dry_run:
        return dry_run(src)
    _need_hub()
    return publish(src)


if __name__ == "__main__":
    sys.exit(main())
