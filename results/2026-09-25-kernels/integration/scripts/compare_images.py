#!/usr/bin/env python3
"""Compare two image_dump.sh outputs (base = canonical-e13, new = review-e14).

  compare_images.py DIR BASE_TAG NEW_TAG REPO COMMIT > compare.json
Reports: python-tree file changes; dist-info (package versions) equality; vllm_exl3_c machine code
per kernel (the .text.<kernel> sections of the embedded cubins, read with
kernel_study/p2b_srcsort/text_identity.sections: is every base kernel byte-identical in new, which
kernels are new); the new image's baked /opt/dsv41-patch and sitecustomize vs REPO's docker/patch at
COMMIT (git show, not the work tree)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def sha_map(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        digest, name = line.split(None, 1)
        out[name.strip().lstrip("./") if name.startswith("./") else name.strip()] = digest
    return out


def text_sections(elf_dir: Path) -> dict[str, list[bytes]]:
    """kernel name -> machine code of each .text.<kernel> section across the extracted cubins."""
    from text_identity import sections

    funcs: dict[str, list[bytes]] = defaultdict(list)
    for cubin in sorted(elf_dir.glob("*.cubin")):
        for name, body in sections(str(cubin)).items():
            if name.startswith(".text."):
                funcs[name[len(".text."):]].append(body)
    return funcs


def demangle(names: list[str]) -> list[str]:
    try:
        res = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True, check=True)
        return res.stdout.splitlines()
    except Exception:  # noqa: BLE001
        return names


def main() -> int:
    d, base, new, repo, commit = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
    sys.path.insert(0, str(Path(repo) / "kernel_study" / "p2b_srcsort"))
    fb, fn = sha_map(d / f"{base}.files.sha256"), sha_map(d / f"{new}.files.sha256")
    changed = sorted(k for k in fb.keys() & fn.keys() if fb[k] != fn[k])
    out: dict = {
        "files": {"base": len(fb), "new": len(fn), "changed": changed,
                  "added": sorted(fn.keys() - fb.keys()), "removed": sorted(fb.keys() - fn.keys())},
        "distinfo_equal": (d / f"{base}.distinfo.txt").read_text() == (d / f"{new}.distinfo.txt").read_text(),
        "so_sha256": {t: (d / f"{t}.so.sha256").read_text().split()[0] for t in (base, new)},
    }
    sb, sn = text_sections(d / f"{base}.elf"), text_sections(d / f"{new}.elf")
    differ = sorted(k for k in sb if sorted(sb[k]) != sorted(sn.get(k, [])))
    added = sorted(sn.keys() - sb.keys())
    out["machine_code"] = {
        "cubins": {t: len(list((d / f"{t}.elf").glob("*.cubin"))) for t in (base, new)},
        "base_kernels": len(sb), "new_kernels": len(sn),
        "base_text_bytes": sum(len(b) for v in sb.values() for b in v),
        "base_kernels_identical": len(sb) - len(differ),
        "base_kernels_differ": dict(zip(differ, demangle(differ))),
        "new_only": demangle(added),
    }
    # the new image's baked patch dir / sitecustomize vs the repo at COMMIT
    ls = subprocess.run(["git", "-C", repo, "ls-tree", "-r", "--name-only", commit, "docker/patch"],
                        capture_output=True, text=True, check=True).stdout.split()
    want = {}
    for p in ls:
        blob = subprocess.run(["git", "-C", repo, "show", f"{commit}:{p}"], capture_output=True, check=True).stdout
        want[p[len("docker/patch/"):]] = hashlib.sha256(blob).hexdigest()
    got, base_patch = sha_map(d / f"{new}.patch.sha256"), sha_map(d / f"{base}.patch.sha256")
    extra = sorted(got.keys() - want.keys())
    out["baked_patch_vs_commit"] = {
        "equal": want == got, "files": len(want),
        "differ": sorted(k for k in want.keys() & got.keys() if want[k] != got[k]),
        "missing": sorted(want.keys() - got.keys()), "extra": extra,
        # COPY never deletes: files an older layer baked into /opt/dsv41-patch stay. run.sh
        # bind-mounts docker/patch over the whole dir, so the serve never sees them.
        "extra_all_inherited_unchanged_from_base": all(base_patch.get(k) == got[k] for k in extra),
    }
    bp = out["baked_patch_vs_commit"]
    baked_ok = not bp["differ"] and not bp["missing"] and bp["extra_all_inherited_unchanged_from_base"]
    site = (d / f"{new}.site.sha256").read_text().split()[0]
    out["baked_sitecustomize_equals_commit"] = site == want.get("sitecustomize.py")
    out["pass"] = bool(not out["files"]["changed"] and not out["files"]["added"] and not out["files"]["removed"]
                   and out["distinfo_equal"] and sb and not differ and baked_ok
                   and out["baked_sitecustomize_equals_commit"])
    print(json.dumps(out, indent=1))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
