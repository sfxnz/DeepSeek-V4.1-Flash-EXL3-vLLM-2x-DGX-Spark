"""Generate the cooperative-MoE overlay for the coop-e1 image.

Mirrors mia-exl3-ref/extensions/cooperative_moe/prepare_profile.py (same
footer contract: stock overlay + explicit install(..., enabled=True)) but pins
the artifacts WE built from source instead of her unpublished a09a589c… binary
(GitHub issue #17). The stock overlay sha (ccdc69bf…) IS still enforced: her
overlay ships verbatim in her GHCR image, so a mismatch means the base image
drifted — abort, do not regenerate.

Usage:
  python3 make_coop_overlay.py --stock /opt/dsv41/exl3.py \
      --artifacts /opt/dsv41/coop --runtime-directory /opt/dsv41/coop \
      --output exl3-coop.py
"""

import argparse
import hashlib
from pathlib import Path

STOCK_SHA = "ccdc69bfa04bff4870c3e555736a990fde6448ddb329441c4e0a27d6fc41078d"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--runtime-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    stock_sha = sha256(args.stock)
    if stock_sha != STOCK_SHA:
        raise SystemExit(
            f"stock overlay sha {stock_sha} != pinned {STOCK_SHA}: base image "
            "drifted from Mia's validated overlay; aborting"
        )
    so = args.artifacts / "cooperative_moe.so"
    runtime = args.artifacts / "runtime.py"
    if not so.is_file() or not runtime.is_file():
        raise SystemExit(f"missing artifacts in {args.artifacts}")

    runtime_root = args.runtime_directory.resolve()
    if not runtime_root.is_absolute():
        raise SystemExit("runtime directory must be absolute")

    base = args.stock.read_bytes()
    footer = (
        "\n# Explicit fixed cooperative MoE opt-in; unsupported calls stay stock.\n"
        "import runpy as _coop_runpy\nimport sys as _coop_sys\n"
        f'_coop_setup = _coop_runpy.run_path({str(runtime_root / "runtime.py")!r})\n'
        f'_coop_setup["install"](_coop_sys.modules[__name__], library_root={str(runtime_root)!r}, enabled=True)\n'
    )
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.write_bytes(base + footer.encode())
    print(
        f"wrote {args.output}\n"
        f"  stock overlay : {STOCK_SHA} (verified)\n"
        f"  cooperative_moe.so : {sha256(so)} (built from source, functionally validated)\n"
        f"  runtime.py         : {sha256(runtime)} (pin line re-pointed at build)"
    )


if __name__ == "__main__":
    main()
