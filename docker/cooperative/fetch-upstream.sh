#!/usr/bin/env bash
# Vendor Mia's pinned exllamav3_ext header tree for the cooperative_moe build.
# Host-side helper (equivalent of her archive_upstream.sh, which needs a git
# checkout; the codeload tarball needs none). Run from docker/:
#   bash cooperative/fetch-upstream.sh
set -euo pipefail

pin=02aef45cd681b960a00afcd0749a4ab99e6c1bfe
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
out="$here/upstream"
mkdir -p -- "$out"
test -z "$(find "$out" -mindepth 1 -maxdepth 1 -name 'exllamav3*' -print -quit)" || {
  echo "upstream/ already contains an exllamav3 tree; refusing to overwrite" >&2
  exit 1
}

# codeload serves any commit sha, not just refs.
curl -fsSL "https://codeload.github.com/turboderp-org/exllamav3/tar.gz/${pin}" \
  | tar -xz -C "$out"
test -d "$out/exllamav3-${pin}/exllamav3/exllamav3_ext"
# Normalize to the layout her build.sh expects: EXTRACTED_TREE/exllamav3/exllamav3_ext
mv "$out/exllamav3-${pin}" "$out/exllamav3"
# Build context wants one file: repack only the exllamav3 tree (headers, ~2 MB).
tar -czf "$out/exllamav3.tar.gz" -C "$out" exllamav3
rm -rf "$out/exllamav3"
sha256sum "$out/exllamav3.tar.gz"
