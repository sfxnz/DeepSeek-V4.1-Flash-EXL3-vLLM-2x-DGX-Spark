# docker/patch/attic

Patches no build and no boot path uses. Nothing here is referenced by any
`docker/Dockerfile*`, `sitecustomize.py` or `apply_*.py`. The verdicts live in
[flags.md](../../../flags.md) ("Rejected / unwired" table and the round
entries). The `tests/test_*.py` files for these patches still load them from
here, so their anchors keep being checked.

They are still copied into the image (`COPY patch`) and scp'd to the worker
with `$PATCH_DIR`, but nothing imports them. To re-arm one, move it back and
wire it in `sitecustomize.py` or a Dockerfile.

| file | what | why it is here |
|---|---|---|
| `engram_helper.py`, `upstream_apply_engram_patch.py` | upstream Tech2Wild/Kai disk-Engram port (2026-09-10) | the build runs `apply_engram_disk.py` instead; the two only reference each other |
| `engram_prefetch.py` (v1), `engram_prefetch_v2.py` | earlier next-step fadvise prefetch | superseded by `engram_prefetch_v3.py`, which defines its v1/v2 markers itself and imports neither (flags.md round 11, R21) |
| `prefer_b12x_bmm.py` | b12x before emulation for the MXFP8 BMM (MLA wo_a) | unwired; `sitecustomize.py` wires `prefer_b12x_mxfp8.py` only |
| `widen_mla_kv_buf.py` | MLA decode KV buffer widen | unwired (flags.md rejected table) |
| `sm120_wo_a.py` | b12x wo_a dense GEMM | unwired: waves 21.23/23.00, not faster than torch.bmm |
| `c1_graph_safe_adaptive.py` | extra graphs / pin-budget | unwired: failed L.A.I.L (22.5, 21.1) |
| `widen_p2b_{fma,cp16,cpasync,ldg,pf4,nocoop}.py` | alternate p2b decode kernels | rejected or reverted (flags.md p2b row, `evidence/p2b-*`) |

Kept in `docker/patch/` on purpose:

- `widen_p2b_mma.py`: `docker/Dockerfile.mma` builds it.
- `widen_mla_tile32.py`: `sitecustomize.py` imports it. The block is a no-op
  on the live image (`DSV4_CAND_WINDOW = 64`), but moving it would change the
  boot path.
- `pfg8_*.py`, `g8_stream_feed.py`: `docker/Dockerfile.g8` and
  `sitecustomize.py` use them.
- `essay_corpus.json`, `essay_markov.json`: loaded by `sm120_page.py` and
  tested in `tests/test_sm120_page.py`.
