# p2b src-sort microbench (moe-expert-dedup step 2a)

Checks `DSV41_P2B_SRC_SORT=1` (`docker/patch/widen_p2b_srcsort.py`) in
isolation. Shape is DSpark-3 verify on one TP=2 rank: m=4 rows, top-6, 384
experts, hidden 5120, inter 1152, K=2 MCG, swiglu_limit 10.

| File | Needs GPU | What |
|---|---|---|
| `make_bench.py` | no | Builds `build/` from `tests/fixtures/p2b_moe.pin.cu` and the `docker/Dockerfile` p2b patch chain, so no untracked `vllm-exl3-patched` tree is needed. |
| `sass_identity.sh` | no | Compiles the chain with and without srcsort for sm_121a in a no-GPU container with an 8 GB memory cap. Requires every SORT=0 kernel `.text` to be byte-identical to the unpatched chain (`text_identity.py`). |
| `driver.py` | yes, serve down | Bitwise check and cold/warm timing, off vs on, at duplicate rates {0, 25, 50}%. |

## Why sorting is not `evidence/p2b-ldg`

`p2b-ldg` kept the row-major item order and switched every trellis load from
`__ldcs` to `__ldg`. A repeated expert in row r+1 comes about one grid wave
after row r (~17.7 MB of gate/up stream in between), so reuse needed the lines
to survive in L2 that long. It regressed (L.A.I.L 22.87/21.22 vs 23.44/22.37).

Src-sort keeps `__ldcs`. It changes only which block takes which item.
Duplicates of one (expert, group, gate|up) get adjacent item indices, so
neighbouring blocks stream the same lines at about the same time. The reuse
distance is ~0 instead of ~1 wave.

It adds no launch or barrier. That matters because `evidence/p2b-nocoop` lost
5% with 9 launches instead of 8 grid.syncs. When off, the kernel is
byte-identical.

## Run (GPU campaign, serve down)

```bash
kernel_study/p2b_srcsort/sass_identity.sh            # CPU only; exit 0 = IDENTICAL x6
docker run --rm --gpus all -v "$PWD:/repo" -w /repo --entrypoint python3 \
  dsv41-flash-exl3-sm121:canonical-e12 kernel_study/p2b_srcsort/driver.py \
  --out results/2026-09-24-review/moe-dedup/microbench.json
```

## Gates

- `check`: `onehot_bitexact` must be true at every duplicate rate. With a
  one-hot routing weight, the float atomics add zeros, so the result does not
  depend on their order. `full_rw_bitexact` is informational. If it is false,
  `full_rw_off_repeat_bitexact` must also be false, which shows the atomics
  already vary run to run.
- dup 0 cold: `reduction_pct` must be >= -1. Sorting must not cost anything
  when there is nothing to share.
- A serve arm needs a cold `reduction_pct` >= 3 at the duplicate rate nearest
  the census mean, and a census mean dup >= 10% (`tools/moe_census.py`).
- Ceiling for reference: if the second read of a duplicate were free, the
  saving would be about dup x 0.78 (streaming share), i.e. about 19% at 25%
  duplicates. A result far below that means the hits are not landing. The next
  arm would then be an L2 `evict_last` policy on duplicate items only. It is
  not built.
