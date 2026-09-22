# Cooperative MoE — source build on Mia's image (`coop-e1`)

Status: **BUILT** — `cooperative_moe.so` sha256
`ca94632f6c6bd06a88ef2424c9fb23320e1ca81f811827cb11749d4b6e037441`,
image `ghcr.io/miaai-lab/deepseek-v4.1-flash-exl3-2x-dgx-sparks:coop-e1`
(ID `sha256:d5291f0872b891fa79d72963a8709f16e35fc9c0b520b5dd4d57ecc178836390`).
Built with **no GPU** and the live serve untouched; host MemAvailable never
dropped below ~12 GiB (floor: abort below 10 GiB).

All `file:line` citations below refer to Mia's kit at
`/home/sfxnz/projects/experiments/mia-exl3-ref` (MIT) unless prefixed
`ours:` (this repo).

---

## 0. Study answers (task 1)

### (a) Exact build inputs and commands

Inputs (her `extensions/cooperative_moe/build.sh`):

- `native/cooperative_moe.cu` — copied to `<out>/goal50_fixed_coop.cu`
  (build.sh:23). C ABI wrapper; pins H=5120, I=1152, TOPK=6, ROWS_MAX=8;
  selects wide/narrow kernel variants per Blackwell auto-selection rules
  (cooperative_moe.cu:12-30).
- `native/cooperative_moe_kernel.cuh` — copied to
  `<upstream>/exllamav3/exllamav3_ext/quant/goal50_fixed_coop_kernel.cuh`
  (build.sh:24-25). The kernel; includes upstream `util.h`, `util.cuh`,
  `compat.cuh`, `exl3_gemv_kernel.cuh`, `hadamard_inner.cuh`,
  `exl3_moe_coop.cuh` (cooperative_moe_kernel.cuh:22-28).
- `native/exl3_moe_coop.cuh` — copied to
  `<upstream>/exllamav3/exllamav3_ext/quant/exl3_moe_coop.cuh`
  (build.sh:26-27). Param struct + activation constants; ATen declarations
  compiled out via `GOAL50_COOP_NATIVE_ONLY` (exl3_moe_coop.cuh:6-8) →
  **no torch headers needed, standalone C ABI**.
- Upstream tree: exllamav3 @ `02aef45cd681b960a00afcd0749a4ab99e6c1bfe`
  (archive_upstream.sh:4), `exllamavav3/exllamav3_ext` subtree only.
- `runtime.py` — copied verbatim into the output dir (build.sh:28).

Compile command (build.sh:33-38), reproduced exactly in our build:

```
nvcc -std=c++17 -O3 --use_fast_math -lineinfo --expt-relaxed-constexpr \
  -gencode arch=compute_121a,code=sm_121a -shared -Xcompiler -fPIC --ptxas-options=-v \
  -I <out>/upstream/exllamav3/exllamav3_ext \
  <out>/goal50_fixed_coop.cu -o <out>/cooperative_moe.so
```

Her toolchain notes: the recipe image has no `git` (build.sh:4-5,
archive_upstream.sh:7-8) — headers must be archived host-side; her own
Dockerfile fetches upstream via GitHub tarball + curl (Dockerfile:91). The
vLLM image keeps CUDA headers under `$PY_SITE/nvidia/cu13/include`, which
must be on `CPATH` (her Dockerfile:84-99). Our build ran her `build.sh`
unmodified inside her GHCR image with those env additions (see
`docker/Dockerfile.coop` step 3).

### (b) Load-time verification — is the sha pin enforced at runtime? **YES.**

`runtime.py:19` defines `SHA256 = "a09a589c…08d78"`;
`runtime.py:83-85` (`CoopLaunch.__init__`) **asserts**
`sha256(cooperative_moe.so) == SHA256` immediately before `C.CDLL(path)`.
So a rebuilt .so is rejected **at load time**, inside
`process_weights_after_loading` (runtime.py:175-184) — i.e. at model load on
every rank. Additionally `prepare_profile.py:24,42-44` refuses to generate
an overlay from artifacts whose binary/adapter hashes differ
(`BINARY_SHA`/`ADAPTER_SHA`, prepare_profile.py:15-16), and her quickstart
forbids repinning without re-running the 54-case GPU gate
(docs/cooperative-moe-quickstart.md:111-118).

**Can our fork skip/replace with a functional gate? Yes, safely:**

1. The pin is a plain module constant; our build re-points it to the digest
   of the .so built in the same image layer (Dockerfile.coop step 4,
   fail-closed sed: build aborts unless exactly one pin line changed and no
   `a09a589c` residue remains). The load-time assert **stays active** —
   against our binary. Tampering with the .so after image build still fails
   loudly at load.
2. Overlay generation: ours:`cooperative/make_coop_overlay.py` mirrors her
   prepare_profile.py (same footer contract) but pins our artifacts. It still
   enforces her **stock overlay** sha `ccdc69bf…` — a mismatch means her
   GHCR base drifted, which must abort, not regenerate.
3. Acceptance shifts from "binary provenance" to **functional gates**
   (§3 below): ABI check (`goal50_coop_abi()==1`, runtime.py:86), the
   boot log line, the 17×19=323 smoke, her CPU dispatch suite (113 checks),
   and — in the coop maintenance window — her 54-case GPU integration gate
   (test_cuda_integration.py) before any real traffic. This is exactly the
   "re-run the GPU gate + explicit pin update" promotion path her docs
   describe (docs/cooperative-moe.md:126-129, quickstart:282-287), minus the
   unpublished binary (GitHub issue #17).

Not bit-reproducible, by design: `-lineinfo` embeds paths and the GNU
build-id varies per link (artifacts/README.md:7-10; build.sh:29-32). Her
docs explicitly say a matching hash is *not expected* from a clean rebuild
(quickstart:114-118).

### (c) What `EXL3_OVERLAY_HOST` expects

- An **absolute path to a Python overlay file on the host** (her start.sh:168
  defaults to `overlay/exl3.py`; must exist, start.sh:565). The launcher
  copies it to the worker (start.sh:1306) and installs it into the container
  at `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py`
  (her Dockerfile:152-160 does the image-baked equivalent; the runtime mount
  overrides it).
- The cooperative profile = her stock overlay (sha `ccdc69bf…`, =
  prepare_profile.py:14 `STOCK_SHA`) + a footer that runs
  `runtime.py` and calls `install(module, library_root=…, enabled=True)`
  (prepare_profile.py:27-36). The footer's `library_root` must be a
  **container-visible absolute directory containing both
  `cooperative_moe.so` and `runtime.py`, identical on BOTH ranks**
  (quickstart:120-125; default `/root/.cache/vllm/cooperative_moe`,
  runtime.py:157 — reachable because each host's vllm cache is mounted at
  `/root/.cache/vllm`).
- `.env` beats the process env: an existing `EXL3_OVERLAY_HOST` in `.env`
  overrides a command-line choice (quickstart:205-209) — replace it there.
- `DSV41_COOPERATIVE_MOE=1` alone activates **nothing** (README.md:12-14,
  runtime.py:160-161 — the generated overlay passes `enabled=True`, which is
  what actually arms it).

**In our coop-e1 image**: the overlay and artifacts are baked at
`/opt/dsv41/coop/` (image-local, not under `/root/.cache/vllm` — her
launcher mounts host caches over that path, which would shadow baked files).
Serve-time selection is still `EXL3_OVERLAY_HOST=/opt/dsv41/coop/exl3-coop.py`
if the launcher bind-mounts it, or stage the two artifact files + generated
overlay via her cache-mount route (quickstart §4) — see §4 below.

### (d) Eligibility gating + serial-stream contract

Layer eligibility — `runtime.py:33-55` (`layer_eligible`), all must hold:
- uniform K2 or K3 across gate/up/down (`_exl3_k_gate ∈ {2,3}`,
  `_exl3_k_up == _exl3_k_down == bits`)
- mul1, not mcg (`_exl3_mul1` true, `_exl3_mcg` false)
- hidden 5120, local intermediate 1152 (TP2)
- 1..384 local experts, `_exl3_ptrs` present, `_exl3_fused_temps` not None
- every expert of the layer agrees (runtime.py:51-55)

These attrs are set by **her overlay only** (overlay/exl3.py:1512-1516) —
see §1 for why that locks us out of canonical-e12.

Call eligibility — `runtime.py:58-73` (`call_eligible`): 2-D input,
1..8 rows × 5120, ids/weights `(rows, 6)` int64/float, CUDA tensors on the
native's device, finite non-negative routing limit. Everything else (K4 MTP,
prefill batches, >8 rows, mixed metadata) → stock dispatch (runtime.py:187-192).

Serial-stream contract — `runtime.py:162-168` (`install`): **raises unless
`DSV41_EXL3_SERIAL_STREAMS=1` AND `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`**.
Rationale (her .env.example): exllamav3 keeps ONE lock buffer per device; two
EXL3 GEMMs in flight on different streams deadlock (her boots 11-14,
2026-09-12). Scratch is allocated after weight load, never during graph
capture (runtime.py:80-82 asserts not stream-capturing; README table).
No retry after a partial CUDA launch failure — it raises (runtime.py:149-151).

### (e) Her measured claims + caveats

Claims (docs/cooperative-moe.md:26-34, paired A/B, TP2, 600k ctx, DSpark k=3):

| Workload | Stock | Coop | Δ |
|---|---:|---:|---:|
| Poetry decode (3-seed median) | 23.62 | 29.26 tok/s | **+23.9%** |
| Coding decode (3-seed median) | 38.76 | 42.96 | +10.8% |
| Incident reasoning (3-seed median) | 31.92 | 41.26 | +29.3% |
| Reference C1 (greedy, 1 pass) | 31.45 | 40.23 | +27.9% |
| Reference C2 combined (2 streams) | 45.87 | 61.06 | +33.1% |
| Uncached 32K prefill | 1137.76 | 1135.35 | −0.2% (unchanged) |

Caveats (same doc): stock reference had **one** repetition vs three for coop
(:39-40); all nine paired full responses differed — not an identical-output
control (:46-47); acceptance is workload-dependent (23-24% poetry,
:42-43); arithmetic is **not bit-exact** with stock — 35/645,120 elements
failed strict rtol=atol=1e-3 on actual inputs (max err 0.1123% of stock
peak, :93-97); synthetic-scale strict failures passed only the
peak-normalized 0.3% criterion (:99-105); sanitizer coverage = first 18
launches of 4 bounded runs (:107-109); near-600K prefill and prolonged
burn-in not repeated (:130-131). Decode ≤8 physical rows only; K4 MTP
stays stock.

### (f) Rollback path

Her quickstart §8 (docs/cooperative-moe-quickstart.md:260-278): stop both
ranks, restore the saved pre-coop `.env` (`original.env` from step 1),
`env -u` the shell overrides, restart. For our lane specifically:
`EXL3_OVERLAY_HOST` back to the stock `overlay/exl3.py` path (or unset her
override), keep the stream knobs or restore per the saved env, boot the
**`2.9bpw`** tag (not coop-e1). Versioned coop artifacts may remain staged —
their presence activates nothing (README.md:60-62, quickstart:277-278).
Our layer adds: `docker tag`/re-pull the base `2.9bpw` image is already
local, so rollback needs no registry access.

---

## 1. Why the coop lane runs on HER image, not canonical-e12

Her runtime patches `Exl3MoEMethod.process_weights_after_loading` and
`apply_exl3_fused_moe` (runtime.py:171-172,194-195) — symbols of **her
overlay exl3.py** (sha `ccdc69bf…`) — and gates every layer on
`_exl3_k_gate/_exl3_k_up/_exl3_k_down/_exl3_mul1/_exl3_mcg` attrs that only
her overlay sets (overlay/exl3.py:1512-1516).

Our canonical-e12 stack serves EXL3 through the **vllm_exl3 plugin fork**
(`VLLM_PLUGINS=vllm_exl3`, ours:`docker/Dockerfile:39`,
ours:`run.sh:278`), whose module
`/usr/local/lib/python3.12/dist-packages/vllm_exl3/exl3.py` (sha
`afc6da3e2213e9580046f13e8cd561e5f12e19df2ba93ea551a30ce1e3305a87`) does
expose `Exl3MoEMethod`/`apply_exl3_fused_moe`/`_exl3_ptrs`/`_exl3_inners`/
`_exl3_fused_temps` (verified in-image) but sets **none** of the
K/mul1/mcg attrs — it keeps `_exl3_codebook_flags` instead
(vllm_exl3/exl3.py:2044-2060, in-image). Consequence: `layer_eligible()`
returns False on every layer; the adapter installs (log line prints!) but
**no layer ever dispatches to the coop kernel** — silently inert. Porting
would mean re-implementing her attr contract inside our plugin fork: real
work with correctness risk, not a boot flag. Decision (parent course
correction): the mul1 lane boots **her GHCR image**; our canonical stack is
untouched. Her MCG pack is ineligible anyway (kernel is mul1-only,
runtime.py:41); this is for the Mia-AiLab 2.9bpw pack (uniform K2/K3 mul1 —
in the HF cache on both sparks as
`models--Mia-AiLab--DeepSeek-V4.1-Flash-EXL3-2.9bpw`, quant_method=exl3
verified).

---

## 2. Build recipe (what was executed)

1. **Vendor sources** (byte-for-byte from her kit): ours:`docker/cooperative/`
   — `build.sh`, `runtime.py`, `prepare_profile.py` (reference; superseded
   by `make_coop_overlay.py`), her CPU tests, `native/` (cu + cuh + MIT
   license notice).
2. **Vendor upstream headers** — ours:`docker/cooperative/fetch-upstream.sh`:
   codeload tarball of exllamav3 @ `02aef45c…` (her archive_upstream.sh pin;
   codeload serves arbitrary SHAs, no git needed), normalized to
   `upstream/exllamav3.tar.gz` (sha256
   `70730c69a1be1528dd8e24afa540994569cd3d6f3f5b4f110b2478625185dc79`,
   754 entries).
3. **Build** (no GPU; MAX_JOBS=2; host MemAvailable monitored ≥12 GiB
   throughout; abort floor 10 GiB never approached):

   ```bash
   docker build -f docker/Dockerfile.coop --build-arg MAX_JOBS=2 \
     -t ghcr.io/miaai-lab/deepseek-v4.1-flash-exl3-2x-dgx-sparks:coop-e1 docker/
   ```

   Inside: extract headers → run **her unmodified `build.sh`** (nvcc
   CUDA 13.0 r13.0, `-gencode arch=compute_121a,code=sm_121a`, cu13 headers
   on CPATH per her Dockerfile:84-99) → re-point the runtime.py pin (§0b) →
   install to `/opt/dsv41/coop/` → generate `exl3-coop.py` overlay → CPU
   self-checks → cleanup.

### Recorded digests

| Artifact | sha256 |
|---|---|
| `cooperative_moe.so` (built from source) | `ca94632f6c6bd06a88ef2424c9fb23320e1ca81f811827cb11749d4b6e037441` |
| `runtime.py` (pin re-pointed) | `078a1fe9a3ec2df676b1185c3ae9f136de069bf21df9c413d0c6580b6021680a` |
| overlay `exl3-coop.py` | `d8ec90a4946e6e09b5145cecd732864dda74bf39754cae05a9fe0ca4eba5bbf8` |
| upstream headers tarball | `70730c69a1be1528dd8e24afa540994569cd3d6f3f5b4f110b2478625185dc79` |
| base image (her validated ID) | `sha256:4cdba4e946da2d19bf5b5a20c6d3a1a4bf421fa4d6db5082f271a986168176cb` |
| **coop-e1 image** | `sha256:d5291f0872b891fa79d72963a8709f16e35fc9c0b520b5dd4d57ecc178836390` |

In-image verification already run (CPU-only, `--entrypoint bash`, no
`--gpus`): `goal50_coop_abi() == 1` via ctypes; overlay byte-verified =
stock overlay + footer; her `test_dispatch.py` →
`{'status': 'pass', 'checks': 113}`; ptxas log shows all kernels compiled
for `sm_121a`, 0 spill stores/loads on the rot kernel.

Re-shipping to spark2 (when the coop boot is scheduled): her
`IMAGE_SHIP=rsync` path (quickstart §2 uses `docker pull` on both nodes —
coop-e1 is local-only, so ship the tar:
`docker save coop-e1 | ssh spark2 docker load`).

---

## 3. Functional validation plan (coop boot, later — separate from the
mul1 stock baseline boot)

The next two boots are **one lever each**:

1. **Boot A — mul1 lane stock baseline** (her `2.9bpw` image, no coop
   overlay): proves her stack + her pack + our knobs on our hardware;
   captures baseline numbers for the A/B. Config: ours:`docker/coop-env-preview.txt`.
2. **Boot B — coop**: same but `EXL3_OVERLAY_HOST` → coop overlay. Never
   enable coop in the same boot that changes anything else.

Sequence for Boot B (adapts her quickstart §§4-7 to our image):

1. Pre-flight (no GPU): `docker run --rm --entrypoint sha256sum <coop-e1>
   /opt/dsv41/coop/cooperative_moe.so` → must print `ca94632f…`.
2. Stage artifacts identically on both ranks — either bake route
   (`/opt/dsv41/coop/`, present in the image on both nodes after
   `docker save|load`) or her cache-mount route (quickstart:127-143).
   The overlay's `library_root` must point at the rank-visible directory.
3. **GPU gate** (maintenance window, serve down — per parent constraint
   this is *scheduled by the operator*, not by this doc): her
   `test_cuda_integration.py` 54-case suite per node, requiring final
   `status: pass` + exit 0 (quickstart:146-174). Our coop-e1 ships the test
   at `/opt/dsv41/coop/test_cuda_integration.py` (copied into
   docker/cooperative/). **This gate is the acceptance criterion that
   replaces the binary pin.**
4. Boot with `EXL3_OVERLAY_HOST=/opt/dsv41/coop/exl3-coop.py` (or staged
   path) + `DSV41_EXL3_SERIAL_STREAMS=1` +
   `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` + the rest of the Boot-A knobs.
5. Verify activation on BOTH ranks:
   `docker logs <rank> 2>&1 | grep -F 'Fixed-shape cooperative MoE enabled'`
   (the exact log line, runtime.py:198-200) and overlay digest match
   (quickstart:236-240).
6. Smoke: `/health`, then the 17×19 chat completion → expect `323`,
   `finish_reason: stop` (quickstart:247-258).
7. Correctness/quality: ours:`tests/correctness.sh` equivalent against the
   coop port + our four-number harness (ours:`tools/four_numbers.sh`,
   HTTP-only) for the A/B vs Boot A. Prose/coding decode per her protocol
   (docs/cooperative-moe.md:49-84) if we want comparable numbers.

**Abort criteria (our floors, both boots):** MemAvailable < 10 GiB at any
point pre/post boot → stop, roll back. Any coop-path CUDA launch error
(runtime.py:149-151 raises; no auto-retry) → roll back. Missing activation
log line on either rank → treat as inert, roll back. Quality regression in
correctness gate → roll back (numerical tolerance context: §0e).

## 4. Rollback

Boot A → our normal serve: `./stop.sh` in her recipe (or her `stop.sh`),
then our `run.sh` as usual — nothing about our stack changed. Boot B →
Boot A config (drop the `EXL3_OVERLAY_HOST` override, restore any env from
the saved pre-boot copy; her quickstart §8 pattern, §0f above). Images stay
local; no registry dependency.

## 5. Files (this change)

- ours:`docker/Dockerfile.coop` — build layer (base = her `2.9bpw` GHCR tag).
- ours:`docker/cooperative/` — vendored kit (native/, build.sh, runtime.py,
  prepare_profile.py, her tests), `fetch-upstream.sh`,
  `make_coop_overlay.py` (pin-repointed overlay generator),
  `upstream/exllamav3.tar.gz` (pinned headers), this BUILD.md,
  `../coop-env-preview.txt`.
