#!/usr/bin/env python3
"""Full Viterbi re-encode of the routed experts, resumable across both Sparks.

Serving format is 2.0bpw-mcg's: 2.0 bpw, MCG codebook (p2b cb=1), K=2, the same
tensor names, shapes, dtypes and shard assignment, stock (non-G8) trellis
layout. The existing loader and p2b kernels (coop path included) read it
unchanged. Every unit is checked against the same shard of the reference pack:
identical safetensors layout (keys, dtypes, shapes, data offsets), byte-equal
non-expert tensors and .mcg markers.

Encoder: exllamav3 1.5.1 quantize_tiles (tail-biting Viterbi, image
dsv41-quant151) through the recipe's own _quantize_fast (meta-H q_fallback,
skip_g_scale), then refit_scales with H = I (requant_probe.refit_identity; the
s1 probe measured it equal to upstream refit_scales within 1e-8 relerr). The
refit is kept per tensor only when it does not raise relerr. Source: the
official MXFP4 snapshot, read-only. s1 probe: relerr 0.2616 vs 0.3773 stock.

Work unit = one output shard (model-000NN, 1152 tensors). Shared state is on
spark1 under OUT/state:
  units.json        unit -> shard file (written by the first worker)
  claims/<NN>/      atomic mkdir = claim; owner.json, heartbeat.json inside
  done/<NN>.json    verified unit: node, sha256, size, relerr stats
  manifest.json     aggregate of done/, refreshed after every unit
  DONE              written once every unit has a done record
spark1 claims locally in ascending order. spark2 claims over ssh in
descending order. A worker resumes its own unfinished claims. Outputs go to
OUT/shards/ on the node that encoded them, with OUT/units/<NN>.json holding
per-tensor rows. OUT/PLAN.txt has the operating commands.

Subcommands:
  run         worker loop (in the dsv41-quant151 container, GPU in-process)
  encode      one unit without claiming (correctness check)
  status      progress, in-flight units, per-node rate, ETA (host, stdlib)
  launch-cmd  print the docker command that starts a worker on a node
  assemble    gather all units on this node, build snapshots/<rev> (host)
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import socket
import struct
import subprocess
import sys
import time
import traceback
import zlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from pack_meta import is_routed_expert_tensor, is_routed_expert_weight  # noqa: E402

OUT = Path("/home/sfxnz/projects/data/dsv41-requant-viterbi")
SRC_REL = "models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277"
PACK_REPO = "models--sfxnz--DeepSeek-V4.1-Flash-EXL3"
REF_REV = "2.0bpw-mcg"  # the served experts; lmhead-mxfp8 links its shards 3-42 here
BASE_REV = "2.0bpw-mcg-lmhead-mxfp8"
NEW_REV = "2.0bpw-mcg-viterbi-lmhead-mxfp8"
IMAGE = "dsv41-quant151"
CONTAINER = "dsv41-requant"
STATE_HOST = "spark1"
K = 2
CODEBOOK = "mcg"
TILE_CHUNK = 4608  # tiles per quantize_tiles call; bitwise equal to 256
GATE_RATIO = 0.97  # unit mean relerr must be <= 0.97x stock (requant-probe.md gate)
TENSORS_PER_UNIT = 1152
EXPERT_SUFFIXES = (".trellis", ".suh", ".svh")
HEARTBEAT_EVERY = 32
STALE_HEARTBEAT_S = 15 * 60
_SHARD_RE = re.compile(r"^model-(\d{5})-of-(\d{5})\.safetensors$")


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def default_hub() -> Path:
    """HF hub dir: /cache/huggingface/hub in the container, ~/.cache/... on a host."""
    if os.environ.get("HF_HUB"):
        return Path(os.environ["HF_HUB"])
    in_container = Path("/cache/huggingface/hub")
    return in_container if in_container.is_dir() else Path.home() / ".cache/huggingface/hub"


# ---------------------------------------------------------------- pure logic


def units_from_index(weight_map: dict[str, str]) -> dict[str, str]:
    """unit ('00003') -> shard file, for shards holding routed-expert weights."""
    out: dict[str, str] = {}
    for name, fname in weight_map.items():
        if not is_routed_expert_weight(name):
            continue
        m = _SHARD_RE.match(fname)
        if m is None:
            raise ValueError(f"unexpected shard name {fname}")
        out[m.group(1)] = fname
    return dict(sorted(out.items()))


def ordered(units, order: str) -> list[str]:
    if order not in ("asc", "desc"):
        raise ValueError(f"order must be asc or desc, not {order}")
    return sorted(units, reverse=order == "desc")


def tensor_seed(stem: str) -> int:
    """Per-tensor RNG seed (su/sv sign flips): independent of node and order."""
    return zlib.crc32(stem.encode())


def read_header(path: Path) -> tuple[bytes, dict]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        raw = fh.read(n)
    return raw, json.loads(raw)


def layout_problems(new: dict, ref: dict) -> list[str]:
    """Differences in keys, dtypes, shapes or data offsets between two headers."""
    new = {k: v for k, v in new.items() if k != "__metadata__"}
    ref = {k: v for k, v in ref.items() if k != "__metadata__"}
    probs = []
    missing, extra = sorted(set(ref) - set(new)), sorted(set(new) - set(ref))
    if missing:
        probs.append(f"missing {len(missing)} keys, e.g. {missing[:3]}")
    if extra:
        probs.append(f"extra {len(extra)} keys, e.g. {extra[:3]}")
    for key in sorted(set(new) & set(ref)):
        for field in ("dtype", "shape", "data_offsets"):
            if new[key][field] != ref[key][field]:
                probs.append(f"{key}.{field}: {new[key][field]} != {ref[key][field]}")
    return probs


def summarize(xs) -> dict:
    xs = sorted(float(x) for x in xs)
    if not xs:
        return {"n": 0}

    def q(p: float) -> float:
        return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]

    return {"n": len(xs), "mean": sum(xs) / len(xs), "min": xs[0], "p50": q(0.5),
            "p90": q(0.9), "p99": q(0.99), "max": xs[-1]}


def unit_gate(rows: list[dict], ratio: float = GATE_RATIO) -> dict:
    """Gate fixed before the run: every relerr finite, and the unit mean relerr
    at most ratio x the stock pack's mean on the same tensors."""
    final = [r["final"] for r in rows]
    stock = [r["stock"] for r in rows]
    finite = all(math.isfinite(x) for x in final + stock)
    mf = sum(final) / len(final) if final else float("nan")
    ms = sum(stock) / len(stock) if stock else float("nan")
    ok = bool(rows) and finite and mf <= ratio * ms
    return {"ok": ok, "finite": finite, "mean_final": mf, "mean_stock": ms,
            "ratio": mf / ms if ms else float("nan"), "limit": ratio,
            "mse_ratio": (mf / ms) ** 2 if ms else float("nan")}


def unit_stats(rows: list[dict]) -> dict:
    out = {k: summarize(r[k] for r in rows) for k in ("final", "viterbi", "stock")}
    out["refit_kept"] = sum(1 for r in rows if r["refit"])
    for kind in ("w1", "w2", "w3"):
        sub = [r["final"] for r in rows if r["tensor"].endswith("." + kind)]
        out[f"final_{kind}_mean"] = sum(sub) / len(sub) if sub else None
    return out


def sha256_file(path: Path, bufsize: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def eta_seconds(remaining_tensors: float, rates: dict[str, float]) -> float | None:
    """Seconds to finish at the summed per-node rates (tensors/s)."""
    total = sum(r for r in rates.values() if r and r > 0)
    return remaining_tensors / total if total > 0 else None


# ---------------------------------------------------------------- shared state


class LocalStore:
    """State dir on this node's disk."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def mkdir(self, rel: str) -> bool:
        try:
            (self.root / rel).mkdir()
            return True
        except FileExistsError:
            return False

    def write(self, rel: str, text: str) -> None:
        p = self.root / rel
        tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
        tmp.write_text(text)
        os.replace(tmp, p)

    def read(self, rel: str) -> str | None:
        try:
            return (self.root / rel).read_text()
        except FileNotFoundError:
            return None

    def ls(self, rel: str = "") -> list[str]:
        try:
            return sorted(x for x in os.listdir(self.root / rel) if not x.startswith("."))
        except FileNotFoundError:
            return []


class SshStore(LocalStore):
    """Same state dir on another node, one shell command per operation.

    mkdir is still the atomic claim: it runs on the state node's filesystem.
    """

    def __init__(self, ssh: list[str], root: Path, runner=subprocess.run):
        super().__init__(root)
        self.ssh = list(ssh)
        self.runner = runner

    def _sh(self, script: str, stdin: str | None = None):
        r = self.runner(self.ssh + [script], input=stdin, capture_output=True, text=True, timeout=120)
        if r.returncode == 255:
            raise RuntimeError(f"ssh failed: {r.stderr.strip()[-300:]}")
        return r

    def mkdir(self, rel: str) -> bool:
        p = shlex.quote(str(self.root / rel))
        out = self._sh(f"mkdir -- {p} 2>/dev/null && echo CLAIMED || {{ test -d {p} && echo EXISTS; }}").stdout.strip()
        if out in ("CLAIMED", "EXISTS"):
            return out == "CLAIMED"
        raise RuntimeError(f"remote mkdir {rel} failed")

    def write(self, rel: str, text: str) -> None:
        p = self.root / rel
        tmp, dst = shlex.quote(str(p.with_name(f".{p.name}.{os.getpid()}.tmp"))), shlex.quote(str(p))
        r = self._sh(f"cat > {tmp} && mv -f -- {tmp} {dst}", stdin=text)
        if r.returncode:
            raise RuntimeError(f"remote write {rel} failed: {r.stderr.strip()[-300:]}")

    def read(self, rel: str) -> str | None:
        p = shlex.quote(str(self.root / rel))
        r = self._sh(f"if [ -f {p} ]; then cat -- {p}; else exit 3; fi")
        if r.returncode == 3:
            return None
        if r.returncode:
            raise RuntimeError(f"remote read {rel} failed: {r.stderr.strip()[-300:]}")
        return r.stdout

    def ls(self, rel: str = "") -> list[str]:
        p = shlex.quote(str(self.root / rel))
        r = self._sh(f"if [ -d {p} ]; then ls -1A -- {p}; fi")
        if r.returncode:
            raise RuntimeError(f"remote ls {rel} failed: {r.stderr.strip()[-300:]}")
        return sorted(x for x in r.stdout.split() if not x.startswith("."))


def ssh_argv(host: str, ssh_dir: Path) -> list[str]:
    """ssh that works as uid 1000 inside the container (config + known_hosts by path)."""
    return ["ssh", "-F", str(ssh_dir / "config"), "-o", f"UserKnownHostsFile={ssh_dir / 'known_hosts'}",
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15", host]


def make_store(args) -> LocalStore:
    root = Path(args.out) / "state"
    if getattr(args, "state_host", None):
        return SshStore(ssh_argv(args.state_host, Path(args.ssh_dir).expanduser()), root)
    return LocalStore(root)


def init_state(store: LocalStore, units: dict[str, str]) -> None:
    for d in ("claims", "done"):
        store.mkdir(d)
    known = store.read("units.json")
    if known is None:
        store.write("units.json", json.dumps(units, indent=1) + "\n")
    elif json.loads(known) != units:
        raise SystemExit("state/units.json disagrees with the source index")


def pick_unit(units, order: str, store: LocalStore, node: str, can_run=lambda u: True):
    """Next unit for this node: (unit, resumed) or (None, None).

    Skips units with a done record and units claimed by another node. A claim
    this node already owns (a previous run died mid-unit) is resumed.
    """
    done = set(store.ls("done"))
    for u in ordered(units, order):
        if f"{u}.json" in done or not can_run(u):
            continue
        if store.mkdir(f"claims/{u}"):
            store.write(f"claims/{u}/owner.json",
                        json.dumps({"node": node, "host": socket.gethostname(), "claimed": utcnow()}) + "\n")
            return u, False
        owner = json.loads(store.read(f"claims/{u}/owner.json") or "{}")
        if owner.get("node") == node:
            return u, True
    return None, None


def load_done(store: LocalStore) -> dict[str, dict]:
    out = {}
    for f in store.ls("done"):
        if f.endswith(".json"):
            text = store.read(f"done/{f}")
            if text:
                out[f[: -len(".json")]] = json.loads(text)
    return out


def build_manifest(units: dict[str, str], done: dict[str, dict]) -> dict:
    means = [done[u]["stats"]["final"]["mean"] for u in sorted(done)]
    stock = [done[u]["stats"]["stock"]["mean"] for u in sorted(done)]
    mf = sum(means) / len(means) if means else None
    ms = sum(stock) / len(stock) if stock else None
    return {
        "revision": NEW_REV,
        "encoder": "exllamav3 1.5.1 quantize_tiles (Viterbi) + refit_scales(H=I), 2.0bpw MCG K=2",
        "units_total": len(units),
        "units_done": len(done),
        "complete": set(done) == set(units),
        "relerr_mean_final": mf,
        "relerr_mean_stock": ms,
        "mse_ratio_vs_stock": (mf / ms) ** 2 if mf and ms else None,
        "units": {u: {k: v for k, v in done[u].items() if k != "rows"} for u in sorted(done)},
        "updated": utcnow(),
    }


def refresh_manifest(store: LocalStore, units: dict[str, str]) -> dict:
    man = build_manifest(units, load_done(store))
    store.write("manifest.json", json.dumps(man, indent=1) + "\n")
    if man["complete"] and store.read("DONE") is None:
        store.write("DONE", json.dumps({k: man[k] for k in (
            "revision", "units_total", "relerr_mean_final", "relerr_mean_stock", "mse_ratio_vs_stock")}
            | {"completed": utcnow()}, indent=1) + "\n")
    return man


def status_report(store: LocalStore, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    units = json.loads(store.read("units.json") or "{}")
    done = load_done(store)
    inflight, rates = {}, {}
    for u in store.ls("claims"):
        if u in done:
            continue
        owner = json.loads(store.read(f"claims/{u}/owner.json") or "{}")
        hb = json.loads(store.read(f"claims/{u}/heartbeat.json") or "{}")
        age = now - parse_utc(hb["time"]) if hb.get("time") else None
        inflight[u] = {"node": owner.get("node"), "claimed": owner.get("claimed"),
                       "tensors_done": hb.get("tensors_done", 0), "tensors": hb.get("tensors", TENSORS_PER_UNIT),
                       "sec_per_tensor": hb.get("sec_per_tensor"), "heartbeat_age_s": age,
                       "stale": age is None or age > STALE_HEARTBEAT_S}
        if not inflight[u]["stale"] and hb.get("sec_per_tensor"):
            rates[owner.get("node")] = 1.0 / hb["sec_per_tensor"]
    per_node: dict[str, list[float]] = {}
    for rec in done.values():
        per_node.setdefault(rec["node"], []).append(rec["sec_per_tensor"])
    for node, spt in per_node.items():
        rates.setdefault(node, 1.0 / sorted(spt)[len(spt) // 2])
    remaining = sum(TENSORS_PER_UNIT for u in units if u not in done and u not in inflight)
    remaining += sum(max(0, v["tensors"] - v["tensors_done"]) for v in inflight.values())
    active = {n: r for n, r in rates.items() if any(v["node"] == n and not v["stale"] for v in inflight.values())}
    eta = eta_seconds(remaining, active)
    return {"units_total": len(units), "units_done": len(done), "done_by_node": {
                n: sum(1 for r in done.values() if r["node"] == n) for n in sorted(per_node)},
            "inflight": inflight, "rates_tensors_per_s": rates, "active_nodes": sorted(active),
            "remaining_tensors": remaining, "eta_s": eta,
            "eta_utc": datetime.fromtimestamp(now + eta, timezone.utc).strftime("%Y-%m-%dT%H:%MZ") if eta else None,
            "DONE": store.read("DONE") is not None}


# ---------------------------------------------------------------- GPU encode (container)


class Log:
    def __init__(self, path: Path | None):
        self.path = path

    def __call__(self, msg: str) -> None:
        line = f"{utcnow()} {msg}"
        print(line, flush=True)
        if self.path is not None:
            with open(self.path, "a") as fh:
                fh.write(line + "\n")


def find_src(srcs: list[Path], fname: str) -> Path | None:
    for d in srcs:
        if (Path(d) / fname).is_file():
            return Path(d) / fname
    return None


def _natural(name: str):
    return tuple(int(p) if p.isdigit() else p for p in name.split("."))


def verify_written(path: Path, ref: Path, tensors: dict) -> tuple[list[str], bool]:
    """Layout vs the reference shard, byte-equal non-expert tensors and markers,
    and a reload of every expert tensor. Returns (problems, header_bytes_equal)."""
    import torch
    from safetensors import safe_open

    raw_new, new = read_header(path)
    raw_ref, old = read_header(ref)
    probs = layout_problems(new, old)
    if path.stat().st_size != ref.stat().st_size:
        probs.append(f"size {path.stat().st_size} != ref {ref.stat().st_size}")
    if probs:
        return probs, raw_new == raw_ref
    base_new, base_ref = 8 + len(raw_new), 8 + len(raw_ref)
    with open(path, "rb") as fn, open(ref, "rb") as fr:
        for key, meta in new.items():
            if key == "__metadata__" or key.endswith(EXPERT_SUFFIXES):
                continue
            a, b = meta["data_offsets"]
            fn.seek(base_new + a)
            fr.seek(base_ref + a)
            if fn.read(b - a) != fr.read(b - a):
                probs.append(f"{key}: bytes differ from the reference pack")
    with safe_open(str(path), framework="pt") as fh:
        for key in fh.keys():
            if key.endswith(EXPERT_SUFFIXES) and not torch.equal(fh.get_tensor(key), tensors[key]):
                probs.append(f"{key}: reload differs from the encoded tensor")
    return probs, raw_new == raw_ref


def encode_unit(u: str, fname: str, srcs: list[Path], ref_dir: Path, out: Path, device: str = "cuda:0",
                tile_chunk: int = TILE_CHUNK, limit: int | None = None, check_first: int = 0,
                heartbeat=None, log=print) -> dict:
    """Encode one shard; write OUT/shards/<fname> and OUT/units/<u>.json."""
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    from safetensors import safe_open
    from safetensors.torch import save_file

    from quantize_experts_exl3 import _dequant_t, _load_index, _quantize_fast
    from requant_probe import package_version, refit_identity, relerr

    if os.environ.get("DSV41_PACK_PF_G8", "0") == "1":
        raise SystemExit("DSV41_PACK_PF_G8=1 would write the G8 layout; the serve pack is stock")
    src_file = find_src(srcs, fname)
    if src_file is None:
        raise FileNotFoundError(f"no source {fname} under {srcs}")
    ref_file = ref_dir / fname
    names = sorted(n for n, f in _load_index(Path(srcs[0]))["weight_map"].items() if f == fname)
    if set(names) != {k for k in read_header(src_file)[1] if k != "__metadata__"}:
        raise SystemExit(f"{fname}: index and header disagree")
    experts = sorted((n for n in names if is_routed_expert_weight(n)), key=_natural)
    if limit:
        experts = experts[:limit]
    dev = torch.device(device)

    def recon(trellis, suh, svh, shape):
        o = torch.empty(shape, dtype=torch.half, device=dev)
        ext.reconstruct_had_slice(o, trellis.to(dev), suh.to(dev).half(), svh.to(dev).half(), K, True, False, 0)
        return o

    def encode(w, stem, chunk):
        # _quantize_fast regularizes its input in place when it is already an
        # fp32 CUDA tensor; hand it a copy so w stays the source weight.
        torch.manual_seed(tensor_seed(stem))
        return _quantize_fast([w.clone()], K, device, h_cache, codebook=CODEBOOK, tile_chunk=chunk)[0]

    upstream_refit = None
    if check_first:
        from exllamav3.modules.quant.exl3_lib.quantize import refit_scales as upstream_refit

    h_cache: dict = {}
    tensors: dict = {}
    rows: list[dict] = []
    t0 = time.time()
    started = utcnow()
    if heartbeat is not None:
        heartbeat({"time": started, "tensors_done": 0, "tensors": len(experts), "sec_per_tensor": None})
    with safe_open(str(src_file), framework="pt") as fh, safe_open(str(ref_file), framework="pt") as rf:
        for n in names:
            if not is_routed_expert_tensor(n):
                tensors[n] = fh.get_tensor(n)
        for i, wname in enumerate(experts):
            stem = wname[: -len(".weight")]
            w = _dequant_t(fh.get_tensor(wname), fh.get_tensor(stem + ".scale"), device).to(dev)
            enc = encode(w, stem, tile_chunk)
            trellis, su, sv = enc["trellis"], enc["suh"].to(dev), enc["svh"].to(dev)
            q = recon(trellis, su, sv, w.shape)
            e_vit = relerr(q, w)
            _, r, c = refit_identity(w, q.float())
            su_f, sv_f = (su.float() * r).half(), (sv.float() * c).half()
            e_fit = relerr(recon(trellis, su_f, sv_f, w.shape), w)
            keep = math.isfinite(e_fit) and e_fit <= e_vit
            if keep:
                su, sv = su_f, sv_f
            st = {x: rf.get_tensor(f"{stem}.{x}") for x in ("trellis", "suh", "svh")}
            row = {"tensor": stem, "shape": list(w.shape), "viterbi": e_vit, "final": e_fit if keep else e_vit,
                   "stock": relerr(recon(st["trellis"], st["suh"], st["svh"], w.shape), w), "refit": keep}
            if i < check_first:
                ref_enc = encode(w, stem, 256)
                row["chunk256_bitwise"] = all(torch.equal(ref_enc[x], enc[x]) for x in ("trellis", "suh", "svh"))
                _, su_u, sv_u, _, _ = upstream_refit(w, q.float(), torch.eye(w.shape[0], device=dev),
                                                     enc["suh"].to(dev).float(), enc["svh"].to(dev).float())
                row["upstream_refit"] = relerr(recon(trellis, su_u.flatten(), sv_u.flatten(), w.shape), w)
            tensors[stem + ".trellis"] = trellis
            tensors[stem + ".suh"] = su.cpu().contiguous()
            tensors[stem + ".svh"] = sv.cpu().contiguous()
            tensors[stem + "." + CODEBOOK] = enc[CODEBOOK]
            rows.append(row)
            k = i + 1
            if k % HEARTBEAT_EVERY == 0 or k == len(experts):
                spt = (time.time() - t0) / k
                mf = sum(x["final"] for x in rows) / k
                ms = sum(x["stock"] for x in rows) / k
                log(f"unit {u} {k}/{len(experts)} {spt:.3f} s/tensor relerr {mf:.4f} (stock {ms:.4f}) "
                    f"unit eta {(len(experts) - k) * spt / 60:.1f} min")
                if heartbeat is not None:
                    heartbeat({"time": utcnow(), "tensors_done": k, "tensors": len(experts),
                               "sec_per_tensor": spt, "relerr_mean": mf, "stock_mean": ms})
            del w, q, enc
    encode_s = time.time() - t0
    gate = unit_gate(rows)
    checks = [r for r in rows if "chunk256_bitwise" in r]
    if checks and not all(r["chunk256_bitwise"] for r in checks):
        gate["ok"] = False
        gate["chunk256"] = "trellis/scales differ between tile_chunk values"
    (out / "shards").mkdir(parents=True, exist_ok=True)
    (out / "units").mkdir(parents=True, exist_ok=True)
    final = out / "shards" / fname
    rec = {"unit": u, "file": fname, "tensors": len(rows), "stats": unit_stats(rows), "gate": gate,
           "started": started, "encode_s": encode_s, "sec_per_tensor": encode_s / max(len(rows), 1),
           "tile_chunk": tile_chunk, "exllamav3": package_version("exllamav3"), "torch": torch.__version__,
           "src": str(src_file), "ref": str(ref_file)}
    if limit or not gate["ok"]:
        # Partial (limit) or failed units never land in shards/.
        rec["written"] = False
        tag = "partial" if limit else "failed"
        (out / "units" / f"{u}.{tag}.json").write_text(json.dumps(rec | {"rows": rows}, indent=1) + "\n")
        if not gate["ok"]:
            raise RuntimeError(f"unit {u} gate failed: {gate}")
        return rec
    tmp = final.with_name(final.name + ".tmp")
    save_file(tensors, str(tmp))
    probs, hdr_equal = verify_written(tmp, ref_file, tensors)
    del tensors
    rec["header_bytes_equal_ref"] = hdr_equal
    if probs:
        rec["problems"] = probs[:50]
        (out / "units" / f"{u}.failed.json").write_text(json.dumps(rec | {"rows": rows}, indent=1) + "\n")
        raise RuntimeError(f"unit {u} verification failed: {probs[:5]}")
    os.chmod(tmp, 0o644)
    os.replace(tmp, final)
    rec.update(written=True, size=final.stat().st_size, sha256=sha256_file(final), finished=utcnow(),
               wall_s=time.time() - t0)
    tmpj = out / "units" / f".{u}.json.tmp"
    tmpj.write_text(json.dumps(rec | {"rows": rows}, indent=1) + "\n")
    os.replace(tmpj, out / "units" / f"{u}.json")
    return rec


def reuse_local(out: Path, u: str, fname: str) -> dict | None:
    """This node's verified output of unit u (earlier run or encode), if the file still matches."""
    j, f = out / "units" / f"{u}.json", out / "shards" / fname
    if not (j.is_file() and f.is_file()):
        return None
    rec = json.loads(j.read_text())
    if not (rec.get("written") and rec.get("gate", {}).get("ok")):
        return None
    if rec.get("size") != f.stat().st_size or rec.get("sha256") != sha256_file(f):
        return None
    rec.pop("rows", None)
    return rec


def _gpu_lock(path: str | None, log):
    """Hold an flock for the worker's lifetime (e.g. spark2's .gpu.lock)."""
    if not path:
        return None
    fh = open(path, "a")
    log(f"waiting for GPU lock {path}")
    fcntl.flock(fh, fcntl.LOCK_EX)
    log(f"holding GPU lock {path}")
    return fh


def cmd_run(args) -> int:
    out = Path(args.out)
    for d in ("shards", "units", "logs"):
        (out / d).mkdir(parents=True, exist_ok=True)
    log = Log(out / "logs" / f"run-{args.node}.log")
    node_lock = open(out / f".run-{args.node}.lock", "a")
    try:
        fcntl.flock(node_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log(f"another {args.node} worker holds {out}/.run-{args.node}.lock")
        return 1
    if not args.state_host:
        (out / "state").mkdir(exist_ok=True)
    store = make_store(args)
    srcs = [Path(s) for s in args.src]
    units = units_from_index(json.loads((srcs[0] / "model.safetensors.index.json").read_text())["weight_map"])
    init_state(store, units)
    gpu_lock = _gpu_lock(args.gpu_lock, log)  # noqa: F841 (held until exit)
    log(f"worker {args.node} order={args.order} units={len(units)} "
        f"state={args.state_host or 'local'}:{out / 'state'} tile_chunk={args.tile_chunk}")
    failures: dict[str, int] = {}
    while True:
        if (out / "STOP").exists() or (out / f"STOP.{args.node}").exists():
            log("STOP file present; exiting after the last finished unit")
            break
        u, resumed = pick_unit(units, args.order, store, args.node,
                               can_run=lambda x: find_src(srcs, units[x]) is not None)
        if u is None:
            log("no claimable unit left for this node")
            break
        log(f"unit {u} {units[u]} {'resumed' if resumed else 'claimed'}")
        rec = reuse_local(out, u, units[u])
        if rec is not None:
            log(f"unit {u} verified output already on this node, reused (sha256 {rec['sha256'][:16]})")
        else:
            try:
                rec = encode_unit(u, units[u], srcs, Path(args.ref), out, tile_chunk=args.tile_chunk, log=log,
                                  heartbeat=lambda hb, u=u: store.write(
                                      f"claims/{u}/heartbeat.json", json.dumps(hb | {"node": args.node}) + "\n"))
            except Exception:
                failures[u] = failures.get(u, 0) + 1
                log(f"unit {u} FAILED ({failures[u]}x):\n{traceback.format_exc()}")
                if failures[u] >= 2:
                    log(f"unit {u} failed twice; stopping (claim kept for resume)")
                    return 2
                continue
        rec.update(node=args.node, host=socket.gethostname(), recorded=utcnow())
        store.write(f"done/{u}.json", json.dumps(rec, indent=1) + "\n")
        man = refresh_manifest(store, units)
        log(f"unit {u} DONE sha256 {rec['sha256'][:16]} relerr {rec['stats']['final']['mean']:.4f} "
            f"(stock {rec['stats']['stock']['mean']:.4f}) {rec['sec_per_tensor']:.3f} s/tensor; "
            f"{man['units_done']}/{man['units_total']} units done")
    man = refresh_manifest(store, units)
    log(f"exit: {man['units_done']}/{man['units_total']} done; DONE={'yes' if store.read('DONE') else 'no'}")
    return 0


def cmd_encode(args) -> int:
    out = Path(args.out)
    srcs = [Path(s) for s in args.src]
    units = units_from_index(json.loads((srcs[0] / "model.safetensors.index.json").read_text())["weight_map"])
    rec = encode_unit(args.unit, units[args.unit], srcs, Path(args.ref), out, tile_chunk=args.tile_chunk,
                      limit=args.limit, check_first=args.check_first, log=Log(None))
    print(json.dumps(rec, indent=1))
    return 0


# ---------------------------------------------------------------- host commands


def cmd_status(args) -> int:
    rep = status_report(make_store(args))
    if args.json:
        print(json.dumps(rep, indent=1))
        return 0
    print(f"units {rep['units_done']}/{rep['units_total']} done {rep['done_by_node']}  DONE={rep['DONE']}")
    for u, v in sorted(rep["inflight"].items()):
        age = v["heartbeat_age_s"]
        spt = v["sec_per_tensor"]
        print(f"  in flight {u} on {v['node']}: {v['tensors_done']}/{v['tensors']} tensors, "
              f"{spt and round(spt, 3)} s/tensor, heartbeat {age and round(age)} s ago{' STALE' if v['stale'] else ''}")
    rates = {n: round(1 / r, 3) for n, r in rep["rates_tensors_per_s"].items()}
    print(f"s/tensor by node {rates}; active {rep['active_nodes']}; remaining {rep['remaining_tensors']} tensors; "
          f"ETA {rep['eta_utc'] or 'n/a'}")
    return 0


def launch_argv(node: str, order: str, out: Path = OUT, hf_cache: str = "/home/sfxnz/.cache/huggingface",
                state_host: str | None = None, ssh_dir: str = "/home/sfxnz/.ssh", gpu_lock: str | None = None,
                image: str = IMAGE) -> list[str]:
    """docker run for one worker. Code runs from the frozen copy in OUT/code."""
    o = str(out)
    argv = ["docker", "run", "-d", "--name", CONTAINER, "--restart", "no", "--gpus", "all",
            "--network", "host" if state_host else "none", "--memory", "24g", "--user", "1000:1000",
            "-e", "HOME=/tmp", "-e", "PYTHONPATH=/usr/local/lib/python3.12/dist-packages",
            "-e", "PYTHONUNBUFFERED=1",
            "-v", f"{hf_cache}:/cache/huggingface:ro", "-v", f"{o}:{o}", "-v", f"{o}/code:/repo:ro"]
    if state_host:
        argv += ["-v", "/usr/bin/ssh:/usr/bin/ssh:ro", "-v", f"{ssh_dir}:{ssh_dir}:ro"]
    if gpu_lock:
        argv += ["-v", f"{gpu_lock}:{gpu_lock}"]
    argv += ["--entrypoint", "python3", image, "-S", "/repo/tools/requant_full.py", "run",
             "--node", node, "--order", order, "--out", o]
    if state_host:
        argv += ["--state-host", state_host, "--ssh-dir", ssh_dir]
    if gpu_lock:
        argv += ["--gpu-lock", gpu_lock]
    return argv


def cmd_launch_cmd(args) -> int:
    print(shlex.join(launch_argv(args.node, args.order, Path(args.out), state_host=args.state_host,
                                 gpu_lock=args.gpu_lock)))
    return 0


def assembly_entries(base: Path, unit_files: set[str]) -> list[tuple[str, str, str]]:
    """(name, action, target) for the new snapshot, mirroring the base snapshot.

    Unit shards are hard links to OUT/shards (same filesystem; symlinks out of
    the HF cache would dangle in the serve container). Other entries keep the
    base's relative links, or link to the base's own files (model-00043 lm_head
    MXFP8). The index is copied: names and shard map are unchanged.
    """
    out = []
    for p in sorted(base.iterdir()):
        name = p.name
        if name in unit_files:
            out.append((name, "hardlink", name))
        elif name == "model.safetensors.index.json":
            out.append((name, "copy", str(p)))
        elif p.is_symlink():
            out.append((name, "symlink", os.readlink(p)))
        else:
            out.append((name, "symlink", f"../{base.name}/{name}"))
    missing = unit_files - {n for n, _, _ in out}
    if missing:
        raise SystemExit(f"base snapshot lacks unit shards {sorted(missing)[:4]}")
    return out


def cmd_assemble(args) -> int:
    out = Path(args.out)
    store = make_store(args)
    if store.read("DONE") is None:
        raise SystemExit("state/DONE missing: the re-encode is not complete")
    man = json.loads(store.read("manifest.json"))
    shards = out / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    for u, rec in sorted(man["units"].items()):
        f = shards / rec["file"]
        if not f.is_file():
            if rec["node"] == args.node:
                raise SystemExit(f"{f} missing on its own node")
            subprocess.run(["rsync", "-a", "--partial", f"{rec['node']}:{f}", str(f)], check=True)
        if f.stat().st_size != rec["size"] or sha256_file(f) != rec["sha256"]:
            raise SystemExit(f"{f}: size/sha256 differ from the manifest")
        print(f"ok {rec['file']} ({rec['node']})", flush=True)
    snaps = Path(args.hub) / PACK_REPO / "snapshots"
    base, snap = snaps / args.base, snaps / args.name
    snap.mkdir(exist_ok=True)
    unit_files = {rec["file"] for rec in man["units"].values()}
    for name, action, target in assembly_entries(base, unit_files):
        dst = snap / name
        if dst.exists() or dst.is_symlink():
            continue
        if action == "hardlink":
            os.link(shards / target, dst)
        elif action == "copy":
            shutil.copy2(target, dst)
        else:
            os.symlink(target, dst)
    (snap / "requant-viterbi-manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    idx = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    have: dict[str, str] = {}
    for f in sorted(snap.glob("model-*-of-*.safetensors")):
        for k in read_header(f)[1]:
            if k != "__metadata__":
                have[k] = f.name
    if have != idx:
        raise SystemExit(f"index and shard headers disagree ({len(idx)} vs {len(have)} tensors)")
    for rec in man["units"].values():
        if (snap / rec["file"]).stat().st_ino != (shards / rec["file"]).stat().st_ino:
            raise SystemExit(f"{rec['file']} is not the verified unit output")
    print(f"assembled {snap}: {len(have)} tensors, {len(unit_files)} re-encoded shards")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    hub = default_hub()

    def common(p, gpu: bool) -> None:
        p.add_argument("--out", default=str(OUT))
        p.add_argument("--state-host", default=None, help="node holding OUT/state (default: this node)")
        p.add_argument("--ssh-dir", default="~/.ssh")
        if gpu:
            p.add_argument("--src", action="append", default=None,
                           help="source snapshot dir(s), first match wins")
            p.add_argument("--ref", default=str(hub / PACK_REPO / "snapshots" / REF_REV))
            p.add_argument("--tile-chunk", type=int, default=TILE_CHUNK)

    p = sub.add_parser("run")
    common(p, True)
    p.add_argument("--node", required=True)
    p.add_argument("--order", choices=("asc", "desc"), required=True)
    p.add_argument("--gpu-lock", default=None)
    p = sub.add_parser("encode")
    common(p, True)
    p.add_argument("--unit", required=True)
    p.add_argument("--limit", type=int, default=None, help="first N expert tensors only (no shard written)")
    p.add_argument("--check-first", type=int, default=0,
                   help="also compare tile_chunk 256 and upstream refit_scales on the first N tensors")
    p = sub.add_parser("status")
    common(p, False)
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("launch-cmd")
    common(p, False)
    p.add_argument("--node", required=True)
    p.add_argument("--order", choices=("asc", "desc"), required=True)
    p.add_argument("--gpu-lock", default=None)
    p = sub.add_parser("assemble")
    common(p, False)
    p.add_argument("--node", required=True)
    p.add_argument("--hub", default=str(hub))
    p.add_argument("--base", default=BASE_REV)
    p.add_argument("--name", default=NEW_REV)
    args = ap.parse_args(argv)
    if getattr(args, "src", "unset") is None:
        args.src = [str(hub / SRC_REL)]
    return {"run": cmd_run, "encode": cmd_encode, "status": cmd_status, "launch-cmd": cmd_launch_cmd,
            "assemble": cmd_assemble}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
