"""Token -> request map dedup across KV-cache groups (DSV41_ATTN_T2R_DEDUP=1).

build_attn_metadata (vllm/v1/worker/gpu/attn_utils.py) makes one
CommonAttentionMetadata per KV-cache group, all carrying that call's
query_start_loc tensors, and every group's builder (SWA, compressor, sparse
MLA) calls token_to_req_indices(its own persistent buffer): arange +
repeat_interleave + copy, ~8 small kernels. The r3 trace has 21 of these per
decode step (3 groups before the draft graph, 18 before the target graph),
queued back to back on the critical path.

With the lever, a group whose metadata carries the previous group's
query_start_loc and query_start_loc_cpu tensor OBJECTS (weakrefs; every
build_attn_metadata call passes its own) and the same token counts copies the
previous result into its own buffer: one kernel instead of eight. Every
builder keeps its own buffer, so captured graph addresses and values are the
stock ones.

Safety: the first DSV41_ATTN_T2R_VERIFY reuses outside graph capture (default
8, minimum 1) also run the stock method into a scratch buffer and compare. A
mismatch puts the stock values into the buffer, prints one LOG_DISARMED line,
and every later call takes the stock method. LOG_ENGAGED is printed with the
first verified reuse (the profile run's metadata builds already reuse, so it
prints at boot). A [t2r-census] line (computes, reuses, reuses per compute)
follows when verification ends and at every power-of-two compute count from
1024, so the serve log shows whether decode steps still reuse: 21 groups in 2
preps a step is 19 reuses per 2 computes = 9.5.

Top-level imports are stdlib only. decode_levers.install() calls install()
when the env is on.
"""

from __future__ import annotations

import copy
import os
import weakref

# Boot-log markers for tools/engagement_audit.py.
LOG_ENGAGED = "dsv41: attention t2r dedup reuse bit-exact"
LOG_DISARMED = "dsv41: attention t2r dedup DISABLED ->"

DEFAULT_VERIFY = 8
CENSUS_FROM = 1024
_STATE = {"armed": True, "verify_left": DEFAULT_VERIFY, "engaged": False, "computes": 0, "reuses": 0}


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return (env.get("DSV41_ATTN_T2R_DEDUP", "0") or "0") == "1"


def verify_calls(env=None) -> int:
    """Initial reuses checked against the stock method; at least one."""
    env = os.environ if env is None else env
    return max(1, int(env.get("DSV41_ATTN_T2R_VERIFY", "") or DEFAULT_VERIFY))


def t2r_reuse(prev, qsl, qsl_cpu, num_tokens: int, num_mapped: int, buffer) -> bool:
    """Whether the previous group's token -> request map can be copied."""
    if prev is None:
        return False
    qref, cref, nt, nm, view = prev
    return (
        qref() is qsl
        and cref() is qsl_cpu
        and nt == num_tokens
        and nm == num_mapped
        and view.dtype == buffer.dtype
        and view.device == buffer.device
        and buffer.shape[0] >= max(num_mapped, num_tokens)
    )


def census_due(computes: int) -> bool:
    """Every power of two from CENSUS_FROM on (a handful of lines per serve)."""
    return computes >= CENSUS_FROM and computes & (computes - 1) == 0


def _census(tag: str) -> None:
    c, r = _STATE["computes"], _STATE["reuses"]
    print(
        "[t2r-census] %s computes=%d reuses=%d reuses/compute=%.2f" % (tag, c, r, r / max(c, 1)),
        flush=True,
    )


def _disarm(exc: BaseException) -> None:
    if _STATE["armed"]:
        _STATE["armed"] = False
        print("dsv41: attention t2r dedup DISABLED -> stock map per group: %r" % (exc,), flush=True)


def install(env=None) -> str:
    """Wrap CommonAttentionMetadata.token_to_req_indices."""
    import torch
    from vllm.v1.attention.backend import CommonAttentionMetadata

    cls = CommonAttentionMetadata
    if getattr(cls.token_to_req_indices, "_dsv41_t2r_dedup", False):
        return "already installed"
    _STATE["verify_left"] = verify_calls(env)
    stock = cls.token_to_req_indices
    last = [None]

    def verify(cam, buffer, n: int) -> None:
        probe = copy.copy(cam)
        probe._token_to_req_indices_cache = None
        ref = torch.empty_like(buffer[:n])
        stock(probe, ref)
        if not torch.equal(ref, buffer[:n]):
            buffer[:n].copy_(ref)
            _disarm(RuntimeError(f"copied map != stock map (tokens {n})"))
            return
        _STATE["verify_left"] -= 1
        if not _STATE["engaged"]:
            _STATE["engaged"] = True
            print(
                "dsv41: attention t2r dedup reuse bit-exact (tokens %d; %d computes, %d reuses so far)"
                % (n, _STATE["computes"], _STATE["reuses"]),
                flush=True,
            )
        if _STATE["verify_left"] == 0:
            _census("verified")

    def token_to_req_indices(self, buffer):
        if self._token_to_req_indices_cache is not None or not _STATE["armed"]:
            return stock(self, buffer)
        num_tokens = self.num_actual_tokens
        num_mapped = int(self.query_start_loc_cpu[-1])
        prev = last[0]
        if t2r_reuse(prev, self.query_start_loc, self.query_start_loc_cpu, num_tokens, num_mapped, buffer):
            n = max(num_mapped, num_tokens)
            view = prev[4]
            if buffer.data_ptr() != view.data_ptr():
                buffer[:n].copy_(view[:n])
            self._token_to_req_indices_cache = buffer[:n]
            _STATE["reuses"] += 1
            if _STATE["verify_left"] > 0 and not torch.cuda.is_current_stream_capturing():
                verify(self, buffer, n)
            return self._token_to_req_indices_cache[:num_tokens]
        out = stock(self, buffer)
        _STATE["computes"] += 1
        last[0] = (
            weakref.ref(self.query_start_loc),
            weakref.ref(self.query_start_loc_cpu),
            num_tokens,
            num_mapped,
            self._token_to_req_indices_cache,
        )
        if census_due(_STATE["computes"]):
            _census("steady")
        return out

    token_to_req_indices._dsv41_t2r_dedup = True
    cls.token_to_req_indices = token_to_req_indices
    return f"CommonAttentionMetadata.token_to_req_indices wrapped (verify={_STATE['verify_left']})"
