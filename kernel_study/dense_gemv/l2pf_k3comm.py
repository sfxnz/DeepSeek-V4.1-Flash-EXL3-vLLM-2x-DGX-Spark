# Verbatim copy of docker/patch/l2pf_kernel.py from branch k3/comm (commit fbaee56), the TMA L2
# prefetch kernel DSV41_AR_L2_PREFETCH ships; used by l2_regime_bench.py to reproduce that lever's
# L2 state. Do not edit here.
"""TMA L2 prefetch kernel for ar_l2_prefetch.py (Triton, inline PTX, compiled on first use).

One program of BLOCK lanes; lane i issues cp.async.bulk.prefetch.L2.global for 16 KiB
chunks i, i + BLOCK, ... of [base, base + nbytes). The requests are fire-and-forget;
the kernel ends once they are issued. Same instruction stream as the C++ kernel of
kernel_study/comm/l2_prefetch_window.py (bulk60 and its Triton twin tri60 both
measured -21.7 us per window). Top-level imports are stdlib only.
"""

from __future__ import annotations

CHUNK = 16384
BLOCK = 128
PTX = "{ .reg .pred p; setp.ne.s32 p, $2, 0; @p cp.async.bulk.prefetch.L2.global [$1], $2; mov.u32 $0, 0; }"


def _build():
    import triton
    import triton.language as tl

    @triton.jit
    def _l2_prefetch_kernel(base, nbytes, sink, CHUNK: tl.constexpr, BLOCK: tl.constexpr):
        for start in range(0, tl.cdiv(nbytes, CHUNK * BLOCK)):
            idx = (start * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
            off = idx * CHUNK
            size = tl.minimum(nbytes - off, CHUNK)
            size = tl.where(off < nbytes, size, 0).to(tl.int32) & ~15
            r = tl.inline_asm_elementwise(
                asm="{ .reg .pred p; setp.ne.s32 p, $2, 0; @p cp.async.bulk.prefetch.L2.global [$1], $2; mov.u32 $0, 0; }",
                constraints="=r,l,r",
                args=[base + off, size],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
            tl.store(sink + tl.arange(0, BLOCK), r, mask=tl.arange(0, BLOCK) < 0)

    return _l2_prefetch_kernel


def launcher(torch):
    """launch(tensor, nbytes): prefetch the first nbytes of tensor on the current stream."""
    state: dict = {}

    def launch(t, nbytes: int) -> None:
        if "k" not in state:
            state["k"] = _build()
        sink = state.get("sink")
        if sink is None or sink.device != t.device:
            sink = state["sink"] = torch.empty(BLOCK, dtype=torch.int32, device=t.device)
        state["k"][(1,)](t.data_ptr(), int(nbytes), sink, CHUNK=CHUNK, BLOCK=BLOCK, num_warps=4)

    return launch
