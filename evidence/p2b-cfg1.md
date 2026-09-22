# p2b CFG=1 occupancy

`docker/patch/widen_p2b_cfg1.py` switches fused-MoE decode tiles from CFG=0
to CFG=1 and launches 256-thread cooperative blocks. Apply after
`widen_p2b_shapes.py` and `widen_p2b_mrow.py`. Idempotent. m-row work lists
stay. BITS stays. The MMA-over-m regroup (`widen_p2b_mma.py`) stays unwired.

CFG=1 is occupancy. GB10 has 48 SMs and 1536 max threads per SM. A 512-thread
block can resident at most 3 blocks per SM. A 256-thread block can resident
at most 6. The occupancy query and `cudaLaunchCooperativeKernel` must use 256
or the extra blocks never launch. MMA-over-m was a different lever. It decoded
B once and reused it across rows, raised register pressure, measured 10.2 tok/s,
and was reverted. CFG=1 does not add a row loop or extra `acc0[MAX_M]` tiles.

Flipping only `run_gemv_tile<BITS, 1, 0>` is not enough. CFG=1 is WK=8, WNT=4,
COLS=64, THREADS=256. The host and shared memory still assumed COLS=32 and 512
threads.

Exact replacements in `csrc/p2b_moe.cu`:

- `run_gemv_tile<BITS, 1, 0>` to `run_gemv_tile<BITS, 1, 1>` (gate/up and down)
- `num_groups_gate = inter / 32` to `inter / 64`
- `num_groups_down = hidden / 32` to `hidden / 64`
- `__shared__ float sh_red[16][1][32]` to `sh_red[8][1][64]`
- `float (*sh_red)[1][32]` to `float (*sh_red)[1][64]`
- `__launch_bounds__(512)` to `__launch_bounds__(256, 4)`
  (minBlocks=4: GB10 has 65536 regs/SM; CFG=1 compiled at 102-111 regs → 2 blocks/SM.
  4 blocks needs ≤64 regs/thread and 1024 threads/SM.)
- `cudaOccupancyMaxActiveBlocksPerMultiprocessor(..., 512, 0)` to `256`
- `cudaLaunchCooperativeKernel(..., dim3(512), ...)` to `dim3(256)`
