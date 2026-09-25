# dual-rail NCCL prep (2026-09-24)

These are read-only facts from both nodes (sysfs, `ip`, `ibv_devinfo`). spark1 and spark2 are identical.

| HCA | state | netdev | IP spark1 / spark2 | PCI | PCIe | port_xmit_data (spark1 / spark2) |
|---|---|---|---|---|---|---|
| rocep1s0f1 | ACTIVE 200G | enp1s0f1np1 | 10.100.8.1 / 10.100.8.2 | 0000:01:00.1 | 32 GT/s x4 | 1.17e12 / 3.52e12 |
| roceP2p1s0f1 | ACTIVE 200G | enP2p1s0f1np1 | 10.100.9.1 / 10.100.9.2 | 0002:01:00.1 | 32 GT/s x4 | 0 / 0 (idle) |
| rocep1s0f0, roceP2p1s0f0 | DOWN | — | — | —.0 | — | 0 |

- Netdev MTU is 1500, which gives RoCE active_mtu 1024 (max 4096).
- Each rail is its own /24 point-to-point link.

## Corrections applied

- The dual-rail arms set **NCCL_CROSS_NIC=0**, so rails stay matched. Pairing rail0 with rail1 cannot route between 10.100.8/24 and 10.100.9/24.
- The HCAs sit in different PCI domains, so **MERGE_NICS may be a no-op**. There is an arm with `NCCL_IB_MERGE_NICS=0`, and the INFO log (`NET/IB : Using [0]… [1]…`, channel→NET mapping) shows what NCCL did.
- Only the ACTIVE f1 HCAs are ever listed.

## Tools

- The recipe image has no nccl-tests and no MPI. `tools/nccl_allreduce_sweep.py` is a torch.distributed bf16 all_reduce sweep, 8 B to 256 MiB, doubling, with nccl-tests busbw (= algbw at 2 ranks). It uses the same libnccl as the serve (2.30.7). `compare` applies the gate: busbw at least 10% higher at 32/64/128 MiB, and 8 B-64 KiB latency no more than 5% worse.
- `tools/nccl_dualrail.sh` runs the arms single (live: rocep1s0f1, CROSS_NIC=1, KEEP set), single_bare, dual, dual_keep and dual_keep_nomerge. Each arm runs one rank per node in `dsv41-flash-exl3-sm121:canonical-e12` (`--gpus all --network host --device /dev/infiniband`). The script refuses while `dsv41-flash-exl3`/`dsv41-quant` runs on either node (verified: it refuses against the live serve) and when any f1 rail is not ACTIVE. Logs and JSON go to `results/2026-09-24-review/research/nccl-dualrail-<ts>/`.
- nccl-tests alternative, if wanted: inside the image, `apt-get install -y openmpi-bin libopenmpi-dev`, then `git clone https://github.com/NVIDIA/nccl-tests && make MPI=1 NCCL_HOME=/usr/local/lib/python3.12/dist-packages/nvidia/nccl CUDA_HOME=/usr/local/cuda`, then run `mpirun -H 10.100.8.1,10.100.8.2 -np 2 -x NCCL_IB_HCA -x NCCL_CROSS_NIC … ./build/all_reduce_perf -b 8 -e 256M -f 2 -g 1`. This needs an ssh launcher between containers, which is why the torch sweep is the default.

## Serve wiring (for step 2)

- `run.sh` now takes `NCCL_CROSS_NIC` (default 1, same as before) and `NCCL_IB_MERGE_NICS` (default unset) through `FORWARD_ENVS`, so head and worker always match (tested).
- `HCA=rocep1s0f1,roceP2p1s0f1` already reaches both ranks. No change is needed in recipe.yaml unless a dual-rail default is adopted later; that would go through `kit/render.py`, AGENTS.md and flags.md:37.
- MTU 9000 (active_mtu 4096) is host netplan config outside `~/projects`. It needs explicit user authorization and must match on both ends. It stays an optional step 3.

Expected impact (corrections): prefill +2-3% (the AR is ~5% of an 8k chunk at single-rail ~11-13 GB/s); decode flat at best. This is a low-priority item to batch into any serve-down window.
