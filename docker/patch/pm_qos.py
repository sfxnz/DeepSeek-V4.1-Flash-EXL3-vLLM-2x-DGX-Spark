"""Hold a PM QoS CPU wake-up latency request while the serve runs.

DSV41_PM_QOS_US=N (default unset = off; run.sh FORWARD_ENVS). sitecustomize calls
install() in every Python process of the container; each holds its own request for
the process lifetime (the kernel applies the minimum over all open requests, and
drops a request when its file descriptor closes). run.sh passes
--device /dev/cpu_dma_latency to both ranks' containers only when the env is set.
The request is host-wide: while the serve runs, no CPU enters an idle state whose
exit latency exceeds N us.

Why (k3 comm). GB10's cpuidle table has LPI-1/2/3 with 42/231/433 us exit latency
(menu governor, LPI-3 is the most used state). A CUDA graph host node runs on a
driver thread that the GPU wakes when it reaches the node; NCCL puts every captured
network collective behind one (hostStreamPlanCallback uploads its proxy ops), then
wakes its proxy thread (condition variable). After the host has been idle for a few
ms, as between two graph replays, that wake-up takes hundreds of us: the r3 serve
trace shows 235 us (rank 0) and 427 us (rank 1) of device idle before the draft
graph's first all-reduce although the graph was launched ~44.7 ms earlier, and the
profile books 1.13 ms/step of graph startup. kernel_study/comm/host_node_latency.py
on spark2 (graph with a root host node, 20 ms of host idle between replays, same
process, alternating): median host-node latency 657-725 us without a request, 14.8
us with DSV41_PM_QOS_US=20; the first of 82 chained nodes 3-488 us vs 2.2 us.
A 50 us request (LPI-1 still allowed) did not help (359-379 us), so the value must
be below 42.

Numerics: none (CPU power-state policy only). Cost: idle cores stay in WFI.
Top-level imports are stdlib only.
"""

from __future__ import annotations

import os
import struct

LOG_ENGAGED = "dsv41: pm qos engaged"
LOG_DISARMED = "dsv41: pm qos DISARMED"

ENV = "DSV41_PM_QOS_US"
DEVICE = "/dev/cpu_dma_latency"
MAX_US = 2000

_held: list[int] = []  # open request fds; never closed while the process lives


def requested_us(env) -> int | None:
    """None when off (unset or empty), else the latency bound in us (0..MAX_US)."""
    raw = (env.get(ENV) or "").strip()
    if not raw:
        return None
    us = int(raw)
    if not 0 <= us <= MAX_US:
        raise ValueError(f"{ENV}={us} outside 0..{MAX_US}")
    return us


def install(env=None, log=print, device: str = DEVICE) -> str:
    """'off' | 'armed' | 'disarmed'."""
    env = os.environ if env is None else env

    def say(msg):
        log(msg, flush=True) if log is print else log(msg)

    try:
        us = requested_us(env)
    except ValueError as exc:
        say(f"dsv41: pm qos DISARMED: {exc}")
        return "disarmed"
    if us is None:
        return "off"
    if _held:
        return "armed"
    try:
        fd = os.open(device, os.O_WRONLY)
        try:
            os.write(fd, struct.pack("i", us))
        except OSError:
            os.close(fd)
            raise
    except OSError as exc:
        say(f"dsv41: pm qos DISARMED: {exc!r} (run.sh adds --device {DEVICE} when {ENV} is set)")
        return "disarmed"
    _held.append(fd)
    say(f"dsv41: pm qos engaged: {device} <= {us} us, held by pid {os.getpid()}")
    return "armed"
