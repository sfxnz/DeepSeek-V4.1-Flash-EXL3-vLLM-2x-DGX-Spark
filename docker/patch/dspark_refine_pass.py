#!/usr/bin/env python3
"""Prefix-condition a second DSpark draft pass on pass-1 samples.

Query offset 0 is the bonus token. Offsets 1..N-1 are a mask token, so later
hiddens attend to noise. After pass-1 sample, write draft[i] into offset 1+i
and replay the same _generate_draft body. Offset 0 stays the bonus.
"""

from __future__ import annotations


def dspark_refine_pass_from_env(env_flag: int) -> bool:
    return int(env_flag) == 1


def refine_query_indices(
    num_reqs: int, num_query_per_req: int, n_spec: int
) -> list[list[int]]:
    """input_ids index for draft[r, i] -> query offset 1+i. Offset 0 omitted."""
    nq = int(num_query_per_req)
    ns = int(n_spec)
    if ns < 2:
        return [[] for _ in range(int(num_reqs))]
    return [
        [r * nq + (1 + i) for i in range(ns - 1)] for r in range(int(num_reqs))
    ]


def apply_refine_fill_list(
    input_ids: list[int], drafts: list[list[int]], num_query_per_req: int
) -> list[int]:
    """CPU fill used by tests. draft[r][i] writes offset 1+i. Offset 0 stays."""
    out = list(input_ids)
    nq = int(num_query_per_req)
    for r, draft in enumerate(drafts):
        if len(draft) < 2:
            continue
        base = r * nq
        for i, tok in enumerate(draft[:-1]):
            out[base + 1 + i] = int(tok)
    return out


def refine_query_index(max_num_reqs: int, num_query_per_req: int, n_spec: int, device):
    import torch

    ns = int(n_spec)
    rows = torch.arange(int(max_num_reqs), dtype=torch.int64, device=device)[:, None]
    if ns < 2:
        return rows[:, :0]
    cols = torch.arange(1, ns, dtype=torch.int64, device=device)
    return rows * int(num_query_per_req) + cols


def apply_refine_fill(input_ids, draft_tokens, refine_idx, num_reqs: int):
    n_spec = int(draft_tokens.shape[1])
    n = int(num_reqs)
    if n_spec < 2 or n <= 0:
        return input_ids
    idx = refine_idx[:n].reshape(-1)
    src = draft_tokens[:n, : n_spec - 1].reshape(-1).to(dtype=input_ids.dtype)
    input_ids.index_copy_(0, idx, src)
    return input_ids
