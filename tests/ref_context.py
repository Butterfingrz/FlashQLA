# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Naive references for CP-context construction, for exact tensor comparison.

Each function recomputes, from first principles, the index tensors the real
builders produce for a *given* split -- not the split decision itself, which is a
tuned model (autocp / heuristic) with no ground-truth answer.
"""


def _cdiv(x, d):
    return (x + d - 1) // d


def ref_intra_context(cu_list, chunk_size, max_local_chunks):
    """Expected intra partition: each raw seq with more than ``max_local_chunks``
    chunks is cut into equal ``max_local_chunks``-chunk pieces (last one shorter)."""
    step = max_local_chunks * chunk_size
    cp_cu, ht, ht_bwd, c2r = [], [], [], []
    r2c = [0]
    for i in range(len(cu_list) - 1):
        s, e = cu_list[i], cu_list[i + 1]
        starts = list(range(s, e, step)) if _cdiv(e - s, chunk_size) > max_local_chunks else [s]
        for j, st in enumerate(starts):
            cp_cu.append(st)
            ht.append(j == len(starts) - 1)
            ht_bwd.append(j == 0)
            c2r.append(i)
        r2c.append(len(cp_cu))
    cp_cu.append(cu_list[-1])
    return dict(cp_cu=cp_cu, r2c=r2c, c2r=c2r, ht=ht, ht_bwd=ht_bwd)


def ref_inter_context(cu_list, world_size, rank):
    """Expected inter partition for one rank: its card owns the global token range
    ``[rank*part, (rank+1)*part)``; boundaries are the global ones falling inside."""
    total = cu_list[-1]
    assert total % world_size == 0
    part = total // world_size
    lo, hi = part * rank, part * (rank + 1)

    local_cu = [0, *(b - lo for b in cu_list if lo < b < hi), part]
    seq_start = max(b for b in cu_list if b <= lo)   # start of the seq owning the first token
    last_rank = (min(b for b in cu_list if b >= hi) - 1) // part  # rank owning the last token
    return dict(
        local_cu=local_cu,
        pre_num_conv_tokens=lo - seq_start,
        pre_num_ranks=rank - seq_start // part,
        is_first_rank=(rank == seq_start // part),
        post_num_ranks=last_rank - rank,
        is_last_rank=(rank == last_rank),
    )
