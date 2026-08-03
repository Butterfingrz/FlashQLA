# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


def all_gather_into_tensor(
    inp: torch.Tensor,
    out: torch.Tensor | None = None,
    group: "ProcessGroup | None" = None,
    async_op: bool = False,
):
    world_size = dist.get_world_size(group=group)
    if out is None:
        out = torch.empty(world_size, *inp.shape, device=inp.device, dtype=inp.dtype)
    handle = dist.all_gather_into_tensor(out, inp, group=group, async_op=async_op)
    return out, handle


def pack_hm(S_ext: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    assert S_ext.shape[:2] == M.shape[:2], (S_ext.shape, M.shape)
    assert S_ext.dtype == M.dtype
    return torch.cat([S_ext, M], dim=-1)


def unpack_hm(hm: torch.Tensor, v_head_dim: int):
    S_ext = hm[..., :v_head_dim].contiguous()
    M = hm[..., v_head_dim:].contiguous()
    return S_ext, M
