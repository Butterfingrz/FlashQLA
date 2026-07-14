# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Inter-card CP 的分布式通信原语（S3，移植自 fla/ops/cp/comm.py）+ 状态打包。

单层 inter-card CP 每层前向做一次 `all_gather_into_tensor`：各卡把本卡聚合
`(S_ext_r, M_r)` 打包成一个 `hm` 张量后 all-gather，再各自取相关邻居做 `inter_scan`。
"""

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
    """跨 rank all-gather。返回 `(out[world_size, *inp.shape], handle)`。"""
    world_size = dist.get_world_size(group=group)
    if out is None:
        out = torch.empty(world_size, *inp.shape, device=inp.device, dtype=inp.dtype)
    handle = dist.all_gather_into_tensor(out, inp, group=group, async_op=async_op)
    return out, handle


def pack_hm(S_ext: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """把本卡聚合 `S_ext[H,K,V]` 与 `M[H,K,K]` 打包成 `hm[H,K,V+K]`（对齐 fla 的 hm 布局）。

    要求同 dtype（生产用 fp32，保 M 链精度）。
    """
    assert S_ext.shape[:2] == M.shape[:2], (S_ext.shape, M.shape)
    assert S_ext.dtype == M.dtype
    return torch.cat([S_ext, M], dim=-1)


def unpack_hm(hm: torch.Tensor, v_head_dim: int):
    """`pack_hm` 的逆：从 `hm[..., K, V+K]` 拆回 `(S_ext[..., K, V], M[..., K, K])`。"""
    S_ext = hm[..., :v_head_dim]
    M = hm[..., v_head_dim:]
    return S_ext, M
