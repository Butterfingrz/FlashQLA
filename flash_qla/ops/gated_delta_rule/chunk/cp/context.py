# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Inter-card CP 的上下文与序列切分（S1，移植自 fla/ops/cp/context.py）。

`build_cp_context` 把**全局** varlen `cu_seqlens` 按 token 等长连续切片分到各 rank，产出
**rank-local** 的 `cu_seqlens` 与跨卡元数据（`pre_num_ranks`/`post_num_ranks`/
`is_first_rank`/`is_last_rank`），供 `cp_preprocess_fwd/bwd` 决定各卡如何交换状态。

纯 CPU / 索引逻辑，不含内核。单进程模拟（无 dist）可用 `get_cp_cu_seqlens(..., world_size=W,
rank=r, group=None)` 逐 rank 构造。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from flash_qla.utils import tensor_cache

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


@dataclass
class FLACPContext:
    """算子级 inter-card CP 上下文（rank-local）。"""
    group: "ProcessGroup | None" = None
    cu_seqlens: torch.Tensor | None = None          # rank-local，GPU int32/int64
    cu_seqlens_cpu: torch.Tensor | None = None       # 同数据在 CPU（host 索引用）
    is_last_rank: bool | None = None
    pre_num_ranks: int | None = None                 # 前面同属本卡首序列的 rank 数（fwd merge 循环数）
    is_first_rank: bool | None = None
    post_num_ranks: int | None = None                # 后面同属本卡末序列的 rank 数（bwd merge 循环数）
    conv1d_kernel_size: int | None = None            # conv halo 用（本轮范围外，仅占位）
    pre_num_conv_tokens: int | None = None

    def copy_for_backward(self) -> "FLACPContext":
        return FLACPContext(
            group=self.group,
            cu_seqlens=self.cu_seqlens.clone() if self.cu_seqlens is not None else None,
            cu_seqlens_cpu=self.cu_seqlens_cpu.clone() if self.cu_seqlens_cpu is not None else None,
            is_last_rank=self.is_last_rank,
            pre_num_ranks=self.pre_num_ranks,
            is_first_rank=self.is_first_rank,
            post_num_ranks=self.post_num_ranks,
            conv1d_kernel_size=self.conv1d_kernel_size,
            pre_num_conv_tokens=self.pre_num_conv_tokens,
        )

    @property
    def num_seqs(self) -> int:
        return 0 if self.cu_seqlens is None else len(self.cu_seqlens) - 1

    @property
    def is_cp_enabled(self) -> bool:
        return self.group is not None


@tensor_cache
def get_cp_cu_seqlens(
    cu_seqlens: torch.LongTensor,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    world_size: int | None = None,
    rank: int | None = None,
    group: "dist.ProcessGroup | None" = None,
    conv1d_kernel_size: int | None = None,
) -> FLACPContext:
    """为某个 rank 计算 rank-local cu_seqlens + 跨卡元数据。

    全局序列按 token 等长连续切片：`part_len = total // world_size`，rank 拥有
    `[rank*part_len, (rank+1)*part_len)`（要求 total 能被 world_size 整除，均衡 CP 的常见约定）。
    单进程模拟可显式传 `world_size`/`rank` 且 `group=None`（无需 dist init）。
    """
    # 1. 环境信息
    if world_size is None:
        assert group is not None
        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)

    # 2. CPU 上算（避免 D2H 同步 + int64 向量化）
    if cu_seqlens_cpu is None:
        cu_seqlens_cpu = cu_seqlens.cpu()
    cu_seqlens_cpu = cu_seqlens_cpu.to(dtype=torch.long)

    total_tokens = cu_seqlens_cpu[-1].item()
    part_len = total_tokens // world_size
    rank_start = part_len * rank
    rank_end = rank_start + part_len

    # 3. 定位与本 rank 区间 [rank_start, rank_end) 重叠的序列
    start_seq_idx = torch.searchsorted(cu_seqlens_cpu[1:], rank_start, side="right")
    end_seq_idx = torch.searchsorted(cu_seqlens_cpu[:-1], rank_end, side="left")
    subset_cu_seqlens = cu_seqlens_cpu[start_seq_idx: end_seq_idx + 1]

    # 4. rank-local cu_seqlens（clamp 到区间后减 rank_start，unique_consecutive 去重）
    local_cu_seqlens_cpu = (
        subset_cu_seqlens.clamp(min=rank_start, max=rank_end) - rank_start
    ).unique_consecutive().to(torch.int32)
    local_cu_seqlens_gpu = local_cu_seqlens_cpu.to(
        device=cu_seqlens.device, non_blocking=True)

    # 5. 跨卡元数据
    first_seq_global_start = cu_seqlens_cpu[start_seq_idx].item()
    last_seq_global_end = cu_seqlens_cpu[end_seq_idx].item()

    pre_num_conv_tokens = max(0, rank_start - first_seq_global_start)

    first_rank_of_first_seq = first_seq_global_start // part_len
    pre_num_ranks = rank - first_rank_of_first_seq
    is_first_rank = (rank == first_rank_of_first_seq)

    last_rank_of_last_seq = (last_seq_global_end - 1) // part_len
    post_num_ranks = last_rank_of_last_seq - rank
    is_last_rank = (rank == last_rank_of_last_seq)

    return FLACPContext(
        group=group,
        cu_seqlens=local_cu_seqlens_gpu,
        cu_seqlens_cpu=local_cu_seqlens_cpu,
        is_last_rank=is_last_rank,
        pre_num_ranks=pre_num_ranks,
        is_first_rank=is_first_rank,
        post_num_ranks=post_num_ranks,
        conv1d_kernel_size=conv1d_kernel_size,
        pre_num_conv_tokens=pre_num_conv_tokens,
    )


def build_cp_context(
    cu_seqlens: torch.Tensor,
    group: "ProcessGroup",
    conv1d_kernel_size: int | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
) -> FLACPContext:
    """真分布式入口：从当前进程的 `group` 取 world_size/rank，构造本 rank 的 CP 上下文。"""
    return get_cp_cu_seqlens(
        cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu, group=group,
        conv1d_kernel_size=conv1d_kernel_size)
