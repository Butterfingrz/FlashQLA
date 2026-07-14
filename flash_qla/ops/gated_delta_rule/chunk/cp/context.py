# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import tilelang

from flash_qla.utils import tensor_cache

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

_COMPUTE_VERSION = tilelang.contrib.nvcc.get_target_compute_version()
if _COMPUTE_VERSION == "9.0":
    ARCH = "SM90"
elif _COMPUTE_VERSION == "10.0":
    ARCH = "SM100"
elif _COMPUTE_VERSION == "10.3":
    ARCH = "SM103"
elif _COMPUTE_VERSION == "12.0":
    ARCH = "SM120"
else:
    raise ValueError(
        f"FlashQLA now support sm90, sm100 and sm103 only. Found compute version: {_COMPUTE_VERSION}"
    )

MULTI_PROCESSOR_COUNT = torch.cuda.get_device_properties().multi_processor_count


@dataclass
class FlashQLACPContext:
    type: str = "inter"  # "inter" | "intra"

    # --- common ---
    cu_seqlens: torch.Tensor | None = None

    # --- inter-card ---
    group: "ProcessGroup | None" = None
    cu_seqlens_cpu: torch.Tensor | None = None 
    is_last_rank: bool | None = None
    pre_num_ranks: int | None = None
    is_first_rank: bool | None = None
    post_num_ranks: int | None = None
    conv1d_kernel_size: int | None = None
    pre_num_conv_tokens: int | None = None

    # --- intra-card ---
    use_intra_cp: bool = False
    intra_cp_cu_seqlens: torch.Tensor | None = None 
    seq_map_r2c: torch.Tensor | None = None
    seq_map_c2r: torch.Tensor | None = None
    ht_mask: torch.Tensor | None = None
    ht_mask_bwd: torch.Tensor | None = None

    def copy_for_backward(self) -> "FlashQLACPContext":
        return FlashQLACPContext(
            type=self.type,
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
    def is_inter_cp_enabled(self) -> bool:
        return self.type == "inter"

    @property
    def is_intra_cp_enabled(self) -> bool:
        return self.type == "intra" and self.use_intra_cp

    def get_fwd_scan_tensors(self, num_v_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
        dev_idx = self.cu_seqlens.device.index
        return (
            _create_scan_seq_map(self.pre_num_ranks, dev_idx),
            _create_scan_fb_mask(self.pre_num_ranks, num_v_heads, dev_idx),
        )

    def get_bwd_scan_tensors(self, num_v_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
        dev_idx = self.cu_seqlens.device.index
        return (
            _create_scan_seq_map(self.post_num_ranks, dev_idx),
            _create_scan_fb_mask(self.post_num_ranks, num_v_heads, dev_idx),
        )


# ---------------------------------------------------------------------------
# build inter-card context
# ---------------------------------------------------------------------------
@tensor_cache
def _calc_inter_cp_seqs(
    cu_seqlens: torch.LongTensor,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    world_size: int | None = None,
    rank: int | None = None,
    group: "dist.ProcessGroup | None" = None,
    conv1d_kernel_size: int | None = None,
) -> FlashQLACPContext:
    if world_size is None:
        assert group is not None
        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)

    if cu_seqlens_cpu is None:
        cu_seqlens_cpu = cu_seqlens.cpu()
    cu_seqlens_cpu = cu_seqlens_cpu.to(dtype=torch.long)

    total_tokens = cu_seqlens_cpu[-1].item()
    assert total_tokens % world_size == 0, (
        f"inter-card CP requires total tokens ({total_tokens}) divisible by "
        f"world_size ({world_size}); pad/reshape the global sequence to a multiple of world_size."
    )
    part_len = total_tokens // world_size
    rank_start = part_len * rank
    rank_end = rank_start + part_len

    start_seq_idx = torch.searchsorted(cu_seqlens_cpu[1:], rank_start, side="right")
    end_seq_idx = torch.searchsorted(cu_seqlens_cpu[:-1], rank_end, side="left")
    subset_cu_seqlens = cu_seqlens_cpu[start_seq_idx: end_seq_idx + 1]

    local_cu_seqlens_cpu = (
        subset_cu_seqlens.clamp(min=rank_start, max=rank_end) - rank_start
    ).unique_consecutive().to(torch.int32)
    local_cu_seqlens_gpu = local_cu_seqlens_cpu.to(
        device=cu_seqlens.device, non_blocking=True)

    first_seq_global_start = cu_seqlens_cpu[start_seq_idx].item()
    last_seq_global_end = cu_seqlens_cpu[end_seq_idx].item()

    pre_num_conv_tokens = max(0, rank_start - first_seq_global_start)

    first_rank_of_first_seq = first_seq_global_start // part_len
    pre_num_ranks = rank - first_rank_of_first_seq
    is_first_rank = (rank == first_rank_of_first_seq)

    last_rank_of_last_seq = (last_seq_global_end - 1) // part_len
    post_num_ranks = last_rank_of_last_seq - rank
    is_last_rank = (rank == last_rank_of_last_seq)

    return FlashQLACPContext(
        type="inter",
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
) -> FlashQLACPContext:
    return _calc_inter_cp_seqs(
        cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu, group=group,
        conv1d_kernel_size=conv1d_kernel_size)

# ---------------------------------------------------------------------------
# build intra-card context
# ---------------------------------------------------------------------------
@tensor_cache
def _calc_intra_cp_seqs(
    raw_cu_seqlens: torch.LongTensor,
    chunk_size: int,
    num_v_heads: int,
    is_bwd: bool = False,
) -> FlashQLACPContext:
    device = raw_cu_seqlens.device
    seqlen_dtype = raw_cu_seqlens.dtype
    raw_cu_seqlens_list = raw_cu_seqlens.tolist()
    raw_batch_size = len(raw_cu_seqlens_list) - 1
    seqlens = [raw_cu_seqlens_list[i + 1] - raw_cu_seqlens_list[i] for i in range(raw_batch_size)]
    num_chunks = [tilelang.cdiv(x, chunk_size) for x in seqlens]

    # autocp
    H = num_v_heads
    # Latency model: T = a·L_cp + b·(B·H·Lc/P) / L_cp + c
    # Minimizing T yields the theoretical optimum: L_cp* ∝ √(B·H·Lc / P), where P = MULTI_PROCESSOR_COUNT, L_cp = max_local_chunks
    # Scaled by empirical factor (3) and aligned to the nearest power of 2 for optimal SM scheduling & memory alignment.

    max_local_chunks = 2 ** round(
        math.log2(math.sqrt(H * sum(num_chunks) / MULTI_PROCESSOR_COUNT) * 3)
    )

    # Set min to 4 to ensure multi-stage pipelining in fused_gdr;
    max_local_chunks = max(max_local_chunks, 4)

    use_cp = False
    cp_cu_seqlens = []
    ht_mask = []
    ht_mask_bwd = []
    seq_map_c2r = []
    seq_map_r2c = [0]
    max_local_tokens = max_local_chunks * chunk_size
    for i, c in enumerate(num_chunks):
        s = raw_cu_seqlens_list[i]
        e = raw_cu_seqlens_list[i + 1]
        if c > max_local_chunks:
            first = True
            while s < e:
                cp_cu_seqlens.append(s)
                ht_mask.append(False)
                ht_mask_bwd.append(first)
                first = False
                seq_map_c2r.append(i)
                s += max_local_tokens
            ht_mask[-1] = True
        else:
            cp_cu_seqlens.append(s)
            ht_mask.append(True)
            ht_mask_bwd.append(True)
            seq_map_c2r.append(i)
        seq_map_r2c.append(len(cp_cu_seqlens))
    cp_cu_seqlens.append(raw_cu_seqlens_list[-1])

    # Disable CP when sequences are too short or B * H naturally saturates SM occupancy.
    # CP has fixed overhead (warmup + correct_initial_states) that only pays off
    # when the longest sequence has enough chunks to amortize the cost.

    Be = sum(num_chunks) / max(num_chunks)

    if ARCH == "SM90" or ARCH == "SM120":
        use_cp = Be * H <= 40 or (Be * H <= 56 and max(num_chunks) >= 128)
    elif ARCH in ["SM100", "SM103"]:
        # SM100 uses separate thresholds for fwd and bwd:
        # - bwd kernel does more work per chunk (higher arithmetic intensity), so GPU
        #   under-utilization appears at fewer chunks (>=64 vs >=256 for fwd). It also
        #   runs prepare_dh (fused_gdr_dh) which itself benefits from CP parallelism,
        #   further lowering the break-even point.
        # - fwd has two tiers: moderate head count (Be*H<=56) needs very long sequences
        #   (>=256 chunks) to justify CP overhead; very low head count (Be*H<=32) allows
        #   slightly shorter sequences (>=192 chunks).
        if is_bwd:
            use_cp = Be * H <= 56 and max(num_chunks) >= 16
        else:
            use_cp = (Be * H <= 56 and max(num_chunks) >= 256) or (
                Be * H <= 32 and max(num_chunks) >= 192
            )
    else:
        raise ValueError(
            f"FlashQLA now support sm90, sm100 and sm103 only. Found compute version: {_COMPUTE_VERSION}"
        )

    if use_cp:
        cp_cu_seqlens = torch.tensor(
            cp_cu_seqlens, dtype=seqlen_dtype, device=device, requires_grad=False
        )
        seq_map_c2r = torch.tensor(seq_map_c2r, dtype=seqlen_dtype, device=device)
        seq_map_r2c = torch.tensor(
            seq_map_r2c, dtype=seqlen_dtype, device=device, requires_grad=False
        )
        ht_mask = torch.tensor(
            ht_mask, dtype=torch.bool, device=device, requires_grad=False
        )
        ht_mask_bwd = torch.tensor(
            ht_mask_bwd, dtype=torch.bool, device=device, requires_grad=False
        )
    else:
        cp_cu_seqlens, seq_map_r2c, seq_map_c2r, ht_mask, ht_mask_bwd = (
            None, None, None, None, None,
        )

    return FlashQLACPContext(
        type="intra",
        cu_seqlens=raw_cu_seqlens,
        use_intra_cp=use_cp,
        intra_cp_cu_seqlens=cp_cu_seqlens,
        seq_map_r2c=seq_map_r2c,
        seq_map_c2r=seq_map_c2r,
        ht_mask=ht_mask,
        ht_mask_bwd=ht_mask_bwd,
    )

def build_intra_cp_context(
    cp_context: "FlashQLACPContext | None",
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None,
    auto_cp: bool = True,
    is_bwd: bool = False,
) -> "FlashQLACPContext":
    if cp_context is not None:
        return cp_context

    batch_size, num_tokens = k.shape[0], k.shape[1]
    if not auto_cp or batch_size > 1:
        return FlashQLACPContext(type="intra", cu_seqlens=cu_seqlens, use_intra_cp=False)

    num_v_heads = v.shape[2]
    if cu_seqlens is None:
        cu_seqlens = _create_cu_seqlens(batch_size, num_tokens, k.device.index)

    return _calc_intra_cp_seqs(
        raw_cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        num_v_heads=num_v_heads,
        is_bwd=is_bwd,
    )


# ---------------------------------------------------------------------------
# indexing tensors generation
# ---------------------------------------------------------------------------
@tensor_cache
def _create_cu_seqlens(
    batch_size: int,
    num_tokens: int,
    device_idx: int,
):
    return (
        torch.arange((batch_size + 1), dtype=torch.int32, device=f"cuda:{device_idx}")
        * num_tokens
    )


@tensor_cache
def _create_scan_seq_map(num_ranks: int, device_idx: int) -> torch.Tensor:
    seq_map = torch.zeros(2, dtype=torch.int32, device=f"cuda:{device_idx}")
    seq_map[1] = num_ranks + 1
    return seq_map


@tensor_cache
def _create_scan_fb_mask(num_ranks: int, num_v_heads: int, device_idx: int) -> torch.Tensor:
    return torch.ones((num_ranks + 1, num_v_heads), dtype=torch.bool, device=f"cuda:{device_idx}")