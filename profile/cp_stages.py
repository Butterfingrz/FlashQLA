# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
from __future__ import annotations

import torch
import torch.distributed as dist

from flash_qla.ops.gated_delta_rule.chunk import (
    fused_gdr_h,
    fused_gdr_dh,
    correct_initial_states,
    correct_terminal_states,
)
from flash_qla.ops.gated_delta_rule.chunk.cp.comm import (
    all_gather_into_tensor,
    pack_hm,
    unpack_hm,
)


# ---------------------------------------------------------------------------
# Forward stages
# ---------------------------------------------------------------------------
def inter_card_cp_prepare_hm(
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,          # A = kkt_solve(...)，[1, T, H, chunk_size]
    g: torch.Tensor,
    beta: torch.Tensor,
    cp_context,               # FlashQLACPContext（rank-local）
    state_v_first: bool = False,
    output_mt_first: bool = False,
    initial_state: torch.Tensor | None = None,
):
    cu = cp_context.cu_seqlens
    cu_cpu = cp_context.cu_seqlens_cpu.tolist()
    N = len(cu_cpu) - 1
    Hv = v.shape[2]
    chunk_size = a.shape[-1]

    num_warmup = torch.zeros((N, Hv), dtype=cu.dtype, device=k.device)
    num_warmup[-1, :] = (cu_cpu[-1] - cu_cpu[-2] + chunk_size - 1) // chunk_size
    if output_mt_first and N > 1:
        num_warmup[0, :] = (cu_cpu[1] - cu_cpu[0] + chunk_size - 1) // chunk_size

    _, ht, mt = fused_gdr_h(
        k=k, v=v, a=a, g=g, b=beta,
        initial_state=initial_state, output_final_state=True, output_h=False,
        cu_seqlens=cu, num_warmup_chunks=num_warmup, state_v_first=state_v_first,
    )
    if output_mt_first:
        return ht[-1], mt[-1], mt[0]
    return ht[-1], mt[-1]


def inter_card_cp_all_gather_hm(
    S_ext_r: torch.Tensor,  # [H, K, V]
    M_r: torch.Tensor,      # [H, K, K]
    V: int,                 # value dim
    group: dist.ProcessGroup,
    cp_context,             # FlashQLACPContext（rank-local）
) -> tuple[torch.Tensor, torch.Tensor]:
    hm = pack_hm(S_ext_r, M_r)                                  # [H, K, V+K]
    ag_hm, _ = all_gather_into_tensor(hm, group=group)          # [W, H, K, V+K]
    rank = dist.get_rank(group=group)
    pre = cp_context.pre_num_ranks
    buf = ag_hm[rank - pre: rank + 1]                           # [pre+1, H, K, V+K]
    S_buf, M_buf = unpack_hm(buf, V)
    return S_buf, M_buf


def inter_card_cp_correct_initial_states(
    S_buf: torch.Tensor,  # [J+1, H, K, V]
    M_buf: torch.Tensor,  # [J+1, H, K, K]
    cp_context,
    state_v_first: bool = False,
):
    Hv = S_buf.shape[1]
    seq_map, fb_mask = cp_context.get_fwd_scan_tensors(Hv)
    cp_h0_all = correct_initial_states(
        raw_h0=None,
        ht_buffer=S_buf,
        mt_buffer=M_buf,
        fallback_mask=fb_mask,
        seq_map_r2c=seq_map,
        state_v_first=state_v_first,
    )
    return cp_h0_all[S_buf.shape[0] - 1]


# ---------------------------------------------------------------------------
# Backward stages
# ---------------------------------------------------------------------------
def inter_card_cp_prepare_dhm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    cp_context,
    state_v_first: bool = False,
    dht: torch.Tensor | None = None,
) -> torch.Tensor:
    cu = cp_context.cu_seqlens
    cu_cpu = cp_context.cu_seqlens_cpu.tolist()
    N = len(cu_cpu) - 1
    Hv = v.shape[2]
    chunk_size = a.shape[-1]

    num_warmup = torch.zeros((N, Hv), dtype=cu.dtype, device=k.device)
    num_warmup[0, :] = (cu_cpu[1] - cu_cpu[0] + chunk_size - 1) // chunk_size

    _, dh0_buffer = fused_gdr_dh(
        q=q, k=k, a=a, g=g, b=beta, do=do,
        dht=dht,
        output_dh0=True,
        output_dh=False,
        scale=scale,
        cu_seqlens=cu,
        num_warmup_chunks=num_warmup,
        state_v_first=state_v_first,
    )
    return dh0_buffer[0]  # [H, K, V]


def inter_card_cp_all_gather_dhm(
    dS_ext_r: torch.Tensor,  # [H, K, V]
    M_r: torch.Tensor,       # [H, K, K]
    V: int,
    group: dist.ProcessGroup,
    cp_context,
) -> tuple[torch.Tensor, torch.Tensor]:
    hm = pack_hm(dS_ext_r, M_r)                         # [H, K, V+K]
    ag_hm, _ = all_gather_into_tensor(hm, group=group)   # [W, H, K, V+K]
    rank = dist.get_rank(group=group)
    post = cp_context.post_num_ranks
    buf = ag_hm[rank: rank + 1 + post]                   # [post+1, H, K, V+K]
    dS_buf, M_buf = unpack_hm(buf, V)
    return dS_buf, M_buf


def inter_card_cp_correct_terminal_states(
    dS_buf: torch.Tensor,  # [J+1, H, K, V]
    M_buf: torch.Tensor,   # [J+1, H, K, K]
    cp_context,
    state_v_first: bool = False,
):
    Hv = dS_buf.shape[1]
    seq_map, fb_mask = cp_context.get_bwd_scan_tensors(Hv)
    cp_dht_all = correct_terminal_states(
        raw_dht=None,
        dht_buffer=dS_buf,
        mt_buffer=M_buf,
        fallback_mask=fb_mask,
        seq_map_r2c=seq_map,
        state_v_first=state_v_first,
    )
    return cp_dht_all[0]
