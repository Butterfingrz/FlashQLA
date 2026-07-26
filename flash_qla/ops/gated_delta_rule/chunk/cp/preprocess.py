# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""单层 inter-card CP 的前向/后向 pre_process（当前主线）。

前向：每张卡在自己的切片上算精确聚合 `(M_r, S_ext_r)`（h0=0）→ all_gather → `inter_scan`
还原本卡首序列的 incoming 状态 `initial_state_r`。
后向：对称——每张卡算 `(dS_ext_r, M_r^T)` → all_gather → 反向 `inter_scan`
还原本卡末序列的 terminal 梯度 `dht_r`。

设计见 docs/inter_card_cp.md「单层方案」§3.4。
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from .comm import all_gather_into_tensor, pack_hm, unpack_hm

def inter_card_cp_prepare_hm(
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,          # A = kkt_solve(...)，[1, T, H, chunk_size]
    g: torch.Tensor,          # 已 chunk_local_cumsum
    beta: torch.Tensor,
    cp_context,               # FLACPContext（rank-local）
    state_v_first: bool = False,
    output_mt_first: bool = False,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回本卡最后一个 chunk 的 `(S_ext_r, M_r)`，用于跨卡 all_gather。

    initial_state: 用户初态（仅 first rank 有意义），传给 fused_gdr_h 使 S_ext 包含 h0 的贡献。
    """

    from flash_qla.ops.gated_delta_rule.chunk import fused_gdr_h

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
    S_ext_r: torch.Tensor,  # [H, K, V]，本卡最后一个 chunk 的 S_ext
    M_r: torch.Tensor,      # [H, K, K]，本卡最后一个 chunk 的 M
    V: int,                # value dim
    group: dist.ProcessGroup,
    cp_context,  # FLACPContext（rank-local）
) -> tuple[torch.Tensor, torch.Tensor]:
    """跨卡 all_gather，返回 J+1 个 (S, M)：前 J 个是 pre-neighbors，
    最后一个是本 rank 自身（exclusive scan 不读 last element）。
    """
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

    from flash_qla.ops.gated_delta_rule.chunk import correct_initial_states

    Hv = S_buf.shape[1]
    seq_map, fb_mask = cp_context.get_fwd_scan_tensors(Hv, S_buf.device)
    cp_h0_all = correct_initial_states(
        raw_h0=None,
        ht_buffer=S_buf,
        mt_buffer=M_buf,
        fallback_mask=fb_mask,
        seq_map_r2c=seq_map,
        state_v_first=state_v_first,
    )
    return cp_h0_all[S_buf.shape[0] - 1]

def inter_card_cp_preprocess_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,          # A = kkt_solve(...)，[1, T, H, chunk_size]
    g: torch.Tensor,          # 已 chunk_local_cumsum
    beta: torch.Tensor,
    cp_context,               # FLACPContext（rank-local）
    state_v_first: bool = False,
    output_mt_first: bool = False,
    initial_state: torch.Tensor | None = None,
):
    """返回本卡首序列的跨卡 incoming 初态 `raw_h0`（[N_local, H, K, V] fp32，仅 [0] 非零），
    首 rank 返回 initial_state（序列真正起点）。

    若 output_mt_first=True，同时返回首序列的 M（供 backward 跨卡 scan 用）。
    返回 `(raw_h0, M_r_first)` 或 `(None, M_r_first)`。
    """


    group = cp_context.group
    cu = cp_context.cu_seqlens
    cu_cpu = cp_context.cu_seqlens_cpu.tolist()
    N = len(cu_cpu) - 1
    Hv = v.shape[2]
    K = k.shape[3]
    V = v.shape[3]

    hm_result = inter_card_cp_prepare_hm(
        k=k, v=v, a=a, g=g, beta=beta, cp_context=cp_context,
        state_v_first=state_v_first, output_mt_first=output_mt_first,
        initial_state=initial_state,
    )
    if output_mt_first:
        S_ext_r, M_r, M_r_first = hm_result
    else:
        S_ext_r, M_r = hm_result
        M_r_first = None

    S_ext_r = S_ext_r.float()
    M_r = M_r.float()

    S_buf, M_buf = inter_card_cp_all_gather_hm(
        S_ext_r=S_ext_r, M_r=M_r, V=V, group=group, cp_context=cp_context
    )

    # first rank does not need correction
    if cp_context.is_first_rank:
        if output_mt_first:
            return initial_state, M_r_first
        return initial_state

    initial_state_r = inter_card_cp_correct_initial_states(
        S_buf=S_buf, M_buf=M_buf, cp_context=cp_context,
        state_v_first=state_v_first,
    )

    if state_v_first:
        raw_h0 = torch.zeros((N, Hv, V, K), dtype=torch.float32, device=k.device)
    else:
        raw_h0 = torch.zeros((N, Hv, K, V), dtype=torch.float32, device=k.device)
    # first partial sequence set to initial_state_r
    raw_h0[0] = initial_state_r
    # other sequences set to initial_state
    if initial_state is not None and N > 1:
        raw_h0[1:] = initial_state[1:]
    if output_mt_first:
        return raw_h0, M_r_first
    return raw_h0


# ============================================================================
# Backward
# ============================================================================

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

    from flash_qla.ops.gated_delta_rule.chunk import fused_gdr_dh

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
    M_r: torch.Tensor,       # [H, K, K]（首序列的 M，未转置）
    V: int,
    group: dist.ProcessGroup,
    cp_context,
) -> tuple[torch.Tensor, torch.Tensor]:
    """跨卡 all_gather backward，返回 J+1 个 (dS, M)：第一个是本 rank 自身
    （reverse exclusive scan 不读 element 0），后 J 个是 post-neighbors（自然顺序）。
    """
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

    from flash_qla.ops.gated_delta_rule.chunk import correct_terminal_states

    Hv = dS_buf.shape[1]
    seq_map, fb_mask = cp_context.get_bwd_scan_tensors(Hv, dS_buf.device)
    cp_dht_all = correct_terminal_states(
        raw_dht=None,
        dht_buffer=dS_buf,
        mt_buffer=M_buf,
        fallback_mask=fb_mask,
        seq_map_r2c=seq_map,
        state_v_first=state_v_first,
    )
    return cp_dht_all[0]


def inter_card_cp_preprocess_bwd(
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
    M_r_first: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
):
    """返回本卡末序列的跨卡 incoming 终态梯度 `corrected_dht`（[N, Hv, K, V] fp32，仅 [N-1] 非零），
    末 rank 返回 None。

    M_r_first: 首序列的传递矩阵 [H, K, K]，由 forward 缓存。
    """


    group = cp_context.group
    cu_cpu = cp_context.cu_seqlens_cpu.tolist()
    N = len(cu_cpu) - 1
    Hv = v.shape[2]
    K = k.shape[3]
    V = v.shape[3]

    dS_ext_r = inter_card_cp_prepare_dhm(
        q=q, k=k, v=v, a=a, g=g, beta=beta, do=do,
        scale=scale, cp_context=cp_context, state_v_first=state_v_first,
        dht=dht,
    )

    dS_ext_r = dS_ext_r.float()
    M_r_f = M_r_first.float() if M_r_first is not None else torch.zeros(
        (Hv, K, K), dtype=torch.float32, device=k.device)

    dS_buf, M_buf = inter_card_cp_all_gather_dhm(
        dS_ext_r=dS_ext_r, M_r=M_r_f, V=V, group=group, cp_context=cp_context,
    )

    if cp_context.is_last_rank:
        return None

    terminal_state_r = inter_card_cp_correct_terminal_states(
        dS_buf=dS_buf, M_buf=M_buf, cp_context=cp_context,
        state_v_first=state_v_first,
    )

    if state_v_first:
        corrected_dht = torch.zeros((N, Hv, V, K), dtype=torch.float32, device=k.device)
    else:
        corrected_dht = torch.zeros((N, Hv, K, V), dtype=torch.float32, device=k.device)
    corrected_dht[N - 1] = terminal_state_r
    if dht is not None and N > 1:
        corrected_dht[:N - 1] = dht[:N - 1]
    return corrected_dht
