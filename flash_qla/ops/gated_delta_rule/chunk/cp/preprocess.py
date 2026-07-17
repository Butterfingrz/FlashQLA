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
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回本卡最后一个 chunk 的 `(S_ext_r, M_r)`，用于跨卡 all_gather。
    S_ext_r = ht[-1]，M_r = mt[-1]，均为 fp32。

    若 output_mt_first=True，同时返回首序列的 mt[0]（供 backward 跨卡 scan 用）。
    此时 num_warmup 对首尾两条序列都设全量，一趟 kernel 同时取到。
    """
    assert not state_v_first, "单层 inter-card CP 暂只支持 state_v_first=False"
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
        initial_state=None, output_final_state=True, output_h=False,
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
    """跨卡 all_gather 本 rank 的 (S_ext, M) 到所有 rank，返回邻居的 (S_ext, M)。
    W = group size，H = num_v_heads，K = num_k_heads，V = value dim。
    """
    hm = pack_hm(S_ext_r, M_r)                                  # [H, K, V+K]
    ag_hm, _ = all_gather_into_tensor(hm, group=group)          # [W, H, K, V+K]
    rank = dist.get_rank(group=group)
    pre = cp_context.pre_num_ranks
    neigh = ag_hm[rank - pre: rank]                             # [pre, H, K, V+K]（从前往后）
    S_neigh, M_neigh = unpack_hm(neigh, V)
    return S_neigh, M_neigh

def inter_card_cp_correct_initial_states(
    S_neigh: torch.Tensor,  # [J, H, K, V]，邻居的 S_ext
    M_neigh: torch.Tensor,  # [J, H, K, K]
    Hv: int,
    K: int,
    V: int,
    state_v_first: bool = False,
):
    """返回本卡首序列的跨卡 incoming 初态 `raw_h0`（[N_local, H, K, V] fp32，仅 [0] 非零），
    首 rank 返回 None（序列真正起点，无 incoming）。

    cu_seqlens 取自 `cp_context.cu_seqlens`（rank-local）。
    """
    assert not state_v_first, "单层 inter-card CP 暂只支持 state_v_first=False"
    # 复用 correct_initial_states kernel 做跨卡 exclusive prefix scan。
    # kernel 输出 cp_h0[i] = 处理前 i 个元素后的状态（exclusive），而我们需要处理完
    # 所有 J 个邻居后的结果（inclusive）。在末尾 pad 一个 dummy 元素使 num_iters = J+1，
    # 循环体跑 J 次恰好覆盖真实邻居，cp_h0[J] 即为 inclusive scan 结果。
    # dummy 位置的 ht/mt 不会被 kernel 循环体读取（只在 last_idx 写出）。
    from flash_qla.ops.gated_delta_rule.chunk import correct_initial_states
    
    J = S_neigh.shape[0]
    pad_S = torch.zeros((1, Hv, K, V), dtype=S_neigh.dtype, device=S_neigh.device)
    pad_M = torch.zeros((1, Hv, K, K), dtype=M_neigh.dtype, device=M_neigh.device)
    ht_buf = torch.cat([S_neigh, pad_S], dim=0)                 # [J+1, H, K, V]
    mt_buf = torch.cat([M_neigh, pad_M], dim=0)                 # [J+1, H, K, K]
    fb_mask = torch.ones((J + 1, Hv), dtype=torch.bool, device=S_neigh.device)
    seq_map = torch.tensor([0, J + 1], dtype=torch.int32, device=S_neigh.device)
    cp_h0_all = correct_initial_states(
        raw_h0=None,
        ht_buffer=ht_buf,
        mt_buffer=mt_buf,
        fallback_mask=fb_mask,
        seq_map_r2c=seq_map,
        state_v_first=state_v_first,
    )
    initial_state_r = cp_h0_all[J]                               # [H, K, V] inclusive scan result
    return initial_state_r

def inter_card_cp_preprocess_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,          # A = kkt_solve(...)，[1, T, H, chunk_size]
    g: torch.Tensor,          # 已 chunk_local_cumsum
    beta: torch.Tensor,
    cp_context,               # FLACPContext（rank-local）
    state_v_first: bool = False,
    output_mt_first: bool = False,
):
    """返回本卡首序列的跨卡 incoming 初态 `raw_h0`（[N_local, H, K, V] fp32，仅 [0] 非零），
    首 rank 返回 None（序列真正起点，无 incoming）。

    若 output_mt_first=True，同时返回首序列的 M（供 backward 跨卡 scan 用）。
    返回 `(raw_h0, M_r_first)` 或 `(None, M_r_first)`。
    """
    assert not state_v_first, "单层 inter-card CP 暂只支持 state_v_first=False"

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
    )
    if output_mt_first:
        S_ext_r, M_r, M_r_first = hm_result
    else:
        S_ext_r, M_r = hm_result
        M_r_first = None

    S_ext_r = S_ext_r.float()
    M_r = M_r.float()

    S_neigh, M_neigh = inter_card_cp_all_gather_hm(
        S_ext_r=S_ext_r, M_r=M_r, V=V, group=group, cp_context=cp_context
    )

    if cp_context.is_first_rank:
        if output_mt_first:
            return None, M_r_first
        return None

    initial_state_r = inter_card_cp_correct_initial_states(
        S_neigh=S_neigh, M_neigh=M_neigh, Hv=Hv, K=K, V=V, state_v_first=state_v_first
    )

    raw_h0 = torch.zeros((N, Hv, K, V), dtype=torch.float32, device=k.device)
    raw_h0[0] = initial_state_r
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
) -> torch.Tensor:
    """返回本卡首序列的 `dS_ext_r`（dht=0 时的 dh0），用于跨卡 backward all_gather。

    M_r 由 forward 缓存传入（见 function.py），此处不再重算。
    """
    assert not state_v_first, "单层 inter-card CP 暂只支持 state_v_first=False"
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
        dht=None,
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
    """跨卡 all_gather backward 聚合量，返回 post-neighbors（已翻转为 scan 顺序）。

    forward 取 pre-neighbors [r-pre:r]；backward 取 post-neighbors [r+1:r+1+post]
    并翻转——scan 从最远 rank 开始。
    """
    hm = pack_hm(dS_ext_r, M_r)                         # [H, K, V+K]
    ag_hm, _ = all_gather_into_tensor(hm, group=group)   # [W, H, K, V+K]
    rank = dist.get_rank(group=group)
    post = cp_context.post_num_ranks
    neigh = ag_hm[rank + 1: rank + 1 + post]             # [post, H, K, V+K]
    neigh = neigh.flip(0)
    dS_neigh, M_neigh = unpack_hm(neigh, V)
    return dS_neigh, M_neigh


def inter_card_cp_correct_terminal_states(
    dS_neigh: torch.Tensor,  # [J, H, K, V]，post-neighbors，已翻转
    M_neigh: torch.Tensor,   # [J, H, K, K]，post-neighbors，已翻转
    Hv: int,
    K: int,
    V: int,
    state_v_first: bool = False,
):
    """反向 prefix scan：`dh = M^T @ dh + dS_ext`，等价于用 M^T 做前向 scan。

    复用 `correct_initial_states`（与 forward 相同的 kernel），传入 M^T。
    """
    assert not state_v_first, "单层 inter-card CP 暂只支持 state_v_first=False"
    from flash_qla.ops.gated_delta_rule.chunk import correct_initial_states

    J = dS_neigh.shape[0]
    M_neigh_T = M_neigh.transpose(-1, -2).contiguous()

    pad_dS = torch.zeros((1, Hv, K, V), dtype=dS_neigh.dtype, device=dS_neigh.device)
    pad_M = torch.zeros((1, Hv, K, K), dtype=M_neigh_T.dtype, device=M_neigh_T.device)
    dht_buf = torch.cat([dS_neigh, pad_dS], dim=0)       # [J+1, H, K, V]
    mt_buf = torch.cat([M_neigh_T, pad_M], dim=0)        # [J+1, H, K, K]
    fb_mask = torch.ones((J + 1, Hv), dtype=torch.bool, device=dS_neigh.device)
    seq_map = torch.tensor([0, J + 1], dtype=torch.int32, device=dS_neigh.device)

    cp_dht_all = correct_initial_states(
        raw_h0=None,
        ht_buffer=dht_buf,
        mt_buffer=mt_buf,
        fallback_mask=fb_mask,
        seq_map_r2c=seq_map,
        state_v_first=state_v_first,
    )
    return cp_dht_all[J]  # [H, K, V] inclusive scan result


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
):
    """返回本卡末序列的跨卡 incoming 终态梯度 `corrected_dht`（[N, Hv, K, V] fp32，仅 [N-1] 非零），
    末 rank 返回 None。

    M_r_first: 首序列的传递矩阵 [H, K, K]，由 forward 缓存。
    """
    assert not state_v_first, "单层 inter-card CP 暂只支持 state_v_first=False"

    group = cp_context.group
    cu_cpu = cp_context.cu_seqlens_cpu.tolist()
    N = len(cu_cpu) - 1
    Hv = v.shape[2]
    K = k.shape[3]
    V = v.shape[3]

    dS_ext_r = inter_card_cp_prepare_dhm(
        q=q, k=k, v=v, a=a, g=g, beta=beta, do=do,
        scale=scale, cp_context=cp_context, state_v_first=state_v_first,
    )

    dS_ext_r = dS_ext_r.float()
    M_r_f = M_r_first.float() if M_r_first is not None else torch.zeros(
        (Hv, K, K), dtype=torch.float32, device=k.device)

    dS_neigh, M_neigh = inter_card_cp_all_gather_dhm(
        dS_ext_r=dS_ext_r, M_r=M_r_f, V=V, group=group, cp_context=cp_context,
    )

    if cp_context.is_last_rank:
        return None

    terminal_state_r = inter_card_cp_correct_terminal_states(
        dS_neigh=dS_neigh, M_neigh=M_neigh, Hv=Hv, K=K, V=V, state_v_first=state_v_first,
    )

    corrected_dht = torch.zeros((N, Hv, K, V), dtype=torch.float32, device=k.device)
    corrected_dht[N - 1] = terminal_state_r
    return corrected_dht
