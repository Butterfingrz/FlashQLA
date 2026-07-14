# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""单层 inter-card CP 的前向 pre_process（当前主线）。

每张卡在自己的切片上：算精确聚合 `(M_r, S_ext_r)`（h0=0）→ all_gather → `inter_scan`
还原本卡首序列的 incoming 状态 `initial_state_r`。随后主前向以它为种子跑普通串行前向
（不做 intra 子序列并行）。设计见 docs/inter_card_cp.md「单层方案」。

聚合复用现有 `fused_gdr_h`（全量 warmup → 精确 `(ht=S_ext, mt=M)`），不新写 kernel。
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from .comm import all_gather_into_tensor, pack_hm, unpack_hm
from .scan import inter_scan


def inter_card_cp_preprocess_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,          # A = kkt_solve(...)，[1, T, H, chunk_size]
    g: torch.Tensor,          # 已 chunk_local_cumsum
    beta: torch.Tensor,
    cp_context,               # FLACPContext（rank-local）
    state_v_first: bool = False,
):
    """返回本卡首序列的跨卡 incoming 初态 `raw_h0`（[N_local, H, K, V] fp32，仅 [0] 非零），
    首 rank 返回 None（序列真正起点，无 incoming）。

    cu_seqlens 取自 `cp_context.cu_seqlens`（rank-local）。
    """
    assert not state_v_first, "单层 inter-card CP 暂只支持 state_v_first=False"
    # 延迟导入，避免与 chunk/__init__ 循环依赖
    from flash_qla.ops.gated_delta_rule.chunk import fused_gdr_h

    group = cp_context.group
    cu = cp_context.cu_seqlens
    cu_cpu = cp_context.cu_seqlens_cpu.tolist()
    N = len(cu_cpu) - 1
    Hv = v.shape[2]
    K = k.shape[3]
    V = v.shape[3]
    chunk_size = a.shape[-1]

    # ① 精确聚合：全量 warmup（每 raw 序列的 chunk 数）→ fused_gdr_h 得 (ht=S_ext, mt=M)
    nchunks = [(cu_cpu[i + 1] - cu_cpu[i] + chunk_size - 1) // chunk_size for i in range(N)]
    num_warmup = torch.empty((N, Hv), dtype=cu.dtype, device=k.device)
    for i in range(N):
        num_warmup[i, :] = nchunks[i]

    _, ht, mt = fused_gdr_h(
        k=k, v=v, a=a, g=g, b=beta,
        initial_state=None, output_final_state=True, output_h=False,
        cu_seqlens=cu, num_warmup_chunks=num_warmup, state_v_first=state_v_first,
    )
    # 末条 raw 序列的聚合（varlen 边界重置性质：整片聚合 == 末序列聚合）
    S_ext_r = ht[-1].float()   # [H, K, V]
    M_r = mt[-1].float()       # [H, K, K]

    # ② 跨卡交换：pack → all_gather
    hm = pack_hm(S_ext_r, M_r)                                  # [H, K, V+K]
    ag_hm, _ = all_gather_into_tensor(hm, group=group)          # [W, H, K, V+K]

    # ③ inter_scan：首序列若为跨 rank 延续，链前 pre_num_ranks 张卡还原 incoming
    if cp_context.is_first_rank:
        return None
    rank = dist.get_rank(group=group)
    pre = cp_context.pre_num_ranks
    neigh = ag_hm[rank - pre: rank]                             # [pre, H, K, V+K]（从前往后）
    S_neigh, M_neigh = unpack_hm(neigh, V)
    initial_state_r = inter_scan(M_neigh, S_neigh)              # [H, K, V] fp32

    raw_h0 = torch.zeros((N, Hv, K, V), dtype=torch.float32, device=k.device)
    raw_h0[0] = initial_state_r
    return raw_h0
