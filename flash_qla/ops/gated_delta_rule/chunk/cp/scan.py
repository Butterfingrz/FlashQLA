# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Inter-card CP 的两级 blocked scan 原语（torch）。

设计见 docs/inter_card_cp.md §3/§7。这里只有阶段一的两个跨块小算子：

  - ``reduce_local`` （阶段 1a）：把本卡所有 sub-seq 的小 ``(mt, ht)`` 归约成本卡
    末条序列的聚合 ``(M_r, S_ext_r)``，用于跨卡 all-gather。
  - ``inter_scan``   （阶段 1b）：对 all-gather 到的各卡聚合做 exclusive prefix
    scan，还原本卡首条序列的 incoming 状态。

阶段二（卡内 scan）直接复用现有 ``correct_initial_states`` / ``correct_terminal_states``，
不在此文件。

线性递推 block = ``(M, b)``，语义 ``state_out = M @ state_in + b``。结合律
（a 先、b 后）：``combine(a, b) = (M_b @ M_a, M_b @ b_a + b_b)``。

这两个算子**保留输入 dtype**（不隐式降精度）。生产路径应传入 **fp32** 缓冲（M 链在
bf16 下会累积显著误差，见文档 §2.4）；L0 纯数学验证传入 fp64 以获得机器精度对拍。

张量按 head 批处理：``M[H, K, K]``、``b/S[H, K, V]``。
"""

from __future__ import annotations

import torch


def seq_reset_from_seq_map(seq_map_r2c, num_subseqs: int) -> list:
    """由 ``seq_map_r2c``（raw→cp 边界，[raw_batch+1]）推出每个 sub-seq 是否起一条新 raw 序列。"""
    starts = set(int(x) for x in seq_map_r2c[:-1].tolist())
    return [j in starts for j in range(num_subseqs)]


def reduce_local(mt, ht, seq_reset, fallback=None):
    """阶段 1a：把本卡各 sub-seq 的 ``(mt, ht)`` 归约成本卡末条 raw 序列的聚合 ``(M_r, S_ext_r)``。

    尊重 raw 序列边界（``seq_reset[i]=True`` 表示 sub-seq i 起新 raw 序列，清零累加器）；
    非 fallback（强衰减）的 block 把 incoming 视作 0（M 项置零），但自身 ht 仍累加。

    .. important::
        输入必须是**精确/足量 warmup** 的 per-subseq ``(M, f)``（诊断：全量 warmup 时本函数
        链乘 == 整段精确，ratio 0~2e-3）。**不要**喂 intra-card 为卡内 h0 重建而**截断**的
        warmup 缓冲（``get_warmup_chunks`` 只跑尾部 chunk）——截断缓冲缺前段贡献，链乘会崩
        （见 inter_card_cp_sim.py 的 L2b 探针）。共用 prepare_h 时，inter-card 用于聚合的
        warmup 需按自身尺度配足（见 docs/inter_card_cp.md §3）。

    Args:
        mt: ``[P, H, K, K]`` 各 sub-seq 精确 M（转移矩阵）。fp32（生产）/ fp64（L0）。
        ht: ``[P, H, K, V]`` 各 sub-seq 精确 f（h0=0 终态）。
        seq_reset: 长度 P 的 bool 序列。
        fallback: ``[P, H]`` bool 或 None（None 等价全 True，即精确）。

    Returns:
        ``(M_r[H, K, K], S_ext_r[H, K, V])``，dtype 同输入。
    """
    P, H, K, _ = mt.shape
    V = ht.shape[-1]
    dtype = mt.dtype
    eye = torch.eye(K, dtype=dtype, device=mt.device).expand(H, K, K)
    M = eye.clone()
    b = torch.zeros(H, K, V, dtype=dtype, device=ht.device)
    for i in range(P):
        if seq_reset[i]:
            M = eye.clone()
            b = torch.zeros(H, K, V, dtype=dtype, device=ht.device)
        Mi = mt[i]
        if fallback is not None:
            Mi = Mi * fallback[i].to(dtype).view(H, 1, 1)
        b = ht[i] + Mi @ b
        M = Mi @ M
    return M, b


def inter_scan(neigh_M, neigh_S):
    """阶段 1b：跨卡 exclusive prefix scan，还原本卡首条序列的 incoming 状态。

    调用方已按链式顺序选出邻居卡的聚合：
      - 前向：ranks ``[r-pre_num_ranks .. r-1]``，从前往后；
      - 后向：ranks ``[r+post_num_ranks .. r+1]``，由调用方翻转成从后往前。

    Args:
        neigh_M: ``[J, H, K, K]`` 按链式顺序的邻居 M。
        neigh_S: ``[J, H, K, V]`` 按链式顺序的邻居 S_ext。

    Returns:
        ``S[H, K, V]``（dtype 同输入）：本卡首条序列的 incoming 状态（J==0 时为 0）。
    """
    J, H, K, _ = neigh_M.shape
    V = neigh_S.shape[-1]
    S = torch.zeros(H, K, V, dtype=neigh_S.dtype, device=neigh_M.device)
    for j in range(J):
        S = neigh_M[j] @ S + neigh_S[j]
    return S
