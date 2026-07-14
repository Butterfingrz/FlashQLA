# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""单层 inter-card CP 的 autograd Function（当前主线）。

把 inter-card 整条前向逻辑收拢于此，与核心 `ChunkGatedDeltaRuleFunction`（intra/非 CP）
**完全分离**。公共入口 `chunk_gated_delta_rule(cp_context=...)` 分派到这里。

前向：自算 g_c/A → `inter_card_cp_preprocess_fwd`（跨卡 pre+all_gather+inter_scan）→
主前向 `fused_gdr_fwd`（无 intra 子序列并行）。后向下一轮实现。
"""

from __future__ import annotations

import torch

from flash_qla.utils import input_guard, l2norm_fwd
from flash_qla.ops.utils import chunk_local_cumsum

from .preprocess import inter_card_cp_preprocess_fwd


class CPChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        state_v_first: bool,
        use_qk_l2norm_in_kernel: bool,
        cp_context,
    ):
        # 懒导入 chunk/__init__ 的符号，避免循环依赖
        from flash_qla.ops.gated_delta_rule.chunk import CHUNK_SIZE, kkt_solve, fused_gdr_fwd

        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        cu = cp_context.cu_seqlens
        g = chunk_local_cumsum(g=g, cu_seqlens=cu, chunk_size=CHUNK_SIZE)
        A = kkt_solve(k=k, b=beta, cu_seqlens=cu, chunk_size=CHUNK_SIZE)

        # 跨卡还原本卡首序列 incoming 初态（首 rank 为 None）
        raw_h0 = inter_card_cp_preprocess_fwd(
            k=k, v=v, a=A, g=g, beta=beta,
            cp_context=cp_context, state_v_first=state_v_first,
        )

        o, _, _ = fused_gdr_fwd(
            q=q, k=k, v=v, a=A, g=g, b=beta,
            scale=scale,
            initial_state=raw_h0,
            output_final_state=False,
            output_h=False,
            output_o=True,
            cu_seqlens=cu,
            cp_seq_map=None,
            raw_cu_seqlens=None,
            state_v_first=state_v_first,
        )

        # 保存供后向（下一轮实现）
        ctx.save_for_backward(q, k, q_rstd, k_rstd, v, g, beta, A, cu)
        ctx.scale = scale
        ctx.state_v_first = state_v_first
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.cp_context = cp_context
        return o.to(q.dtype)

    @staticmethod
    @input_guard
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, do: torch.Tensor):
        raise NotImplementedError(
            "inter-card CP backward 尚未实现（当前只支持前向）。"
            "若需梯度，请暂勿在 cp_context 前向后调用 backward。"
        )
