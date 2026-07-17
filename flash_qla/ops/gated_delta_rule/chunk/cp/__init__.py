# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Inter-card CP（跨卡上下文并行）。设计见 docs/inter_card_cp.md。

当前主线为**单层** inter-card CP（fla 风格，无 intra 子序列并行）：
  build_cp_context(S1) → inter_card_cp_preprocess_fwd(pre_process + all_gather + inter_scan) → 主前向。
  后向对称：inter_card_cp_preprocess_bwd(prepare_dhm + all_gather + reverse inter_scan) → 主后向。
`reduce_local` 属两层方案（暂 park），单层不使用。
"""

from .scan import reduce_local, inter_scan, seq_reset_from_seq_map
from .context import FLACPContext, build_cp_context, get_cp_cu_seqlens
from .comm import all_gather_into_tensor, pack_hm, unpack_hm
from .preprocess import inter_card_cp_preprocess_fwd
from .preprocess import inter_card_cp_prepare_hm, inter_card_cp_all_gather_hm, inter_card_cp_correct_initial_states
from .preprocess import inter_card_cp_preprocess_bwd
from .preprocess import inter_card_cp_prepare_dhm, inter_card_cp_all_gather_dhm, inter_card_cp_correct_terminal_states
from .function import CPChunkGatedDeltaRuleFunction

__all__ = [
    # S1 上下文/切分
    "FLACPContext", "build_cp_context", "get_cp_cu_seqlens",
    # S2 scan 原语
    "inter_scan", "reduce_local", "seq_reset_from_seq_map",
    # S3 通信 + 打包
    "all_gather_into_tensor", "pack_hm", "unpack_hm",
    # 单层前向 pre_process + autograd Function
    "inter_card_cp_preprocess_fwd",
    "inter_card_cp_prepare_hm", "inter_card_cp_all_gather_hm", "inter_card_cp_correct_initial_states",
    # 单层后向 pre_process
    "inter_card_cp_preprocess_bwd",
    "inter_card_cp_prepare_dhm", "inter_card_cp_all_gather_dhm", "inter_card_cp_correct_terminal_states",
    "CPChunkGatedDeltaRuleFunction",
]
