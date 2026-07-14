# 最简 debug 脚本：单进程模拟 build_cp_context / get_cp_cu_seqlens
#
# 运行：
#   python debug_cp_context.py
# 或用 pdb 逐行 debug：
#   python -m pdb debug_cp_context.py
#
# 无需 dist init / 无需 GPU（纯 CPU 索引逻辑）。核心思路：
# get_cp_cu_seqlens 支持单进程模拟——显式传 world_size/rank 且 group=None。

import torch

from flash_qla.ops.gated_delta_rule.chunk.cp.context import get_cp_cu_seqlens


def main():
    world_size = 4

    # 全局 varlen cu_seqlens（前缀和形式）。这里造 3 条序列：长度 8 / 16 / 8，共 32 token。
    # total_tokens=32 能被 world_size=4 整除 -> part_len=8，每张卡拥有 8 个连续 token。
    cu_seqlens = torch.tensor([0, 4096, 24576, 32768], dtype=torch.long)

    print(f"global cu_seqlens = {cu_seqlens.tolist()}")
    print(f"world_size = {world_size}, part_len = {cu_seqlens[-1].item() // world_size}\n")

    for rank in range(world_size):
        # group=None + 显式 world_size/rank => 单进程模拟某个 rank
        ctx = get_cp_cu_seqlens(
            cu_seqlens,
            world_size=world_size,
            rank=rank,
            group=None,
        )

        print(f"===== rank {rank} =====")
        print(f"  local cu_seqlens : {ctx.cu_seqlens_cpu.tolist()}")
        print(f"  num_seqs         : {ctx.num_seqs}")
        print(f"  is_first_rank    : {ctx.is_first_rank}")
        print(f"  is_last_rank     : {ctx.is_last_rank}")
        print(f"  pre_num_ranks    : {ctx.pre_num_ranks}")
        print(f"  post_num_ranks   : {ctx.post_num_ranks}")
        print(f"  pre_num_conv_tok : {ctx.pre_num_conv_tokens}")
        print()


if __name__ == "__main__":
    main()
