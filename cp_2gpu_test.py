# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""单层 inter-card CP 真多卡验证（独立脚本，不污染 tests）。

  torchrun --nproc_per_node=2 cp_2gpu_test.py

各 rank 用**同一 seed** 生成同一条全局序列，切自己的 token 片，用 build_cp_context 建上下文，
走 chunk_gated_delta_rule(cp_context=...)；输出与**单卡** auto_cp=False golden 的对应片对齐。
"""

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from flash_qla.ops.gated_delta_rule import chunk_gated_delta_rule
from flash_qla.ops.gated_delta_rule.chunk.cp import build_cp_context

# ===== 可配置（直接改这里做轻量调试）=====
RTOL = 1e-2
SEED = 1234
DTYPE = torch.bfloat16
# 每个 case：(全局 cu_seqlens, Hk, Hv, g_scale)
CASES = [
    ([0, 2048], 2, 2, 1.0 / 16),          # 单序列跨全卡（弱衰减）
    ([0, 512, 1536, 2048], 2, 2, 1.0 / 16),  # 多序列含跨卡
    ([0, 2048], 2, 4, 1.0 / 16),          # GQA
    ([0, 2048], 2, 2, 1.0),               # 单序列跨全卡（中衰减）
    ([0, 512, 1536, 2048], 2, 2, 1.0),
    ([0, 2048], 2, 4, 1.0),
]
# =========================================


def run_case(cu_global, Hk, Hv, g_scale, dev, rank, W, seed=SEED):
    T = cu_global[-1]
    K = V = 128
    torch.manual_seed(seed)  # 所有 rank 同 seed → 同一全局序列
    q = F.normalize(torch.randn(1, T, Hk, K, device=dev, dtype=DTYPE), p=2, dim=-1)
    k = F.normalize(torch.randn(1, T, Hk, K, device=dev, dtype=DTYPE), p=2, dim=-1)
    v = torch.randn(1, T, Hv, V, device=dev, dtype=DTYPE)
    beta = torch.randn(1, T, Hv, device=dev, dtype=torch.float32).sigmoid()
    g = F.logsigmoid(torch.randn(1, T, Hv, device=dev, dtype=torch.float32)) * g_scale
    scale = K ** -0.5
    cu_g = torch.tensor(cu_global, device=dev, dtype=torch.int32)

    # 单卡 golden（整条，串行）
    o_ref, _ = chunk_gated_delta_rule(
        q, k, v, g, beta, scale=scale, cu_seqlens=cu_g,
        output_final_state=False, auto_cp=False)

    # CP：本 rank token 片 [lo,hi)
    part = T // W
    lo, hi = rank * part, (rank + 1) * part
    ctx = build_cp_context(cu_g, group=dist.group.WORLD)
    o_loc, _ = chunk_gated_delta_rule(
        q[:, lo:hi], k[:, lo:hi], v[:, lo:hi], g[:, lo:hi], beta[:, lo:hi],
        scale=scale, cp_context=ctx, output_final_state=False)

    ref_slice = o_ref[:, lo:hi]
    err = (o_loc.float() - ref_slice.float()).abs().max().item()
    ref = ref_slice.float().abs().max().item()
    ratio = err / (ref + 1e-30)

    t = torch.tensor([ratio], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    max_ratio = t.item()
    ok = max_ratio <= RTOL
    if rank == 0:
        print(f"[{'PASS' if ok else 'FAIL'}] cu={cu_global} Hk/Hv={Hk}/{Hv} "
              f"g×{g_scale:<7} max_ratio={max_ratio:.2e}")
    return ok


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    W = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dev = f"cuda:{local}"

    if rank == 0:
        print("=" * 80)
        print(f"单层 inter-card CP 真 {W} 卡验证（vs 单卡 auto_cp=False golden，rtol={RTOL}）")
        print("=" * 80)

    results = []
    for cu_global, Hk, Hv, gs in CASES:
        results.append(run_case(cu_global, Hk, Hv, gs, dev, rank, W))
    if rank == 0:
        n = sum(results)
        print("-" * 80)
        print(f"总计 {n}/{len(results)} 通过")
    dist.destroy_process_group()
    if rank == 0 and sum(results) != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
